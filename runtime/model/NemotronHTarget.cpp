#include "model/NemotronHTarget.hpp"

#include "model/ExpertStore.hpp"
#include "ops/Mamba2.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace splash::model {

namespace {

constexpr uint32_t kVerifyRows = ExecutionLimits::targetVerifyRows;
// A full batch's verify rows, which the decode scratch holds (a step's storage
// lanes are at most a batch's, QwenTarget::decodeStorageLanes).
constexpr uint64_t kScratchRows = uint64_t{ExecutionLimits::maximumBatchWidth} * kVerifyRows;
// A prefill chunk's rows and the two slices a sequence's Mamba-2 rows run
// past its last row: the last slice's and the fold's (addPrefillMamba).
constexpr uint64_t kPrefillScratchRows = uint64_t{ExecutionLimits::prefillTokenBudget} + 2 * kVerifyRows;
constexpr uint64_t kAlignment = 256;

uint64_t alignUp(uint64_t bytes) { return (bytes + kAlignment - 1) / kAlignment * kAlignment; }

// The bytes of each tensor of rows the decode and prefill scratch hold, in
// order: packed, y, gated, latent, MoE latent, shared mid.
std::array<uint64_t, 6> rowTensorBytes(uint64_t rows) {
  const NemotronHLayout layout;
  return {rows * layout.mambaInProjectionRows * 2, rows * layout.mambaInnerSize * 4,
          rows * layout.mambaInnerSize * 2,        rows * layout.moeLatentSize * 2,
          rows * layout.moeLatentSize * 2,         rows * layout.sharedIntermediateSize * 2};
}

std::vector<uint64_t> decodeTensorBytes(const QwenTargetGeometry &) {
  const auto rows = rowTensorBytes(kScratchRows);
  return {rows.begin(), rows.end()};
}

// The prefill scratch's: its rows', then two one-layer state cells
// (convolution, recurrent each) and a one-layer tape of two slots.
std::vector<uint64_t> prefillTensorBytes(const QwenTargetGeometry &g) {
  const auto rows = rowTensorBytes(kPrefillScratchRows);
  std::vector<uint64_t> bytes(rows.begin(), rows.end());
  const ops::GdnStateStrides state = g.stateLayout.strides();
  for (int cell = 0; cell < 2; ++cell) bytes.insert(bytes.end(), {state.convolutionLayerBytes, state.recurrentLayerBytes});
  bytes.push_back(2 * mamba2TapeLayerBytes(g));
  return bytes;
}

uint64_t totalBytes(const std::vector<uint64_t> &sizes) {
  uint64_t total = 0;
  for (const uint64_t bytes : sizes) total += alignUp(bytes);
  return total;
}

std::vector<metal::MetalBuffer> place(metal::MetalBackend &backend, const metal::MetalBuffer &buffer,
                                      const std::vector<uint64_t> &sizes) {
  std::vector<metal::MetalBuffer> views;
  uint64_t cursor = 0;
  for (const uint64_t bytes : sizes) {
    views.push_back(backend.view(buffer, cursor, bytes));
    cursor += alignUp(bytes);
  }
  return views;
}

// The buffers a latent MoE block runs on: its rows' normalized input, the
// layer's input it adds to and its output, a row of hidden values it sums
// the shared expert into; the runtime's MoE scratch, each lane's verify rows
// (a verify's) and the target's latent rows.
struct MoeRun final {
  metal::CommandGraph &graph;
  uint32_t rows;
  metal::MetalBuffer normalized, input, output, sum, verifyRows;
  const ops::MoeScratch &moe;
  metal::MetalBuffer latent, moeLatent, sharedMid;
};

// The router reads the normalized rows, the routed experts run in the latent
// of the block's down projection (ops::MoE), and the shared expert reads the
// normalized rows: output = input + shared + latent up of the routed
// experts' sum, with `project(projection, in, out, residual)` the step's
// projections.
template <class Project>
void addMoe(const MoeRun &run, const NemotronMoeWeights &block, const ops::MoePlan &plan, Project project) {
  project(block.latentDown, run.normalized, run.latent, metal::MetalBuffer{});
  ops::MoE::add(run.graph,
                {.input = run.latent, .output = run.moeLatent, .scratch = run.moe, .verifyRows = run.verifyRows,
                 .routerInput = run.normalized},
                block.experts, plan);
  project(block.sharedUp, run.normalized, run.sharedMid, metal::MetalBuffer{});
  ops::MoE::addReluSquared(run.graph, run.sharedMid, uint64_t{run.rows} * block.sharedUp.outputSize);
  project(block.sharedDown, run.sharedMid, run.sum, run.input);
  project(block.latentUp, run.moeLatent, run.output, run.sum);
}

} // namespace

NemotronHTarget::NemotronHTarget(const NemotronHWeights &weights, const QwenTargetGeometry &geometry,
                                 metal::MetalBackend &backend, const ops::ExecutionPlans &operators)
    : QwenTarget(weights, geometry, backend, operators), weights_(weights) {
  // ops::Mamba2 keeps the state in fp32.
  if (geometry_.stateLayout.format != ops::GdnStateFormat::Float32)
    throw std::invalid_argument("Nemotron-H keeps its Mamba-2 state in fp32");
  const std::vector<uint64_t> decodeSizes = decodeTensorBytes(geometry_), prefillSizes = prefillTensorBytes(geometry_);
  decode_ = backend_.allocateBuffer(totalBytes(decodeSizes), metal::BufferStorage::Shared, "nemotron-h-decode-scratch");
  prefill_ =
      backend_.allocateBuffer(totalBytes(prefillSizes), metal::BufferStorage::Shared, "nemotron-h-prefill-scratch");
  const std::vector<metal::MetalBuffer> d = place(backend_, decode_, decodeSizes);
  scratch_ = {d[0], d[1], d[2], d[3], d[4], d[5]};
  const std::vector<metal::MetalBuffer> p = place(backend_, prefill_, prefillSizes);
  prefillScratch_ = {p[0], p[1], p[2], p[3], p[4], p[5], {p[6], p[8]}, {p[7], p[9]}, p[10]};
}

QwenTargetScratchBytes NemotronHTarget::scratchBytes(const QwenTargetGeometry &geometry) {
  return {totalBytes(prefillTensorBytes(geometry)), totalBytes(decodeTensorBytes(geometry))};
}

QwenTargetScratchBytes NemotronHTarget::allocatedScratchBytes() const noexcept {
  return {prefill_.sizeBytes(), decode_.sizeBytes()};
}

// ---- prefill ----

std::vector<metal::MetalBuffer> NemotronHTarget::addPrefill(metal::CommandGraph &graph,
                                                            std::span<const QwenTargetPrefillChunk> chunks,
                                                            const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                            QwenPrefillLayers layers, ops::AneFfn *aneFfn) {
  if (spilled.active()) throw std::logic_error("Nemotron-H attends no spilled K/V");
  std::vector<PrefillStep> steps = prefillSteps(graph, chunks, kv, spilled, layers, aneFfn);
  for (PrefillStep &step : steps) {
    for (uint32_t index = 0; index < layers.first; ++index) {
      if (NemotronHLayout::block(index) == NemotronBlock::Mamba) ++step.gdnLayer;
      if (NemotronHLayout::block(index) == NemotronBlock::Attention) ++step.attentionLayer;
    }
  }
  // Layer by layer, each chunk's in turn: a chunk's sequences read the states
  // and K/V the chunks before it wrote.
  for (uint32_t index = layers.first; index < layers.end; ++index)
    for (PrefillStep &step : steps) {
      const NemotronHLayerWeights &layer = weights_.layers[index];
      const metal::MetalBuffer input = step.buffers.hidden[index & 1];
      ops::Normalization::addRms(step.graph, input, layer.norm, step.buffers.normalized, geometry_.hiddenSize,
                                 step.rows);
      const metal::MetalBuffer output = step.buffers.hidden[(index & 1) ^ 1];
      std::visit([&](const auto &block) { addPrefillBlock(step, block, input, output); }, layer.block);
      forEachPrefillCapture(step, index, [&](uint32_t slot, uint32_t source, uint32_t destination, uint32_t rows) {
        addCapture(step.graph, slot, output, source, step.buffers.captured, destination, rows);
      });
    }
  std::vector<metal::MetalBuffer> outputs;
  if (layers.end != geometry_.layers) return outputs;
  for (PrefillStep &step : steps) {
    requireLayerPartition(step.gdnLayer, step.attentionLayer);
    outputs.push_back(step.buffers.hidden[geometry_.layers & 1]);
  }
  return outputs;
}

namespace {

// A prefill step's projection of its rows, adding to `residual` when given.
struct PrefillProject final {
  const ops::Linear &linear;
  metal::CommandGraph &graph;
  uint32_t rows;
  ops::LinearScratch scratch;
  void operator()(const ops::Projection &projection, metal::MetalBuffer input, metal::MetalBuffer output,
                  metal::MetalBuffer residual) const {
    if (residual)
      linear.addPrefillResidual(graph, std::move(input), projection, std::move(residual), std::move(output), rows,
                                scratch);
    else
      linear.addPrefill(graph, std::move(input), projection, std::move(output), rows, scratch);
  }
};

} // namespace

void NemotronHTarget::addPrefillBlock(PrefillStep &step, const NemotronMambaWeights &block, metal::MetalBuffer input,
                                      metal::MetalBuffer output) const {
  const PrefillScratch &s = prefillScratch_;
  const uint32_t layer = step.gdnLayer++;
  const PrefillProject project{operators_.linear(), step.graph, step.rows, step.buffers.linearScratch};
  project(block.in, step.buffers.normalized, s.packed, {});
  for (const QwenTargetPrefillSequence &sequence : step.sequences) addPrefillMamba(step, block, layer, sequence);
  ops::Mamba2::addGroupNorm(step.graph, s.y, block.mixer.norm, s.gated, weights_.layout.rmsEpsilon,
                            mamba2Shape(geometry_), step.rows);
  project(block.out, s.gated, std::move(output), std::move(input));
}

// A sequence's rows of layer `layer`, in slices of up to eight rows that end
// at each state it captures: slice k writes the state before its rows (after
// slice k - 1's, which it folds from the tape) into the next cell, a capture's
// views when its first row is the capture's, else the one-layer cell it does
// not read; a last dispatch folds the last slice into the sequence's state
// out. A slice runs eight rows from its first, so the rows past it (the next
// slice's, which it writes again, or the scratch's past the chunk) are
// computed and dropped.
// ponytail: eight rows a dispatch and a 2K-row scratch (~230 MB); a chunked
// scan kernel when prefill speed or memory matters.
void NemotronHTarget::addPrefillMamba(PrefillStep &step, const NemotronMambaWeights &block, uint32_t layer,
                                      const QwenTargetPrefillSequence &sequence) const {
  const PrefillScratch &s = prefillScratch_;
  const NemotronHLayout &layout = weights_.layout;
  std::vector<uint32_t> boundaries{0};
  for (const QwenTargetStateCapture &capture : sequence.stateCaptures) {
    if (capture.row <= boundaries.back() || capture.row >= sequence.rows)
      throw std::invalid_argument("Nemotron-H prefill state captures out of order");
    boundaries.push_back(capture.row);
  }
  boundaries.push_back(sequence.rows);
  struct Slice final {
    uint32_t begin, rows;
    const QwenTargetStateCapture *capture;  // whose state the slice writes
  };
  std::vector<Slice> slices;
  for (size_t b = 0; b + 1 < boundaries.size(); ++b)
    for (uint32_t row = boundaries[b]; row < boundaries[b + 1]; row += kVerifyRows)
      slices.push_back({row, std::min(kVerifyRows, boundaries[b + 1] - row),
                        row == boundaries[b] && b ? &sequence.stateCaptures[b - 1] : nullptr});
  const uint64_t tapeBytes = mamba2TapeLayerBytes(geometry_);
  metal::MetalBuffer convolution = sequence.convolutionIn[layer], recurrent = sequence.recurrentIn[layer];
  // The one-layer cell the state is in, 2 for none of them.
  uint32_t cell = 2;
  for (size_t k = 0; k <= slices.size(); ++k) {
    const bool fold = k == slices.size();
    // The fold runs past the sequence's rows, whose outputs it drops.
    const uint32_t row = sequence.rowBegin + (fold ? sequence.rows : slices[k].begin);
    metal::MetalBuffer convolutionOut, recurrentOut;
    if (fold) {
      convolutionOut = sequence.convolutionOut[layer], recurrentOut = sequence.recurrentOut[layer];
    } else if (const QwenTargetStateCapture *capture = slices[k].capture) {
      convolutionOut = capture->convolution[layer], recurrentOut = capture->recurrent[layer];
    } else {
      // The cell this slice does not read.
      cell = cell == 0 ? 1 : 0;
      convolutionOut = s.convolution[cell], recurrentOut = s.recurrent[cell];
    }
    ops::Mamba2::addPrefillStep(step.graph, rowsOf<uint16_t>(s.packed, row, kVerifyRows, layout.mambaInProjectionRows),
                                block.mixer, {convolution, recurrent, convolutionOut, recurrentOut}, s.tape,
                                (k + 1) % 2 * tapeBytes, k % 2 * tapeBytes, k ? slices[k - 1].rows : 0,
                                rowsOf<float>(s.y, row, kVerifyRows, layout.mambaInnerSize), mamba2Shape(geometry_));
    if (fold || slices[k].capture) cell = 2;
    convolution = convolutionOut, recurrent = recurrentOut;
  }
}

void NemotronHTarget::addPrefillBlock(PrefillStep &step, const QwenAttentionWeights &block, metal::MetalBuffer input,
                                      metal::MetalBuffer output) const {
  addPrefillMixer(step, block, {std::move(output), std::move(input)});
}

void NemotronHTarget::addPrefillBlock(PrefillStep &step, const NemotronMoeWeights &block, metal::MetalBuffer input,
                                      metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const PrefillScratch &s = prefillScratch_;
  addMoe({step.graph, step.rows, b.normalized, std::move(input), std::move(output), b.gdnOutput, {}, b.moe, s.latent,
          s.moeLatent, s.sharedMid},
         block, operators_.moePrefill(block.shape, step.rows),
         PrefillProject{operators_.linear(), step.graph, step.rows, b.linearScratch});
}

// ---- verify ----

void NemotronHTarget::addProjection(VerifyStep &step, const ops::Projection &projection, metal::MetalBuffer input,
                                    metal::MetalBuffer output, metal::MetalBuffer residual) const {
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan plan = linear.decodePlan(
      projection, step.lanes, residual ? ops::LinearEpilogue::Residual : ops::LinearEpilogue::None);
  linear.add(step.graph,
             {.input = std::move(input), .output = std::move(output), .residual = std::move(residual),
              .scratch = step.buffers.linearScratch},
             projection, plan);
}

void NemotronHTarget::addVerify(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
                                const kv::PageStorage &kv, SpilledKvReads &spilled,
                                std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes) const {
  if (spilled.active()) throw std::logic_error("Nemotron-H attends no spilled K/V");
  VerifyStep step = verifyStep(graph, buffers, kv, spilled, chunks, lanes);
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const NemotronHLayerWeights &layer = weights_.layers[index];
    const metal::MetalBuffer input = buffers.hidden[index & 1];
    ops::Normalization::addRms(step.graph, input, layer.norm, buffers.normalized, geometry_.hiddenSize, step.rows);
    const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
    std::visit([&](const auto &block) { addBlock(step, block, input, output); }, layer.block);
    if (const auto slot = geometry_.captureSlot(index))
      addCapture(step.graph, *slot, output, 0, buffers.capturedTargetHidden, 0, step.rows);
  }
  requireLayerPartition(step.gdnLayer, step.attentionLayer);
  addHeadBatch(step.graph, buffers.hidden[geometry_.layers & 1], buffers.finalHidden, buffers.logits, step.lanes,
               buffers.linearScratch);
}

void NemotronHTarget::addBlock(VerifyStep &step, const NemotronMambaWeights &block, metal::MetalBuffer input,
                               metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const uint32_t layer = step.gdnLayer++;
  addProjection(step, block.in, b.normalized, scratch_.packed);
  ops::Mamba2::addDecode(step.graph, scratch_.packed, block.mixer, b.currentGdnStates,
                         b.nextGdnStates, b.gdnTape, std::span(b.gdnTapeLanes).first(step.lanes), scratch_.y,
                         mamba2Shape(geometry_), layer, geometry_.stateLayout.strides());
  ops::Mamba2::addGroupNorm(step.graph, scratch_.y, block.mixer.norm, scratch_.gated, weights_.layout.rmsEpsilon,
                            mamba2Shape(geometry_), step.rows);
  addProjection(step, block.out, scratch_.gated, std::move(output), std::move(input));
}

void NemotronHTarget::addBlock(VerifyStep &step, const QwenAttentionWeights &block, metal::MetalBuffer input,
                               metal::MetalBuffer output) const {
  addVerifyMixer(step, block, {}, {std::move(output), std::move(input)});
}

void NemotronHTarget::addBlock(VerifyStep &step, const NemotronMoeWeights &block, metal::MetalBuffer input,
                               metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  addMoe({step.graph, step.rows, b.normalized, std::move(input), std::move(output), b.gdnOutput, b.verifyRows, b.moe,
          scratch_.latent, scratch_.moeLatent, scratch_.sharedMid},
         block, operators_.moeDecode(block.shape, step.lanes),
         [&](const ops::Projection &projection, metal::MetalBuffer in, metal::MetalBuffer out,
             metal::MetalBuffer residual) { addProjection(step, projection, std::move(in), std::move(out), std::move(residual)); });
}

void NemotronHTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                                   metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t lanes,
                                   ops::LinearScratch scratch) const {
  addNormalizedHead(graph, weights_.finalNorm, std::move(hidden), std::move(finalHidden), std::move(logits), lanes,
                    scratch);
}

QwenTargetScratchBytes qwenTargetScratchBytes(const NemotronHWeights &, const QwenTargetGeometry &geometry, uint32_t,
                                              uint64_t) {
  return NemotronHTarget::scratchBytes(geometry);
}

} // namespace splash::model
