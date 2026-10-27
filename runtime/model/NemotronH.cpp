#include "NemotronH.hpp"

#include "model/ExpertStore.hpp"
#include "model/GgufTarget.hpp"
#include "model/NemotronHTarget.hpp"
#include "model/QwenTargetLoader.hpp"
#include "model/SparseRowStore.hpp"

#include <algorithm>

namespace splash::model {

namespace {

// A layer's weights; its MoE experts read their slots from `slots`, every
// expert resident in its own.
NemotronHLayerWeights readLayer(WeightFile &file, const NemotronHLayout &layout, uint32_t index,
                                const metal::MetalBuffer &slots) {
  const auto f32 = [&](uint32_t width, std::string_view label) { return readNorm(file, width, true, label).buffer; };
  NemotronHLayerWeights layer;
  layer.norm = readNorm(file, layout.hiddenSize, true, "block-norm");
  layer.norm.epsilon = layout.rmsEpsilon;
  switch (NemotronHLayout::block(index)) {
  case NemotronBlock::Mamba: {
    NemotronMambaWeights m;
    m.in = readBlockProjection(file, layout.mambaInProjectionRows, layout.hiddenSize, "ssm-in");
    m.mixer.convolution = f32(layout.convolutionDimension * kGdnConvolutionTaps, "ssm-conv");
    m.mixer.convolutionBias = f32(layout.convolutionDimension, "ssm-conv-bias");
    m.mixer.dtBias = f32(layout.gdnValueHeads, "ssm-dt-bias");
    m.mixer.a = f32(layout.gdnValueHeads, "ssm-a");
    m.mixer.d = f32(layout.gdnValueHeads, "ssm-d");
    m.mixer.norm = f32(layout.mambaInnerSize, "ssm-norm");
    m.out = readBlockProjection(file, layout.hiddenSize, layout.mambaInnerSize, "ssm-out");
    layer.block = std::move(m);
    break;
  }
  case NemotronBlock::Attention: {
    ops::BlockWeights qkv;
    uint32_t offset = 0;
    for (const std::string_view label : {"attn-q", "attn-k", "attn-v"}) {
      ops::QuantizedSegment segment = readQuantizedSegment(file, label);
      segment.columnOffset = offset;
      offset += segment.outputSize;
      qkv.segments.push_back(std::move(segment));
    }
    if (offset != layout.packedFullWidth) throw WeightStoreError("Nemotron-H attention q | k | v has unexpected rows");
    QwenAttentionWeights a;
    a.inputProjection = {offset, layout.hiddenSize, std::move(qkv)};
    a.outputProjection = readBlockProjection(file, layout.hiddenSize, layout.attentionWidth, "attn-output");
    layer.block = std::move(a);
    break;
  }
  case NemotronBlock::Moe: {
    NemotronMoeWeights m;
    const uint32_t experts = NemotronHLayout::layerExperts(index), width = NemotronHLayout::layerExpertWidth(index);
    m.experts.router = readQuantizedSegment(file, "router");
    m.experts.selectionBias = f32(experts, "exp-probs-bias");
    m.latentDown = readBlockProjection(file, layout.moeLatentSize, layout.hiddenSize, "latent-down");
    m.experts.up.routed = readQuantizedSegment(file, "experts-up");
    m.experts.down.routed = readQuantizedSegment(file, "experts-down");
    m.latentUp = readBlockProjection(file, layout.hiddenSize, layout.moeLatentSize, "latent-up");
    m.sharedUp = readBlockProjection(file, layout.sharedIntermediateSize, layout.hiddenSize, "shared-up");
    m.sharedDown = readBlockProjection(file, layout.hiddenSize, layout.sharedIntermediateSize, "shared-down");
    m.experts.slotTable = slots;
    m.experts.slots = experts;
    const ops::QuantizedSegment &up = m.experts.up.routed, &down = m.experts.down.routed;
    m.shape = {.hiddenSize = layout.moeLatentSize,
               .experts = experts,
               .expertsPerToken = layout.expertsPerToken,
               .expertIntermediateSize = width,
               .expertFormat = up.formatId,
               .registerFormats = quant_register_tensor(up.formatId, up.inputSize) &&
                                  quant_register_tensor(down.formatId, down.inputSize),
               .router = ops::MoeRouter::SigmoidBiased,
               .sharedSlices = 0,
               .sharedGate = false,
               .activation = ops::MoeActivation::ReluSquared,
               .routerInputSize = layout.hiddenSize,
               .routedScale = layout.routedScale};
    if (!m.shape.valid()) throw WeightStoreError("Nemotron-H MoE block has no compiled shape");
    layer.block = std::move(m);
    break;
  }
  }
  return layer;
}

} // namespace

NemotronHWeights loadNemotronHWeights(metal::MetalBackend &backend, const NemotronHLayout &layout,
                                      const QwenTargetFiles &files) {
  const auto *loader = std::get_if<std::reference_wrapper<GgufTargetLoader>>(&files);
  if (!loader) throw WeightStoreError(std::string(layout.family) + " loads from a GGUF only");
  GgufTargetLoader &gguf = loader->get();
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  NemotronHWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  const metal::MetalBuffer slots = residentExpertSlots(backend, layout.experts);
  for (uint32_t index = 0; index < layout.layers; ++index) {
    WeightFile file = gguf.layer(index);
    result.layers.push_back(readLayer(file, layout, index, slots));
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = gguf.head();
    result.finalNorm = readNorm(file, layout.hiddenSize, true, "final-norm");
    result.finalNorm.epsilon = layout.rmsEpsilon;
    result.logitsProjection = readBlockProjection(file, layout.vocabularySize, layout.hiddenSize, "logits");
    result.logitsProjection.destination = ops::FloatOutput::Float32;
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = gguf.embedding();
    result.tokenEmbedding = readBlockEmbedding(file, layout.vocabularySize, layout.hiddenSize, "embedding");
    file.finish();
    result.files.push_back(file.record());
  }
  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

QwenTargetGeometry qwenTargetGeometry(const NemotronHWeights &weights, std::span<const uint32_t> captureLayers) {
  QwenTargetGeometry geometry(weights.layout, captureLayers);
  geometry.kvLayout = weights.layout.kvLayout();
  geometry.stateLayout = weights.layout.gdnStateLayout();
  // The runtime sizes its MoE scratch for the geometry's shape, of the most
  // experts and the widest; each block runs its own (NemotronMoeWeights).
  geometry.moeExpertFormat = GGUF_FMT_COUNT;
  geometry.moeRegisterFormats = true;
  for (const NemotronHLayerWeights &layer : weights.layers)
    std::visit(
        [&](const auto &block) {
          using Block = std::decay_t<decltype(block)>;
          if constexpr (std::is_same_v<Block, NemotronMoeWeights>) {
            for (const ops::Projection *p : {&block.latentDown, &block.latentUp, &block.sharedUp, &block.sharedDown})
              geometry.include(*p);
            geometry.moeExpertFormat = block.shape.expertFormat;
            geometry.moeRegisterFormats &= block.shape.registerFormats;
          } else if constexpr (std::is_same_v<Block, NemotronMambaWeights>) {
            geometry.include(block.in);
            geometry.include(block.out);
          } else {
            geometry.include(block.inputProjection);
            geometry.include(block.outputProjection);
          }
        },
        layer.block);
  geometry.prefillProjections = geometry.decodeProjections;
  geometry.include(weights.logitsProjection);
  for (auto *shapes : {&geometry.prefillProjections, &geometry.decodeProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()), shapes->end());
  }
  return geometry;
}

std::unique_ptr<QwenTarget> makeQwenTarget(const NemotronHWeights &weights, const QwenTargetGeometry &geometry,
                                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                           uint32_t) {
  return std::make_unique<NemotronHTarget>(weights, geometry, backend, operators);
}

SparseRowStoreOptions streamedTableRows(const NemotronHLayout &) { return {}; }

} // namespace splash::model
