#include "Vision.hpp"

#include "Checked.hpp"
#include "metal/abi/DenseAttention.h"
#include "metal/abi/Vision.h"
#include "ops/BufferExtent.hpp"
#include "ops/DenseAttention.hpp"
#include "ops/InklingVision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace splash::ops {
namespace {

using metal::CommandGraph;
using metal::MetalBuffer;

constexpr uint64_t kBf16Bytes = 2;
constexpr uint64_t kFloatBytes = 4;
// RADIO's own kernels' tiles (kernels/shared/vision.metal, _radio and _d80).
constexpr uint32_t kGemmRowTile = 64;
constexpr uint32_t kGemmColumnTile = 128;
constexpr uint32_t kMergerRowTile = 32;
constexpr uint32_t kMergerColumnTile = 256;
constexpr uint32_t kKeyTile = 128;
static_assert(kRadioVisionLayout.hiddenSize % kGemmColumnTile == 0 &&
                  kRadioVisionLayout.paddedIntermediateSize % kGemmColumnTile == 0 &&
                  kRadioVisionLayout.mergedHiddenSize % kMergerColumnTile == 0 &&
                  kRadioVisionLayout.mergerIntermediateSize % kMergerColumnTile == 0 &&
                  kRadioVisionLayout.patchDimension % kGemmColumnTile == 0,
              "RADIO vision kernels' tile assumptions");

// The one vision tower the kernels are specialized for; its projections'
// sizes are DenseLinear's to check (requireWeights).
constexpr VisionLayout kTower{};
// Queries and keys of the tower's heads as the dense attention reads them.
constexpr uint32_t kQkDimension = splash_dense_attention_padded(kTower.headDimension);

// Scratch tensor byte sizes for one encoder sized to maximumPatches. The
// projections read whole row tiles, attention reads whole key tiles, and the
// merger reads the normalized rows as (patches / 4, 4608). Attention is
// RADIO's alone, a placeholder value here.
std::array<uint64_t, 12> qwenScratchLayout(const VisionLayout &layout, uint32_t maximumPatches) {
  const uint64_t rows = DenseLinear::paddedRows(maximumPatches);
  const uint64_t padded = DenseAttention::paddedKeys(maximumPatches);
  const uint64_t mergedRows = DenseLinear::paddedRows(maximumPatches / 4);
  const uint64_t hidden = layout.hiddenSize;
  const uint64_t headRows = uint64_t{layout.heads} * padded;
  return {
      rows * layout.patchDimension * kBf16Bytes,                 // Patches
      uint64_t{maximumPatches} * layout.headDimension * kFloatBytes, // RopeCos
      uint64_t{maximumPatches} * layout.headDimension * kFloatBytes, // RopeSin
      rows * hidden * kFloatBytes,                               // Hidden
      std::max(rows * hidden, mergedRows * layout.mergedHiddenSize) *
          kBf16Bytes,                                            // Normalized
      std::max(rows * 3 * hidden, mergedRows * layout.mergedHiddenSize) *
          kBf16Bytes,                                            // Qkv
      rows * layout.paddedIntermediateSize * kBf16Bytes,         // Intermediate
      headRows * kQkDimension * kBf16Bytes,                      // Queries
      headRows * kQkDimension * kBf16Bytes,                      // Keys
      headRows * layout.headDimension * kBf16Bytes,              // Values
      kBf16Bytes,                                                // Attention (RADIO's only)
      rows * hidden * kBf16Bytes,                                // Context
  };
}

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept {
  return (value + multiple - 1) / multiple * multiple;
}

// Query and key rows hold the head dimension padded with zeros to the 16 a
// matmul's K extent takes: RADIO's 80 as it is.
uint32_t qkDimension(const VisionLayout &layout) noexcept {
  return roundUp(layout.headDimension, 16);
}

// The bias RADIO's bias-free GEMMs read: zeros for the widest of them, at
// the arena's end after the scratch tensors.
uint64_t zeroBiasBytes(const VisionLayout &layout) noexcept {
  return uint64_t{std::max({layout.hiddenSize, layout.mergerIntermediateSize, layout.outputHiddenSize})} *
         kBf16Bytes;
}

// RADIO's scratch tensor byte sizes for one encoder sized to maximumPatches. The
// tower's tokens are the patches and its prefix; GEMM tiles read and write
// whole 64-row tiles, attention reads whole 128-key tiles, and the merger
// reads the normalized rows as (patches / 4, merged) and writes its up
// projection's (patches / 4, mergerIntermediate) over the q/k/v rows. RADIO
// has no rope, so its rope tables are a placeholder value, and its patch
// rows are two frames wide for a video's tubelet.
std::array<uint64_t, 12> radioScratchLayout(const VisionLayout &layout, uint32_t maximumPatches) {
  constexpr bool radio = true;
  const uint32_t tokens = maximumPatches + layout.prefixTokens;
  const uint64_t rows = roundUp(tokens, kGemmRowTile);
  const uint64_t padded = roundUp(tokens, kKeyTile);
  const uint64_t mergedRows = roundUp(maximumPatches / 4, kMergerRowTile);
  const uint64_t hidden = layout.hiddenSize;
  const uint64_t headRows = uint64_t{layout.heads} * padded;
  const uint64_t rope = radio ? kFloatBytes : uint64_t{maximumPatches} * layout.headDimension * kFloatBytes;
  return {
      rows * layout.patchDimension * (radio ? 2 : 1) * kBf16Bytes, // Patches
      rope,                                                      // RopeCos
      rope,                                                      // RopeSin
      rows * hidden * kFloatBytes,                               // Hidden
      std::max(rows * hidden, mergedRows * layout.mergedHiddenSize) *
          kBf16Bytes,                                            // Normalized
      std::max(rows * 3 * hidden, mergedRows * layout.mergerIntermediateSize) *
          kBf16Bytes,                                            // Qkv
      rows * layout.paddedIntermediateSize * kBf16Bytes,         // Intermediate
      headRows * qkDimension(layout) * kBf16Bytes,               // Queries
      headRows * qkDimension(layout) * kBf16Bytes,               // Keys
      headRows * layout.headDimension * kBf16Bytes,              // Values
      headRows * layout.headDimension * kBf16Bytes,              // Attention
      rows * hidden * kBf16Bytes,                                // Context
  };
}

std::array<uint64_t, 12> scratchLayout(const VisionLayout &layout, uint32_t maximumPatches, bool radio) {
  return radio ? radioScratchLayout(layout, maximumPatches) : qwenScratchLayout(layout, maximumPatches);
}

// Whether the layout is RADIO's; throws unless the kernels compile its tower:
// Qwen3.5's 27 blocks on the shared dense operators, or RADIO on the Nemotron
// fork's own kernels. Only the language-space projection width varies with
// the text model, and whether RADIO has its final norm.
bool requireLayout(const VisionLayout &layout) {
  VisionLayout tower = layout;
  tower.outputHiddenSize = 0;
  tower.finalNorm = false;
  if (isRadioVision(tower)) {
    if (!layout.outputHiddenSize || layout.outputHiddenSize % kMergerColumnTile)
      throw std::invalid_argument("no vision kernels for the tower");
    return true;
  }
  tower.outputHiddenSize = kTower.outputHiddenSize;
  if (tower != kTower || !layout.outputHiddenSize || layout.finalNorm)
    throw std::invalid_argument("no vision kernels for the tower");
  return false;
}

// Every weight of the tower the kernels read, in each of its layout's blocks:
// bf16 [output][input] matrices with a bias per output, a LayerNorm's weight
// and bias per hidden value, and the learned position of every cell of the
// grid.
void requireQwenWeights(const VisionWeights &model) {
  const VisionLayout &layout = model.layout;
  if (model.blocks.size() != layout.depth) {
    throw std::invalid_argument("vision tower holds " + std::to_string(model.blocks.size()) + " blocks, needs " +
                                std::to_string(layout.depth));
  }
  const uint32_t hidden = layout.hiddenSize, merged = layout.mergedHiddenSize;
  DenseLinear::requireWeights(model.patchEmbedding, hidden, layout.patchDimension, "vision patch embedding");
  requireBytes(model.positionTable,
               uint64_t{layout.positionGridSide} * layout.positionGridSide * hidden * kBf16Bytes,
               "vision position table");
  for (const VisionBlock &block : model.blocks) {
    Normalization::requireLayer(block.norm1, hidden, "vision attention norm");
    DenseLinear::requireWeights(block.qkv, 3 * hidden, hidden, "vision q/k/v");
    DenseLinear::requireWeights(block.projection, hidden, hidden, "vision attention projection");
    Normalization::requireLayer(block.norm2, hidden, "vision MLP norm");
    DenseLinear::requireWeights(block.upProjection, layout.paddedIntermediateSize, hidden, "vision up projection");
    DenseLinear::requireWeights(block.downProjection, hidden, layout.paddedIntermediateSize, "vision down projection");
  }
  Normalization::requireLayer(model.mergerNorm, hidden, "vision merger norm");
  DenseLinear::requireWeights(model.mergerUpProjection, merged, merged, "vision merger up projection");
  DenseLinear::requireWeights(model.mergerDownProjection, layout.outputHiddenSize, merged,
                              "vision merger down projection");
}

// Every weight of RADIO's tower its kernels read, in each of its layout's blocks:
// bf16 [output][input] matrices with a bias per output, a LayerNorm's weight
// and bias per hidden value, and the learned position of every cell of the
// grid. RADIO's patch embedding and merger read no bias, it reads its prefix
// tokens, and its merger norm is an RMS weight per merged value.
void requireRadioWeights(const VisionWeights &model) {
  constexpr bool radio = true;
  const VisionLayout &layout = model.layout;
  if (model.blocks.size() != layout.depth) {
    throw std::invalid_argument("vision tower holds " + std::to_string(model.blocks.size()) + " blocks, needs " +
                                std::to_string(layout.depth));
  }
  const uint64_t hidden = layout.hiddenSize, merged = layout.mergedHiddenSize;
  const uint64_t mergerIntermediate = layout.mergerIntermediateSize;
  const auto requireAffine = [](const VisionAffine &affine, uint64_t output, uint64_t input,
                                const std::string &name, bool bias = true) {
    requireBytes(affine.weight, output * input * kBf16Bytes, name + " weight");
    if (bias) requireBytes(affine.bias, output * kBf16Bytes, name + " bias");
  };
  const auto requireNorm = [&](const VisionNorm &norm, const std::string &name) {
    requireBytes(norm.weight, hidden * kBf16Bytes, name + " weight");
    requireBytes(norm.bias, hidden * kBf16Bytes, name + " bias");
  };
  requireAffine(model.patchEmbedding, hidden, layout.patchDimension, "vision patch embedding", !radio);
  if (radio) requireAffine(model.videoPatchEmbedding, hidden, 2 * layout.patchDimension, "vision video patch embedding", false);
  requireBytes(model.positionTable,
               uint64_t{layout.positionGridSide} * layout.positionGridSide * hidden * kBf16Bytes,
               "vision position table");
  if (radio) requireBytes(model.prefixTokens, uint64_t{layout.prefixTokens} * hidden * kBf16Bytes, "vision prefix tokens");
  for (const VisionBlock &block : model.blocks) {
    requireNorm(block.norm1, "vision attention norm");
    requireAffine(block.qkv, 3 * hidden, hidden, "vision q/k/v");
    requireAffine(block.projection, hidden, hidden, "vision attention projection");
    requireNorm(block.norm2, "vision MLP norm");
    requireAffine(block.upProjection, layout.paddedIntermediateSize, hidden, "vision up projection");
    requireAffine(block.downProjection, hidden, layout.paddedIntermediateSize, "vision down projection");
  }
  if (layout.finalNorm) requireNorm(model.finalNorm, "vision final norm");
  if (radio)
    requireBytes(model.mergerNorm.weight, merged * kBf16Bytes, "vision merger norm weight");
  else
    requireNorm(model.mergerNorm, "vision merger norm");
  requireAffine(model.mergerUpProjection, mergerIntermediate, merged, "vision merger up projection", !radio);
  requireAffine(model.mergerDownProjection, layout.outputHiddenSize, mergerIntermediate,
                "vision merger down projection", !radio);
}

void requireWeights(const VisionWeights &model, bool radio) {
  radio ? requireRadioWeights(model) : requireQwenWeights(model);
}

} // namespace

// Whole pages, each tower's the sum of its tensors' (alignUp), so the arena
// Metal allocates is as large (arenaBytes).
uint64_t ImageEncoder::scratchBytes(const VisionLayout &layout, uint32_t maximumPatches) {
  return layout.tower == VisionTower::Inkling ? InklingVision::scratchBytes(layout, maximumPatches)
                                              : Vision::scratchBytes(layout, maximumPatches);
}

std::shared_ptr<ImageEncoder> ImageEncoder::make(metal::MetalBackend &backend, const VisionWeights &model,
                                                 uint32_t maximumPatches) {
  if (model.layout.tower == VisionTower::Inkling)
    return std::make_shared<InklingVision>(backend, model, maximumPatches);
  return std::make_shared<Vision>(backend, model, maximumPatches);
}

uint64_t Vision::scratchBytes(const VisionLayout &layout,
                              uint32_t maximumPatches) {
  const bool radio = requireLayout(layout);
  if (!maximumPatches || maximumPatches % 4) {
    throw std::invalid_argument(
        "vision scratch must cover a positive multiple of four patches");
  }
  uint64_t total = 0;
  for (uint64_t bytes : scratchLayout(layout, maximumPatches, radio)) {
    const uint64_t aligned = alignUp(bytes);
    if (aligned > std::numeric_limits<uint64_t>::max() - total) {
      throw std::overflow_error("vision scratch byte count overflows");
    }
    total += aligned;
  }
  if (radio)
    total += alignUp(zeroBiasBytes(layout));
  return total;
}

uint32_t Vision::embeddingRows(ImageGrid grid) noexcept {
  return roundUp(grid.mergedTokens(), kMergerRowTile);
}

Vision::Vision(metal::MetalBackend &backend, const VisionWeights &model,
               uint32_t maximumPatches)
    : model_(model), maximumPatches_(maximumPatches), radio_(requireLayout(model.layout)),
      dense_(backend.capabilities()) {
  const uint64_t total = scratchBytes(model.layout, maximumPatches);
  requireWeights(model, radio_);
  // New backend buffers are zero-filled, so padding rows read by whole tiles
  // start finite.
  arena_ = backend.allocateBuffer(total, metal::BufferStorage::Shared,
                                  "vision-scratch");
  uint64_t cursor = 0;
  const auto layout = scratchLayout(model.layout, maximumPatches, radio_);
  for (uint32_t index = 0; index < layout.size(); ++index) {
    scratch_[index] = backend.view(arena_, cursor, layout[index]);
    cursor += alignUp(layout[index]);
  }
  if (radio_) {
    zeros_ = backend.view(arena_, cursor, zeroBiasBytes(model.layout));
    cursor += alignUp(zeroBiasBytes(model.layout));
  }
  if (cursor != total)
    throw std::logic_error("vision scratch arena mismatch");
}

void Vision::addGemm(CommandGraph &graph, const char *pipeline,
                     const MetalBuffer &input, const VisionAffine &weights,
                     const MetalBuffer &output, const MetalBuffer &residual,
                     uint32_t outputSize, uint32_t inputSize, uint32_t rows,
                     uint32_t tileRows, uint32_t tileColumns) const {
  graph.add(pipeline, {input, weights.weight, weights.bias, output, residual},
            VisionGemmParams{outputSize, inputSize},
            {(rows + tileRows - 1) / tileRows, outputSize / tileColumns, 1});
}

void Vision::addNorm(CommandGraph &graph, const MetalBuffer &input,
                     const VisionNorm &weights, const MetalBuffer &output,
                     uint32_t rows) const {
  graph.add("vision_layer_norm", {input, weights.weight, weights.bias, output},
            VisionNormParams{model_.layout.hiddenSize}, {rows, 1, 1});
}

void Vision::encode(CommandGraph &graph, ImageGrid grid,
                    const MetalBuffer &pixels,
                    const MetalBuffer &embeddings) const {
  const VisionLayout &layout = model_.layout;
  if (!grid.valid() || grid.patches() > maximumPatches_) {
    throw std::invalid_argument("image grid exceeds the vision encoder");
  }
  if (radio_) {
    encodeRadio(graph, grid, pixels, embeddings);
    return;
  }
  if (grid.frames != 1) throw std::invalid_argument("only RADIO encodes video");
  const auto tokens = static_cast<uint32_t>(grid.patches());
  requireBytes(pixels, grid.pixelBytes(layout.patchSize), "image pixel");
  requireBytes(embeddings, uint64_t{grid.mergedTokens()} * layout.outputHiddenSize * kBf16Bytes,
               "image embedding");

  const uint32_t padded = DenseAttention::paddedKeys(tokens);
  const uint32_t merged = grid.mergedTokens();
  const uint32_t hiddenSize = layout.hiddenSize;
  const float epsilon = static_cast<float>(SPLASH_VISION_NORM_EPSILON);
  const VisionGridParams gridParams{grid.height, grid.width};
  const VisionQkvParams qkvParams{tokens, padded};
  // vision_qkv_prepare's per-head planes of every padded token: queries and
  // keys padded to 80 dimensions, values of 72. The output goes straight to
  // the context rows the projection reads, each head's 72 dimensions after
  // the heads before it.
  const auto plane = [&](Scratch tensor, uint32_t width) {
    return HeadTensor{scratch(tensor), padded * width, width};
  };
  const DenseAttentionTensors attention{plane(Scratch::Queries, kQkDimension), plane(Scratch::Keys, kQkDimension),
                                        plane(Scratch::Values, layout.headDimension),
                                        {scratch(Scratch::Context), layout.headDimension, layout.hiddenSize}};
  const float scale = 1.0F / std::sqrt(static_cast<float>(layout.headDimension));
  const MetalBuffer &hidden = scratch(Scratch::Hidden);
  const MetalBuffer &normalized = scratch(Scratch::Normalized);
  const MetalBuffer &qkv = scratch(Scratch::Qkv);
  const MetalBuffer &context = scratch(Scratch::Context);

  graph.add("vision_patchify", {pixels, scratch(Scratch::Patches)}, gridParams,
            {tokens, 1, 1});
  // The positions start the residual stream the patch embedding adds to.
  graph.add("vision_prepare_positions",
            {model_.positionTable, hidden, scratch(Scratch::RopeCos),
             scratch(Scratch::RopeSin)},
            gridParams, {tokens, 1, 1});
  dense_.add(graph, model_.patchEmbedding, hiddenSize, layout.patchDimension, scratch(Scratch::Patches),
             hidden, tokens, DenseEpilogue::Residual);

  for (const VisionBlock &block : model_.blocks) {
    Normalization::addLayer(graph, hidden, block.norm1, normalized, hiddenSize, tokens, epsilon);
    dense_.add(graph, block.qkv, 3 * hiddenSize, hiddenSize, normalized, qkv, tokens);
    // Padded key tokens are zeroed by the prepare pass so every 128-key tile
    // is finite under the softmax mask.
    graph.add("vision_qkv_prepare",
              {qkv, scratch(Scratch::RopeCos), scratch(Scratch::RopeSin),
               scratch(Scratch::Queries), scratch(Scratch::Keys),
               scratch(Scratch::Values)},
              qkvParams, {padded, 1, 1});
    // Every patch attends every patch of the image.
    DenseAttention::add(graph, attention, layout.heads, layout.headDimension, tokens, scale);
    dense_.add(graph, block.projection, hiddenSize, hiddenSize, context, hidden, tokens, DenseEpilogue::Residual);
    Normalization::addLayer(graph, hidden, block.norm2, normalized, hiddenSize, tokens, epsilon);
    dense_.add(graph, block.upProjection, layout.paddedIntermediateSize, hiddenSize, normalized,
               scratch(Scratch::Intermediate), tokens, DenseEpilogue::GeluTanh);
    dense_.add(graph, block.downProjection, hiddenSize, layout.paddedIntermediateSize,
               scratch(Scratch::Intermediate), hidden, tokens, DenseEpilogue::Residual);
  }

  // The merger reads the normalized rows as (merged, 4608): four consecutive
  // block-major patches form one merged token.
  Normalization::addLayer(graph, hidden, model_.mergerNorm, normalized, hiddenSize, tokens, epsilon);
  dense_.add(graph, model_.mergerUpProjection, layout.mergedHiddenSize, layout.mergedHiddenSize, normalized,
             qkv, merged, DenseEpilogue::GeluErf);
  dense_.add(graph, model_.mergerDownProjection, layout.outputHiddenSize, layout.mergedHiddenSize, qkv,
             embeddings, merged);
}

// RADIO on the Nemotron fork's own kernels, as 1.7.0's fork encoded it.
void Vision::encodeRadio(CommandGraph &graph, ImageGrid grid, const MetalBuffer &pixels,
                         const MetalBuffer &embeddings) const {
  const VisionLayout &layout = model_.layout;
  const auto patches = static_cast<uint32_t>(grid.patches());
  const uint32_t tokens = patches + layout.prefixTokens;
  requireBytes(pixels, grid.pixelBytes(layout.patchSize), "image pixel");
  requireBytes(embeddings, uint64_t{embeddingRows(grid)} * layout.outputHiddenSize * kBf16Bytes,
               "image embedding");

  const uint32_t padded = roundUp(tokens, kKeyTile);
  const uint32_t merged = grid.mergedTokens();
  const VisionGridParams gridParams{grid.height, grid.width};
  const VisionQkvParams qkvParams{tokens, padded};
  const VisionAttentionParams attentionParams{
      tokens, padded, 1.0F / std::sqrt(static_cast<float>(layout.headDimension))};
  const MetalBuffer &hidden = scratch(Scratch::Hidden);
  const MetalBuffer &normalized = scratch(Scratch::Normalized);
  const MetalBuffer &qkv = scratch(Scratch::Qkv);
  const MetalBuffer &context = scratch(Scratch::Context);

  const bool video = grid.frames == 2;
  graph.add(video ? "vision_patchify_radio_video" : "vision_patchify_radio", {pixels, scratch(Scratch::Patches)},
            gridParams, {patches, 1, 1});
  graph.add("vision_prepare_positions_radio", {model_.positionTable, hidden}, gridParams, {patches, 1, 1});
  addGemm(graph, "vision_gemm_m64n128_residual", scratch(Scratch::Patches),
          {(video ? model_.videoPatchEmbedding : model_.patchEmbedding).weight, zeros_}, hidden, hidden,
          layout.hiddenSize, grid.frames * layout.patchDimension, patches, kGemmRowTile, kGemmColumnTile);
  // The prefix rows, past the embedding's last tile over them.
  graph.add("vision_prefix_radio", {model_.prefixTokens, hidden}, gridParams,
            {roundUp(tokens, kGemmRowTile) - patches, 1, 1});

  for (const VisionBlock &block : model_.blocks) {
    addNorm(graph, hidden, block.norm1, normalized, tokens);
    addGemm(graph, "vision_gemm_m64n128", normalized, block.qkv, qkv, hidden,
            3 * layout.hiddenSize, layout.hiddenSize, tokens, kGemmRowTile,
            kGemmColumnTile);
    // Padded key tokens are zeroed by the prepare pass so every 128-key tile
    // is finite under the softmax mask.
    graph.add("vision_qkv_prepare_radio",
              {qkv, scratch(Scratch::Queries), scratch(Scratch::Keys), scratch(Scratch::Values)}, qkvParams,
              {padded, 1, 1});
    graph.add("vision_attention_d80",
              {scratch(Scratch::Queries), scratch(Scratch::Keys),
               scratch(Scratch::Values), scratch(Scratch::Attention)},
              attentionParams,
              {(tokens + SPLASH_VISION_ATTENTION_ROWS - 1) /
                   SPLASH_VISION_ATTENTION_ROWS,
               layout.heads, 1},
              {SPLASH_VISION_ATTENTION_THREADS, 1, 1});
    graph.add("vision_attention_gather_d80", {scratch(Scratch::Attention), context},
              qkvParams, {tokens, 1, 1});
    addGemm(graph, "vision_gemm_m64n128_residual", context, block.projection, hidden,
            hidden, layout.hiddenSize, layout.hiddenSize, tokens, kGemmRowTile,
            kGemmColumnTile);
    addNorm(graph, hidden, block.norm2, normalized, tokens);
    addGemm(graph, "vision_gemm_m64n128_gelu_erf", normalized,
            block.upProjection,
            scratch(Scratch::Intermediate), hidden,
            layout.paddedIntermediateSize, layout.hiddenSize, tokens,
            kGemmRowTile, kGemmColumnTile);
    addGemm(graph, "vision_gemm_m64n128_residual", scratch(Scratch::Intermediate),
            block.downProjection, hidden, hidden, layout.hiddenSize,
            layout.paddedIntermediateSize, tokens, kGemmRowTile,
            kGemmColumnTile);
  }

  // The merger reads the normalized rows as (merged, mergedHiddenSize): four
  // consecutive block-major patches form one merged token, normed as one
  // merged row.
  if (layout.finalNorm)
    graph.add("vision_layer_norm_f32", {hidden, model_.finalNorm.weight, model_.finalNorm.bias, hidden},
              VisionNormParams{layout.hiddenSize}, {patches, 1, 1});
  graph.add("vision_rms_norm", {hidden, model_.mergerNorm.weight, normalized},
            VisionNormParams{layout.mergedHiddenSize}, {merged, 1, 1});
  addGemm(graph, "vision_gemm_m32n256_relu2", normalized, {model_.mergerUpProjection.weight, zeros_}, qkv,
          hidden, layout.mergerIntermediateSize, layout.mergedHiddenSize, merged, kMergerRowTile,
          kMergerColumnTile);
  addGemm(graph, "vision_gemm_m32n256", qkv, {model_.mergerDownProjection.weight, zeros_}, embeddings, hidden,
          layout.outputHiddenSize, layout.mergerIntermediateSize, merged, kMergerRowTile, kMergerColumnTile);
}

uint64_t VideoPruning::flagBytes(const VideoPruningShape &shape) noexcept {
  return uint64_t{shape.tubelets} * shape.grid.mergedTokens() * kFloatBytes;
}

void VideoPruning::add(CommandGraph &graph, const VideoPruningShape &shape, const MetalBuffer &encoded,
                       const MetalBuffer &text, const MetalBuffer &cuts, const MetalBuffer &dissimilarity,
                       const MetalBuffer &keep, const MetalBuffer &output) {
  const uint32_t rows = shape.grid.mergedTokens();
  const uint64_t total = uint64_t{shape.tubelets} * rows;
  if (!shape.tubelets || shape.grid.frames != 2 || !shape.width || !shape.textTokens || shape.kept < rows ||
      shape.kept > total || total + shape.textTokens > UINT32_MAX)
    throw std::invalid_argument("invalid video pruning");
  const uint32_t stride = Vision::embeddingRows(shape.grid);
  requireBytes(encoded, uint64_t{shape.tubelets} * stride * shape.width * kBf16Bytes, "video tubelet rows");
  requireBytes(text, uint64_t{shape.textTokens} * shape.width * kBf16Bytes, "video text rows");
  requireBytes(cuts, uint64_t{shape.tubelets} * sizeof(uint32_t), "video text cuts");
  requireBytes(dissimilarity, flagBytes(shape), "video dissimilarity");
  requireBytes(keep, flagBytes(shape), "video keep flag");
  requireBytes(output, (uint64_t{shape.textTokens} + shape.kept) * shape.width * kBf16Bytes, "video rows");
  const VideoPruningParams params{shape.tubelets, rows, stride, shape.width, shape.kept, shape.textTokens};
  const auto groups = static_cast<uint32_t>(total);
  graph.add("video_pruning_dissimilarity", {encoded, dissimilarity}, params, {groups, 1, 1});
  graph.add("video_pruning_keep", {dissimilarity, keep}, params,
            {(groups + CommandGraph::kDefaultThreads - 1) / CommandGraph::kDefaultThreads, 1, 1});
  graph.add("video_pruning_place", {encoded, text, cuts, keep, output}, params,
            {groups + shape.textTokens, 1, 1});
}

} // namespace splash::ops
