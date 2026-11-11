#pragma once

// Nemotron-H's target (model/NemotronH.hpp): its single-block layers, each
// behind its own RMS norm (epsilon 1e-5), added to one residual. Its Mamba-2
// state lives in the runtime's GDN state cells (GdnStateLayout of Mamba-2
// geometry) and folds a step's retained rows from its tape as GDN's does; it
// has no auxiliary state and no draft. Its blocks run through ops::Mamba2,
// the Qwen families' attention mixers and ops::MoE, on the runtime's arena and
// the target's own decode scratch (scratchBytes). A prefill runs layer by
// layer as the Qwen families' does; its Mamba-2 blocks run each sequence's
// rows eight at a time, between two one-layer state cells of its own, and
// fold the last rows into the sequence's state at its end.

#include "model/NemotronH.hpp"
#include "model/QwenTarget.hpp"

namespace splash::model {

class NemotronHTarget final : public QwenTarget {
public:
  NemotronHTarget(const NemotronHWeights &weights, const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
                  const ops::ExecutionPlans &operators);

  // The decode and prefill scratch the constructor allocates.
  [[nodiscard]] static QwenTargetScratchBytes scratchBytes(const QwenTargetGeometry &geometry);
  [[nodiscard]] QwenTargetScratchBytes allocatedScratchBytes() const noexcept override;

  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph,
                                                           std::span<const QwenTargetPrefillChunk> chunks,
                                                           const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                           QwenPrefillLayers layers, ops::AneFfn *aneFfn) override;
  void addVerify(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers, const kv::PageStorage &kv,
                 SpilledKvReads &spilled, std::span<const kv::ChunkedPrefillParams> chunks,
                 uint32_t lanes) const override;
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer finalHidden,
                    metal::MetalBuffer logits, uint32_t lanes, ops::LinearScratch scratch) const override;

private:
  // Views of the decode scratch, each for a full batch's verify rows.
  struct Scratch final {
    metal::MetalBuffer packed, y, gated, latent, moeLatent, sharedMid;
  };

  // Views of the prefill scratch: rows for a chunk and the slices past it;
  // two one-layer state cells and a one-layer tape of two slots.
  struct PrefillScratch final {
    metal::MetalBuffer packed, y, gated, latent, moeLatent, sharedMid;
    std::array<metal::MetalBuffer, 2> convolution, recurrent;
    metal::MetalBuffer tape;
  };

  void addPrefillBlock(PrefillStep &step, const NemotronMambaWeights &block, metal::MetalBuffer input,
                       metal::MetalBuffer output) const;
  void addPrefillBlock(PrefillStep &step, const QwenAttentionWeights &block, metal::MetalBuffer input,
                       metal::MetalBuffer output) const;
  void addPrefillBlock(PrefillStep &step, const NemotronMoeWeights &block, metal::MetalBuffer input,
                       metal::MetalBuffer output) const;
  void addPrefillMamba(PrefillStep &step, const NemotronMambaWeights &block, uint32_t layer,
                       const QwenTargetPrefillSequence &sequence) const;

  void addProjection(VerifyStep &step, const ops::Projection &projection, metal::MetalBuffer input,
                     metal::MetalBuffer output, metal::MetalBuffer residual = {}) const;
  void addBlock(VerifyStep &step, const NemotronMambaWeights &block, metal::MetalBuffer input,
                metal::MetalBuffer output) const;
  void addBlock(VerifyStep &step, const QwenAttentionWeights &block, metal::MetalBuffer input,
                metal::MetalBuffer output) const;
  void addBlock(VerifyStep &step, const NemotronMoeWeights &block, metal::MetalBuffer input,
                metal::MetalBuffer output) const;

  const NemotronHWeights &weights_;
  metal::MetalBuffer decode_, prefill_;
  Scratch scratch_;
  PrefillScratch prefillScratch_;
};

} // namespace splash::model
