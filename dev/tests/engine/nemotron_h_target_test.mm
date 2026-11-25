// Nemotron-H's target (model/NemotronHTarget.cpp) end to end on a small GGUF
// of the family: Bolt's 88 blocks (40 Mamba-2, 8 NoPE attention, 40 latent
// MoE) at Super's routing (no per-layer tables) with a 256-wide hidden row,
// 16 experts of 64 and seeded Q8_0 projections, loaded as an installation
// holds it (target/, no draft) and run through the runtime: a prefill command
// of two prompts of different lengths, then a decode step of both lanes,
// with its routed experts resident and again streamed through the smallest
// store (ungated experts through the expert store). What a merge of the
// shared target code can miss shows here and not in the operators' tests,
// which build their own plans: each run must succeed (a target whose
// addPrefill misses a step the shared loop needs fails its first prefill),
// its logits must be finite, and the streamed run's must be the resident
// run's within bounds.
//
//   nemotron-h-target METALLIB

#include "Inkling2Fixtures.hpp"
#include "TestFiles.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Types.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ExpertStore.hpp"
#include "model/ModelFactory.hpp"
#include "model/ModelSizing.hpp"
#include "model/NemotronH.hpp"
#include "model/QwenState.hpp"
#include "model/Runtime.hpp"
#include "model/RuntimeInstrumentation.hpp"
#include "model/SsdReaders.hpp"
#include "ops/PageStorage.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

constexpr uint32_t kSeed = 3511;
// Two prompts: one past a page and a Mamba-2 verify block, one shorter.
constexpr std::array<uint32_t, 2> kPromptRows{300, 70};

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

// Bolt's blocks and every width the GGUF loader takes from the family
// (NemotronHGguf::kLayout: the Mamba-2 inner width, the latent and the shared
// expert), a 256-wide hidden row, a small vocabulary and 16 experts of 64, 4
// a token: not Bolt's 9, so that every MoE layer has the target's experts.
model::NemotronHLayout fixtureLayout() {
  model::QwenHybridLayout family = model::NemotronHLayout{};
  family.hiddenSize = 256;
  family.vocabularySize = 1024;
  family.experts = 16;
  family.expertsPerToken = 4;
  family.expertIntermediateSize = 64;
  family.family = "Nemotron-H-fixture";
  return model::NemotronHLayout(family);
}

using gguf_fixtures::Tensor;
namespace test_gguf = gguf_fixtures::test_gguf;

// The family's tensors as llama.cpp's converter writes them, of seeded values:
// Q8_0 projections, head and token table scaled as a checkpoint's
// (gguf_fixtures::scaleBlocks), F32 the rest: norms 1, the convolution's taps
// a quarter, A negative, dt's bias and D small, the router's scores small and
// its bias apart by a step an expert.
std::vector<Tensor> fixtureTensors(const model::NemotronHLayout &g) {
  const model::NemotronHLayout k{};
  std::vector<Tensor> tensors;
  const auto add = [&](const std::string &name, std::vector<uint64_t> dims, uint32_t type = model::ggml::kF32) {
    tensors.push_back({name, std::move(dims), type, {}});
  };
  const uint64_t hidden = g.hiddenSize, q8 = gguf_fixtures::kQ8_0;
  const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
  for (uint32_t layer = 0; layer < g.layers; ++layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    add(p + "attn_norm.weight", {hidden});
    switch (model::NemotronHLayout::block(layer)) {
    case model::NemotronBlock::Mamba:
      add(p + "ssm_in.weight", {hidden, k.mambaInProjectionRows}, q8);
      add(p + "ssm_conv1d.weight", {model::kGdnConvolutionTaps, g.convolutionDimension});
      add(p + "ssm_conv1d.bias", {g.convolutionDimension});
      add(p + "ssm_dt.bias", {g.gdnValueHeads});
      add(p + "ssm_a", {g.gdnValueHeads});
      add(p + "ssm_d", {g.gdnValueHeads});
      add(p + "ssm_norm.weight", {k.mambaInnerSize});
      add(p + "ssm_out.weight", {k.mambaInnerSize, hidden}, q8);
      break;
    case model::NemotronBlock::Attention:
      add(p + "attn_q.weight", {hidden, g.attentionWidth}, q8);
      add(p + "attn_k.weight", {hidden, kvRows}, q8);
      add(p + "attn_v.weight", {hidden, kvRows}, q8);
      add(p + "attn_output.weight", {g.attentionWidth, hidden}, q8);
      break;
    case model::NemotronBlock::Moe:
      add(p + "ffn_gate_inp.weight", {hidden, g.experts});
      add(p + "exp_probs_b.bias", {g.experts});
      add(p + "ffn_latent_down.weight", {hidden, k.moeLatentSize}, q8);
      add(p + "ffn_up_exps.weight", {k.moeLatentSize, g.expertIntermediateSize, g.experts}, q8);
      add(p + "ffn_down_exps.weight", {g.expertIntermediateSize, k.moeLatentSize, g.experts}, q8);
      add(p + "ffn_latent_up.weight", {k.moeLatentSize, hidden}, q8);
      add(p + "ffn_up_shexp.weight", {hidden, k.sharedIntermediateSize}, q8);
      add(p + "ffn_down_shexp.weight", {k.sharedIntermediateSize, hidden}, q8);
      break;
    }
  }
  add("output.weight", {hidden, g.vocabularySize}, q8);
  add("output_norm.weight", {hidden});
  add("token_embd.weight", {hidden, g.vocabularySize}, q8);
  uint32_t seed = kSeed;
  for (Tensor &tensor : tensors) {
    uint64_t rows = 1;
    for (size_t i = 1; i < tensor.dims.size(); ++i) rows *= tensor.dims[i];
    const uint64_t inputs = tensor.dims[0], count = inputs * rows;
    std::mt19937 rng(++seed);
    if (tensor.type == q8) {
      tensor.data = gguf_fixtures::fixture(gguf_fixtures::Fmt(gguf_format_of(tensor.type)), uint32_t(rows),
                                           uint32_t(inputs), seed);
      gguf_fixtures::scaleBlocks(tensor, inputs, rng);
      continue;
    }
    tensor.data = gguf_fixtures::floatValues(count, seed); // 0.5 to 1 in size
    const std::string &name = tensor.name;
    const auto scale = [&](float factor) {
      for (uint64_t i = 0; i < count; ++i)
        gguf_fixtures::storeFloat(tensor.data, i, gguf_fixtures::loadFloat(tensor.data, i) * factor);
    };
    if (name.ends_with("norm.weight")) {
      for (uint64_t i = 0; i < count; ++i) gguf_fixtures::storeFloat(tensor.data, i, 1.0f);
    } else if (name.ends_with("ssm_a")) {
      for (uint64_t i = 0; i < count; ++i)
        gguf_fixtures::storeFloat(tensor.data, i, -std::abs(gguf_fixtures::loadFloat(tensor.data, i)));
    } else if (name.ends_with("ssm_conv1d.weight")) {
      scale(0.25f);
    } else if (name.ends_with("ffn_gate_inp.weight")) {
      scale(0.1f * gguf_fixtures::projectionScale(inputs));
    } else if (name.ends_with("exp_probs_b.bias")) {
      for (uint64_t e = 0; e < count; ++e) gguf_fixtures::storeFloat(tensor.data, e, 0.1f * float((e * 7) % count));
    } else {
      scale(0.1f); // conv and dt's biases, D
    }
  }
  return tensors;
}

// The keys llama.cpp writes for nemotron_h_moe and the loader checks
// (NemotronHGguf::requireMetadata and arrays).
std::vector<test_gguf::Key> fixtureMetadata(const model::NemotronHLayout &g) {
  const model::NemotronHLayout k{};
  const std::string arch(model::NemotronHLayout::ggufArchitecture);
  std::vector<test_gguf::Key> keys{test_gguf::stringKey("general.architecture", arch)};
  const auto key = [&](const char *name, uint32_t value) {
    keys.push_back(test_gguf::uint32Key(arch + "." + name, value));
  };
  const auto perLayer = [&](const char *name, auto value) {
    std::vector<int32_t> values;
    for (uint32_t layer = 0; layer < g.layers; ++layer) values.push_back(int32_t(value(layer)));
    keys.push_back(test_gguf::int32ArrayKey(arch + "." + name, values));
  };
  key("block_count", g.layers);
  key("embedding_length", g.hiddenSize);
  key("attention.head_count", g.attentionQueryHeads);
  key("attention.key_length", g.attentionHeadDimension);
  key("attention.value_length", g.attentionHeadDimension);
  keys.push_back(test_gguf::float32Key(arch + ".attention.layer_norm_rms_epsilon", float(k.rmsEpsilon)));
  key("ssm.conv_kernel", model::kGdnConvolutionTaps);
  key("ssm.group_count", g.gdnKeyHeads);
  key("ssm.time_step_rank", g.gdnValueHeads);
  key("ssm.state_size", g.gdnHeadDimension);
  key("ssm.inner_size", k.mambaInnerSize);
  key("expert_count", g.experts);
  key("expert_shared_feed_forward_length", k.sharedIntermediateSize);
  key("expert_shared_count", 1);
  key("moe_latent_size", k.moeLatentSize);
  key("expert_gating_func", 2); // sigmoid
  keys.push_back(test_gguf::boolKey(arch + ".expert_weights_norm", true));
  keys.push_back(test_gguf::float32Key(arch + ".expert_weights_scale", k.routedScale));
  const auto width = [&](uint32_t layer) { return model::NemotronHLayout::layerExpertWidth(g, layer); };
  perLayer("feed_forward_length", width);
  perLayer("expert_feed_forward_length", width);
  perLayer("attention.head_count_kv", [&](uint32_t layer) {
    return model::NemotronHLayout::block(layer) == model::NemotronBlock::Attention ? g.attentionKvHeads : 0;
  });
  perLayer("expert_used_count", [&](uint32_t layer) {
    return model::NemotronHLayout::block(layer) == model::NemotronBlock::Moe ? g.expertsPerToken : 0;
  });
  return keys;
}

model::ModelDescriptor descriptor(const model::NemotronHLayout &layout) {
  return model::makeModelDescriptor("nemotron-h-target-test", layout, model::DFlashDraftLayout{},
                                    ops::visionLayout(layout.visionTower, layout.hiddenSize), model::TargetSource::Gguf,
                                    model::VisionSource::None);
}

std::vector<uint32_t> prompt(uint32_t rows, uint32_t lane, uint32_t vocabulary) {
  std::vector<uint32_t> ids(rows);
  for (uint32_t i = 0; i < rows; ++i) ids[i] = (i * 37 + 11 + 101 * lane) % vocabulary;
  return ids;
}

// A prefill command of the prompts, one a lane, then a decode step of every
// lane: each lane's logits after its prompt and after the token it selected.
std::vector<std::vector<float>> run(metal::MetalBackend &backend, model::LoadedModel &model,
                                    const std::vector<std::vector<uint32_t>> &prompts, uint32_t vocabulary) {
  const ops::ExecutionPlans operators(backend.capabilities(), ops::Int8Activations::Off);
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1, queryHostMemory, 0);
  MemoryGovernor::Scope claim(governor, governor.ledger().open({.cls = ClaimClass::Floor}));
  const kv::Layout layout = model.targetKvLayout(kv::Format::BFloat16);
  const uint32_t extent = layout.maximumExtentPages();
  const auto lanes = static_cast<uint32_t>(prompts.size());
  const uint32_t lanePages = (*std::max_element(kPromptRows.begin(), kPromptRows.end()) + 2 * kv::kPageTokens - 1) /
                             kv::kPageTokens;
  const uint32_t pageCount = (lanes * lanePages + extent - 1) / extent * extent;
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, pageCount, extent);
  for (uint32_t index = 0; index < pageCount / extent; ++index)
    require(static_cast<bool>(pages.allocateExtent(index)), "could not allocate the KV extents");
  model::QwenStateStorage states(backend, governor.allocationAdmission(),
                                 model.stateLayout(ops::GdnStateFormat::Float32), nullptr);
  model::RuntimeContext context{backend, model, pages, states, operators};
  model::Runtime executor(context);

  std::vector<std::vector<uint32_t>> tables(lanes);
  BatchPlan prefill{.kind = WorkKind::Prefill, .decodeStage = DecodeStage::Regular};
  std::vector<ModelBatchItem> items;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    EngineRequest request;
    request.id = lane + 1;
    request.prompt = prompts[lane];
    request.maxNewTokens = 2;
    // A first token that happens to be a stop token must not end the lane before its decode step.
    request.flags = RequestIgnoreEndOfSequence;
    executor.beginColdRequest(request.modelView(), lane);
    tables[lane].resize(lanePages);
    std::iota(tables[lane].begin(), tables[lane].end(), lane * lanePages);
    const auto rows = static_cast<uint32_t>(prompts[lane].size());
    prefill.items.push_back({request.id, rows, 0});
    ModelBatchItem &item = items.emplace_back(ModelBatchItem{request.id, 0, rows, tables[lane], 1});
    item.inputTokens = prompts[lane];
  }
  std::vector<std::vector<float>> logits;
  const auto keep = [&](uint32_t lane) {
    const std::span<const float> last = model::RuntimeInstrumentation::lastLogits(executor, lane);
    require(last.size() >= vocabulary, "the run kept fewer logits than the vocabulary");
    // Every logit, not only the selected one: a NaN elsewhere would slip past the comparison's maximum.
    require(std::all_of(last.begin(), last.begin() + vocabulary, [](float v) { return std::isfinite(v); }),
            "a run's logits are not finite");
    logits.emplace_back(last.begin(), last.begin() + vocabulary);
  };
  const std::vector<ModelStepResult> prefilled = executor.prefill(prefill, items);
  require(prefilled.size() == lanes, "a prefill of several sequences returned another count of results");
  BatchPlan decode{.kind = WorkKind::Decode, .decodeStage = DecodeStage::Regular};
  std::vector<ModelBatchItem> steps;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    require(prefilled[lane].failure.empty() && prefilled[lane].firstToken,
            "a lane's prefill failed: " + prefilled[lane].failure);
    keep(lane);
    decode.items.push_back({items[lane].requestId, 0});
    steps.push_back(ModelBatchItem{items[lane].requestId, prompts[lane].size(), 0, tables[lane], 1});
  }
  const std::vector<ModelStepResult> decoded = executor.decode(decode, steps);
  require(decoded.size() == lanes, "a decode step returned another count of results");
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    require(decoded[lane].failure.empty(), "a lane's decode step failed: " + decoded[lane].failure);
    keep(lane);
    executor.end(items[lane].requestId);
  }
  return logits;
}

// The largest difference of two runs' logits over the largest logit.
double relativeDifference(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b) {
  double difference = 0.0, largest = 0.0;
  for (size_t run = 0; run < a.size(); ++run)
    for (size_t i = 0; i < a[run].size(); ++i) {
      difference = std::max(difference, std::abs(double(a[run][i]) - double(b[run][i])));
      largest = std::max(largest, std::abs(double(a[run][i])));
    }
  return difference / std::max(largest, 1e-30);
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: nemotron-h-target METALLIB\n";
      return 2;
    }
    try {
      // Declared before the backend and the models, so it is removed after they close its file.
      const splash::test::TemporaryDirectory directory("nemotron-h-target");
      const std::filesystem::path &root = directory.path();
      const model::NemotronHLayout layout = fixtureLayout();
      std::filesystem::create_directories(root / "target");
      splash::test::writeFile(root / "target" / "nemotron-h-fixture.gguf",
                              test_gguf::file(fixtureMetadata(layout), fixtureTensors(layout)));
      metal::MetalBackend backend(argv[1]);
      const model::ModelDescriptor described = descriptor(layout);
      std::vector<std::vector<uint32_t>> prompts;
      for (uint32_t lane = 0; lane < kPromptRows.size(); ++lane)
        prompts.push_back(prompt(kPromptRows[lane], lane, layout.vocabularySize));

      std::vector<std::vector<float>> resident;
      {
        model::LoadedModel model = model::loadModel(backend, root, described);
        resident = run(backend, model, prompts, layout.vocabularySize);
      }
      std::printf("nemotron-h-target: a prefill of prompts of %u and %u rows and a decode step of both, resident: "
                  "finite ok\n",
                  kPromptRows[0], kPromptRows[1]);

      model::LoadedModel streamed = model::loadModel(
          backend, root, described,
          {.expertMemoryBytes = model::streamableWeights(root, described, {.streamExperts = true}).expertMinimumBytes,
           .readers = std::make_shared<model::SsdReaders>(model::kSsdReaders)});
      require(streamed.expertStore() != nullptr, "the fixture's routed experts did not stream");
      const double difference = relativeDifference(resident, run(backend, streamed, prompts, layout.vocabularySize));
      // The smallest store holds fewer slots than experts, so both the prefill's and the decode step's loads miss.
      const model::ExpertStoreStats stats = streamed.expertStore()->stats();
      require(stats.slots < stats.experts && stats.uses[0].loads > stats.uses[0].residentLoads &&
                  stats.uses[1].loads > stats.uses[1].residentLoads,
              "the streamed run's prefill or decode step read no routed experts");
      require(difference <= 1e-3, "the streamed run's logits are " + std::to_string(difference) +
                                      " of the largest apart from the resident run's");
      std::printf("nemotron-h-target: the same with the experts streamed through the smallest store: within %.2g of "
                  "the resident logits ok\n",
                  difference);
      std::printf("nemotron-h-target: passed\n");
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "nemotron-h-target: " << error.what() << '\n';
      return 1;
    }
  }
}
