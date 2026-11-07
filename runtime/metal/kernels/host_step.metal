// The system memory scope, which Metal's public headers leave out: the only
// fence under which a running kernel sees a store of the CPU. Device-scope
// atomics, volatile and coherent(device) loads never do (M5 Max, M5 Pro, M3
// Max), nor do coherent(device) loads or relaxed device-scope atomic loads
// behind a device-scope seq_cst fence (M5 Max, M5 Pro, M6).
#pragma METAL internals : enable
#include <metal_stdlib>

using namespace metal;

constexpr constant thread_scope kSystemScope = static_cast<thread_scope>(3);

// Keeps the GPU awake through a host step (metal/MetalBackend.mm,
// encodeHostStep): one thread spins until the runner's release counter
// reaches release.x, at most release.y loads, and sees the runner's store
// within 2 us; it then writes release.x to seen[0] and the loads it made
// before to seen[1], from which the backend learns that the spin ends at its
// release while it runs (checkHostStepSpin). The buffer
// after it still waits for the step's event, which the runner raises before
// the counter: if the spin ends at its bound, the GPU only idles as it would
// without it.
kernel void host_step_wait(volatile coherent(system) device const uint *released [[buffer(0)]],
                           constant uint2 &release [[buffer(1)]], device uint *seen [[buffer(2)]]) {
  for (uint n = 0; n < release.y; ++n) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, kSystemScope);
    if (as_type<int>(released[0] - release.x) >= 0) {
      seen[1] = n;
      seen[0] = release.x;
      return;
    }
  }
}

// An inline host step (HostStepGpu::Inline) hands the runner what the GPU
// wrote before it while its buffer runs on: a buffer's end is what makes the
// GPU's stores visible to the CPU, and the CPU reads stale values without
// it. host_step_publish stores each word of `words` words back through the
// system scope, from `from`, the same memory read as the GPU sees it.
kernel void host_step_publish(device const uint *from [[buffer(0)]], volatile coherent(system) device uint *to [[buffer(1)]],
                              constant uint &words [[buffer(2)]], uint t [[thread_position_in_grid]],
                              uint threads [[threads_per_grid]]) {
  for (uint i = t; i < words; i += threads) to[i] = from[i];
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, kSystemScope);
}

// Then raises the step's flag to `value`, which the runner waits for in place
// of the event a buffer's end signals.
kernel void host_step_ready(volatile coherent(system) device uint *flag [[buffer(0)]], constant uint &value [[buffer(1)]]) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, kSystemScope);
  flag[0] = value;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, kSystemScope);
}
