# CUDA Graph-safe MoE weight transport

This repository contains an experimental NVIDIA UVM + vLLM transport for
streaming a fixed set of MoE expert weights through CUDA Graph replay while
preserving each model `Parameter`'s original virtual address.

The validated path is the driver-backed `uvm_original_va` static-alias
backend. It builds the mapping once before Graph capture, copies the selected
weights to stable upload VAs on each forward, and lets the MoE kernels keep
reading the original `Parameter.data_ptr()` addresses. CUDA Graph replay does
not issue mapping ioctls or change PTEs.

Start with [the transport design and reproduction guide](docs/cudagraph_static_alias_transport.md).

Current design and experiment entry points:

- [original-VA Graph integration](experiments/vllm_uvm_original_va_graph_integration/README.md)
- [driver-level original-VA probes](experiments/uvm_static_alias_original_va/README.md)
- [complete technical design](docs/moe_phase_aware_offload_technical_design.md)
- [EuroSys evaluation plan](docs/eurosys_moe_offload_evaluation_plan.md)

The tree also contains a CUDA-VMM static-alias foundation and an experimental
dual-topology phase mode. They are kept build-consistent with the integration,
while all formal paper measurements are being regenerated under the EuroSys
protocol.
