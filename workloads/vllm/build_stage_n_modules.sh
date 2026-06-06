#!/usr/bin/env bash
# Build the patched Stage N kernel/user modules.
#
# nvidia-uvm.ko depends on nvUvmInterface* symbols exported by nvidia.ko.
# Building only NV_KERNEL_MODULES=nvidia-uvm can fail in modpost with
# unresolved nvUvmInterface* symbols, so this script always builds both
# nvidia and nvidia-uvm in the same Kbuild invocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KERNEL_OPEN_DIR="$REPO_ROOT/kernel-module/nvidia-module/kernel-open"

JOBS="${JOBS:-$(nproc)}"
NV_KERNEL_MODULES="${NV_KERNEL_MODULES:-nvidia nvidia-uvm}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

case " $NV_KERNEL_MODULES " in
  *" nvidia "*) ;;
  *)
    echo "ERROR: NV_KERNEL_MODULES must include nvidia because nvidia-uvm depends on nvUvmInterface* exports." >&2
    echo "       Use: NV_KERNEL_MODULES=\"nvidia nvidia-uvm\"" >&2
    exit 1
    ;;
esac

case " $NV_KERNEL_MODULES " in
  *" nvidia-uvm "*) ;;
  *)
    echo "ERROR: NV_KERNEL_MODULES must include nvidia-uvm for Stage N." >&2
    exit 1
    ;;
esac

echo "Building kernel modules: $NV_KERNEL_MODULES"
if [ "$FORCE_REBUILD" = "1" ]; then
  echo "FORCE_REBUILD=1: cleaning kernel-open build outputs first..."
  make -C "$KERNEL_OPEN_DIR" clean
fi
make -C "$KERNEL_OPEN_DIR" -j"$JOBS" modules NV_KERNEL_MODULES="$NV_KERNEL_MODULES"

echo "Building vLLM UVM allocator shim..."
if [ "$FORCE_REBUILD" = "1" ]; then
  make -C "$SCRIPT_DIR/vllm/uvm_test" clean || true
fi
make -C "$SCRIPT_DIR/vllm/uvm_test" uvm

echo "Build complete:"
stat -c '  %y %s %n' \
  "$KERNEL_OPEN_DIR/nvidia.ko" \
  "$KERNEL_OPEN_DIR/nvidia-uvm.ko" \
  "$SCRIPT_DIR/vllm/uvm_test/uvm_allocator.so"
