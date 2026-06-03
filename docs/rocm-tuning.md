# ROCm Tuning Guide

## GPU Configuration

### Enable Large Bar (Resizable BAR)
```bash
# Check current BAR size
cat /sys/class/drm/card*/device/virtual_size
# Should show >256MB for optimal performance
```

### GPU Memory Overdrive
```bash
# Enable GPU memory overdrive for maximum VRAM access
echo 1 | sudo tee /sys/class/drm/card*/device/mem_info_vram_used
```

## ROCm Environment Variables

```bash
# Optimize kernel dispatch latency
export HIP_LAUNCH_BLOCKING=0
export GPU_MAX_HW_QUEUES=8

# Enable flash attention for supported models
export ROCM_ENABLE_FLASH_ATTENTION=1

# Memory allocation strategy
export HIP_ALLOC_POLICY=bestfit

# Debug (if needed)
export HIP_DB=0
export AMD_LOG_LEVEL=0
```

## Kernel Optimization

### PagedAttention Block Size

| Block Size | Throughput | Latency | Memory Efficiency |
|-----------|-----------|---------|-------------------|
| 8 | High | Low | High |
| 16 | High | Medium | High (default) |
| 32 | Medium | Low | Medium |
| 64 | Low | Low | Low |

Recommendation: `--block-size 16` for most workloads.

### Batch Size Tuning

The scheduler auto-tunes batch size based on:
- `--max-num-seqs` (concurrent sequences)
- `--max-num-batched-tokens` (total tokens per batch)
- Available KV-cache memory

For MI300X (192GB HBM3), recommended starting values:
- Small models (7B-13B): `--max-num-seqs 512 --max-num-batched-tokens 16384`
- Large models (70B): `--max-num-seqs 256 --max-num-batched-tokens 8192`
- Huge models (405B+): `--max-num-seqs 64 --max-num-batched-tokens 4096`

## Multi-GPU (Tensor Parallelism)

### Topology Detection
```bash
# Check GPU topology
rocm-smi --showtopology
# Infinity Fabric connections show as "XGMI" links
```

### Recommended Configurations

| Model | GPUs | Command |
|-------|------|---------|
| Llama-3-8B | 1× MI300X | `--tensor-parallel 1` |
| Llama-3-70B | 4× MI300X | `--tensor-parallel 4` |
| Mixtral-8x22B | 4× MI300X | `--tensor-parallel 4` |
| Llama-3-405B | 8× MI300X | `--tensor-parallel 8` |

## Quantization

### AWQ (Recommended)
```bash
rocm-serve --model TheBloke/Llama-2-70B-Chat-AWQ --quantization awq
```

### GPTQ
```bash
rocm-serve --model TheBloke/Llama-2-70B-Chat-GPTQ --quantization gptq
```

### When to Quantize
- **INT4 (AWQ/GPTQ)**: 40-60% memory reduction, ~5-15% throughput loss
- **INT8**: 50% memory reduction, <5% throughput loss
- **FP16/BF16**: Full precision, best quality, requires more VRAM
