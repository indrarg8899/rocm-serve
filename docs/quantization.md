# Quantization Guide

## Supported Methods

| Method | Bits | Speed | Quality | Memory Savings |
|--------|------|-------|---------|----------------|
| AWQ | 4-bit | Fast | Excellent | ~75% |
| GPTQ | 4-bit | Fast | Very Good | ~75% |
| GGUF | 4-8-bit | Medium | Good | 50-75% |
| FP8 | 8-bit | Fast | Near-lossless | 50% |

## AWQ (Activation-aware Weight Quantization)

AWQ is the recommended quantization method for rocm-serve. It preserves model quality by protecting salient weights during quantization.

### Using AWQ Models
```bash
# Download pre-quantized AWQ model from HuggingFace
rocm-serve --model TheBloke/Llama-2-70B-Chat-AWQ --quantization awq
```

### Performance
- Llama-2-70B-AWQ on 2× MI300X: ~12,000 tokens/sec
- Memory usage: ~35GB (vs ~140GB for FP16)
- Quality retention: >99% on standard benchmarks

## GPTQ (GPT Quantization)

GPTQ provides strong 4-bit quantization with optional group quantization.

```bash
rocm-serve --model TheBloke/Llama-2-70B-Chat-GPTQ --quantization gptq
```

## FP8 Quantization

MI300X has native FP8 support via the Matrix Core units. FP8 provides the best throughput with minimal quality loss.

```bash
rocm-serve --model meta-llama/Llama-3-70B-Instruct --dtype float16
# FP8 is automatically used for matrix operations when supported
```

## Choosing a Quantization Method

1. **VRAM is the bottleneck** → AWQ or GPTQ (4-bit)
2. **Quality matters most** → FP16/BF16 or FP8
3. **Need maximum throughput** → AWQ on fewer GPUs
4. **Running multiple models** → GGUF (flexible quantization levels)
