"""
rocm-serve configuration: CLI arguments and server settings.

Centralized configuration with sensible defaults for MI300X hardware.
All settings can be overridden via CLI arguments or environment variables.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

import torch


def str2bool(v: str) -> bool:
    """Parse boolean CLI arguments."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


@dataclass
class ServerConfig:
    """
    Configuration for rocm-serve inference server.

    Defaults are tuned for a single MI300X (192GB HBM3) running
    a 70B parameter model in BF16. Adjust for your hardware.
    """

    # ── Model ──────────────────────────────────────────────────────
    model_name: str = ""
    dtype: torch.dtype = torch.bfloat16
    quantization: Optional[str] = None  # "awq", "gptq", "gguf"
    trust_remote_code: bool = True

    # ── Tensor Parallelism ─────────────────────────────────────────
    tensor_parallel: int = 1
    pipeline_parallel: int = 1

    # ── Server ─────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # ── Memory & Batching ──────────────────────────────────────────
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    max_model_len: Optional[int] = None
    block_size: int = 16  # PagedAttention block size in tokens

    # ── Scheduling ─────────────────────────────────────────────────
    preemption_policy: str = "recompute"  # "recompute" or "swap"
    scheduling_policy: str = "fcfs"  # "fcfs" or "fair_share"

    # ── Prefix Caching ─────────────────────────────────────────────
    enable_prefix_caching: bool = False
    prefix_cache_block_size: int = 64

    # ── Speculative Decoding ───────────────────────────────────────
    enable_speculative: bool = False
    speculative_draft_model: Optional[str] = None
    speculative_num_tokens: int = 5

    # ── API ────────────────────────────────────────────────────────
    api_key: Optional[str] = None
    disable_log_requests: bool = False

    @classmethod
    def from_args(cls) -> "ServerConfig":
        """Parse CLI arguments and return config."""
        parser = argparse.ArgumentParser(
            prog="rocm-serve",
            description="Production LLM inference server for AMD ROCm/MI300X",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  rocm-serve --model meta-llama/Llama-3-70B-Instruct --tensor-parallel 4
  rocm-serve --model mistralai/Mixtral-8x22B-Instruct-v0.1 --quantization awq
  rocm-serve --model meta-llama/Llama-3-8B-Instruct --port 8001
            """,
        )

        # Model
        parser.add_argument("--model", type=str, required=True,
                            help="HuggingFace model ID or local path")
        parser.add_argument("--dtype", type=str, default="bfloat16",
                            choices=["float16", "bfloat16", "float32", "auto"],
                            help="Data type for model weights")
        parser.add_argument("--quantization", type=str, default=None,
                            choices=["awq", "gptq", "gguf"],
                            help="Quantization method")
        parser.add_argument("--trust-remote-code", type=str2bool, default=True,
                            help="Trust remote code from HuggingFace")

        # Tensor parallelism
        parser.add_argument("--tensor-parallel", type=int, default=1,
                            help="Number of GPUs for tensor parallelism")
        parser.add_argument("--pipeline-parallel", type=int, default=1,
                            help="Number of GPUs for pipeline parallelism")

        # Server
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--log-level", type=str, default="info",
                            choices=["debug", "info", "warning", "error"])

        # Memory & batching
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                            help="Fraction of GPU memory for KV-cache (0.0-1.0)")
        parser.add_argument("--max-num-seqs", type=int, default=256,
                            help="Maximum concurrent sequences per batch")
        parser.add_argument("--max-num-batched-tokens", type=int, default=8192,
                            help="Maximum total tokens per batch")
        parser.add_argument("--max-model-len", type=int, default=None,
                            help="Maximum sequence length (model default if not set)")
        parser.add_argument("--block-size", type=int, default=16,
                            help="PagedAttention block size in tokens")

        # Scheduling
        parser.add_argument("--preemption-policy", type=str, default="recompute",
                            choices=["recompute", "swap"],
                            help="Preemption strategy for long requests")
        parser.add_argument("--scheduling-policy", type=str, default="fcfs",
                            choices=["fcfs", "fair_share"],
                            help="Request scheduling policy")

        # Prefix caching
        parser.add_argument("--enable-prefix-caching", type=str2bool, default=False,
                            help="Enable automatic prefix caching across requests")

        # Speculative decoding
        parser.add_argument("--enable-speculative", type=str2bool, default=False,
                            help="Enable speculative decoding")
        parser.add_argument("--speculative-draft-model", type=str, default=None,
                            help="Draft model for speculative decoding")
        parser.add_argument("--speculative-num-tokens", type=int, default=5,
                            help="Number of speculative tokens per step")

        # API
        parser.add_argument("--api-key", type=str, default=None,
                            help="API key for authentication")
        parser.add_argument("--disable-log-requests", type=str2bool, default=False,
                            help="Disable per-request logging")

        args = parser.parse_args()

        # Parse dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if args.dtype == "auto":
            # Auto-select: bfloat16 if available (MI300X native), else float16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = dtype_map[args.dtype]

        return cls(
            model_name=args.model,
            dtype=dtype,
            quantization=args.quantization,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel=args.tensor_parallel,
            pipeline_parallel=args.pipeline_parallel,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            block_size=args.block_size,
            preemption_policy=args.preemption_policy,
            scheduling_policy=args.scheduling_policy,
            enable_prefix_caching=args.enable_prefix_caching,
            enable_speculative=args.enable_speculative,
            speculative_draft_model=args.speculative_draft_model,
            speculative_num_tokens=args.speculative_num_tokens,
            api_key=args.api_key,
            disable_log_requests=args.disable_log_requests,
        )
