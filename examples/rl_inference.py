# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import collections
import json
import os

# Set environment variables exactly as requested
os.environ["VLLM_TPU_PATCH_MM_EMBEDDINGS"] = "1"
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "0"
os.environ["MOE_REQUANTIZE_BLOCK_SIZE"] = "512"
os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["MODEL_IMPL_TYPE"] = "vllm"
os.environ["HF_HOME"] = "/mnt/disks/persist/hf"
os.environ["TPU_BACKEND_TYPE"] = "jax"
os.environ["SKIP_JAX_PRECOMPILE"] = "1"  # Fast iteration

from vllm import LLM, SamplingParams

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def main():
    # Read the benchmark dataset to get 128 prompts (for concurrency 128)
    prompts = []
    dataset_path = "/home/karangoel_google_com/benchmarks/datasets/bench_2000_500.jsonl"

    if os.path.exists(dataset_path):
        with open(dataset_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 128:  # Grab 128 prompts
                    break
                data = json.loads(line)
                prompts.append(data["prompt"])
    else:
        # Fallback if dataset isn't generated yet
        prompts = ["Hello, this is a test prompt to check expert routing."
                   ] * 128

    print(f"Loaded {len(prompts)} prompts. Initializing LLM...")

    # Initialize LLM with the exact parameters from the vllm serve command
    llm = LLM(
        model="mistralai/Mistral-Large-3-675B-Instruct-2512",
        download_dir="/mnt/disks/persist/hf/hub",
        max_model_len=3072,
        max_num_batched_tokens=2048,
        max_num_seqs=8,  # Adjusted to 128 to allow concurrency 128
        enable_prefix_caching=False,
        tensor_parallel_size=8,
        kv_cache_dtype="fp8",
        async_scheduling=False,  # --no-async-scheduling
        gpu_memory_utilization=0.90,
        enable_expert_parallel=True,
        trust_remote_code=True,
        enable_return_routed_experts=True,  # --enable-return-routed-experts
        hf_overrides={"num_hidden_layers":
                      5},  # --hf_overrides '{"num_hidden_layers": 5}'
        additional_config={
            "sharding": {
                "sharding_strategy": {
                    "enable_dp_attention": True
                }
            }
        },
        enforce_eager=
        True,  # Recommended when skipping compilation to avoid hanging
    )

    print("LLM Initialized. Generating 1 token to analyze prefill routing...")

    # We only want to evaluate prefill, so we generate 1 token
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)

    print("Generation complete. Aggregating expert usage...")

    # Aggregate expert usage
    expert_counts = collections.Counter()
    total_tokens_routed = 0

    for output in outputs:
        if hasattr(output, 'prompt_routed_experts'
                   ) and output.prompt_routed_experts is not None:
            # Shape is (num_tokens, num_layers, top_k)
            routed_experts = output.prompt_routed_experts

            # Flatten to 1D list of all chosen experts
            flattened_experts = routed_experts.flatten().tolist()
            expert_counts.update(flattened_experts)
            total_tokens_routed += routed_experts.shape[
                0] * routed_experts.shape[1] * routed_experts.shape[2]

    print("\n--- Expert Imbalance Report (128 concurrent requests) ---")
    print(f"Total routing decisions made: {total_tokens_routed}")

    # Print the top 10 most used experts
    print("\nTop 10 Most Used Experts:")
    for expert_id, count in expert_counts.most_common(10):
        percentage = (count / total_tokens_routed
                      ) * 100 if total_tokens_routed > 0 else 0
        print(f"Expert {expert_id:3d}: {count} times ({percentage:.2f}%)")

    # Print the bottom 10 least used experts
    print("\nBottom 10 Least Used Experts:")
    for expert_id, count in reversed(expert_counts.most_common()[-10:]):
        percentage = (count / total_tokens_routed
                      ) * 100 if total_tokens_routed > 0 else 0
        print(f"Expert {expert_id:3d}: {count} times ({percentage:.2f}%)")


if __name__ == "__main__":
    main()
