# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



def patch_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content
    for target, replacement in replacements:
        if target not in content:
            print(f"Warning: target string not found in {filepath}:\n{target}")
            continue
        content = content.replace(target, replacement)

    if content == original_content:
        print(f"No changes made to {filepath}")
    else:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Successfully patched {filepath}")


# 1. Patch outputs.py
replacements_outputs = [
    ("        num_cached_tokens: int | None = None,\n        *,\n        kv_transfer_params: dict[str, Any] | None = None,",
     "        num_cached_tokens: int | None = None,\n        prompt_routed_experts: np.ndarray | None = None,\n        *,\n        kv_transfer_params: dict[str, Any] | None = None,"
     ),
    ("        self.num_cached_tokens = num_cached_tokens\n        self.kv_transfer_params = kv_transfer_params",
     "        self.num_cached_tokens = num_cached_tokens\n        self.kv_transfer_params = kv_transfer_params\n        self.prompt_routed_experts = prompt_routed_experts"
     ),
    ("        self.finished |= next_output.finished\n        self.kv_transfer_params = next_output.kv_transfer_params",
     "        self.finished |= next_output.finished\n        self.kv_transfer_params = next_output.kv_transfer_params\n        if getattr(self, 'prompt_routed_experts', None) is None:\n            self.prompt_routed_experts = getattr(next_output, 'prompt_routed_experts', None)"
     ),
    ("            f\"lora_request={self.lora_request}, \"\n            f\"num_cached_tokens={self.num_cached_tokens})\"",
     "            f\"lora_request={self.lora_request}, \"\n            f\"num_cached_tokens={self.num_cached_tokens}, \"\n            f\"prompt_routed_experts={getattr(self, 'prompt_routed_experts', None)})\""
     )
]

patch_file("/workspace/vllm/vllm/outputs.py", replacements_outputs)

# 2. Patch output_processor.py
replacements_output_processor = [
    ("        # Routed experts accumulation (prompt + sample chunks)\n        self.routed_experts_chunks: list[np.ndarray] = []",
     "        # Routed experts accumulation (prompt + sample chunks)\n        self.prompt_routed_experts: np.ndarray | None = None\n        self.routed_experts_chunks: list[np.ndarray] = []"
     ),
    ("            if engine_core_output.routed_experts is not None:\n                req_state.routed_experts_chunks.append(\n                    engine_core_output.routed_experts\n                )",
     "            if engine_core_output.routed_experts is not None:\n                if req_state.is_prefilling:\n                    req_state.prompt_routed_experts = (\n                        engine_core_output.routed_experts\n                    )\n                else:\n                    req_state.routed_experts_chunks.append(\n                        engine_core_output.routed_experts\n                    )"
     ),
    ("        return RequestOutput(\n            request_id=external_req_id,  # request_id is what was provided externally\n            lora_request=self.lora_request,\n            prompt=self.prompt,\n            prompt_token_ids=prompt_token_ids,\n            prompt_logprobs=prompt_logprobs,\n            outputs=cast(list[CompletionOutput], outputs),\n            finished=finished,\n            kv_transfer_params=kv_transfer_params,\n            num_cached_tokens=self.num_cached_tokens,\n            metrics=self.stats,\n        )",
     "        return RequestOutput(\n            request_id=external_req_id,  # request_id is what was provided externally\n            lora_request=self.lora_request,\n            prompt=self.prompt,\n            prompt_token_ids=prompt_token_ids,\n            prompt_logprobs=prompt_logprobs,\n            outputs=cast(list[CompletionOutput], outputs),\n            finished=finished,\n            kv_transfer_params=kv_transfer_params,\n            num_cached_tokens=self.num_cached_tokens,\n            metrics=self.stats,\n            prompt_routed_experts=self.prompt_routed_experts,\n        )"
     )
]

patch_file("/workspace/vllm/vllm/v1/engine/output_processor.py",
           replacements_output_processor)
