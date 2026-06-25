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

import time

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.sparse_core.dense_gather_reduce import \
    dense_gather_reduce


@jax.jit(static_argnums=(3, ))
def xla_dense_gather_reduce(x, indices, topk_weights, reduce_group_size):
    token_hidden_full = x[indices]
    cur_sorted = token_hidden_full.reshape(
        (-1, reduce_group_size, x.shape[-1]))
    cur_topk_weights = jnp.expand_dims(topk_weights, axis=-1)
    cur_weighted = cur_sorted.astype(jnp.float32) * cur_topk_weights.astype(
        jnp.float32)
    mask = jnp.full((indices.shape[0], ),
                    True).reshape(-1, reduce_group_size, 1)
    cur_masked = jnp.where(mask, cur_weighted, 0.0)
    out = cur_masked.sum(axis=-2)
    return out.astype(x.dtype)


class DenseGatherReduceBenchmark(parameterized.TestCase):

    _results = []

    _benchmark_cases = [
        dict(
            out_size=16384,
            hidden_size=8192,
            dtype=jnp.bfloat16,
            reduce_group_size=8,
        ),
        # Mistral 3: prefill seq len=2k, bs=16, topk=4, embedding=7168
        dict(
            out_size=131072,
            hidden_size=7168,
            dtype=jnp.bfloat16,
            reduce_group_size=4,
        ),
        # Mistral 3: prefill seq len=2k, bs=8, topk=4, embedding=7168
        dict(
            out_size=65536,
            hidden_size=7168,
            dtype=jnp.bfloat16,
            reduce_group_size=4,
        ),
    ]

    @parameterized.parameters(*_benchmark_cases)
    def test_benchmark_dense_gather_reduce(self, out_size, hidden_size, dtype,
                                           reduce_group_size):
        prefix = f'[{self._testMethodName}/{out_size}/{hidden_size}/{reduce_group_size}/{dtype}]'
        tpu_info = pltpu.get_tpu_info()
        print("TPU INFO:", tpu_info)
        if tpu_info.sparse_core is not None:
            print("SPARSE CORE INFO:", tpu_info.sparse_core)

        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size),
                              jnp.float32).astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights_2d = jax.random.normal(
            key, (out_size // reduce_group_size, reduce_group_size),
            jnp.float32).astype(dtype)

        # Warmup
        xla_out = xla_dense_gather_reduce(x, indices, topk_weights_2d,
                                          reduce_group_size)
        xla_out.block_until_ready()
        pallas_out = dense_gather_reduce(x, indices, topk_weights_2d,
                                         reduce_group_size)
        pallas_out.block_until_ready()

        # Benchmark XLA
        num_runs = 50
        t0 = time.perf_counter()
        for _ in range(num_runs):
            xla_out = xla_dense_gather_reduce(x, indices, topk_weights_2d,
                                              reduce_group_size)
            xla_out.block_until_ready()
        xla_time = (time.perf_counter() -
                    t0) / num_runs * 1e6  # to microseconds

        # Benchmark Pallas
        t0 = time.perf_counter()
        for _ in range(num_runs):
            pallas_out = dense_gather_reduce(x, indices, topk_weights_2d,
                                             reduce_group_size)
            pallas_out.block_until_ready()
        pallas_time = (time.perf_counter() -
                       t0) / num_runs * 1e6  # to microseconds

        print(f'{prefix} XLA: {xla_time:.2f} us')
        print(f'{prefix} Pallas: {pallas_time:.2f} us')

        self.__class__._results.append({
            'out_size': out_size,
            'hidden_size': hidden_size,
            'reduce_group_size': reduce_group_size,
            'xla_us': xla_time,
            'pallas_us': pallas_time,
            'speedup': xla_time / pallas_time,
        })

        # Correctness check.
        np.testing.assert_allclose(pallas_out, xla_out, atol=1e-2, rtol=1e-2)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if not cls._results:
            return

        print('\n=== BENCHMARK RESULTS ===')
        print(
            '| Shape (Out x Hidden) | Reduce Group Size (topk) | XLA (us) | Pallas'
            ' (us) | Speedup |')
        print('| :--- | :--- | :--- | :--- | :--- |')
        for r in cls._results:
            shape = f'{r["out_size"]} x {r["hidden_size"]}'
            speedup_str = f'{r["speedup"]:.2f}x'
            print(f'| {shape} | {r["reduce_group_size"]} | {r["xla_us"]:.2f} |'
                  f' {r["pallas_us"]:.2f} | {speedup_str} |')
        print('=========================\n')


if __name__ == '__main__':
    absltest.main()
