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

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.sparse_core.ragged_gather_reduce import \
    dense_gather_reduce

jax.config.parse_flags_with_absl()


def reference_dense_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Reference implementation of dense gather reduce."""
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    out = jnp.sum(out, axis=1).astype(jnp.bfloat16)
    return out


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class DenseGatherReduceTest(jtu.JaxTestCase):
    _test_cases = [
        dict(out_size=o, hidden_size=h, dtype=d, reduce_group_size=rg)
        for o, h, d, rg in itertools.chain(
            itertools.product(
                [400, 840],
                [128, 512, 8192],
                [jnp.bfloat16, jnp.float32],
                [8, 5],
            ),
            itertools.product(
                [16384],
                [7168],
                [jnp.bfloat16],
                [8],
            ),
            itertools.product(
                [16384],
                [6144],
                [jnp.bfloat16],
                [8],
            ),
            itertools.product(
                [20480],
                [4096],
                [jnp.bfloat16],
                [10],
            ),
        )
    ]

    @parameterized.parameters(*_test_cases)
    def test_sc_dense_gather_reduce(self, out_size, hidden_size, dtype,
                                    reduce_group_size):
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(key, (out_size, ), jnp.bfloat16)

        actual = dense_gather_reduce(x, indices, topk_weights,
                                     reduce_group_size)
        # Correctness check.
        desired = reference_dense_gather_reduce(x, indices, topk_weights,
                                                reduce_group_size)
        np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
