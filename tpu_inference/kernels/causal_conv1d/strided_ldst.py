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

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def load_large_to_compact(vmem_ref,
                          dst_dtype: jnp.dtype | None = None) -> jax.Array:
    row_size, col_size = vmem_ref.shape
    src_dtype = vmem_ref.dtype
    vmem_ref = vmem_ref.bitcast(jnp.uint32).reshape(-1, col_size)
    tpu_info = pltpu.get_tpu_info()
    num_sublanes = tpu_info.num_sublanes

    strided_vmem_ref = vmem_ref.reshape(-1, num_sublanes, col_size)

    vreg_list = []
    for i in range(strided_vmem_ref.shape[0]):
        for j in range(num_sublanes):
            vreg = strided_vmem_ref[i, j:j + 1]
            vreg_list.append(vreg)
    out = jnp.stack(vreg_list, axis=0)

    if dst_dtype is None or dst_dtype == src_dtype:
        out = pltpu.bitcast(out, src_dtype)
    else:
        packing = 4 // src_dtype.itemsize
        unpacked_list = []
        for p in range(packing):
            unpacked = pltpu.unpack_elementwise(out,
                                                index=p,
                                                packed_dtype=src_dtype,
                                                unpacked_dtype=dst_dtype)
            unpacked_list.append(unpacked)
        out = jnp.stack(unpacked_list, axis=1)
    return out.reshape(row_size, 1, col_size)


def store_compact_to_large(vreg: jax.Array, vmem_ref):
    src_dtype = vreg.dtype
    dst_dtype = vmem_ref.dtype

    tpu_info = pltpu.get_tpu_info()
    num_sublanes = tpu_info.num_sublanes

    col_size = vreg.shape[-1]
    if src_dtype != dst_dtype:
        assert src_dtype.itemsize == 4
        assert dst_dtype.itemsize == 2
        packing = 4 // dst_dtype.itemsize

        vreg_list = []
        for row_start in range(0, vreg.shape[0], packing):
            unpacked_list = [vreg[row_start + i] for i in range(packing)]
            packed = pltpu.pack_elementwise(unpacked_list,
                                            packed_dtype=dst_dtype)
            vreg_list.append(packed)
    else:
        vreg = pltpu.bitcast(vreg, jnp.uint32).reshape(-1, col_size)
        vreg_list = jnp.split(vreg, vreg.shape[0], axis=0)

    vmem_ref = vmem_ref.bitcast(jnp.uint32).reshape(-1, col_size)
    strided_vmem_ref = vmem_ref.reshape(-1, num_sublanes, col_size)

    for i in range(strided_vmem_ref.shape[0]):
        for j in range(num_sublanes):
            strided_vmem_ref[i, j:j + 1] = vreg_list[i * num_sublanes + j]


def load_compact_to_large(vmem_ref: jax.Array) -> jax.Array:
    assert vmem_ref.dtype.itemsize == 4
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes

    vreg_list = []
    col_size = vmem_ref.shape[-1]
    lanes_per_col = col_size // num_lanes
    squeezed_ref = vmem_ref.reshape(-1, num_lanes)
    for i in range(lanes_per_col):
        vreg = squeezed_ref[i::lanes_per_col]
        vreg_list.append(vreg)
    return jnp.concatenate(vreg_list, axis=-1)


def store_large_to_compact(vreg: jax.Array, vmem_ref: jax.Array):
    raise NotImplementedError()
