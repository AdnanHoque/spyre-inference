# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batched multi-sequence decode over a head-major KV cache.

The reduction is ``batched_decode``'s, chunk for chunk; only the page read differs.
One index row per page, as token-major, but this layout stores the page head-major
already, so the permute token-major does per chunk disappears.

Not the folded ``(page, kv_head)`` rows the per-sequence kernel gathers: the two move the
same bytes, but folding costs ``num_kv_heads`` times the index entries, and gather time
scales with entries rather than bytes -- measured at ~2x the kernel time for 8 kv heads.
"""

import torch

from spyre_inference import envs
from spyre_inference.v1.attention.ops.tile_loop import USE_FOR_EACH_TILE, walk_tiles

# Read at module scope like the walk-mode flag: process-wide choices.
ENTRY_LOCAL_DECODE = envs.SPYRE_ATTN_ENTRY_LOCAL_DECODE
# TEMPORARY WORKAROUND (torch-spyre#4603): chunk-major index walked one chunk per trip.
TEMP_SPLIT_INDEX = envs.SPYRE_ATTN_TEMP_SPLIT_INDEX


def _entry_local_update(
    carry,
    sc,
    v_page,
    entries,
    blocks_per_chunk,
    num_seqs,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
):
    """One chunk's per-block-slot online-softmax update; state kept across chunks."""

    def pv(probs):
        return torch.matmul(
            probs.reshape(entries, num_kv_heads, num_queries_per_kv, block_size), v_page
        ).reshape(blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv, head_size)

    sc = torch.clamp(sc, min=torch.finfo(sc.dtype).min)
    pos_max = torch.amax(sc, dim=-1, keepdim=True)
    if carry is None:
        probs = torch.exp(sc - pos_max)
        return pos_max, torch.sum(probs, dim=-1, keepdim=True), pv(probs)
    tile_max, tile_sum, tile_output = carry
    rescale = torch.exp(-torch.relu(pos_max - tile_max))
    new_max = torch.maximum(tile_max, pos_max)
    probs = torch.exp(sc - new_max)
    return (
        new_max,
        tile_sum * rescale + torch.sum(probs, dim=-1, keepdim=True),
        tile_output * rescale + pv(probs),
    )


def _merge_entry_local(
    tile_max,
    tile_sum,
    tile_output,
    num_seqs,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
):
    """The one final merge over the ``blocks_per_chunk`` slots held in the carry."""
    merged_max = torch.amax(tile_max, dim=0, keepdim=True)
    weight = torch.exp(tile_max - merged_max)
    merged_sum = torch.sum(weight * tile_sum, dim=0)
    merged_out = torch.sum(weight * tile_output, dim=0)
    return (merged_out / merged_sum).reshape(num_seqs, num_kv_heads * num_queries_per_kv, head_size)


def batched_decode_head_major_kernel(
    query,
    rep_row_ids,
    k_pages,
    v_pages,
    chunk_page_ids,
    mask_by_chunk,
    scale,
    num_seqs,
    blocks_per_chunk,
    num_kv_heads,
    num_queries_per_kv,
    block_size,
    head_size,
    logits_soft_cap=0.0,
    out=None,
):
    """Shapes as in ``batched_decode_kernel``, except k/v_pages are the unfolded
    head-major cache, [num_pages_total, num_kv_heads, block_size, head_size].

    One index row per page rather than per (page, kv_head): the gather then splits on the
    axis that stays the matmul's batch dim 0, as token-major's does, and the page still
    arrives head-major so there is no permute either.
    """
    num_heads = num_kv_heads * num_queries_per_kv
    entries = num_seqs * blocks_per_chunk
    # Workaround walk only exists on the tiled path; the Python walk keeps #876's index.
    split_index = TEMP_SPLIT_INDEX and USE_FOR_EACH_TILE
    num_chunks = (
        chunk_page_ids.shape[0] if split_index else chunk_page_ids.shape[0] // blocks_per_chunk
    )
    # One chunk is upstream's flat branch; entry-local only pays across >1 chunk.
    use_entry_local = ENTRY_LOCAL_DECODE and num_chunks > 1
    q = query.index_select(0, rep_row_ids).reshape(
        entries, num_kv_heads, num_queries_per_kv, head_size
    )

    def reduce_chunk(probs, v_page):
        """Sum the chunk's blocks; its slots share one max, so this needs no rescale."""
        chunk_sum = torch.sum(torch.sum(probs, dim=-1, keepdim=True), dim=0, keepdim=True)
        chunk_out = torch.sum(
            torch.matmul(
                probs.reshape(entries, num_kv_heads, num_queries_per_kv, block_size),
                v_page,
            ).reshape(blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv, head_size),
            dim=0,
            keepdim=True,
        )
        return chunk_sum, chunk_out

    def chunk_body(carry, tiles):
        page_ids, mask_rows, k_pages, v_pages, q = tiles
        if split_index:
            # Workaround walk: consume the chunk axis -> [E, 1] pages the gather splits on.
            page_ids = page_ids[0]
            mask_rows = mask_rows.reshape(blocks_per_chunk, num_seqs, *mask_rows.shape[3:])
        # Subscripting, not index_select: behind a 1-D index the entry axis splits only in
        # whole 32-entry sticks, and flattening the int32 tile first needs an unsupported
        # staging layout. Costs the eager path, which the preconditions decline.
        k_page = k_pages[page_ids].reshape(entries, num_kv_heads, block_size, head_size)
        v_page = v_pages[page_ids].reshape(entries, num_kv_heads, block_size, head_size)
        scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale
        if logits_soft_cap > 0.0:
            # Before the mask add: tanh(-inf/cap)*cap is -cap, not -inf, so
            # capping after it would un-mask the padded lanes.
            scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        # Leading-axis split only: torch-spyre rejects merging a permuted axis pair, and
        # the mask's advancing read window must stay unflattened.
        sc = scores.reshape(
            blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv, block_size
        )
        sc = sc + mask_rows
        if use_entry_local:
            return (
                _entry_local_update(
                    carry,
                    sc,
                    v_page,
                    entries,
                    blocks_per_chunk,
                    num_seqs,
                    num_kv_heads,
                    num_queries_per_kv,
                    block_size,
                    head_size,
                ),
                None,
            )
        chunk_max = torch.amax(torch.amax(sc, dim=-1, keepdim=True), dim=0, keepdim=True)

        # The running max drives exp(), not the chunk's own: a chunk wholly past a
        # sequence's length is -inf throughout and exp(-inf - -inf) is NaN.
        # `carry is None` is required for SPYRE_ATTN_FOR_EACH_TILE=0
        if carry is None:
            chunk_sum, chunk_out = reduce_chunk(torch.exp(sc - chunk_max), v_page)
            return (chunk_max, chunk_sum, chunk_out), None

        tile_max, tile_sum, tile_output = carry
        # Read tile_max before the maximum that supersedes it, or the tiled lowering
        # copies the whole carry every trip. Identical to exp(tile_max - new_max).
        rescale = torch.exp(-torch.relu(chunk_max - tile_max))
        new_max = torch.maximum(tile_max, chunk_max)
        chunk_sum, chunk_out = reduce_chunk(torch.exp(sc - new_max), v_page)
        return (
            new_max,
            tile_sum * rescale + chunk_sum,
            tile_output * rescale + chunk_out,
        ), None

    slots = (
        (blocks_per_chunk, num_seqs, num_kv_heads, num_queries_per_kv)
        if use_entry_local
        else (1, num_seqs, num_kv_heads, num_queries_per_kv)
    )
    state_shape = (*slots, 1)
    out_shape = (*slots, head_size)
    state_kwargs = {"dtype": q.dtype, "device": q.device}
    carry, _ = walk_tiles(
        chunk_body,
        (chunk_page_ids, mask_by_chunk, k_pages, v_pages, q),
        dims=(0, 0, None, None, None),
        tile_size=1 if split_index else blocks_per_chunk,
        init=(
            torch.full(state_shape, float("-inf"), **state_kwargs),
            torch.zeros(state_shape, **state_kwargs),
            torch.zeros(out_shape, **state_kwargs),
        ),
    )
    if use_entry_local:
        tile_max, tile_sum, tile_output = carry
        attn = _merge_entry_local(
            tile_max, tile_sum, tile_output, num_seqs, num_kv_heads, num_queries_per_kv, head_size
        )
    else:
        _, tile_sum, tile_output = carry
        attn = (tile_output / tile_sum).reshape(num_seqs, num_heads, head_size)
    if out is not None:
        # Offset 0, so torch-spyre#3770 does not apply; rows past the batch are
        # don't-care and kept finite by the builder.
        out[:num_seqs].copy_(attn)
        return out
    return attn
