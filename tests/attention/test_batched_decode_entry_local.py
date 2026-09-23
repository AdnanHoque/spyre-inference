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

"""Card-free tests for the opt-in entry-local batched-decode body."""

import pytest
import torch

from spyre_inference.v1.attention.ops import batched_decode_head_major as bdhm
from spyre_inference.v1.attention.ops import tile_loop

KV, Q, BLOCK, HEAD = 1, 1, 4, 8
SCALE = 0.3


def _inputs(b, bpc, c):
    """Synthetic #876 metadata: pages [J, B] and mask [J, B, 1, 1, block]."""
    j_total = c * bpc
    torch.manual_seed(0)
    num_pages = 64
    k_pages = torch.randn(num_pages, KV, BLOCK, HEAD)
    v_pages = torch.randn(num_pages, KV, BLOCK, HEAD)
    # Distinct, in-range page ids (e = j*B + s) so a wrong read is observable.
    page_ids = (torch.arange(j_total * b).reshape(j_total, b) + 1).to(torch.int32)
    mask = torch.zeros(j_total, b, 1, 1, BLOCK, dtype=torch.float32)
    return k_pages, v_pages, page_ids, mask, j_total


def _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, cap):
    """Per-sequence softmax over every slot, independent of the kernel's recurrence."""
    out = torch.empty(b, KV, Q, HEAD)
    for s in range(b):
        k = torch.cat([k_pages[int(page_ids[jj, s])] for jj in range(j_total)], dim=0)
        v = torch.cat([v_pages[int(page_ids[jj, s])] for jj in range(j_total)], dim=0)
        k = k.reshape(j_total * BLOCK, KV, HEAD)
        v = v.reshape(j_total * BLOCK, KV, HEAD)
        m = torch.cat([mask[jj, s, 0, 0, :] for jj in range(j_total)])
        for h in range(KV):
            for qq in range(Q):
                sc = (query[s, h, qq] @ k[:, h, :].transpose(0, 1)) * SCALE
                if cap > 0.0:
                    sc = torch.tanh(sc / cap) * cap
                out[s, h, qq] = torch.softmax(sc + m, dim=-1) @ v[:, h, :]
    return out.reshape(b, KV * Q, HEAD)


def _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, cap=0.0):
    return bdhm.batched_decode_head_major_kernel(
        query, rep, k_pages, v_pages, page_ids, mask, SCALE, b, bpc, KV, Q, BLOCK, HEAD,
        logits_soft_cap=cap,
    )


@pytest.fixture(autouse=True)
def python_walk(monkeypatch):
    # The body math is under test; drive the Python walk (carry=None).
    monkeypatch.setattr(tile_loop, "USE_FOR_EACH_TILE", False)


@pytest.mark.parametrize("c", [1, 2])
def test_flag_off_matches_reference(monkeypatch, c):
    monkeypatch.setattr(bdhm, "ENTRY_LOCAL_DECODE", False)
    b, bpc = 2, 2
    query = torch.randn(b, KV, Q, HEAD)
    rep = torch.arange(b).repeat(bpc)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c)
    got = _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc)
    want = _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, 0.0)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_flag_on_one_chunk_is_upstream(monkeypatch):
    b, bpc, c = 2, 2, 1
    query = torch.randn(b, KV, Q, HEAD)
    rep = torch.arange(b).repeat(bpc)
    k_pages, v_pages, page_ids, mask, _ = _inputs(b, bpc, c)
    monkeypatch.setattr(bdhm, "ENTRY_LOCAL_DECODE", False)
    off = _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc)
    monkeypatch.setattr(bdhm, "ENTRY_LOCAL_DECODE", True)
    on = _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc)
    assert torch.equal(on, off)


@pytest.mark.parametrize("c", [2, 3])
def test_flag_on_multi_chunk_masked(monkeypatch, c):
    monkeypatch.setattr(bdhm, "ENTRY_LOCAL_DECODE", True)
    b, bpc, cap = 2, 2, 2.0
    query = torch.randn(b, KV, Q, HEAD)
    rep = torch.arange(b).repeat(bpc)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c)
    # Partial page at (s=0, slot 0); a slot padded in EVERY chunk for s=1 (slots
    # 0, bpc, 2bpc, ...); one slot valid early and masked in a later chunk.
    mask[0, 0, 0, 0, 2:] = float("-inf")
    for jj in range(0, j_total, bpc):
        mask[jj, 1, 0, 0, :] = float("-inf")
    mask[1 + bpc, 0, 0, 0, :] = float("-inf")
    got = _call(query, rep, k_pages, v_pages, page_ids, mask, b, bpc, cap=cap)
    want = _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, cap)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_body_matches_reference_with_init_state():
    """The FET-path init convention ``(-inf, 0, 0)`` threads to the same result."""
    b, bpc, c = 2, 2, 2
    query = torch.randn(b, KV, Q, HEAD)
    rep = torch.arange(b).repeat(bpc)
    k_pages, v_pages, page_ids, mask, j_total = _inputs(b, bpc, c)
    entries = b * bpc
    q = query.index_select(0, rep).reshape(entries, KV, Q, HEAD)

    state_shape = (bpc, b, KV, Q, 1)
    carry = (
        torch.full(state_shape, float("-inf")),
        torch.zeros(state_shape),
        torch.zeros(bpc, b, KV, Q, HEAD),
    )
    for chunk in range(c):
        tile_ids = page_ids[chunk * bpc : (chunk + 1) * bpc]
        tile_mask = mask[chunk * bpc : (chunk + 1) * bpc]
        k_page = k_pages[tile_ids].reshape(entries, KV, BLOCK, HEAD)
        v_page = v_pages[tile_ids].reshape(entries, KV, BLOCK, HEAD)
        scores = torch.matmul(q, k_page.transpose(-2, -1)) * SCALE
        sc = scores.reshape(bpc, b, KV, Q, BLOCK) + tile_mask
        carry = bdhm._entry_local_update(
            carry if chunk else None, sc, v_page, entries, bpc, b, KV, Q, BLOCK, HEAD
        )
    got = bdhm._merge_entry_local(*carry, b, bpc, KV, Q, HEAD)
    want = _reference(query, k_pages, v_pages, page_ids, mask, b, j_total, 0.0)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
