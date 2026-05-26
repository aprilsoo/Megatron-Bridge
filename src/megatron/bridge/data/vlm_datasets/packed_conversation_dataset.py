# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
Dataset-side fixed-token packing for text-only VLM conversation examples.

This module provides :class:`VLMPackedConversationDataset`, a subclass of
:class:`VLMConversationDataset` that performs packing **lazily** using a
streaming token pool. Each ``__getitem__`` call returns a single fixed-length
packed sample (1D tensors plus ``cu_seqlens`` / ``max_seqlen``) ready for THD
attention. The collate function only stacks ``B`` such pre-packed samples
into a ``[1, B*L]`` mega-sequence and merges per-sample ``cu_seqlens`` with
row offsets.

Pool-based streaming pipeline:

1. Maintain a tokenized ``pool`` of up to ``pool_size`` examples.
2. When the ready-packings queue is empty, top up the pool by tokenizing the
   next base examples (cycling forever once the source is exhausted) and run
   :class:`BalanceBatchManager` over the full pool to produce a batch of
   packed samples; the leftover items remain in the pool for the next round.
3. ``__getitem__`` pops one ready packed sample from the queue.

This avoids paying the cost of tokenizing and packing the entire dataset up
front, which is intractable for large corpora.
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Tuple

import torch

from megatron.bridge.data.vlm_datasets.collate import (
    create_multiturn_loss_mask_by_search,
)
from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.data.vlm_datasets.token_utils import extract_skipped_token_ids


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BalanceBatchManager
# ---------------------------------------------------------------------------


class BalanceBatchManager:
    """Token-budget greedy packer.

    Adapted from the user-supplied ``demo_packing_dataset.py``. Only the
    ``packing_token`` strategy is retained; ``not_used_seqs`` are recursively
    re-packed so every example ends up in some packing.
    """

    def __init__(
        self,
        *,
        token_num_limit: int,
        sorted_strategy: str = "bucket",
    ) -> None:
        if token_num_limit <= 0:
            raise ValueError(f"token_num_limit must be positive, got {token_num_limit}")
        if sorted_strategy not in ("bucket", "none"):
            raise ValueError(f"Unknown sorted_strategy={sorted_strategy!r}")
        self.token_num_limit = token_num_limit
        self.sorted_strategy = sorted_strategy

    def _sort(self, pool: list, get_len: Callable[[Any], int]) -> list:
        if not pool:
            return pool
        if self.sorted_strategy == "bucket":
            max_len = max(get_len(s) for s in pool)
            buckets: List[list] = [[] for _ in range(max_len)]
            for s in pool:
                buckets[get_len(s) - 1].append(s)
            res: list = []
            for b in reversed(buckets):
                res.extend(b)
            return res
        # "none" -> stable descending sort
        return sorted(pool, key=get_len, reverse=True)

    def packing_token(
        self,
        pool: list,
        get_len: Callable[[Any], int],
    ) -> Tuple[List[list], list]:
        """Greedy heap-based packing under a token budget.

        Returns:
            Tuple ``(packings, leftover)`` where each packing is a list of
            entries from ``pool``, and ``leftover`` are entries that did not
            fit (each individually <= token_num_limit thanks to caller-side
            truncation).
        """
        pool = self._sort(pool, get_len)
        if not pool:
            return [], []
        seq_sum = sum(get_len(x) for x in pool)
        n_pack = max(1, math.ceil(seq_sum / self.token_num_limit))
        heaps: List[Tuple[int, int, list]] = [(0, i, []) for i in range(n_pack)]
        heapq.heapify(heaps)
        leftover: list = []
        for s in pool:
            length, idd, packing = heapq.heappop(heaps)
            l_s = get_len(s)
            if length + l_s > self.token_num_limit:
                leftover.append(s)
                heapq.heappush(heaps, (length, idd, packing))
                continue
            packing.append(s)
            heapq.heappush(heaps, (length + l_s, idd, packing))
        packings = [p[-1] for p in heaps]
        return packings, leftover

    def pack_all(
        self,
        pool: list,
        get_len: Callable[[Any], int],
    ) -> List[list]:
        """Iteratively re-pack leftover until everything is placed.

        If we cannot make further progress (all remaining items individually
        exceed ``token_num_limit``), each leftover becomes its own packing —
        the caller is expected to have pre-truncated to ``token_num_limit``.
        """
        out: List[list] = []
        cur = pool
        while cur:
            packings, leftover = self.packing_token(cur, get_len)
            out.extend(p for p in packings if p)
            if not leftover:
                break
            if len(leftover) == len(cur):
                for s in leftover:
                    out.append([s])
                break
            cur = leftover
        return out


# ---------------------------------------------------------------------------
# VLMPackedConversationDataset
# ---------------------------------------------------------------------------


_VISUAL_TYPES = {"image", "video", "audio"}


def _conversation_is_text_only(conversation: List[Dict[str, Any]]) -> bool:
    for turn in conversation:
        content = turn.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _VISUAL_TYPES:
                    return False
    return True


def _resolve_pad_id(processor: Any) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        raise ValueError(
            "Tokenizer has neither pad_token_id nor eos_token_id; cannot determine padding token for packed dataset."
        )
    return int(pad_id)


def _tokenize_conversation(processor: Any, conversation: List[Dict[str, Any]]) -> List[int]:
    """Tokenize a single conversation via apply_chat_template.

    Returns a flat ``List[int]`` of token ids without padding/truncation.
    """
    out = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        return_tensors=None,
    )
    # apply_chat_template may return List[int] for a single conversation,
    # or List[List[int]] / BatchEncoding-like for batched inputs.
    if isinstance(out, list) and out and isinstance(out[0], int):
        return out
    if isinstance(out, list) and out and isinstance(out[0], list):
        return list(out[0])
    if hasattr(out, "tolist"):
        ids = out.tolist()
        if ids and isinstance(ids[0], list):
            return ids[0]
        return ids
    if hasattr(out, "input_ids"):
        ids = out["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            return list(ids[0])
        return list(ids)
    raise TypeError(f"Unexpected return type from apply_chat_template: {type(out)}")


class VLMPackedConversationDataset(VLMConversationDataset):
    """Text-only VLM conversation dataset with streaming token-pool packing.

    Tokenization and packing happen lazily on demand. The dataset maintains a
    tokenized ``pool`` of at most ``pool_size`` examples and a queue of ready
    packed samples. Each ``__getitem__`` call:

    1. If the ready queue is empty, tops up the pool from the (cyclic) source
       of base examples until it reaches ``pool_size``, runs the greedy
       token-budget packer once over the pool, pushes the produced packed
       samples onto the queue, and keeps the leftover items in the pool for
       the next refill cycle.
    2. Pops and returns one ready packed sample.

    Each returned dict contains 1D tensors:

    - ``input_ids`` / ``labels`` / ``loss_mask`` / ``position_ids``: ``[L]``
    - ``cu_seqlens``: ``[num_segments + 1]`` int32, where the trailing
      padding region (if any) is recorded as the last segment so that the
      cumulative length equals ``L``.
    - ``max_seqlen``: 0-dim int32 scalar
    - ``visual_inputs``: ``None`` (this dataset is text-only)
    - ``attention_mask``: ``None`` (THD attention uses ``cu_seqlens``)

    The bound :attr:`collate_fn` stacks ``B`` such samples sequence-wise into
    ``[1, B*L]`` and merges per-sample ``cu_seqlens`` with row offsets.

    Args:
        base_examples: List of ``{"conversation": [...]}`` dicts. All
            conversations must be text-only — any ``image``/``video``/
            ``audio`` content part raises ``ValueError``.
        target_length: Logical length exposed by ``__len__`` (driven by
            ``DatasetBuildContext.{train,valid,test}_samples``). The
            streaming pool produces packed samples on demand for any number
            of indexed reads.
        processor: HF AutoProcessor whose tokenizer + chat template are used
            to materialize token ids.
        token_num_limit: Fixed length of every packed sample (typically
            ``model.seq_length``).
        pool_size: Target number of tokenized examples kept in the pool.
            Larger pools yield better packing density at the cost of more
            up-front tokenization per refill cycle.
        sorted_strategy: ``"bucket"`` or ``"none"`` — see
            :class:`BalanceBatchManager`.
        align_multiple: If > 1, ``token_num_limit`` must be divisible by this
            value (e.g. ``lcm(2*cp_size, cp_size*tp_size)``).
    """

    def __init__(
        self,
        base_examples: List[Dict[str, Any]],
        target_length: int,
        processor: Any,
        *,
        token_num_limit: int,
        pool_size: int,
        sorted_strategy: str = "bucket",
        align_multiple: int = 1,
    ) -> None:
        assert isinstance(base_examples, list) and len(base_examples) > 0, "base_examples must be a non-empty list"
        if token_num_limit <= 0:
            raise ValueError(f"token_num_limit must be positive, got {token_num_limit}")
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        if align_multiple > 1 and token_num_limit % align_multiple != 0:
            raise ValueError(
                f"token_num_limit={token_num_limit} must be divisible by align_multiple="
                f"{align_multiple} for CP/SP/TP alignment"
            )

        # Validate text-only invariant up front (cheap; no tokenization).
        for i, ex in enumerate(base_examples):
            conv = ex.get("conversation")
            if not isinstance(conv, list):
                raise ValueError(f"base_examples[{i}] missing 'conversation' list")
            if not _conversation_is_text_only(conv):
                raise ValueError(
                    f"VLMPackedConversationDataset only supports text-only conversations; "
                    f"base_examples[{i}] contains image/video/audio content."
                )

        # Initialize parent (binds a default collate_fn we will override below).
        super().__init__(base_examples, target_length, processor)

        self._skipped_tokens = extract_skipped_token_ids(processor)
        self._pad_id = _resolve_pad_id(processor)
        self._token_num_limit = int(token_num_limit)
        self._pool_size = int(pool_size)
        self._manager = BalanceBatchManager(
            token_num_limit=token_num_limit, sorted_strategy=sorted_strategy
        )
        self._length = int(max(0, target_length))

        # Streaming state.
        self._pool: List[Dict[str, Any]] = []
        self._ready: Deque[Dict[str, torch.Tensor]] = deque()
        self._cursor: int = 0  # next base_examples index to tokenize (cyclic)

        # Stats for periodic logging.
        self._packed_count: int = 0
        self._real_token_total: int = 0
        self._used_token_total: int = 0
        self._refill_count: int = 0

        logger.info(
            "VLMPackedConversationDataset (streaming): %d base examples, "
            "token_num_limit=%d, pool_size=%d",
            len(base_examples),
            token_num_limit,
            pool_size,
        )

        # Override the parent's collate_fn with the packed variant.
        self.collate_fn = self._packed_collate

    # ------------------------------------------------------------------
    # Streaming pool internals
    # ------------------------------------------------------------------

    def _tokenize_one(self, ex: Dict[str, Any]) -> Dict[str, Any] | None:
        """Tokenize a single base example into a pool entry.

        Returns ``None`` for empty tokenizations (caller skips them).
        """
        ids = _tokenize_conversation(self._processor, ex["conversation"])
        if not ids:
            return None
        if len(ids) > self._token_num_limit:
            logger.warning(
                "Example length %d exceeds token_num_limit %d; truncating.",
                len(ids),
                self._token_num_limit,
            )
            ids = ids[: self._token_num_limit]
        mask = create_multiturn_loss_mask_by_search(
            ex, torch.tensor(ids, dtype=torch.long), self._processor, self._skipped_tokens
        )
        return {"ids": ids, "mask": mask}

    def _refill_pool(self) -> None:
        """Top up ``self._pool`` to ``pool_size`` by tokenizing more sources.

        Cycles through ``base_examples`` indefinitely. Stops early only if
        every cycled source produced an empty tokenization (defensive).
        """
        n_base = len(self._base_examples)
        if n_base == 0:
            return
        attempts = 0
        attempt_limit = self._pool_size + n_base  # guard against degenerate inputs
        while len(self._pool) < self._pool_size and attempts < attempt_limit:
            ex = self._base_examples[self._cursor % n_base]
            self._cursor += 1
            attempts += 1
            entry = self._tokenize_one(ex)
            if entry is not None:
                self._pool.append(entry)

    def _produce_more_packings(self) -> None:
        """Refill the pool then run one packing pass into the ready queue."""
        self._refill_pool()
        if not self._pool:
            raise RuntimeError(
                "Token pool is empty after refill; no usable examples in base_examples."
            )

        indexed = list(enumerate(self._pool))
        # One pass of greedy packing — leftovers stay in the pool.
        packings, leftover = self._manager.packing_token(indexed, get_len=lambda x: len(x[1]["ids"]))
        new_pool: List[Dict[str, Any]] = [item for _, item in leftover]

        produced = 0
        for group in packings:
            if not group:
                continue
            sample = self._build_packed_sample(
                [item for _, item in group],
                token_num_limit=self._token_num_limit,
                pad_id=self._pad_id,
                skipped_tokens=self._skipped_tokens,
            )
            self._ready.append(sample)
            produced += 1
            self._packed_count += 1
            self._real_token_total += self._token_num_limit
            used = (
                int(sample["cu_seqlens"][-2].item())
                if sample["cu_seqlens"].numel() >= 2
                else int(sample["cu_seqlens"][-1].item())
            )
            self._used_token_total += used

        if produced == 0:
            # Pool is non-empty but no full packing fit (every item is shorter
            # than budget yet packer chose to leave them). Force progress by
            # emitting each leftover as its own packed sample.
            for item in new_pool:
                sample = self._build_packed_sample(
                    [item],
                    token_num_limit=self._token_num_limit,
                    pad_id=self._pad_id,
                    skipped_tokens=self._skipped_tokens,
                )
                self._ready.append(sample)
                self._packed_count += 1
                self._real_token_total += self._token_num_limit
                used = (
                    int(sample["cu_seqlens"][-2].item())
                    if sample["cu_seqlens"].numel() >= 2
                    else int(sample["cu_seqlens"][-1].item())
                )
                self._used_token_total += used
            new_pool = []

        self._pool = new_pool
        self._refill_count += 1

        # Coverage logging on a coarse cadence to avoid spamming.
        if self._refill_count % 16 == 0:
            cov = 100.0 * self._used_token_total / max(1, self._real_token_total)
            logger.info(
                "VLMPackedConversationDataset: produced %d packings so far "
                "(real-token coverage ~%.1f%%, pool=%d, refills=%d)",
                self._packed_count,
                cov,
                len(self._pool),
                self._refill_count,
            )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_packed_sample(
        items: List[Dict[str, Any]],
        *,
        token_num_limit: int,
        pad_id: int,
        skipped_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Concatenate per-example tensors into one fixed-length packed sample."""
        ids_cat: List[int] = []
        mask_cat: List[float] = []
        pos_cat: List[int] = []
        seg_lens: List[int] = []

        for it in items:
            ids = it["ids"]
            mask = it["mask"]
            assert len(ids) == len(mask), f"len(ids)={len(ids)} != len(mask)={len(mask)} in packed item"
            seg_lens.append(len(ids))
            ids_cat.extend(ids)
            mask_cat.extend(float(m) for m in mask)
            pos_cat.extend(range(len(ids)))

        real_len = len(ids_cat)
        if real_len > token_num_limit:
            # Should not happen because BalanceBatchManager enforces budget.
            raise RuntimeError(f"Internal error: packed real_len {real_len} > token_num_limit {token_num_limit}")

        pad_len = token_num_limit - real_len
        if pad_len > 0:
            ids_cat.extend([pad_id] * pad_len)
            mask_cat.extend([0.0] * pad_len)
            pos_cat.extend(range(pad_len))
            seg_lens.append(pad_len)

        input_ids = torch.tensor(ids_cat, dtype=torch.long)
        loss_mask = torch.tensor(mask_cat, dtype=torch.float)
        position_ids = torch.tensor(pos_cat, dtype=torch.long)

        # Build next-token labels: shift left within each segment;
        # the last position of each segment has no next token -> -100.
        labels = input_ids.clone()
        if labels.numel() > 1:
            labels[:-1] = input_ids[1:]
        labels[-1] = -100

        # Mark each segment boundary (last token of every segment) as -100.
        boundary = torch.cumsum(torch.tensor(seg_lens, dtype=torch.long), dim=0) - 1
        boundary = boundary[boundary < labels.numel()]
        labels[boundary] = -100

        # Mask skipped/special tokens out of loss.
        if skipped_tokens.numel() > 0:
            labels[torch.isin(labels, skipped_tokens)] = -100

        # Shift loss_mask consistently with next-token alignment.
        shifted = torch.zeros_like(loss_mask)
        if loss_mask.numel() > 1:
            shifted[:-1] = loss_mask[1:]
        loss_mask = shifted

        # Final consistency: positions with loss_mask==0 must have label -100.
        labels = labels.masked_fill(loss_mask == 0, -100)

        cu = [0]
        for sl in seg_lens:
            cu.append(cu[-1] + sl)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32)
        max_seqlen = torch.tensor(max(seg_lens), dtype=torch.int32)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
        }

    @staticmethod
    def _packed_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Stack ``B`` pre-packed samples into a ``[1, B*L]`` THD batch.

        The per-sample ``cu_seqlens`` lists are merged with cumulative row
        offsets so the result is a single 1D tensor describing all segments
        across the batch.
        """
        if not batch:
            raise ValueError("Empty batch passed to _packed_collate")
        L = int(batch[0]["input_ids"].shape[0])
        for i, b in enumerate(batch):
            if int(b["input_ids"].shape[0]) != L:
                raise ValueError(
                    f"Inconsistent packed length: batch[0]={L}, batch[{i}]={int(b['input_ids'].shape[0])}"
                )

        input_ids = torch.cat([b["input_ids"] for b in batch], dim=0).unsqueeze(0)
        labels = torch.cat([b["labels"] for b in batch], dim=0).unsqueeze(0)
        loss_mask = torch.cat([b["loss_mask"] for b in batch], dim=0).unsqueeze(0)
        position_ids = torch.cat([b["position_ids"] for b in batch], dim=0).unsqueeze(0)

        cu_pieces: List[torch.Tensor] = []
        offset = 0
        for i, b in enumerate(batch):
            cu = b["cu_seqlens"].to(torch.int32)
            if i == 0:
                cu_pieces.append(cu)
            else:
                cu_pieces.append(cu[1:] + offset)
            offset += L
        cu_seqlens = torch.cat(cu_pieces, dim=0).to(torch.int32)
        max_seqlen = torch.stack([b["max_seqlen"] for b in batch]).max().to(torch.int32)
        cu_seqlens_argmin = torch.tensor(cu_seqlens.numel(), dtype=torch.int32)

        return {
            "input_ids": input_ids,
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": None,
            "visual_inputs": None,
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": cu_seqlens_argmin,
            "max_seqlen": max_seqlen,
        }

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        if self._length == 0:
            raise IndexError("Empty packed dataset")
        # Pool-based streaming: indices are advisory; we serve the next ready
        # packed sample, refilling and packing as needed.
        while not self._ready:
            self._produce_more_packings()
        return self._ready.popleft()
