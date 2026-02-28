"""
Async batch reward gateway: collects concurrent reward computation requests
from multiple agent loop postprocess coroutines and batches the retrieval
calls into single API requests.

This is the reward-side counterpart of batch_search_gateway.py.
Instead of each sample calling bright.compute_score() individually
(each making its own HTTP POST to the retrieval service), this gateway:

1. Collects (solution_str, extra_info) from all concurrent _compute_score calls
2. Extracts expanded queries from all samples
3. Makes ONE batched _batch_search call (one HTTP POST per task)
4. Computes NDCG@10 for each sample
5. Returns scores to each waiting coroutine

This reduces HTTP overhead from N requests to ~12 (one per BRIGHT task).
"""

import asyncio
import json
import logging
import math
import os
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_reward_gateway_instance: Optional["AsyncBatchRewardGateway"] = None


class AsyncBatchRewardGateway:
    """Batches reward computation (search + NDCG@10) for concurrent agent loop samples."""

    def __init__(
        self,
        batch_wait_time: float = 0.15,
        max_batch_size: int = 1024,
    ):
        """Initialize the batch reward gateway.

        Args:
            batch_wait_time: Seconds to wait for more requests before flushing.
            max_batch_size: Maximum requests per batch before immediate flush.
        """
        self.batch_wait_time = batch_wait_time
        self.max_batch_size = max_batch_size
        self._lock = asyncio.Lock()
        self._current_batch: Optional["_PendingRewardBatch"] = None

    async def compute_reward(
        self, solution_str: str, extra_info: dict
    ) -> tuple[float, dict]:
        """Submit a reward computation request. Returns when batch result is ready.

        Args:
            solution_str: Decoded model response string.
            extra_info: Dict with keys: initial_doc, question, task, id, interaction_kwargs.

        Returns:
            Tuple of (reward_score, reward_extra_info dict).
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        batch_to_execute = None

        async with self._lock:
            if self._current_batch is None:
                self._current_batch = _PendingRewardBatch(self)

            self._current_batch.add(solution_str, extra_info, future)

            if self._current_batch.size >= self.max_batch_size:
                batch_to_execute = self._current_batch
                self._current_batch = None
            elif self._current_batch.size == 1:
                asyncio.ensure_future(self._delayed_flush(self._current_batch))

        if batch_to_execute is not None:
            asyncio.ensure_future(batch_to_execute.execute())

        return await future

    async def _delayed_flush(self, batch: "_PendingRewardBatch"):
        """Wait for batch_wait_time, then flush if batch is still the current one."""
        await asyncio.sleep(self.batch_wait_time)
        should_execute = False
        async with self._lock:
            if self._current_batch is batch:
                self._current_batch = None
                should_execute = True
        if should_execute and not batch.executed:
            await batch.execute()


class _PendingRewardBatch:
    """A batch of pending reward computation requests."""

    def __init__(self, gateway: AsyncBatchRewardGateway):
        self.gateway = gateway
        self.solution_strs: list[str] = []
        self.extra_infos: list[dict] = []
        self.futures: list[asyncio.Future] = []
        self.executed = False

    @property
    def size(self) -> int:
        return len(self.solution_strs)

    def add(self, solution_str: str, extra_info: dict, future: asyncio.Future):
        self.solution_strs.append(solution_str)
        self.extra_infos.append(extra_info)
        self.futures.append(future)

    async def execute(self):
        """Execute the batched reward computation and distribute results."""
        if self.executed:
            return
        self.executed = True

        batch_size = self.size
        logger.info(f"[BatchRewardGateway] Computing rewards for batch of {batch_size} samples")

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._compute_rewards)

            for i, future in enumerate(self.futures):
                if not future.done():
                    future.set_result(results[i])

            logger.info(
                f"[BatchRewardGateway] Batch of {batch_size} rewards computed successfully"
            )
        except Exception as e:
            logger.error(
                f"[BatchRewardGateway] Batch reward computation failed: {e}"
            )
            for future in self.futures:
                if not future.done():
                    future.set_exception(e)

    def _compute_rewards(self) -> list[tuple[float, dict]]:
        """Compute rewards for all samples in batch (blocking, runs in thread pool).

        Steps:
        1. Extract expanded queries from each sample's trajectory
        2. ONE batched _batch_search call for all queries
        3. Compute NDCG@10 per sample
        4. Return (score, extra_info) for each

        Returns:
            List of (reward_score, reward_extra_info) tuples.
        """
        from verl.utils.reward_score.bright import (
            DEFAULT_BRIGHT_RETRIEVAL_URL,
            _batch_search,
            extract_expand_query,
            ndcg_at_k,
        )

        retrieval_url = os.getenv("BRIGHT_RETRIEVAL_URL", DEFAULT_BRIGHT_RETRIEVAL_URL)

        # Step 1: Prepare expanded queries for all samples
        expanded_queries = []
        question_ids = []
        tasks = []
        ground_truth_ids_list = []
        valid_indices = []  # Track which samples have valid extra_info

        for i, (solution_str, extra_info) in enumerate(
            zip(self.solution_strs, self.extra_infos)
        ):
            try:
                input_docs = extra_info.get("initial_doc", "")
                query = extra_info.get("question", "")
                task = extra_info.get("task", "")
                sample_id = extra_info.get("id", "")

                if not task or not sample_id:
                    continue

                question_id = sample_id.split(task + "_")[1] if task + "_" in sample_id else sample_id
                ground_truth = extra_info.get("interaction_kwargs", {}).get("ground_truth", [])
                if not isinstance(ground_truth, list):
                    ground_truth = [ground_truth]

                expanded_query = extract_expand_query(solution_str, input_docs, query)

                expanded_queries.append(expanded_query)
                question_ids.append(question_id)
                tasks.append(task)
                ground_truth_ids_list.append(ground_truth)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"[BatchRewardGateway] Failed to prepare sample {i}: {e}")
                continue

        # Default results: score=0, empty extra_info
        results = [(0.0, {"acc": 0.0})] * len(self.solution_strs)

        if not expanded_queries:
            return results

        # Step 2: ONE batched _batch_search for ALL samples
        logger.info(
            f"[BatchRewardGateway] Batch searching {len(expanded_queries)} expanded queries"
        )
        final_all_scores = _batch_search(
            expanded_queries,
            search_host_url=retrieval_url,
            question_ids=question_ids,
            tasks=tasks,
        )

        # Step 3: Map results back to original order and compute NDCG@10
        task_indices = defaultdict(list)
        for j, task in enumerate(tasks):
            task_indices[task].append(j)

        batch_scores = [0.0] * len(valid_indices)
        task_counters = defaultdict(int)

        for (task, qid, inst_id), did_scores in final_all_scores.items():
            counter = task_counters[task]
            if counter < len(task_indices[task]):
                inner_idx = task_indices[task][counter]
                task_counters[task] += 1

                gt_ids = ground_truth_ids_list[inner_idx]
                ndcg = ndcg_at_k(did_scores, gt_ids, k=10)
                batch_scores[inner_idx] = ndcg

        # Map back to original sample order
        for j, orig_idx in enumerate(valid_indices):
            score = batch_scores[j]
            results[orig_idx] = (score, {"acc": score})
            logger.debug(f"[BatchRewardGateway] Sample {orig_idx}: NDCG@10={score:.4f}")

        return results


def get_batch_reward_gateway(
    batch_wait_time: float = 0.15,
    max_batch_size: int = 1024,
) -> AsyncBatchRewardGateway:
    """Get or create the singleton batch reward gateway."""
    global _reward_gateway_instance
    if _reward_gateway_instance is None:
        _reward_gateway_instance = AsyncBatchRewardGateway(
            batch_wait_time=batch_wait_time,
            max_batch_size=max_batch_size,
        )
        logger.info(
            f"[BatchRewardGateway] Created singleton: "
            f"wait={batch_wait_time}s, max_batch={max_batch_size}"
        )
    return _reward_gateway_instance


def reset_batch_reward_gateway():
    """Reset the singleton gateway."""
    global _reward_gateway_instance
    _reward_gateway_instance = None
