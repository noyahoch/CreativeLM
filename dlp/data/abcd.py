"""
ABCD dataset loader: load abcd_*.json and expose items with train/eval split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from dlp.training.data_prep import TaskType, build_contrastive_pair


class ABCDDataset:
    """
    Load ABCD JSON (e.g. abcd_aut.json, abcd_aiden.json) and expose
    items with optional train/eval split and contrastive pairs.
    """

    def __init__(
        self,
        path: str | Path,
        task_type: str | TaskType = "aut",
        train_frac: float = 0.8,
        seed: int = 42,
    ) -> None:
        """
        Args:
            path: Path to abcd_*.json.
            task_type: "aut" or "ps" (or TaskType enum).
            train_frac: Fraction of items for train (rest for eval). 0.0 = all eval.
            seed: Random seed for split.
        """
        self.path = Path(path)
        self.task_type = TaskType(task_type) if isinstance(task_type, str) else task_type
        self.train_frac = train_frac
        self.seed = seed
        self._items: list[dict] = []
        self._train_ids: list[str] = []
        self._eval_ids: list[str] = []
        self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self._items = data if isinstance(data, list) else [data]
        if self.train_frac <= 0 or self.train_frac >= 1:
            self._eval_ids = [item.get("id", str(i)) for i, item in enumerate(self._items)]
            self._train_ids = []
            return
        rng = random.Random(self.seed)
        indices = list(range(len(self._items)))
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * self.train_frac))
        train_idx = set(indices[:n_train])
        for i, item in enumerate(self._items):
            id_ = item.get("id", str(i))
            if i in train_idx:
                self._train_ids.append(id_)
            else:
                self._eval_ids.append(id_)

    @property
    def items(self) -> list[dict]:
        """All dataset items."""
        return self._items

    @property
    def train_ids(self) -> list[str]:
        """IDs in the train split."""
        return self._train_ids

    @property
    def eval_ids(self) -> list[str]:
        """IDs in the eval split."""
        return self._eval_ids

    def train_items(self) -> list[dict]:
        """Items in the train split (same order as train_ids)."""
        id_to_item = {item.get("id", str(i)): item for i, item in enumerate(self._items)}
        return [id_to_item[i] for i in self._train_ids if i in id_to_item]

    def eval_items(self) -> list[dict]:
        """Items in the eval split (same order as eval_ids)."""
        id_to_item = {item.get("id", str(i)): item for i, item in enumerate(self._items)}
        return [id_to_item[i] for i in self._eval_ids if i in id_to_item]

    def contrastive_pairs(self, item_ids: list[str] | None = None) -> list[dict]:
        """
        Build contrastive pairs for steering. If item_ids is None, use all items.
        """
        if item_ids is not None:
            id_set = set(item_ids)
            items = [it for it in self._items if it.get("id", "") in id_set]
        else:
            items = self._items
        return [build_contrastive_pair(it, self.task_type) for it in items]
