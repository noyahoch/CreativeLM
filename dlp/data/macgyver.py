"""
MacGyver dataset loader: load problem_solution_pair.xlsx and expose items
with filtering by solvability / unconventionality and train/eval split.

The MacGyver dataset contains 1,683 real-world verbal problems designed to
trigger innovative object usage and out-of-the-box thinking.  Each item has:
  - Problem: open-ended scenario with a set of available tools
  - Solution: step-by-step reference solution
  - Solvable?: whether the problem can be solved with the given tools
  - Unconventional?: whether the solution requires unconventional tool use
  - Label: quality annotation (efficient, inefficient, infeasible, …)

Reference:
  Tian et al., "MacGyver: Are Large Language Models Creative Problem Solvers?", NAACL 2024.
"""

from __future__ import annotations

import json
import random
from enum import Enum
from pathlib import Path
from typing import Any


class MacGyverSubset(Enum):
    """Pre-defined filtering presets."""

    ALL = "all"
    SOLVABLE = "solvable"
    SOLVABLE_UNCONVENTIONAL = "solvable_unconventional"
    SOLVABLE_CONVENTIONAL = "solvable_conventional"
    UNSOLVABLE = "unsolvable"
    BENCHMARK = "benchmark"


# Relative path from the MacGyver repo root to the benchmark results file.
_BENCH_REL = "data/Benchmark_results/benchmark_results.json"


class MacGyverDataset:
    """
    Load the MacGyver xlsx and expose items with optional filtering and
    train/eval split, following the same conventions as ABCDDataset.
    """

    def __init__(
        self,
        path: str | Path,
        subset: str | MacGyverSubset = "solvable_unconventional",
        train_frac: float = 0.0,
        seed: int = 42,
    ) -> None:
        """
        Args:
            path: Path to ``problem_solution_pair.xlsx``.
            subset: Which slice to keep — see ``MacGyverSubset`` for options.
                    ``"benchmark"`` cross-references IDs from the paper's
                    benchmark_results.json (323 items).
            train_frac: Fraction of items for train (rest → eval).
                        0.0 (default) puts everything in eval.
            seed: Random seed for the train/eval split.
        """
        self.path = Path(path)
        self.subset = MacGyverSubset(subset) if isinstance(subset, str) else subset
        self.train_frac = train_frac
        self.seed = seed

        self._items: list[dict[str, Any]] = []
        self._train_ids: list[str] = []
        self._eval_ids: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    # Loading & filtering
    # ------------------------------------------------------------------

    def _load(self) -> None:
        import pandas as pd

        df = pd.read_excel(self.path)
        df = self._filter(df)

        self._items = []
        for _, row in df.iterrows():
            self._items.append({
                "id": str(int(row["ID"])),
                "problem": str(row["Problem"]),
                "solvable": row.get("Solvable?") == "Yes",
                "unconventional": row.get("Unconventional?") == "unconventional",
                "solution": str(row.get("Solution", "")),
                "label": str(row.get("Label", "")),
            })

        self._split()

    def _filter(self, df: "pd.DataFrame") -> "pd.DataFrame":
        import pandas as pd

        s = self.subset
        if s == MacGyverSubset.ALL:
            return df
        if s == MacGyverSubset.SOLVABLE:
            return df[df["Solvable?"] == "Yes"]
        if s == MacGyverSubset.SOLVABLE_UNCONVENTIONAL:
            return df[
                (df["Solvable?"] == "Yes") & (df["Unconventional?"] == "unconventional")
            ]
        if s == MacGyverSubset.SOLVABLE_CONVENTIONAL:
            return df[
                (df["Solvable?"] == "Yes") & (df["Unconventional?"] == "conventional")
            ]
        if s == MacGyverSubset.UNSOLVABLE:
            return df[df["Solvable?"] == "No"]
        if s == MacGyverSubset.BENCHMARK:
            return self._filter_benchmark(df)
        raise ValueError(f"Unknown subset: {s!r}")

    def _filter_benchmark(self, df: "pd.DataFrame") -> "pd.DataFrame":
        bench_path = self.path.parent.parent / "Benchmark_results" / "benchmark_results.json"
        if not bench_path.exists():
            repo_root = self.path.parents[2]
            bench_path = repo_root / _BENCH_REL
        with open(bench_path, encoding="utf-8") as f:
            bench_ids = set(json.load(f).keys())
        return df[df["ID"].astype(str).isin(bench_ids)]

    def _split(self) -> None:
        if self.train_frac <= 0 or self.train_frac >= 1:
            self._eval_ids = [it["id"] for it in self._items]
            self._train_ids = []
            return
        rng = random.Random(self.seed)
        indices = list(range(len(self._items)))
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * self.train_frac))
        train_idx = set(indices[:n_train])
        for i, item in enumerate(self._items):
            if i in train_idx:
                self._train_ids.append(item["id"])
            else:
                self._eval_ids.append(item["id"])

    # ------------------------------------------------------------------
    # Public API (mirrors ABCDDataset)
    # ------------------------------------------------------------------

    @property
    def items(self) -> list[dict[str, Any]]:
        """All items after filtering."""
        return self._items

    @property
    def train_ids(self) -> list[str]:
        return self._train_ids

    @property
    def eval_ids(self) -> list[str]:
        return self._eval_ids

    def train_items(self) -> list[dict[str, Any]]:
        id_to_item = {it["id"]: it for it in self._items}
        return [id_to_item[i] for i in self._train_ids if i in id_to_item]

    def eval_items(self) -> list[dict[str, Any]]:
        id_to_item = {it["id"]: it for it in self._items}
        return [id_to_item[i] for i in self._eval_ids if i in id_to_item]

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return (
            f"MacGyverDataset(path={str(self.path)!r}, subset={self.subset.value!r}, "
            f"n_items={len(self._items)}, train={len(self._train_ids)}, "
            f"eval={len(self._eval_ids)})"
        )
