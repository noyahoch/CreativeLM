"""
Pipeline: generate AUT outputs and/or run LLM judge -> write results and report.

OpenAI-only generation. For local HF model inference, use methods.py +
scripts/run_aut_inference.py instead.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .generate import generate_aut_outputs
from .judge import AUTJudge


class AUTPipeline:
    """
    AUT generate + judge pipeline: one output_dir, optional JudgeConfig.
    Methods: load_outputs(), generate(), judge(), run_full().
    """

    def __init__(
        self,
        output_dir: str | Path,
        generate_model: str = "gpt-4o-mini",
        generate_temperature: float = 0.7,
        judge_model: str = "gpt-4o-mini",
        judge_temperature: float = 0.0,
        num_uses: int = 10,
        max_uses_per_item: int = 15,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.generate_model = generate_model
        self.generate_temperature = generate_temperature
        self.judge_model = judge_model
        self.judge_temperature = judge_temperature
        self.num_uses = num_uses
        self.max_uses_per_item = max_uses_per_item

    def load_outputs(self, path: str) -> list[dict]:
        """Load AUT outputs from JSON path. Same as load_aut_outputs(path)."""
        return load_aut_outputs(path)

    def generate(
        self,
        objects: list[str] | list[dict] | str,
        output_filename: str = "aut_outputs.json",
    ) -> str:
        """Generate AUT outputs and save to output_dir. Returns path to written file."""
        return run_generate(
            objects,
            str(self.output_dir),
            num_uses=self.num_uses,
            model=self.generate_model,
            temperature=self.generate_temperature,
            output_filename=output_filename,
        )

    def judge(
        self,
        input_path: str,
        object_from_metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Load AUT outputs from input_path, run judge, write outputs. Returns results and report."""
        return run_judge_only(
            input_path,
            str(self.output_dir),
            model=self.judge_model,
            temperature=self.judge_temperature,
            max_uses_per_item=self.max_uses_per_item,
            object_from_metadata=object_from_metadata,
        )

    def run_full(
        self,
        objects: list[str] | list[dict] | str,
        object_from_metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Generate then judge. Returns judge results and report."""
        self.generate(objects)
        return self.judge(
            str(self.output_dir / "aut_outputs.json"),
            object_from_metadata=object_from_metadata,
        )


def load_aut_outputs(path: str) -> list[dict]:
    """
    Load AUT outputs from JSON or JSONL.

    Expected per-item shape (JSON array or JSONL lines):
      - object: str
      - uses: list[str] or str (newline-separated)
      - optional: id, meta_data, etc. (preserved in output)

    Also accepts a dict keyed by id where each value has object + uses
    (e.g. inference_output.json from some pipelines).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = []
        for k, v in data.items():
            if isinstance(v, dict):
                if "object" in v and "uses" in v:
                    items.append({"id": k, **v})
                elif all(isinstance(vv, str) for vv in v.values()):
                    for iter_id, raw in v.items():
                        uses = _raw_output_to_uses(raw)
                        items.append({"id": k, "iter_id": iter_id, "uses": uses})
                else:
                    items.append({"id": k, **v})
            else:
                items.append({"id": k, "uses": v if isinstance(v, list) else [v]})
        return items
    if isinstance(data, list):
        return data
    return [data]


def _raw_output_to_uses(raw: str) -> list[str]:
    """Parse raw model output into list of use strings (bullets or numbered)."""
    uses = []
    for line in raw.split("\n"):
        line = line.strip()
        line = re.sub(r"^[\-\*\•]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"\s*:\s*[\d.]+\s*$", "", line).strip()
        if line:
            uses.append(line)
    return uses[:15]


def run_generate(
    objects: list[str] | list[dict] | str,
    output_dir: str,
    num_uses: int = 10,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    output_filename: str = "aut_outputs.json",
) -> str:
    """
    Generate AUT outputs for given objects and save to output_dir.

    - objects: list of object names, list of {object, id?}, or path to JSON.
    Returns path to written JSON file.
    """
    if isinstance(objects, str):
        path = Path(objects)
        if not path.exists():
            raise FileNotFoundError(objects)
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            objects = data
        elif isinstance(data, dict) and "objects" in data:
            objects = data["objects"]
        else:
            objects = data if isinstance(data, list) else [data]
    items = generate_aut_outputs(
        objects,
        num_uses=num_uses,
        model=model,
        temperature=temperature,
    )
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    with open(out_path, "w") as f:
        json.dump(items, f, indent=2)
    return out_path


def run_judge_only(
    input_path: str,
    output_dir: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_uses_per_item: int = 15,
    object_from_metadata: dict | None = None,
) -> dict[str, Any]:
    """
    Load AUT outputs from input_path, run judge, write judge_output.json and judge_report.json.
    """
    items = load_aut_outputs(input_path)
    if object_from_metadata:
        for it in items:
            i = it.get("id")
            if i is not None and i in object_from_metadata:
                it.setdefault("object", object_from_metadata[i])
    judge = AUTJudge(
        model=model,
        temperature=temperature,
        max_uses_per_item=max_uses_per_item,
    )
    for it in items:
        it.setdefault("object", "")
        if isinstance(it.get("uses"), str):
            it["uses"] = [u.strip() for u in it["uses"].split("\n") if u.strip()]
    results = []
    for it in tqdm(items, desc="Judge"):
        if not it.get("uses"):
            results.append({**it, "scores": [], "avg_creativity": float("nan"), "fluency": 0})
            continue
        batch = judge.rate([it])
        r = batch[0]
        r["avg_creativity"] = round(r["avg_creativity"], 4)
        results.append(r)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "judge_output.json")

    def _to_serializable(obj: Any) -> Any:
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, float) and obj != obj:
            return None
        return obj

    for r in results:
        r["scores"] = [[_to_serializable(x) for x in pair] for pair in r.get("scores", [])]
        if r.get("avg_creativity") != r.get("avg_creativity"):
            r["avg_creativity"] = None
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    avgs = [r["avg_creativity"] for r in results if r["avg_creativity"] == r["avg_creativity"]]
    report = {
        "n_items": len(results),
        "mean_creativity": round(sum(avgs) / len(avgs), 4) if avgs else None,
        "mean_fluency": round(sum(r["fluency"] for r in results) / len(results), 2) if results else None,
        "output_path": out_path,
    }
    report_path = os.path.join(output_dir, "judge_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return {"results": results, "report": report}


def run_pipeline(
    output_dir: str,
    objects: list[str] | list[dict] | str | None = None,
    input_path: str | None = None,
    mode: str = "full",
    num_uses: int = 10,
    generate_model: str = "gpt-4o-mini",
    generate_temperature: float = 0.7,
    judge_model: str = "gpt-4o-mini",
    judge_temperature: float = 0.0,
    max_uses_per_item: int = 15,
    object_from_metadata: dict | None = None,
) -> dict[str, Any]:
    """
    Run generate and/or judge (backward-compatible wrapper).

    Creates AUTPipeline and calls run_full(), generate(), or judge() per mode.
    """
    pipeline = AUTPipeline(
        output_dir=output_dir,
        generate_model=generate_model,
        generate_temperature=generate_temperature,
        judge_model=judge_model,
        judge_temperature=judge_temperature,
        num_uses=num_uses,
        max_uses_per_item=max_uses_per_item,
    )
    if mode in ("full", "generate"):
        if objects is None:
            raise ValueError("objects (or path to JSON) required for mode=%s" % mode)
        pipeline.generate(objects)
        if mode == "generate":
            return {"aut_outputs_path": str(Path(output_dir) / "aut_outputs.json")}
    path_to_judge = input_path if mode == "judge" else str(Path(output_dir) / "aut_outputs.json")
    if not path_to_judge or not os.path.isfile(path_to_judge):
        raise FileNotFoundError("No AUT outputs to judge at %s" % path_to_judge)
    return pipeline.judge(path_to_judge, object_from_metadata=object_from_metadata)
