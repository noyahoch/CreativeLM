"""
CreativityPrism-style AUT benchmark with pluggable inference methods.

AUTBenchmarkRunner loads aut_push_skipped.json, runs one or more inference methods
(baseline, steered, few-shot, etc.) and writes inference_output.json per method
in the shape expected by aut_ttcw_cshort AUTEval.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from dlp.evaluation.methods import AUTInferenceMethod, BaselineMethod, SteeredMethod
from dlp.models import HFLoader
from dlp.utils.config import BenchmarkConfig

# -----------------------------------------------------------------------------
# CreativityPrism aut_ttcw_cshort prompt constants
# -----------------------------------------------------------------------------

SUFFIX_FIRST = "\nList your results in an unordered list with one use per new line (starting with \"-\"); provide at most 10 answers."
SUFFIX_ITER = "\nList your results in an unordered list with one use per new line (starting with \"-\"); provide at most 10 answers; only provide answer, no appology or irrelevant information."

ITERATIONS = {
    "": "",
    "bsr": "Really? Is this the best you can do?",
    "bsrd": "I'm so disappointed with you. I hope this time you put effort into it.",
    "bsrde": "Stop with excuses and do your best this time",
    "bsrdel": "This is your last chance.",
}


def _model_slug(name: str) -> str:
    """e.g. Qwen/Qwen2-7B-Instruct -> qwen2_7b_instruct"""
    s = name.split("/")[-1].lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "model"


class AUTBenchmarkRunner:
    """
    Runs CreativityPrism AUT benchmark with pluggable inference methods.
    Encapsulates prompt building, parsing, and inference loop. Pass a list of
    AUTInferenceMethod instances (e.g. BaselineMethod, SteeredMethod) to run();
    or use config.vectors_path / no_baseline / no_steered for the default baseline + steered behavior.
    """

    def __init__(self, config: BenchmarkConfig | None = None, **kwargs: Any) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = BenchmarkConfig(**kwargs)
        self._loader: HFLoader | None = None

    def _get_loader(self) -> HFLoader:
        if self._loader is None:
            self._loader = HFLoader(model_name_or_path=str(self.config.model_name))
            self._loader.load()
        return self._loader

    @staticmethod
    def create_batched_prompt(
        input_data: list[dict],
        prev_interactions: dict[int, list[dict]],
        curr_iter: str,
        test_size: float = 1e10,
    ) -> list[dict]:
        """Build prompt list for one iteration (CreativityPrism aut_ttcw_cshort style)."""
        all_prompt_data = []
        for dp in input_data:
            skip_dp = False
            if curr_iter == "":
                messages = [{"role": "user", "content": dp["input"]["text"] + SUFFIX_FIRST}]
            elif dp["input"]["others"].get("prompt_type") == "bs":
                history = prev_interactions.get(dp["meta_data"]["id"], [])
                messages = copy.deepcopy(history)
                it_lst = dp["input"]["others"].get("iteration_lst") or {}
                if curr_iter not in it_lst:
                    skip_dp = True
                else:
                    messages.append({"role": "user", "content": it_lst[curr_iter] + SUFFIX_ITER})
            else:
                skip_dp = True
            if not skip_dp:
                all_prompt_data.append({"prompt_id": dp["meta_data"]["id"], "messages": messages, "data": dp})
            if len(all_prompt_data) >= test_size:
                break
        return all_prompt_data

    @staticmethod
    def parse_raw_output(raw_output: str) -> str:
        """Extract cleaned list of uses from model output."""
        if "[assistant]:" in raw_output:
            last_assistant = raw_output.split("[assistant]:")[-1].strip()
            list_lines = [line.strip() for line in last_assistant.split("\n") if line.strip().startswith("-")]
            return "\n".join(list_lines) if list_lines else last_assistant
        if ":" in raw_output:
            return ":".join(raw_output.split(":")[1:])
        return raw_output

    def run_aut_inference(self, method: AUTInferenceMethod) -> dict[str, dict[str, str]]:
        """
        Run full AUT inference (all iterations) with the given method.
        Returns final_results with shape { dp_id: { iter_id: cleaned_output_str } } (string keys for JSON).
        """
        loader = self._get_loader()
        with open(Path(self.config.aut_data_path), encoding="utf-8") as f:
            input_data = json.load(f)

        prev_interactions: dict[int, list[dict]] = {dp["meta_data"]["id"]: [] for dp in input_data}
        final_results: dict[int, dict[str, str]] = {}

        for curr_iter in ITERATIONS:
            all_prompt_data = self.create_batched_prompt(
                input_data, prev_interactions, curr_iter, test_size=self.config.test_size
            )
            if curr_iter == "" and len(all_prompt_data) == 0:
                raise RuntimeError("No prompts for first round. Check aut_push_skipped.json.")

            for prompt_data in tqdm(
                all_prompt_data,
                desc=f"{method.slug} iter={curr_iter or 'first'}",
            ):
                messages = prompt_data["messages"]
                raw_output = method.generate(
                    loader,
                    messages,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                )
                cleaned = self.parse_raw_output(raw_output)
                dp_id = prompt_data["data"]["meta_data"]["id"]
                prompt_type = prompt_data["data"]["input"]["others"].get("prompt_type", "")
                if curr_iter == "":
                    tmp_key = prompt_data["data"]["input"]["others"].get("prompt_type", "bs")
                    final_results[dp_id] = {tmp_key: cleaned}
                else:
                    tmp_key = curr_iter
                    if dp_id not in final_results:
                        final_results[dp_id] = {}
                    final_results[dp_id][tmp_key] = cleaned
                if prompt_type == "bs":
                    prev_interactions[dp_id] = list(messages)
                    prev_interactions[dp_id].append({"role": "assistant", "content": raw_output})

        return {str(k): v for k, v in final_results.items()}

    def run_method(self, method: AUTInferenceMethod) -> Path:
        """Run one inference method and write inference_output.json. Returns output dir."""
        slug = _model_slug(str(self.config.model_name))
        out_dir = Path(self.config.output_dir) / f"aut_{method.slug}_{slug}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = self.run_aut_inference(method)
        (out_dir / "inference_output.json").write_text(json.dumps(results, indent=2))
        return out_dir

    def run(
        self,
        methods: list[AUTInferenceMethod] | None = None,
        no_baseline: bool = False,
        no_steered: bool = False,
    ) -> list[Path]:
        """
        Run one or more inference methods. Returns list of output dirs written.

        If methods is None (default), builds a default list from config:
        - baseline unless no_baseline
        - steered (using config.vectors_path and config.alpha) unless no_steered or vectors_path is unset
        """
        if methods is not None:
            to_run = methods
        else:
            to_run = []
            if not no_baseline:
                to_run.append(BaselineMethod())
            if not no_steered and self.config.vectors_path:
                to_run.append(
                    SteeredMethod(
                        vectors_path=self.config.vectors_path,
                        alpha=self.config.alpha,
                    )
                )
        if not to_run:
            return []

        written: list[Path] = []
        for method in to_run:
            d = self.run_method(method)
            written.append(d)
            print(f"Wrote {d / 'inference_output.json'}")
        return written
