from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import verifiers as vf
from pydantic import BaseModel
from core.config import PINFERENCE_API_BASE_URL, PINFERENCE_API_KEY_VAR, EnvironmentConfig
from core.constants import DEFAULT_JUDGE_MODEL, DEFAULT_SAMPLING_ARGS, STATE_COLUMNS
from core.dataset import AdvancedIFExample, BenchmarkSplits, build_benchmark_splits, build_rollout_input
from core.dotenv_bootstrap import load_project_dotenv
from core.skill_artifact import EMPTY_SKILL_TEMPLATE


def _json_default_rollout(obj: Any) -> Any:
    """Serialize verifiers / Pydantic objects embedded in rollout state for JSONL."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__!r} is not JSON serializable")


def _dumps_rollout_line(record: dict[str, Any]) -> str:
    return json.dumps(record, default=_json_default_rollout) + "\n"


def default_output_dir(env_id: str, model: str, feedback_mode: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_key = model.replace("/", "__")
    return (
        Path("outputs") / "sequence_runs" / f"{env_id}__{model_key}__{feedback_mode}__{timestamp}"
    )


def load_system_prompt(path: str | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


async def run_rollout(
    env: vf.Environment,
    input_row: dict[str, Any],
    model: str,
    sampling_args: dict[str, Any],
) -> dict[str, Any]:
    return await env.run_rollout(
        input=input_row,
        client=vf.ClientConfig(
            api_base_url=PINFERENCE_API_BASE_URL, api_key_var=PINFERENCE_API_KEY_VAR
        ),
        model=model,
        sampling_args=sampling_args,
        state_columns=STATE_COLUMNS,
    )


def build_sequence_configs(
    *,
    feedback_mode: str,
    max_turns: int,
    benchmark_seed: int,
    system_prompt_override: str | None,
    judge_model: str,
    max_examples: int | None = None,
) -> tuple[EnvironmentConfig, EnvironmentConfig]:
    carry_cfg = EnvironmentConfig(
        benchmark_split="carryover_sequences",
        feedback_mode=feedback_mode,
        allow_skill_updates=True,
        max_turns=max_turns,
        split_seed=benchmark_seed,
        system_prompt_override=system_prompt_override,
        judge_model=judge_model,
        max_examples=max_examples,
    )
    transfer_cfg = EnvironmentConfig(
        benchmark_split="transfer_probe",
        feedback_mode="none",
        allow_skill_updates=False,
        max_turns=max_turns,
        split_seed=benchmark_seed,
        system_prompt_override=system_prompt_override,
        judge_model=judge_model,
        max_examples=max_examples,
    )
    return carry_cfg, transfer_cfg


def slice_splits_for_max_examples(
    splits: BenchmarkSplits, max_examples: int | None
) -> tuple[list[list[AdvancedIFExample]], list[AdvancedIFExample]]:
    """Limit carryover to the first sequence and first N tasks; transfer_probe to first N rows."""
    if max_examples is None:
        return splits.carryover_sequences, splits.transfer_probe
    carry = splits.carryover_sequences[:1]
    if carry:
        carry = [carry[0][:max_examples]]
    transfer = splits.transfer_probe[:max_examples]
    return carry, transfer


async def run_sequence_experiment(
    env_id: str,
    model: str,
    feedback_mode: str = "score_only",
    max_turns: int = 6,
    sampling_args: dict[str, Any] | None = None,
    system_prompt_override: str | None = None,
    output_dir: str | Path | None = None,
    benchmark_seed: int = 7,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_examples: int | None = 1,
    env_extra_kwargs: dict[str, Any] | None = None,
    system_prompt_strategy: str = "baseline",
) -> dict[str, Any]:
    sampling = dict(DEFAULT_SAMPLING_ARGS if sampling_args is None else sampling_args)
    env_kwargs = dict(env_extra_kwargs or {})
    carry_cfg, transfer_cfg = build_sequence_configs(
        feedback_mode=feedback_mode,
        max_turns=max_turns,
        benchmark_seed=benchmark_seed,
        system_prompt_override=system_prompt_override,
        judge_model=judge_model,
        max_examples=max_examples,
    )

    carry_env = vf.load_environment(env_id, config=carry_cfg, **env_kwargs)
    transfer_env = vf.load_environment(env_id, config=transfer_cfg, **env_kwargs)
    splits = build_benchmark_splits(carry_cfg)
    carry_sequences, transfer_probe = slice_splits_for_max_examples(splits, max_examples)

    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else default_output_dir(env_id, model, feedback_mode)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "outputs.jsonl"

    record_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for sequence_id, sequence in enumerate(carry_sequences):
            current_skill = EMPTY_SKILL_TEMPLATE
            for task_index, example in enumerate(sequence):
                result = await run_rollout(
                    carry_env,
                    build_rollout_input(example, initial_skill_markdown=current_skill),
                    model,
                    sampling,
                )
                result.update(
                    {
                        "phase": "carryover",
                        "sequence_id": sequence_id,
                        "task_index": task_index,
                        "env": env_id,
                        "model": model,
                        "feedback_mode": feedback_mode,
                        "system_prompt_strategy": system_prompt_strategy,
                    }
                )
                current_skill = str(result.get("current_skill_markdown") or current_skill)
                handle.write(_dumps_rollout_line(result))
                record_count += 1

            for example in transfer_probe:
                for phase, skill_text in (
                    ("transfer_frozen", current_skill),
                    ("transfer_empty_control", EMPTY_SKILL_TEMPLATE),
                ):
                    result = await run_rollout(
                        transfer_env,
                        build_rollout_input(example, initial_skill_markdown=skill_text),
                        model,
                        sampling,
                    )
                    result.update(
                        {
                            "phase": phase,
                            "sequence_id": sequence_id,
                            "task_index": None,
                            "env": env_id,
                            "model": model,
                            "feedback_mode": feedback_mode,
                            "system_prompt_strategy": system_prompt_strategy,
                        }
                    )
                    handle.write(_dumps_rollout_line(result))
                    record_count += 1

    metadata = {
        "env": env_id,
        "model": model,
        "feedback_mode": feedback_mode,
        "system_prompt_strategy": system_prompt_strategy,
        "output_file": str(output_path.resolve()),
        "record_count": record_count,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "env": env_id,
        "model": model,
        "feedback_mode": feedback_mode,
        "results_path": str(output_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "system_prompt_strategy": system_prompt_strategy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential carryover and transfer probes.")
    parser.add_argument("--env", default="advancedif_iter_skill")
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument(
        "--feedback-mode", default="score_only", choices=["score_only", "one_violation", "none"]
    )
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--sampling-temperature", type=float, default=0.2)
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--benchmark-seed", type=int, default=7)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=1,
        help="First sequence only: at most this many carryover tasks and transfer probe rows; "
        "use 0 or negative for no limit (full pilot splits).",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    max_examples: int | None = None if args.max_examples <= 0 else args.max_examples
    result = await run_sequence_experiment(
        env_id=args.env,
        model=args.model,
        feedback_mode=args.feedback_mode,
        max_turns=args.max_turns,
        sampling_args={"temperature": args.sampling_temperature},
        system_prompt_override=load_system_prompt(args.system_prompt_file),
        output_dir=args.output_dir,
        benchmark_seed=args.benchmark_seed,
        judge_model=args.judge_model,
        max_examples=max_examples,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    load_project_dotenv()
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
