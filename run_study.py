from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from core.skill_artifact import skill_word_count

DEFAULT_MODEL = "openai/gpt-4.1-mini"
DEFAULT_JUDGE_MODEL = "z-ai/glm-4.7"
DEFAULT_FEEDBACK_MODE = "score_only"
DEFAULT_MAX_TURNS = 6
DEFAULT_BENCHMARK_SPLIT = "main_eval"
DEFAULT_NUM_EXAMPLES = 8
DEFAULT_SEQUENCE_ENV = "auto"
DEFAULT_GEPA_MAX_CALLS = 120
DEFAULT_GEPA_NUM_TRAIN = 4
DEFAULT_GEPA_NUM_VAL = 4
DEFAULT_SEED = 7

# Prime eval / GEPA throughput (lower if you hit API rate limits or local OOM).
DEFAULT_EVAL_MAX_CONCURRENT_ITER = 48
DEFAULT_EVAL_MAX_CONCURRENT_RLM = 32
DEFAULT_GEPA_MAX_CONCURRENT = 32

# RLM sandbox resources per rollout (prime_sandboxes / Docker).
RLM_SANDBOX_CPU_CORES = 8
RLM_SANDBOX_MEMORY_GB = 16
RLM_SANDBOX_DISK_GB = 40
RLM_SANDBOX_TIMEOUT_MINUTES = 60
RLM_CODE_EXECUTION_TIMEOUT_SEC = 300


@dataclass(frozen=True)
class EvalRunSummary:
    label: str
    env_id: str
    model: str
    run_dir: Path
    metadata_path: Path
    results_path: Path
    num_examples: int
    criterion_satisfaction_mean: float
    all_criteria_pass_rate: float
    first_submission_lift_mean: float
    total_tokens_mean: float
    reward_per_1k_tokens_mean: float
    judge_calls_mean: float
    error_rate: float
    paired_rewards: dict[int, float]
    paired_efficiency: dict[int, float]


@dataclass(frozen=True)
class SequenceRunSummary:
    env_id: str
    output_dir: Path
    metadata_path: Path
    results_path: Path
    carryover_reward_mean: float
    transfer_frozen_mean: float
    transfer_control_mean: float
    transfer_lift_mean: float
    transfer_pairs: int
    task_summaries: list[dict[str, float]]
    skill_length_slope: float
    carryover_reward_slope: float
    first_submission_lift_slope: float


@dataclass(frozen=True)
class GepaRunSummary:
    run_dir: Path
    metadata_path: Path
    best_prompt_path: Path
    best_prompt: str
    best_score: float | None
    num_candidates: int | None
    total_metric_calls: int | None


def model_dir_name(env_id: str, model: str) -> str:
    return f"{env_id}--{model.replace('/', '--')}"


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_string(value: str) -> str:
    return json.dumps(value)


def mean_or_zero(values: list[float]) -> float:
    return mean(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def se(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sample_std(values) / math.sqrt(len(values))


def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    mu = mean(values)
    margin = 1.96 * se(values)
    return (mu - margin, mu + margin)


def linear_slope(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    return numer / denom


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    command_str = " ".join(shlex.quote(part) for part in cmd)
    print(f"$ {command_str}")
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {command_str}\n\n")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {command_str}")


def write_eval_config(
    *,
    path: Path,
    env_id: str,
    model: str,
    judge_model: str,
    feedback_mode: str,
    num_examples: int,
    max_turns: int,
    benchmark_split: str,
    max_concurrent: int,
    system_prompt_override: str | None = None,
) -> None:
    lines = [
        f"model = {json_string(model)}",
        'api_base_url = "https://api.pinference.ai/api/v1"',
        'api_key_var = "PRIME_API_KEY"',
        f"num_examples = {num_examples}",
        "rollouts_per_example = 1",
        f"max_concurrent = {max_concurrent}",
        "save_results = true",
        'env_dir_path = "."',
        "",
        "[[eval]]",
        f"env_id = {json_string(env_id)}",
        "",
        "[eval.sampling_args]",
        "temperature = 0.2",
        "",
        "[eval.env_args]",
        f"benchmark_split = {json_string(benchmark_split)}",
        f"feedback_mode = {json_string(feedback_mode)}",
        "allow_skill_updates = true",
        f"judge_model = {json_string(judge_model)}",
        f"max_turns = {max_turns}",
    ]
    if system_prompt_override is not None:
        lines.append(f"system_prompt_override = {json_string(system_prompt_override)}")
    if env_id == "advancedif_rlm_skill":
        lines.extend(
            [
                'repl_language = "python"',
                f"sub_model = {json_string(model)}",
                'sandbox_docker_image = "python:3.11-slim"',
                f"sandbox_cpu_cores = {RLM_SANDBOX_CPU_CORES}",
                f"sandbox_memory_gb = {RLM_SANDBOX_MEMORY_GB}",
                f"sandbox_disk_size_gb = {RLM_SANDBOX_DISK_GB}",
                "sandbox_gpu_count = 0",
                f"sandbox_timeout_minutes = {RLM_SANDBOX_TIMEOUT_MINUTES}",
                f"code_execution_timeout = {RLM_CODE_EXECUTION_TIMEOUT_SEC}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gepa_config(
    *,
    path: Path,
    model: str,
    reflection_model: str,
    judge_model: str,
    max_turns: int,
    max_calls: int,
    num_train: int,
    num_val: int,
    seed: int,
    run_dir: Path,
    max_concurrent: int = DEFAULT_GEPA_MAX_CONCURRENT,
) -> None:
    lines = [
        f"model = {json_string(model)}",
        f"reflection_model = {json_string(reflection_model)}",
        'api_base_url = "https://api.pinference.ai/api/v1"',
        'api_key_var = "PRIME_API_KEY"',
        'env_dir_path = "."',
        f"run_dir = {json_string(str(run_dir))}",
        "save_results = true",
        "",
        "[env]",
        'env_id = "advancedif_iter_skill"',
        (
            "[env.env_args]\n"
            'benchmark_split = "dev_gepa"\n'
            'feedback_mode = "score_only"\n'
            "allow_skill_updates = true\n"
            f"judge_model = {json_string(judge_model)}\n"
            f"max_turns = {max_turns}\n"
            'initial_skill_markdown = ""'
        ),
        "",
        "[gepa]",
        f"max_calls = {max_calls}",
        f"num_train = {num_train}",
        f"num_val = {num_val}",
        "minibatch_size = 3",
        'state_columns = ["criterion_vector", "judge_history", "first_submission_score"]',
        "",
        "[execution]",
        f"max_concurrent = {max_concurrent}",
        f"seed = {seed}",
        "sampling_args = { temperature = 0.2 }",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_latest_run_dir(base_dir: Path) -> Path:
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_from_row(row: dict[str, Any], metric_name: str) -> float:
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and metric_name in metrics:
        return float(metrics[metric_name] or 0.0)
    value = row.get(metric_name)
    if isinstance(value, (int, float)):
        return float(value)
    if metric_name == "skill_word_count_metric":
        skill = row.get("current_skill_markdown")
        if isinstance(skill, str):
            return float(skill_word_count(skill))
    return 0.0


def reward_per_1k_from_row(row: dict[str, Any]) -> float:
    reward = float(row.get("reward") or 0.0)
    tokens = metric_from_row(row, "total_token_count")
    if tokens <= 0:
        return 0.0
    return 1000.0 * reward / tokens


def summarize_eval_run(label: str, run_dir: Path) -> EvalRunSummary:
    metadata_path = run_dir / "metadata.json"
    results_path = run_dir / "results.jsonl"
    metadata = load_json(metadata_path)
    rows = load_jsonl(results_path)

    reward_values = [float(row.get("reward") or 0.0) for row in rows]
    pass_values = [metric_from_row(row, "all_criteria_pass") for row in rows]
    first_lift_values = [metric_from_row(row, "first_submission_lift") for row in rows]
    token_values = [metric_from_row(row, "total_token_count") for row in rows]
    efficiency_values = [reward_per_1k_from_row(row) for row in rows]
    judge_call_values = [metric_from_row(row, "judge_call_count") for row in rows]
    error_values = [1.0 if row.get("error") else 0.0 for row in rows]

    paired_rewards = {
        int(row["example_id"]): float(row.get("reward") or 0.0)
        for row in rows
        if "example_id" in row
    }
    paired_efficiency = {
        int(row["example_id"]): reward_per_1k_from_row(row) for row in rows if "example_id" in row
    }

    return EvalRunSummary(
        label=label,
        env_id=str(metadata["env_id"]),
        model=str(metadata["model"]),
        run_dir=run_dir,
        metadata_path=metadata_path,
        results_path=results_path,
        num_examples=len(rows),
        criterion_satisfaction_mean=mean_or_zero(reward_values),
        all_criteria_pass_rate=mean_or_zero(pass_values),
        first_submission_lift_mean=mean_or_zero(first_lift_values),
        total_tokens_mean=mean_or_zero(token_values),
        reward_per_1k_tokens_mean=mean_or_zero(efficiency_values),
        judge_calls_mean=mean_or_zero(judge_call_values),
        error_rate=mean_or_zero(error_values),
        paired_rewards=paired_rewards,
        paired_efficiency=paired_efficiency,
    )


def summarize_sequence_run(
    *,
    env_id: str,
    output_dir: Path,
    metadata_path: Path,
    results_path: Path,
) -> SequenceRunSummary:
    rows = load_jsonl(results_path)

    carry_rows = [row for row in rows if row.get("phase") == "carryover"]
    frozen_rows = [row for row in rows if row.get("phase") == "transfer_frozen"]
    control_rows = [row for row in rows if row.get("phase") == "transfer_empty_control"]

    transfer_frozen = {
        (int(row["sequence_id"]), int(row["example_id"])): float(row.get("reward") or 0.0)
        for row in frozen_rows
    }
    transfer_control = {
        (int(row["sequence_id"]), int(row["example_id"])): float(row.get("reward") or 0.0)
        for row in control_rows
    }
    transfer_keys = sorted(set(transfer_frozen) & set(transfer_control))
    transfer_lifts = [transfer_frozen[key] - transfer_control[key] for key in transfer_keys]

    task_indices = sorted(
        {int(row["task_index"]) for row in carry_rows if row.get("task_index") is not None}
    )
    task_summaries: list[dict[str, float]] = []
    for task_index in task_indices:
        rows_at_index = [row for row in carry_rows if int(row.get("task_index", -1)) == task_index]
        task_summaries.append(
            {
                "task_index": float(task_index),
                "count": float(len(rows_at_index)),
                "skill_word_count_mean": mean_or_zero(
                    [metric_from_row(row, "skill_word_count_metric") for row in rows_at_index]
                ),
                "criterion_satisfaction_mean": mean_or_zero(
                    [float(row.get("reward") or 0.0) for row in rows_at_index]
                ),
                "first_submission_lift_mean": mean_or_zero(
                    [metric_from_row(row, "first_submission_lift") for row in rows_at_index]
                ),
            }
        )

    skill_length_slope = linear_slope(
        [(summary["task_index"], summary["skill_word_count_mean"]) for summary in task_summaries]
    )
    carryover_reward_slope = linear_slope(
        [
            (summary["task_index"], summary["criterion_satisfaction_mean"])
            for summary in task_summaries
        ]
    )
    first_submission_lift_slope = linear_slope(
        [
            (summary["task_index"], summary["first_submission_lift_mean"])
            for summary in task_summaries
        ]
    )

    return SequenceRunSummary(
        env_id=env_id,
        output_dir=output_dir,
        metadata_path=metadata_path,
        results_path=results_path,
        carryover_reward_mean=mean_or_zero([float(row.get("reward") or 0.0) for row in carry_rows]),
        transfer_frozen_mean=mean_or_zero([float(row.get("reward") or 0.0) for row in frozen_rows]),
        transfer_control_mean=mean_or_zero(
            [float(row.get("reward") or 0.0) for row in control_rows]
        ),
        transfer_lift_mean=mean_or_zero(transfer_lifts),
        transfer_pairs=len(transfer_lifts),
        task_summaries=task_summaries,
        skill_length_slope=skill_length_slope,
        carryover_reward_slope=carryover_reward_slope,
        first_submission_lift_slope=first_submission_lift_slope,
    )


def summarize_gepa_run(run_dir: Path) -> GepaRunSummary:
    metadata_path = run_dir / "metadata.json"
    best_prompt_path = run_dir / "best_prompt.txt"
    metadata = load_json(metadata_path)
    best_prompt = best_prompt_path.read_text(encoding="utf-8")
    return GepaRunSummary(
        run_dir=run_dir,
        metadata_path=metadata_path,
        best_prompt_path=best_prompt_path,
        best_prompt=best_prompt,
        best_score=(
            float(metadata["best_score"])
            if isinstance(metadata.get("best_score"), (int, float))
            else None
        ),
        num_candidates=(
            int(metadata["num_candidates"])
            if isinstance(metadata.get("num_candidates"), (int, float))
            else None
        ),
        total_metric_calls=(
            int(metadata["total_metric_calls"])
            if isinstance(metadata.get("total_metric_calls"), (int, float))
            else None
        ),
    )


def compare_eval_runs(iter_summary: EvalRunSummary, rlm_summary: EvalRunSummary) -> dict[str, Any]:
    paired_example_ids = sorted(set(iter_summary.paired_rewards) & set(rlm_summary.paired_rewards))
    reward_deltas = [
        rlm_summary.paired_rewards[example_id] - iter_summary.paired_rewards[example_id]
        for example_id in paired_example_ids
    ]
    efficiency_deltas = [
        rlm_summary.paired_efficiency[example_id] - iter_summary.paired_efficiency[example_id]
        for example_id in paired_example_ids
    ]

    reward_ci = confidence_interval_95(reward_deltas)
    efficiency_ci = confidence_interval_95(efficiency_deltas)

    return {
        "paired_examples": len(paired_example_ids),
        "reward_delta_rlm_minus_iter": mean_or_zero(reward_deltas),
        "reward_delta_ci95": reward_ci,
        "efficiency_delta_rlm_minus_iter": mean_or_zero(efficiency_deltas),
        "efficiency_delta_ci95": efficiency_ci,
        "rlm_reward_win_rate": mean_or_zero([1.0 if delta > 0 else 0.0 for delta in reward_deltas]),
        "rlm_efficiency_win_rate": mean_or_zero(
            [1.0 if delta > 0 else 0.0 for delta in efficiency_deltas]
        ),
    }


def choose_sequence_env(comparison: dict[str, Any]) -> str:
    reward_delta = float(comparison["reward_delta_rlm_minus_iter"])
    efficiency_delta = float(comparison["efficiency_delta_rlm_minus_iter"])
    if reward_delta > 0.02:
        return "advancedif_rlm_skill"
    if reward_delta < -0.02:
        return "advancedif_iter_skill"
    if efficiency_delta > 0.05:
        return "advancedif_rlm_skill"
    return "advancedif_iter_skill"


def verdict_from_delta(
    delta: float,
    *,
    positive_threshold: float,
    negative_threshold: float | None = None,
) -> str:
    negative_threshold = (
        negative_threshold if negative_threshold is not None else -positive_threshold
    )
    if delta >= positive_threshold:
        return "Yes"
    if delta <= negative_threshold:
        return "No"
    return "Inconclusive"


def growth_verdict(sequence: SequenceRunSummary) -> str:
    if sequence.transfer_lift_mean > 0.02 and sequence.carryover_reward_slope >= -0.01:
        if abs(sequence.skill_length_slope) <= 5.0:
            return "Weak yes"
        return "Mixed"
    if sequence.transfer_lift_mean < -0.02:
        return "No"
    return "Inconclusive"


def build_report_markdown(
    *,
    study_name: str,
    study_dir: Path,
    model: str,
    judge_model: str,
    feedback_mode: str,
    iter_summary: EvalRunSummary,
    rlm_summary: EvalRunSummary,
    comparison: dict[str, Any],
    sequence_summary: SequenceRunSummary,
    sequence_env: str,
    gepa_eval_summary: EvalRunSummary | None,
    gepa_run_summary: GepaRunSummary | None,
) -> str:
    q1_verdict = verdict_from_delta(
        float(comparison["reward_delta_rlm_minus_iter"]),
        positive_threshold=0.02,
    )
    q2_verdict = verdict_from_delta(
        float(comparison["efficiency_delta_rlm_minus_iter"]),
        positive_threshold=0.05,
    )
    q3_verdict = verdict_from_delta(sequence_summary.transfer_lift_mean, positive_threshold=0.02)
    q4_verdict = growth_verdict(sequence_summary)

    lines = [
        f"# {study_name}",
        "",
        "## Setup",
        "",
        f"- Model: `{model}`",
        f"- Judge model: `{judge_model}`",
        f"- Feedback mode: `{feedback_mode}`",
        f"- Study directory: `{study_dir}`",
        f"- Sequence harness used: `{sequence_env}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Variant | Criterion satisfaction | All-criteria pass | First-submission lift | Total tokens | Reward / 1k tokens | Judge calls | Error rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Iterative | {iter_summary.criterion_satisfaction_mean:.4f} | "
            f"{iter_summary.all_criteria_pass_rate:.4f} | {iter_summary.first_submission_lift_mean:.4f} | "
            f"{iter_summary.total_tokens_mean:.1f} | {iter_summary.reward_per_1k_tokens_mean:.4f} | "
            f"{iter_summary.judge_calls_mean:.4f} | {iter_summary.error_rate:.4f} |"
        ),
        (
            f"| RLM | {rlm_summary.criterion_satisfaction_mean:.4f} | "
            f"{rlm_summary.all_criteria_pass_rate:.4f} | {rlm_summary.first_submission_lift_mean:.4f} | "
            f"{rlm_summary.total_tokens_mean:.1f} | {rlm_summary.reward_per_1k_tokens_mean:.4f} | "
            f"{rlm_summary.judge_calls_mean:.4f} | {rlm_summary.error_rate:.4f} |"
        ),
    ]

    if gepa_eval_summary is not None:
        lines.append(
            (
                f"| GEPA-optimized iterative | {gepa_eval_summary.criterion_satisfaction_mean:.4f} | "
                f"{gepa_eval_summary.all_criteria_pass_rate:.4f} | "
                f"{gepa_eval_summary.first_submission_lift_mean:.4f} | "
                f"{gepa_eval_summary.total_tokens_mean:.1f} | "
                f"{gepa_eval_summary.reward_per_1k_tokens_mean:.4f} | "
                f"{gepa_eval_summary.judge_calls_mean:.4f} | {gepa_eval_summary.error_rate:.4f} |"
            )
        )

    lines.extend(
        [
            "",
            "## Paired Deltas",
            "",
            f"- Paired examples: {int(comparison['paired_examples'])}",
            (
                f"- `reward_delta_rlm_minus_iter`: {float(comparison['reward_delta_rlm_minus_iter']):.4f} "
                f"(95% CI {comparison['reward_delta_ci95'][0]:.4f} to "
                f"{comparison['reward_delta_ci95'][1]:.4f})"
            ),
            (
                f"- `efficiency_delta_rlm_minus_iter`: "
                f"{float(comparison['efficiency_delta_rlm_minus_iter']):.4f} "
                f"(95% CI {comparison['efficiency_delta_ci95'][0]:.4f} to "
                f"{comparison['efficiency_delta_ci95'][1]:.4f})"
            ),
            f"- RLM reward win rate across paired examples: {float(comparison['rlm_reward_win_rate']):.4f}",
            (
                f"- RLM efficiency win rate across paired examples: "
                f"{float(comparison['rlm_efficiency_win_rate']):.4f}"
            ),
            "",
            "## Carryover And Transfer",
            "",
            f"- Carryover reward mean: {sequence_summary.carryover_reward_mean:.4f}",
            f"- Transfer frozen mean: {sequence_summary.transfer_frozen_mean:.4f}",
            f"- Transfer empty-control mean: {sequence_summary.transfer_control_mean:.4f}",
            f"- `transfer_lift_mean`: {sequence_summary.transfer_lift_mean:.4f}",
            f"- Transfer pairs: {sequence_summary.transfer_pairs}",
            f"- Skill length slope by task index: {sequence_summary.skill_length_slope:.4f}",
            f"- Carryover reward slope by task index: {sequence_summary.carryover_reward_slope:.4f}",
            (
                f"- First-submission lift slope by task index: "
                f"{sequence_summary.first_submission_lift_slope:.4f}"
            ),
            "",
            "### Task-Index Means",
            "",
            "| Task index | Skill words | Criterion satisfaction | First-submission lift |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for summary in sequence_summary.task_summaries:
        lines.append(
            (
                f"| {int(summary['task_index'])} | {summary['skill_word_count_mean']:.1f} | "
                f"{summary['criterion_satisfaction_mean']:.4f} | "
                f"{summary['first_submission_lift_mean']:.4f} |"
            )
        )

    lines.extend(
        [
            "",
            "## Direct Answers",
            "",
            (
                f"- Q1. Is the RLM harness + skill iteration setup better or worse than non-RLM iterative "
                f"refinement with the same judge feedback tool? {q1_verdict}. "
                f"The paired reward delta is {float(comparison['reward_delta_rlm_minus_iter']):.4f}."
            ),
            (
                f"- Q2. Is it more token-efficient? {q2_verdict}. "
                f"The efficiency delta is {float(comparison['efficiency_delta_rlm_minus_iter']):.4f}."
            ),
            (
                f"- Q3. Do models in this setup write skills that generalize to other tasks? {q3_verdict}. "
                f"The transfer lift is {sequence_summary.transfer_lift_mean:.4f}."
            ),
            (
                f"- Q4. If multiple tasks are attempted sequentially, with the skill carrying over across "
                f"tasks, do models eventually write skills which are more high-level or less overfit to "
                f"specific tasks? {q4_verdict}. This benchmark only measures that indirectly via transfer "
                f"lift plus task-index trends, not semantic abstraction directly."
            ),
        ]
    )

    if gepa_eval_summary is None or gepa_run_summary is None:
        lines.append(
            "- Q5. How does this compare to prompt optimization strategies like GEPA? Not run in this study."
        )
    else:
        gepa_minus_iter = (
            gepa_eval_summary.criterion_satisfaction_mean - iter_summary.criterion_satisfaction_mean
        )
        gepa_minus_rlm = (
            gepa_eval_summary.criterion_satisfaction_mean - rlm_summary.criterion_satisfaction_mean
        )
        lines.extend(
            [
                (
                    f"- Q5. How does this compare to prompt optimization strategies like GEPA? "
                    f"GEPA main-eval criterion satisfaction is "
                    f"{gepa_eval_summary.criterion_satisfaction_mean:.4f}. "
                    f"`gepa_minus_iter` is {gepa_minus_iter:.4f}, and `gepa_minus_rlm` is "
                    f"{gepa_minus_rlm:.4f}."
                ),
                (
                    f"- GEPA best validation score: {gepa_run_summary.best_score:.4f}"
                    if gepa_run_summary.best_score is not None
                    else "- GEPA best validation score: unavailable."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            (
                f"- Use `{sequence_env}` as the next scaffold to study or train against unless you have an "
                "external reason to privilege the other harness."
            ),
            (
                "- Treat the skill-generalization answer as the strongest signal for whether self-improving "
                "artifacts are worth deeper investment."
            ),
            (
                "- Do not overinterpret Q4: the benchmark can detect transfer and growth patterns, but it "
                "does not directly measure whether the learned skill became semantically more abstract."
            ),
            "",
            "## Artifacts",
            "",
            f"- Iterative eval: `{iter_summary.run_dir}`",
            f"- RLM eval: `{rlm_summary.run_dir}`",
            f"- Sequence run: `{sequence_summary.output_dir}`",
        ]
    )

    if gepa_run_summary is not None:
        lines.extend(
            [
                f"- GEPA optimization: `{gepa_run_summary.run_dir}`",
                (
                    f"- GEPA main eval: `{gepa_eval_summary.run_dir}`"
                    if gepa_eval_summary is not None
                    else "- GEPA main eval: unavailable."
                ),
            ]
        )

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the narrowed AdvancedIF study and write a markdown analysis report."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--feedback-mode",
        default=DEFAULT_FEEDBACK_MODE,
        choices=["score_only", "one_violation", "none"],
    )
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--benchmark-split", default=DEFAULT_BENCHMARK_SPLIT)
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--sequence-env",
        default=DEFAULT_SEQUENCE_ENV,
        choices=["auto", "advancedif_iter_skill", "advancedif_rlm_skill"],
    )
    parser.add_argument("--skip-gepa", action="store_true")
    parser.add_argument("--study-root", default="outputs/studies")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_root = Path(__file__).resolve().parent
    study_root = Path(args.study_root)
    if not study_root.is_absolute():
        study_root = (package_root / study_root).resolve()
    study_dir = study_root / f"initial_study__{args.model.replace('/', '__')}__{timestamp_slug()}"
    configs_dir = study_dir / "configs"
    logs_dir = study_dir / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    iter_config_path = configs_dir / "iter_main_eval.toml"
    rlm_config_path = configs_dir / "rlm_main_eval.toml"
    write_eval_config(
        path=iter_config_path,
        env_id="advancedif_iter_skill",
        model=args.model,
        judge_model=args.judge_model,
        feedback_mode=args.feedback_mode,
        num_examples=args.num_examples,
        max_turns=args.max_turns,
        benchmark_split=args.benchmark_split,
        max_concurrent=DEFAULT_EVAL_MAX_CONCURRENT_ITER,
    )
    write_eval_config(
        path=rlm_config_path,
        env_id="advancedif_rlm_skill",
        model=args.model,
        judge_model=args.judge_model,
        feedback_mode=args.feedback_mode,
        num_examples=args.num_examples,
        max_turns=args.max_turns,
        benchmark_split=args.benchmark_split,
        max_concurrent=DEFAULT_EVAL_MAX_CONCURRENT_RLM,
    )

    run_command(
        [
            "uv",
            "run",
            "prime",
            "eval",
            "run",
            str(iter_config_path),
            "--output-dir",
            str(study_dir),
        ],
        cwd=package_root,
        log_path=logs_dir / "iter_eval.log",
    )
    run_command(
        [
            "uv",
            "run",
            "prime",
            "eval",
            "run",
            str(rlm_config_path),
            "--output-dir",
            str(study_dir),
        ],
        cwd=package_root,
        log_path=logs_dir / "rlm_eval.log",
    )

    iter_run_dir = find_latest_run_dir(
        study_dir / "evals" / model_dir_name("advancedif_iter_skill", args.model)
    )
    rlm_run_dir = find_latest_run_dir(
        study_dir / "evals" / model_dir_name("advancedif_rlm_skill", args.model)
    )
    iter_summary = summarize_eval_run("iterative", iter_run_dir)
    rlm_summary = summarize_eval_run("rlm", rlm_run_dir)
    comparison = compare_eval_runs(iter_summary, rlm_summary)

    if args.sequence_env == "auto":
        sequence_env = choose_sequence_env(comparison)
    else:
        sequence_env = args.sequence_env

    sequence_dir = study_dir / "sequence_run"
    run_command(
        [
            "uv",
            "run",
            "python",
            "sequence_runner.py",
            "--env",
            sequence_env,
            "--model",
            args.model,
            "--feedback-mode",
            args.feedback_mode,
            "--judge-model",
            args.judge_model,
            "--max-turns",
            str(args.max_turns),
            "--benchmark-seed",
            str(args.seed),
            "--output-dir",
            str(sequence_dir),
        ],
        cwd=package_root,
        log_path=logs_dir / "sequence.log",
    )
    sequence_summary = summarize_sequence_run(
        env_id=sequence_env,
        output_dir=sequence_dir,
        metadata_path=sequence_dir / "metadata.json",
        results_path=sequence_dir / "outputs.jsonl",
    )

    gepa_run_summary: GepaRunSummary | None = None
    gepa_eval_summary: EvalRunSummary | None = None
    if not args.skip_gepa:
        gepa_run_dir = study_dir / "gepa_run"
        gepa_config_path = configs_dir / "gepa.toml"
        write_gepa_config(
            path=gepa_config_path,
            model=args.model,
            reflection_model=args.model,
            judge_model=args.judge_model,
            max_turns=args.max_turns,
            max_calls=DEFAULT_GEPA_MAX_CALLS,
            num_train=DEFAULT_GEPA_NUM_TRAIN,
            num_val=DEFAULT_GEPA_NUM_VAL,
            seed=args.seed,
            run_dir=gepa_run_dir,
        )
        run_command(
            ["uv", "run", "prime", "gepa", "run", str(gepa_config_path)],
            cwd=package_root,
            log_path=logs_dir / "gepa.log",
        )
        gepa_run_summary = summarize_gepa_run(gepa_run_dir)

        gepa_eval_config_path = configs_dir / "gepa_main_eval.toml"
        write_eval_config(
            path=gepa_eval_config_path,
            env_id="advancedif_iter_skill",
            model=args.model,
            judge_model=args.judge_model,
            feedback_mode=args.feedback_mode,
            num_examples=args.num_examples,
            max_turns=args.max_turns,
            benchmark_split=args.benchmark_split,
            max_concurrent=DEFAULT_EVAL_MAX_CONCURRENT_ITER,
            system_prompt_override=gepa_run_summary.best_prompt,
        )
        run_command(
            [
                "uv",
                "run",
                "prime",
                "eval",
                "run",
                str(gepa_eval_config_path),
                "--output-dir",
                str(study_dir),
            ],
            cwd=package_root,
            log_path=logs_dir / "gepa_main_eval.log",
        )
        gepa_eval_dir = find_latest_run_dir(
            study_dir / "evals" / model_dir_name("advancedif_iter_skill", args.model)
        )
        if gepa_eval_dir == iter_run_dir:
            raise RuntimeError("Failed to locate the GEPA main-eval run directory.")
        gepa_eval_summary = summarize_eval_run("gepa_main_eval", gepa_eval_dir)

    summary_payload = {
        "study_dir": str(study_dir),
        "model": args.model,
        "judge_model": args.judge_model,
        "feedback_mode": args.feedback_mode,
        "iter_summary": iter_summary.__dict__,
        "rlm_summary": rlm_summary.__dict__,
        "comparison": comparison,
        "sequence_env": sequence_env,
        "sequence_summary": {
            **sequence_summary.__dict__,
            "task_summaries": sequence_summary.task_summaries,
        },
        "gepa_run_summary": None if gepa_run_summary is None else gepa_run_summary.__dict__,
        "gepa_eval_summary": None if gepa_eval_summary is None else gepa_eval_summary.__dict__,
    }
    summary_json_path = study_dir / "study_summary.json"
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, default=str), encoding="utf-8"
    )

    report_text = build_report_markdown(
        study_name="AdvancedIF Initial Study Report",
        study_dir=study_dir,
        model=args.model,
        judge_model=args.judge_model,
        feedback_mode=args.feedback_mode,
        iter_summary=iter_summary,
        rlm_summary=rlm_summary,
        comparison=comparison,
        sequence_summary=sequence_summary,
        sequence_env=sequence_env,
        gepa_eval_summary=gepa_eval_summary,
        gepa_run_summary=gepa_run_summary,
    )
    report_path = study_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nStudy summary: {summary_json_path}")
    print(f"Study report: {report_path}")


if __name__ == "__main__":
    main()
