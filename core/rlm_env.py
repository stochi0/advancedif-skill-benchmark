from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, cast

import verifiers as vf
from verifiers.envs.experimental.rlm_env import RLMEnv
from verifiers.types import State

from core.config import EnvironmentConfig
from core.dataset import AdvancedIFExample, build_dataset, materialize_rlm_context
from core.judge import (
    AdvancedIFAnswerJudge,
    AdvancedIFAnswerRubric,
    ensure_judge_state,
    format_limited_feedback,
    parse_gold_rubrics,
    register_judge_result,
)
from core.prompts import render_rlm_task_prompt
from core.skill_artifact import (
    EMPTY_SKILL_TEMPLATE,
    snapshot_skill,
    validate_skill_markdown,
    write_skill,
)


def example_from_state(state: State) -> AdvancedIFExample:
    info = state.get("info", {}) or {}
    return AdvancedIFExample(
        example_id=int(state.get("example_id", 0)),
        benchmark_name=str(info.get("benchmark_name", "unknown")),
        conversation=list(info.get("conversation", [])),
        gold_rubrics=parse_gold_rubrics(str(state.get("answer", ""))),
    )


class RLMSkillEnv(RLMEnv):
    def __init__(self, config: EnvironmentConfig, **kwargs: Any):
        self.cfg = config
        self.answer_judge = AdvancedIFAnswerJudge(config)
        rubric = AdvancedIFAnswerRubric(self.answer_judge)
        dataset = build_dataset(config)

        repl_language = kwargs.pop("repl_language", "python")
        root_prompt_verbosity = kwargs.pop("root_prompt_verbosity", "medium")
        super().__init__(
            dataset=dataset,
            eval_dataset=dataset,
            rubric=rubric,
            parser=vf.Parser(),
            env_id="advancedif_rlm_skill",
            max_turns=config.max_turns,
            root_tools=[self.submit_candidate_answer, self.update_skill],
            repl_language=repl_language,
            root_prompt_verbosity=root_prompt_verbosity,
            **kwargs,
        )

    async def setup_state(self, state: State, **kwargs: Any) -> State:
        ensure_judge_state(state)

        rollout_tag = uuid.uuid4().hex[:8]
        rollout_dir = (
            self.cfg.resolved_output_root
            / self.env_id
            / f"example_{int(state['example_id']):06d}_{rollout_tag}"
        )
        skill_dir = rollout_dir / "skill_snapshots"
        skill_dir.mkdir(parents=True, exist_ok=True)

        initial_skill = (
            str(state.get("info", {}).get("initial_skill_markdown", ""))
            or self.cfg.initial_skill_markdown
            or EMPTY_SKILL_TEMPLATE
        )
        valid, _message, normalized = validate_skill_markdown(initial_skill)
        current_skill = normalized if valid else EMPTY_SKILL_TEMPLATE

        current_skill_path = write_skill(rollout_dir / "current_skill.md", current_skill)
        state["rollout_output_dir"] = str(rollout_dir.resolve())
        state["skill_snapshot_dir"] = str(skill_dir.resolve())
        state["current_skill_markdown"] = current_skill
        state["current_skill_path"] = current_skill_path
        state["skill_update_count"] = 0
        state["skill_snapshot_paths"] = [snapshot_skill(skill_dir, 0, current_skill)]

        info = dict(state.get("info", {}) or {})
        example = example_from_state(state)
        context_root = self.cfg.resolved_context_parent_dir / rollout_tag
        info["context_dir"] = materialize_rlm_context(context_root, example, current_skill)
        state["info"] = info
        state["prompt"] = [
            vf.UserMessage(
                content=render_rlm_task_prompt(
                    current_skill=current_skill,
                    feedback_mode=self.cfg.feedback_mode,
                    allow_skill_updates=self.cfg.allow_skill_updates,
                )
            )
        ]
        return await super().setup_state(state, **kwargs)

    def _state_for_root_tool(self) -> State:
        ctx = self._root_tool_context_var.get()
        if not isinstance(ctx, dict) or not isinstance(ctx.get("state"), dict):
            raise RuntimeError("RLM root tool state is unavailable.")
        return cast(State, ctx["state"])

    async def submit_candidate_answer(self, candidate_answer: str) -> str:
        state = self._state_for_root_tool()
        candidate = candidate_answer.strip()
        if not candidate:
            return "Judge feedback: candidate_answer must be non-empty."

        example = example_from_state(state)
        result = await self.answer_judge.evaluate(
            example_id=example.example_id,
            conversation=example.conversation,
            gold_rubrics=example.gold_rubrics,
            candidate_answer=candidate,
        )
        register_judge_result(state, stage="submission", candidate_answer=candidate, result=result)
        if state.get("first_submission_score") is None:
            state["first_submission_score"] = result.score
            state["first_submission_answer"] = candidate
        state["last_submission_score"] = result.score
        return format_limited_feedback(result, example.gold_rubrics, self.cfg.feedback_mode)

    async def update_skill(self, skill_markdown: str) -> str:
        state = self._state_for_root_tool()
        if not self.cfg.allow_skill_updates:
            return "Skill updates are disabled for this rollout."

        valid, message, normalized = validate_skill_markdown(skill_markdown)
        if not valid:
            return f"Skill update rejected: {message}"

        state["current_skill_markdown"] = normalized
        state["skill_update_count"] = int(state.get("skill_update_count", 0)) + 1
        state["current_skill_path"] = write_skill(
            Path(str(state["rollout_output_dir"])) / "current_skill.md",
            normalized,
        )
        snapshot_path = snapshot_skill(
            state["skill_snapshot_dir"],
            int(state["skill_update_count"]),
            normalized,
        )
        state["skill_snapshot_paths"].append(snapshot_path)
        return (
            "Skill update accepted and persisted. "
            "If you want the sandbox copy to match, rewrite skill.md locally as well."
        )
