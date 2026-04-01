from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import verifiers as vf
from verifiers.types import State, SystemMessage

from core.config import EnvironmentConfig
from core.dataset import build_dataset
from core.judge import (
    AdvancedIFAnswerJudge,
    AdvancedIFAnswerRubric,
    ensure_judge_state,
    format_limited_feedback,
    parse_gold_rubrics,
    register_judge_result,
)
from core.prompts import ITER_SYSTEM_PROMPT, render_iter_dynamic_skill_message
from core.skill_artifact import (
    EMPTY_SKILL_TEMPLATE,
    snapshot_skill,
    validate_skill_markdown,
    write_skill,
)


class IterSkillEnv(vf.StatefulToolEnv):
    def __init__(self, config: EnvironmentConfig, **kwargs: Any):
        self.cfg = config
        self.answer_judge = AdvancedIFAnswerJudge(config)
        rubric = AdvancedIFAnswerRubric(self.answer_judge)
        dataset = build_dataset(config)
        system_prompt = config.system_prompt_override or ITER_SYSTEM_PROMPT

        super().__init__(
            dataset=dataset,
            eval_dataset=dataset,
            rubric=rubric,
            parser=vf.Parser(),
            system_prompt=system_prompt,
            env_id="advancedif_iter_skill",
            max_turns=config.max_turns,
            **kwargs,
        )
        self.add_tool(self.submit_candidate_answer, args_to_skip=["state"])
        self.add_tool(self.update_skill, args_to_skip=["state"])

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
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

        prompt = list(state["prompt"])
        insertion_idx = 1 if prompt and getattr(prompt[0], "role", None) == "system" else 0
        prompt.insert(
            insertion_idx,
            SystemMessage(
                content=render_iter_dynamic_skill_message(
                    current_skill=current_skill,
                    feedback_mode=self.cfg.feedback_mode,
                )
            ),
        )
        state["prompt"] = prompt
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: State,
        **kwargs: Any,
    ) -> dict:
        if tool_name in {"submit_candidate_answer", "update_skill"}:
            updated = dict(tool_args)
            updated["state"] = state
            return updated
        return super().update_tool_args(tool_name, tool_args, messages, state, **kwargs)

    async def submit_candidate_answer(self, candidate_answer: str, state: State) -> str:
        candidate = candidate_answer.strip()
        if not candidate:
            return "Judge feedback: candidate_answer must be non-empty."

        info = state.get("info", {}) or {}
        gold_rubrics = parse_gold_rubrics(str(state.get("answer", "")))
        result = await self.answer_judge.evaluate(
            example_id=int(state.get("example_id", 0)),
            conversation=list(info.get("conversation", [])),
            gold_rubrics=gold_rubrics,
            candidate_answer=candidate,
        )
        register_judge_result(state, stage="submission", candidate_answer=candidate, result=result)
        if state.get("first_submission_score") is None:
            state["first_submission_score"] = result.score
            state["first_submission_answer"] = candidate
        state["last_submission_score"] = result.score
        return format_limited_feedback(result, gold_rubrics, self.cfg.feedback_mode)

    async def update_skill(self, skill_markdown: str, state: State) -> str:
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
        return "Skill update accepted and persisted."
