from __future__ import annotations

import json
from pathlib import Path

import pytest
from core.config import EnvironmentConfig
from core.dataset import AdvancedIFExample, build_rollout_input
from core.iter_env import IterSkillEnv
from core.rlm_env import RLMSkillEnv
from core.skill_artifact import EMPTY_SKILL_TEMPLATE
from sequence_runner import _dumps_rollout_line, run_sequence_experiment
from verifiers.clients.client import Client
from verifiers.types import Response, ResponseMessage, SystemMessage, ToolCall, UserMessage


def make_dataset_row() -> dict[str, str]:
    return {
        "benchmark_name": "complex_if_single_turn_v5",
        "conversation_history": json.dumps([{"role": "user", "content": "Answer in two bullets."}]),
        "prompt_metadata": json.dumps({"rubrics": json.dumps(["two bullets", "helpful"])}),
    }


class FakeJudgeCompletions:
    async def create(self, model, messages, **kwargs):
        content = messages[0]["content"]
        if "final answer" in content:
            satisfied = [True, True]
        else:
            satisfied = [False, True]

        class Usage:
            prompt_tokens = 5
            completion_tokens = 3

        class Message:
            content = json.dumps({"satisfied": satisfied})

        class Choice:
            message = Message()

        class Resp:
            usage = Usage()
            choices = [Choice()]

        return Resp()


class FakeJudgeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeJudgeCompletions()})()


class MockClient(Client):
    def __init__(self, responses):
        self.responses = list(responses)
        self._client = None

    async def get_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        content, tool_calls = self.responses.pop(0)
        return Response(
            id="test",
            created=0,
            model=model,
            usage=None,
            message=ResponseMessage(
                content=content,
                finish_reason="tool_calls" if tool_calls else "stop",
                is_truncated=False,
                tokens=None,
                tool_calls=tool_calls,
            ),
        )

    def setup_client(self, config):
        return None

    async def to_native_tool(self, tool):
        return None

    async def to_native_prompt(self, messages):
        return [], {}

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        return None

    async def raise_from_native_response(self, response):
        return None

    async def from_native_response(self, response):
        return None

    async def close(self) -> None:
        return None


def tool_call(name: str, arguments: dict[str, str], call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


@pytest.mark.asyncio
async def test_iter_env_round_trip_with_tools(tmp_path: Path):
    cfg = EnvironmentConfig(
        dataset_rows=[make_dataset_row()],
        judge_client=FakeJudgeClient(),
        output_root=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
    )
    env = IterSkillEnv(cfg)
    example = AdvancedIFExample(
        example_id=0,
        benchmark_name="complex_if_single_turn_v5",
        conversation=[{"role": "user", "content": "Answer in two bullets."}],
        gold_rubrics=["two bullets", "helpful"],
    )
    client = MockClient(
        responses=[
            (
                "",
                [
                    tool_call(
                        "submit_candidate_answer", {"candidate_answer": "draft answer"}, "call-1"
                    )
                ],
            ),
            (
                "",
                [tool_call("update_skill", {"skill_markdown": EMPTY_SKILL_TEMPLATE}, "call-2")],
            ),
            ("final answer", None),
        ]
    )
    output = await env.run_rollout(
        input=build_rollout_input(example, initial_skill_markdown=EMPTY_SKILL_TEMPLATE),
        client=client,
        model="test-model",
        sampling_args={"temperature": 0.0},
        state_columns=[
            "current_skill_markdown",
            "skill_snapshot_paths",
            "criterion_vector",
            "final_answer",
        ],
    )

    assert output["reward"] == 1.0
    assert output["final_answer"] == "final answer"
    assert output["criterion_vector"] == [True, True]
    assert len(output["skill_snapshot_paths"]) >= 2


@pytest.mark.asyncio
async def test_rlm_env_setup_state_materializes_context_and_tools(tmp_path: Path, monkeypatch):
    cfg = EnvironmentConfig(
        dataset_rows=[make_dataset_row()],
        judge_client=FakeJudgeClient(),
        output_root=str(tmp_path / "outputs"),
        cache_dir=str(tmp_path / "cache"),
        context_parent_dir=str(tmp_path / "contexts"),
    )
    env = RLMSkillEnv(cfg)
    example = AdvancedIFExample(
        example_id=0,
        benchmark_name="complex_if_single_turn_v5",
        conversation=[{"role": "user", "content": "Answer in two bullets."}],
        gold_rubrics=["two bullets", "helpful"],
    )
    client = MockClient(responses=[])

    async def fake_super_setup_state(self, state, **kwargs):
        return state

    monkeypatch.setattr(
        "verifiers.envs.experimental.rlm_env.RLMEnv.setup_state",
        fake_super_setup_state,
    )

    state = await env.init_state(
        build_rollout_input(example, initial_skill_markdown=EMPTY_SKILL_TEMPLATE),
        client,
        "test-model",
        None,
    )
    state = await env.setup_state(state)

    context_dir = Path(state["info"]["context_dir"])
    assert (context_dir / "trajectory" / "manifest.json").is_file()
    assert (context_dir / "skill.md").is_file()
    assert "submit_candidate_answer" in [tool.__name__ for tool in env.root_tools]
    assert "update_skill" in [tool.__name__ for tool in env.root_tools]

    env._root_tool_context_var.set({"state": state})
    message = await env.submit_candidate_answer("draft answer")
    assert "Judge feedback" in message


@pytest.mark.asyncio
async def test_sequence_runner_writes_outputs(tmp_path: Path, monkeypatch):
    carry_examples = [
        AdvancedIFExample(
            example_id=1,
            benchmark_name="complex_if_single_turn_v5",
            conversation=[{"role": "user", "content": "Task one"}],
            gold_rubrics=["rubric-a"],
        ),
        AdvancedIFExample(
            example_id=2,
            benchmark_name="system_steerability_v2",
            conversation=[{"role": "user", "content": "Task two"}],
            gold_rubrics=["rubric-b"],
        ),
    ]
    transfer_examples = [
        AdvancedIFExample(
            example_id=3,
            benchmark_name="carried_context_multi_turn_eval_v5",
            conversation=[{"role": "user", "content": "Transfer task"}],
            gold_rubrics=["rubric-c"],
        )
    ]

    class FakeSequenceEnv:
        def __init__(self, allow_skill_updates: bool):
            self.allow_skill_updates = allow_skill_updates

        async def run_rollout(self, input, client, model, sampling_args, state_columns):
            skill_text = str(input["info"]["initial_skill_markdown"])
            if self.allow_skill_updates:
                skill_text = skill_text + "\nsequence-note"
            return {
                "reward": 0.75 if self.allow_skill_updates else 0.5,
                "first_submission_lift": 0.1,
                "current_skill_markdown": skill_text,
                "rollout_output_dir": str(tmp_path / "rollouts" / f"example_{input['example_id']}"),
            }

    def fake_load_environment(env_id, config, **kwargs):
        return FakeSequenceEnv(allow_skill_updates=config.allow_skill_updates)

    monkeypatch.setattr("sequence_runner.vf.load_environment", fake_load_environment)
    monkeypatch.setattr(
        "sequence_runner.build_benchmark_splits",
        lambda cfg: type(
            "FakeSplits",
            (),
            {
                "carryover_sequences": [carry_examples],
                "transfer_probe": transfer_examples,
            },
        )(),
    )

    result = await run_sequence_experiment(
        env_id="advancedif_iter_skill",
        model="model-a",
        output_dir=tmp_path / "sequence_run",
        max_examples=None,
    )

    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["record_count"] == 4
    assert Path(result["results_path"]).is_file()


def test_dumps_rollout_line_serializes_verifiers_messages():
    row = {
        "prompt": [SystemMessage(content="sys"), UserMessage(content="hi")],
        "reward": 0.5,
    }
    line = _dumps_rollout_line(row)
    out = json.loads(line)
    assert out["reward"] == 0.5
    assert out["prompt"][0] == {"role": "system", "content": "sys"}
    assert out["prompt"][1] == {"role": "user", "content": "hi"}
