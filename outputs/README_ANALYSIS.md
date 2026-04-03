# AdvancedIF RLM vs Iterative

Scores are highly task-dependent. In this setup, the task is mostly "produce the next assistant turn under hidden rubric constraints" and is not a true long-horizon, long-context workload. Gotta do a similar study for more complex multi-turn envs for a fair comparison..

## What the current task is

For each dataset example, the agent gets a convo history and must output the next assistant response that satisfies hidden rubric criteria.

- Dataset: `facebook/AdvancedIF`
- Environment variants:
  - `advancedif_iter_skill` (iterative, tool-using baseline)
  - `advancedif_rlm_skill` (RLM harness baseline)
- Hidden gold rubric: parsed from example metadata and used only by judge

## What the agent has access to

In both harnesses, the agent can access:

- Conversation context from the example
- A carried `skill.md` artifact (initial template, then mutable if updates are allowed)
- A judge tool for draft-checking
- Optional skill-update tool

### Judge feedback modes

- `score_only`: returns only satisfied-count / score (no rubric text)
- `one_violation`: returns one violated criterion (or all-clear)
- `none`: acknowledgement only (no contentful feedback)

## The whole loop

1. Env initializes state (conversation, skill artifact, counters).
2. Agent drafts a candidate answer.
3. Agent may call `submit_candidate_answer(...)` one or more times.
4. Agent may call `update_skill(...)` to persist a revised skill.
5. Agent outputs final assistant answer.
6. Rubric computes final reward via hidden-judge criterion satisfaction.

## Rewards & metrics 

## Primary reward:

- `criterion_satisfaction_mean`: final fraction of hidden criteria satisfied.

Key metrics:

- `all_criteria_pass_rate`
- `first_submission_lift_mean` (final score - first judged draft score)
- `total_tokens_mean` (policy + judge tokens)
- `reward_per_1k_tokens_mean`
- `judge_calls_mean` and `judge_query_count`
- `error_rate`
- Sequence metrics: carryover reward, transfer frozen/control, `transfer_lift_mean`

## preliminary stats for AdvancedIF


| Metric                         | Iterative | RLM      | Winner        | Takeaway                            |
| ------------------------------ | --------- | -------- | ------------- | ----------------------------------- |
| **Judge Calls (mean)**         | 1.03      | **0.78** | **RLM**       | RLM is more call-efficient          |
| **All-Criteria Pass Rate**     | **0.595** | 0.519    | **Iterative** | More reliable task completion       |
| **Criterion Satisfaction**     | **0.821** | 0.604    | **Iterative** | Better rubric adherence             |
| **Error Rate**                 | **0.077** | 0.154    | **Iterative** | About 2x fewer errors               |
| **Total Tokens**               | **6.0k**  | 10.7k    | **Iterative** | Much cheaper                        |
| **Reward / 1k Tokens**         | **0.314** | 0.061    | **Iterative** | About 5x more efficient             |
| **Reward Wins (count)**        | 16        | 2        | **Iterative** | Dominant in current study mix       |
| **Call Efficiency vs Quality** | Balanced  | Skewed   | **Iterative** | RLM saves calls but loses elsewhere |


