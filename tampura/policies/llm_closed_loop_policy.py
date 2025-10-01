"""LLM-based closed-loop policy that directly selects actions.

This policy uses an LLM to synthesize a closed-loop policy function from offline data, bypassing model learning and MDP solving in TAMPURA entirely. The LLM directly maps (abstract_belief, last_observation, last_action, reward) -> action.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tampura.policies.policy import Policy
from tampura.spec import ProblemSpec
from tampura.structs import AbstractBelief, Action, AliasStore, Belief
from tampura.solvers.llm.training_data_formatter import format_trajectory_for_llm

try:
    from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
    from prpl_llm_utils.code import (
        SyntaxRepromptCheck,
        synthesize_python_function_with_llm,
    )
    from prpl_llm_utils.models import OpenAIModel
    from prpl_llm_utils.structs import Query
except ImportError:
    logging.warning("prpl-llm-utils not available.")


class LLMClosedLoopPolicy(Policy):
    """Policy that uses LLM-synthesized function to directly select actions."""

    def __init__(self, config: Dict[str, Any], problem_spec: ProblemSpec, **kwargs):
        super().__init__(config, problem_spec, **kwargs)

        self.llm_policy_fn = None
        self.llm_policy_code = None
        self.t = 0
        self.last_action: Optional[Action] = None
        self.last_observation_dict: Optional[Dict[str, Any]] = None

        model_name = config.get("llm_model", "gpt-4o-mini")
        cache_db_path = config.get("llm_cache_db", ".llm_cache.db")
        cache = SQLite3PretrainedLargeModelCache(Path(cache_db_path))
        self.llm = OpenAIModel(model_name, cache)

        training_data_dir = config.get("training_data_dir", "./training_data")
        max_trajectories = config.get("max_training_trajectories", 5)
        self.training_examples = format_trajectory_for_llm(training_data_dir, max_trajectories)

    def synthesize_policy_function(
        self,
        initial_belief: Belief,
        store: AliasStore
    ) -> bool:
        """Synthesize a closed-loop policy function using LLM."""
        ab = initial_belief.abstract(store)

        pddl_save_dir = os.path.join(self.config["save_dir"], "pddl_init")
        os.makedirs(pddl_save_dir, exist_ok=True)

        domain_file, problem_file = self.problem_spec.save_pddl(
            ab, default_cost=100, folder=pddl_save_dir, store=store
        )
        with open(domain_file, "r") as f:
            domain_content = f.read()
        with open(problem_file, "r") as f:
            problem_content = f.read()

        ab_strings = []
        for atom in ab.items:
            if hasattr(atom, 'pred_name'):
                if atom.args:
                    ab_strings.append(f"({atom.pred_name} {' '.join(map(str, atom.args))})")
                else:
                    ab_strings.append(f"({atom.pred_name})")

        prompt = self._create_synthesis_prompt(domain_content, problem_content, ab_strings)
        prompt_path = Path(self.config["save_dir"]) / "llm_policy_synthesis_prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompt_path, "w") as f:
            f.write(prompt)

        query = Query(prompt)
        reprompt_checks = [SyntaxRepromptCheck()]

        try:
            logging.info("[LLMClosedLoopPolicy] Synthesizing closed-loop policy function...")
            synthesized_fn = synthesize_python_function_with_llm(
                function_name="get_next_action",
                model=self.llm,
                query=query,
                reprompt_checks=reprompt_checks,
            )

            self.llm_policy_fn = synthesized_fn
            self.llm_policy_code = synthesized_fn.code_str

            save_path = Path(self.config["save_dir"]) / "llm_generated_policy_function.py"
            with open(save_path, "w") as f:
                f.write(self.llm_policy_code)

            logging.info("[LLMClosedLoopPolicy] Policy function synthesized successfully")
            return True

        except Exception as e:
            logging.error(f"[LLMClosedLoopPolicy] Failed to synthesize policy: {e}")
            return False

    def get_action(
        self,
        belief: Belief,
        store: AliasStore,
        last_observation: Optional[Dict[str, Any]] = None
    ) -> Tuple[Action, Dict[str, Any], AliasStore]:
        """Get next action using LLM-synthesized policy."""
        ab = belief.abstract(store)
        reward = self.problem_spec.get_reward(ab, store)

        if reward > 0:
            self.t += 1
            return Action("no-op"), {}, store

        if self.llm_policy_fn is None:
            success = self.synthesize_policy_function(belief, store)
            if not success:
                logging.error("[LLMClosedLoopPolicy] Policy synthesis failed, returning no-op")
                self.t += 1
                return Action("no-op"), {}, store

        ab_strings = []
        for atom in ab.items:
            if hasattr(atom, 'pred_name'):
                if atom.args:
                    ab_strings.append(f"({atom.pred_name} {' '.join(map(str, atom.args))})")
                else:
                    ab_strings.append(f"({atom.pred_name})")

        obs_dict = self.last_observation_dict if self.last_observation_dict is not None else {}
        last_action_dict = None
        if self.last_action is not None:
            last_action_dict = {
                "name": self.last_action.name,
                "args": list(self.last_action.args) if self.last_action.args else []
            }

        try:
            action_data = self.llm_policy_fn(ab_strings, obs_dict, last_action_dict, reward)

            if action_data is None or not isinstance(action_data, dict):
                logging.warning("[LLMClosedLoopPolicy] Invalid action returned, using no-op")
                action = Action("no-op")
            else:
                action_name = action_data.get("name", "no-op")
                action_args = tuple(action_data.get("args", []))
                action = Action(action_name, action_args)

            logging.info(f"[LLMClosedLoopPolicy] t={self.t}, Selected action: {action}")

        except Exception as e:
            logging.error(f"[LLMClosedLoopPolicy] Error calling policy function: {e}")
            action = Action("no-op")

        self.last_action = action
        self.last_observation_dict = last_observation
        self.t += 1
        return action, {}, store

    def _create_synthesis_prompt(
        self,
        domain_content: str,
        problem_content: str,
        initial_ab: list[str]
    ) -> str:
        """Create prompt for synthesizing a closed-loop policy function."""
        ab_str = ", ".join(initial_ab) if initial_ab else "(empty)"

        prompt = f"""You are tasked with generating a Python function that implements a closed-loop policy for a POMDP task and motion planning problem.

## Task Description

You will be given:
1. A PDDL domain defining predicates, actions, and their effects
2. A PDDL problem defining the initial state and goal
3. Training examples showing successful execution traces

Your goal is to generate a Python function that takes the current state (abstract belief, last observation, last action, reward) and returns the SINGLE BEST action to take next.

## Training Examples

Below are examples of successful task executions. Study the patterns:
- How the agent reacts to different observations and abstract beliefs
- What actions are effective in different situations
- How to handle uncertainty and partial observability
- When to gather information vs. when to make progress toward the goal

{self.training_examples}

## New Task

Now, you need to generate a closed-loop policy for this new problem instance:

--- Domain (PDDL) ---
{domain_content}

--- Problem (PDDL) ---
{problem_content}

--- Initial Abstract Belief ---
{ab_str}

## Your Task

Generate a Python function with this signature:

```python
def get_action(abstract_belief, last_observation, last_action, reward):
    '''Select the next action based on current state.

    Args:
        abstract_belief: List of strings representing current abstract belief
        last_observation: Dict with keys that may include: regions_in, holding, robot_pose, collision
        last_action: Dict with "name" and "args" keys representing the last action taken
        reward: Float representing current reward (positive means goal reached)

    Returns:
        Dict with single action
    '''
    # Your code here
    pass
```

## Important Notes

- Check if the goal is already reached (reward > 0) → return no-op
- Identify what information is missing → use information-gathering actions
- Once confident about the state → take goal-directed actions
- Handle partial observability by being cautious when uncertain
- Learn from the successful patterns in the training examples

Generate ONLY the Python function code. Do not include explanations or example usage.
"""
        return prompt
