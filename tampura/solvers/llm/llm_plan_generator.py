"""LLM-based plan generator to replace symK.

This module uses OpenAI's LLM via prpl-llm-utils to generate K candidate plans
for model learning, replacing the traditional symK planner.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tampura.structs import AbstractBelief, Action, AliasStore
from tampura.symbolic import Atom

try:
    from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
    from prpl_llm_utils.code import (
        SyntaxRepromptCheck,
        synthesize_python_function_with_llm,
    )
    from prpl_llm_utils.models import OpenAIModel
    from prpl_llm_utils.structs import Query
except ImportError:
    logging.warning("prpl-llm-utils not available. LLM plan generation will not work.")


class LLMPlanGenerator:
    """Generate plans using LLM instead of symK."""

    def __init__(
        self,
        training_examples: str,
        model_name: str = "gpt-4o-mini",
        cache_db_path: str = ".llm_cache.db",
        num_plans: int = 10,
        save_dir: Optional[str] = None,
    ):
        """Initialize LLM plan generator."""
        self.training_examples = training_examples
        self.num_plans = num_plans
        self.model_name = model_name
        self.save_dir = save_dir

        cache = SQLite3PretrainedLargeModelCache(Path(cache_db_path))
        self.llm = OpenAIModel(model_name, cache)

        self.plan_generator_fn = None
        self.plan_generator_code = None

    def synthesize_plan_generator(
        self, domain_content: str, problem_content: str, initial_abstract_belief: List[str]
    ) -> bool:
        """Synthesize a Python function that generates K plans."""
        ab_str = ", ".join(initial_abstract_belief) if initial_abstract_belief else "(empty)"
        prompt = self._create_synthesis_prompt(
            domain_content, problem_content, ab_str
        )

        if self.save_dir:
            prompt_path = Path(self.save_dir) / "llm_synthesis_prompt.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(prompt_path, "w") as f:
                f.write(prompt)

        query = Query(prompt)

        reprompt_checks = [SyntaxRepromptCheck()]

        try:
            synthesized_fn = synthesize_python_function_with_llm(
                function_name="generate_plans",
                model=self.llm,
                query=query,
                reprompt_checks=reprompt_checks,
            )
            self.plan_generator_fn = synthesized_fn
            self.plan_generator_code = synthesized_fn.code_str

            if self.save_dir and self.plan_generator_code:
                save_path = Path(self.save_dir) / "llm_generated_plan_function.py"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as f:
                    f.write(self.plan_generator_code)

            logging.info("[LLMPlanGenerator] Plan generator function synthesized successfully")
            return True

        except Exception as e:
            logging.error(f"[LLMPlanGenerator] Failed to synthesize plan generator: {e}")
            return False

    def generate_plans(
        self,
        current_abstract_belief: AbstractBelief,
        last_observation: Optional[Dict[str, Any]],
        reward: float,
    ) -> List[List[Action]]:
        """Generate K candidate plans using the synthesized LLM function."""
        if self.plan_generator_fn is None:
            logging.error("[LLMPlanGenerator] Plan generator not synthesized yet")
            return []

        try:
            ab_strings = [self._atom_to_string(atom) for atom in current_abstract_belief.items]
            obs_dict = last_observation if last_observation is not None else {}

            plans_data = self.plan_generator_fn(ab_strings, obs_dict, reward)

            plans = []
            for plan_data in plans_data:
                plan = []
                for action_data in plan_data:
                    action_name = action_data.get("name", "")
                    action_args = action_data.get("args", [])
                    plan.append(Action(action_name, tuple(action_args)))
                plans.append(plan)

            logging.info(f"[LLMPlanGenerator] Generated {len(plans)} plans")
            return plans

        except Exception as e:
            logging.error(f"[LLMPlanGenerator] Error generating plans: {e}")
            return []

    def _create_synthesis_prompt(
        self, domain_content: str, problem_content: str, initial_ab: str
    ) -> str:
        """Create the prompt for LLM code synthesis."""
        prompt = f"""You are tasked with generating a Python function that produces {self.num_plans} candidate action plans for a POMDP task and motion planning problem.

## Task Description

You will be given:
1. A PDDL domain defining predicates, actions, and their effects
2. A PDDL problem defining the initial state and goal
3. Training examples showing successful execution traces

Your goal is to generate a Python function that takes the current state (abstract belief, last observation, reward) and returns {self.num_plans} diverse candidate action plans.

## Training Examples

Below are {self.num_plans} examples of successful task executions. Study the patterns:
- How actions are sequenced
- How observations and current abstract beliefs affect the next action choice
- What predicates indicate progress toward the goal

{self.training_examples}

## New Task

Now, you need to generate plans for this new problem instance:

--- Domain (PDDL) ---
{domain_content}

--- Problem (PDDL) ---
{problem_content}

--- Initial Abstract Belief ---
{initial_ab}

## Your Task

Generate a Python function with this signature:

```python
def generate_plans(abstract_belief, last_observation, reward):
    '''Generate {self.num_plans} diverse candidate action plans.

    Args:
        abstract_belief: List of strings representing current abstract belief
                        e.g., ["(known_pose)", "(in region1)", "(holding target0)"]
        last_observation: Dict with keys: regions_in, holding, robot_pose, collision
                         e.g., {{"regions_in": ["region1"], "holding": "target0", "robot_pose": null, "collision": false}}
        reward: Float representing current reward

    Returns:
        List of {self.num_plans} plans, where each plan is a list of action dicts
        Each action dict has: {{"name": "action_name", "args": ["arg1", "arg2"]}}
    '''
    # Your code here
    pass
```

## Important Notes

1. Generate exactly {self.num_plans} plans
2. Each plan should be a valid sequence of actions from the domain
3. Consider the current abstract belief and observation when generating plans
4. The robot should be able to follow the plan trajectories with high probability
5. Plans should be diverse (different strategies to reach the goal), unless you are confident about the next actions to take
6. Weigh the benefits of information-gathering actions and reason about whether/when to use them in the plans
7. Return empty list if no valid plans can be generated

Generate ONLY the Python function code. Do not include explanations or example usage.
"""
        return prompt

    def _atom_to_string(self, atom: Atom) -> str:
        """Convert Atom to string representation."""
        if not hasattr(atom, "pred_name"):
            return str(atom)

        if not atom.args or len(atom.args) == 0:
            return f"({atom.pred_name})"
        else:
            args_str = " ".join(str(arg) for arg in atom.args)
            return f"({atom.pred_name} {args_str})"