"""Format collected trajectory data for LLM training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class TrainingDataFormatter:
    """Formats trajectory data for prompting LLM."""

    def __init__(self, training_data_dir: str):
        """Initialize formatter."""
        self.training_data_dir = Path(training_data_dir)
        self.trajectories: List[Dict[str, Any]] = []

    def load_trajectories(self, max_trajectories: int = 5) -> List[Dict[str, Any]]:
        """Load trajectory data from all subdirectories."""
        trajectory_files = sorted(self.training_data_dir.glob("*/trajectory.json"))
        for traj_file in trajectory_files[:max_trajectories]:
            with open(traj_file, "r") as f:
                data = json.load(f)
                self.trajectories.append(data)
        return self.trajectories

    def format_for_llm_prompt(self) -> str:
        """Format all trajectories as training examples for LLM prompt."""
        if not self.trajectories:
            self.load_trajectories()
        examples = []
        for idx, traj_data in enumerate(self.trajectories):
            example = self._format_single_trajectory(idx, traj_data)
            examples.append(example)
        return "\n\n".join(examples)

    def _format_single_trajectory(self, idx: int, traj_data: Dict[str, Any]) -> str:
        """Format a single trajectory."""
        metadata = traj_data.get("metadata", {})
        trajectory = traj_data.get("trajectory", [])

        first_timestep = trajectory[0] if trajectory else {}
        domain_file = first_timestep.get("domain_file", "")
        problem_file = first_timestep.get("problem_file", "")
        domain_content = self._read_pddl_file(domain_file)
        problem_content = self._read_pddl_file(problem_file)

        steps_formatted = []
        seen_noop = False
        for step_data in trajectory:
            action = step_data.get("action_taken")
            # Skip no-op actions after the first one
            if action and action.get("name") == "no-op":
                if seen_noop:
                    continue
                seen_noop = True

            step_str = self._format_timestep(step_data)
            if step_str:
                steps_formatted.append(step_str)

        example = f"""
=== Training Example {idx + 1} ===
Task: {metadata.get('task_name', 'unknown')}
Success: {metadata.get('success', False)}
Total Reward: {metadata.get('total_reward', 0.0)}

--- Domain (PDDL) ---
{domain_content}

--- Initial Problem (PDDL) ---
{problem_content}

--- Execution Trace ---
{chr(10).join(steps_formatted)}
"""
        return example

    def _format_timestep(self, step_data: Dict[str, Any]) -> str:
        """Format a single timestep for LLM.

        Clarify that abstract_belief and reward are BEFORE action,
        observation is AFTER action execution.
        """
        timestep = step_data.get("timestep", 0)
        abstract_belief = step_data.get("abstract_belief", [])
        reward = step_data.get("reward", 0.0)
        action = step_data.get("action_taken")
        observation = step_data.get("observation", {})
        next_abstract_belief = step_data.get("next_abstract_belief", [])

        if not action:
            return ""

        ab_str = ", ".join(abstract_belief) if abstract_belief else "(empty)"

        action_name = action.get("name", "unknown")
        action_args = action.get("args", [])
        action_str = f"{action_name}({', '.join(action_args)})" if action_args else action_name

        obs_parts = []
        if observation.get("regions_in"):
            obs_parts.append(f"regions_in={observation['regions_in']}")
        if observation.get("holding") is not None:
            obs_parts.append(f"holding={observation['holding']}")
        if observation.get("robot_pose") is not None:
            pose = observation["robot_pose"]
            obs_parts.append(f"robot_pose=({pose['x']:.2f}, {pose['y']:.2f}, {pose['theta']:.2f})")
        if observation.get("collision") is not None:
            obs_parts.append(f"collision={observation['collision']}")
        obs_str = ", ".join(obs_parts) if obs_parts else "(empty)"

        next_ab_str = ", ".join(next_abstract_belief) if next_abstract_belief else "(empty)"

        return f"""
Timestep {timestep}:
  BEFORE action:
    - Abstract Belief: {ab_str}
    - Reward: {reward}
  ACTION TAKEN: {action_str}
  AFTER action:
    - Observation: {obs_str}
    - Next Abstract Belief: {next_ab_str}
"""

    def _read_pddl_file(self, file_path: str) -> str:
        """Read and return content of a PDDL file."""
        if not file_path:
            return "(No PDDL file available)"

        try:
            with open(file_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return f"(File not found: {file_path})"
        except Exception as e:
            return f"(Error reading file: {e})"


def format_trajectory_for_llm(
    training_data_dir: str, max_trajectories: int = 5
) -> str:
    """Overall function to format trajectories for LLM."""
    formatter = TrainingDataFormatter(training_data_dir)
    formatter.load_trajectories(max_trajectories=max_trajectories)
    return formatter.format_for_llm_prompt()