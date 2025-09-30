from __future__ import annotations

import copy
import logging
import os
import random
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from tampura.solvers.symk import symk_search, symk_translate
from tampura.spec import (
    ProblemSpec,
    beta_quantile,
    compute_cost_modifiers,
    inject_action_costs,
)
from tampura.structs import (
    AbstractBelief,
    AbstractBeliefSet,
    AbstractEffect,
    AbstractRewardModel,
    AbstractRollout,
    AbstractSolution,
    AbstractTransitionModel,
    Action,
    AliasStore,
    Atom,
    Belief,
    Stream,
    eval_expr,
    substitute,
    symbolic_eff,
    symbolic_update,
)
from tampura.symbolic import (
    ACTION_EXT,
    OPT_OBJ,
    VARIABLE_PREFIX,
    VEFFECT_SEPARATOR,
    And,
    Not,
    OneOf,
    negate,
    replace_arg,
)

MAX_COST = 1000


def visualize_exploration_mdp(
    F: AbstractTransitionModel,
    R: AbstractRewardModel,
    cost_modifiers: Dict,
    save_file: str,
    a_b0: AbstractBelief,
    learning_strategy: str,
    last_action: Action = None,
    newly_explored_edges: set = None,
):
    """
    Visualize the complete abstract belief graph that symK planner sees:
    - All abstract belief states as nodes
    - All possible actions as edges with Bayes optimistic costs
    - Exploration statistics (attempts, successes, probabilities)
    - Complete state space visible to the planner
    """
    import pygraphviz as pgv

    # Create directed graph
    graph = pgv.AGraph(directed=True, strict=False, rankdir="TB", size="20,16!")
    last_action_str = f"{last_action.name}({', '.join(map(str, last_action.args))})" if last_action else "None"
    graph.graph_attr.update(
        dpi=600,
        fontsize=10,
        labelloc="t",
        label=(
            f"Abstract Belief Graph (symK Planning View)\\n"
            f"Last Action: {last_action_str}"
        )
    )

    # Collect all abstract beliefs from the transition model and rewards
    all_beliefs = set([a_b0])
    for action, belief_dict in F.effects.items():
        for belief, belief_set in belief_dict.items():
            all_beliefs.add(belief)
            for next_belief in belief_set.all_ab():  # Get all abstract beliefs
                all_beliefs.add(next_belief)

    # Also add any beliefs that have rewards but might not be in transitions yet
    for belief in R.reward.keys():
        all_beliefs.add(belief)

    def belief_str(ab):
        """Create a readable string representation of abstract belief"""
        # Show first few predicates, truncated for readability
        items_str = ", ".join([str(item)[:25] for item in list(ab.items)[:4]])
        return f"S{str(hash(ab))[:5]}\\n{{{items_str}}}"

    # Add belief state nodes
    for belief in all_beliefs:
        belief_id = f"belief_{hash(belief)}"
        label = belief_str(belief)

        # Add reward information if non-zero
        reward_val = R.reward.get(belief, 0)
        if reward_val != 0:
            label += f"\\nR={reward_val:.3f}"

        # Color nodes based on reward and initial state
        if belief == a_b0:
            node_color = "#87CEEB"
            label += "\\n(Initial)"
        elif reward_val > 0:
            node_color = "#98FB98"
        elif reward_val < 0:
            node_color = "#FFB6C1"
        else:
            node_color = "#E0E0E0"

        graph.add_node(belief_id, label=label, shape="box",
                      style="filled", fillcolor=node_color, fontsize=7)

    # Add action edges with comprehensive exploration information
    for action, belief_dict in F.effects.items():
        for source_belief, belief_set in belief_dict.items():
            source_id = f"belief_{hash(source_belief)}"
            total_observed = sum(belief_set.ab_counts.values())

            # Create edges from source to each possible target belief
            for target_belief in belief_set.all_ab():
                target_id = f"belief_{hash(target_belief)}"

                # Get exploration statistics
                observed_count = belief_set.ab_counts.get(target_belief, 0)
                sampling_count = belief_set.sampling_attempts.get(target_belief, 0)
                success_count = belief_set.outcome_successes.get(target_belief, 0)
                attempt_count = belief_set.outcome_attempts.get(target_belief, 0)

                # Calculate probability from observations
                if total_observed > 0:
                    prob = observed_count / total_observed
                else:
                    prob = 0

                # Calculate success rate
                if attempt_count > 0:
                    success_rate = success_count / attempt_count
                else:
                    success_rate = 0

                # Create comprehensive edge label with action name and args
                action_str = f"{action.name}({', '.join(map(str, action.args))})"
                edge_label = f"{action_str}"

                # Find the Bayes optimistic cost for this action from this belief
                action_cost = None
                for cost_modifier, cost in cost_modifiers.items():
                    if (cost_modifier.action == action and
                        set(cost_modifier.pre_facts) == set(source_belief.items)):
                        action_cost = cost
                        break

                if action_cost is not None:
                    edge_label += f"\\nCost: {action_cost}"

                # Add exploration statistics - now showing sampling attempts and success rates
                edge_label += f"\\np={prob:.3f}, s={sampling_count}"
                if attempt_count > 0:
                    edge_label += f"\\nsucc={success_count}/{attempt_count} ({success_rate:.2f})"

                # Check if this edge was newly explored
                edge_key = (action, source_belief, target_belief)
                is_newly_explored = newly_explored_edges and edge_key in newly_explored_edges

                # Color and style edges based on sampling frequency and confidence
                if sampling_count == 0:
                    # Never sampled transitions
                    edge_color = "lightgray"
                    edge_style = "dotted"
                    penwidth = 1
                elif sampling_count <= 2:
                    edge_color = "gray"
                    edge_style = "dashed"
                    penwidth = 1
                elif sampling_count <= 5:
                    edge_color = "orange"
                    edge_style = "solid"
                    penwidth = 2
                else:
                    edge_color = "darkgreen"
                    edge_style = "bold"
                    penwidth = 3

                # Highlight newly explored edges with bright colors
                if is_newly_explored:
                    edge_color = "purple"
                    edge_style = "bold"
                    penwidth = max(penwidth, 4)

                # Highlight high-cost actions with red edges
                if action_cost is not None and action_cost > 50:
                    edge_color = "red"

                # Scale line thickness based on sampling frequency
                if sampling_count > 10:
                    penwidth = min(penwidth + sampling_count // 10, 8)

                # Only show edges that have been sampled or have observed outcomes
                if sampling_count > 0 or observed_count > 0:
                    graph.add_edge(source_id, target_id,
                                  label=edge_label,
                                  color=edge_color,
                                  style=edge_style,
                                  fontsize=6,
                                  penwidth=penwidth)

    graph.draw(save_file, prog="dot")


def normalize(F: AbstractTransitionModel) -> AbstractTransitionModel:
    """Normalizes an abstract transition model such that all transition
    probabilities sum to 1.

    If the counts on a transition are zero, the probability mass is
    split equally between all possible outcomes
    """
    norm_F = copy.deepcopy(F)
    for _, transitions in norm_F.effects.items():
        for _, abs_belief_set in transitions.items():
            total_counts = sum(abs_belief_set.ab_counts.values())

            if total_counts == 0:
                assert False
            else:
                for ab, count in abs_belief_set.ab_counts.items():
                    abs_belief_set.ab_counts[ab] = count / total_counts

    return norm_F


def generate_rollouts(
    a_b0: AbstractBelief,
    F: AbstractTransitionModel,
    sol: AbstractSolution,
    num_rollouts=1,
    max_steps=10,
) -> List[AbstractRollout]:
    rollouts = []

    for _ in range(num_rollouts):
        current_belief = a_b0
        rollout_transitions = []
        for _ in range(max_steps):
            if current_belief not in sol.policy:
                break  # We have reached an unhandled belief, so we stop.

            action = sol.policy[current_belief]
            if current_belief not in F.effects[action]:
                break
            next_belief_set = F.effects[action][current_belief]
            next_belief = next_belief_set.sample()

            rollout_transitions.append((action, current_belief, next_belief))
            current_belief = next_belief

        rollouts.append(AbstractRollout(rollout_transitions))

    return rollouts


def plan_to_rollout(
    spec: ProblemSpec, plan: List[Action], ab_0: AbstractBelief, store: AliasStore
) -> AbstractRollout:
    transitions = []
    ab = ab_0

    for unprocessed_action in plan:
        action_name, veffect_str = unprocessed_action.name.split(ACTION_EXT)
        action = Action(action_name, unprocessed_action.args, detailed_name=unprocessed_action.name)
        s = spec.get_action_schema(action.name)
        effect_items = copy.deepcopy(s.effects)
        if len(veffect_str) > 0:
            actives = [int(v) for v in veffect_str.split(VEFFECT_SEPARATOR)]
            for active, ve in zip(actives, s.verify_effects):
                if isinstance(ve, Atom) or isinstance(ve, Not):
                    if active:
                        effect_items.append(ve)
                    else:
                        effect_items.append(negate(ve))
                elif isinstance(ve, OneOf):
                    for oo_idx, oo_elem in enumerate(ve.components):
                        if oo_idx == active:
                            effect_items.append(oo_elem)
                        else:
                            effect_items.append(negate(oo_elem))
                else:
                    raise NotImplementedError

        action_effect = AbstractEffect(effect_items)
        ab_p = symbolic_update(ab, action, s, action_effect, store)
        transitions.append((action, ab, ab_p))
        ab = ab_p

    return AbstractRollout(transitions)


def get_stream_plan(
    a: Action, ab: AbstractBelief, continuous_arg: str, spec: ProblemSpec, store: AliasStore
):
    """Given a abstract belief and action along with a continuous argument of
    the action that we want to create a new continuous sample of, return the
    shortest sequence of stream executions that generates a new continuous_arg
    sample while satisfying the preconditions of the input action."""

    # Abandon all hope ye who enter
    action_schema = spec.get_action_schema(a.name)
    action_arg_map = {k: v for k, v in zip(action_schema.inputs, a.args)}

    new_arg = continuous_arg.replace(VARIABLE_PREFIX, OPT_OBJ)
    new_preconditions = []
    for p in action_schema.preconditions:
        new_preconditions.append(replace_arg(p, continuous_arg, new_arg))

    full_type_dict = defaultdict(list)
    for arg_type, arg in zip(action_schema.input_types, a.args):
        full_type_dict[arg_type].append(arg)

    # An open stream plan is a set of known facts and a sequence of stream calls
    open_stream_plans = [(full_type_dict, store.certified + ab.items, [], new_preconditions)]
    while len(open_stream_plans) > 0:
        type_dict, known, stream_plan, subbed_preconditions = open_stream_plans.pop(0)
        for stream_schema in spec.stream_schemas:
            for stream_arg_tuple in product(*[type_dict[t] for t in stream_schema.input_types]):
                # Substitute the args into the stream schema certified effects
                continuous_s_arg = stream_schema.output
                new_s_arg = continuous_s_arg.replace(VARIABLE_PREFIX, OPT_OBJ)
                new_subbed_preconditions = []
                for p in subbed_preconditions:
                    new_subbed_preconditions.append(replace_arg(p, continuous_s_arg, new_s_arg))

                subs = {k: v for k, v in zip(stream_schema.inputs, stream_arg_tuple)}

                if eval_expr(And(stream_schema.preconditions), subs, known, store.type_dict):
                    subs[continuous_s_arg] = new_s_arg
                    subbed_certified = [substitute(cert, subs) for cert in stream_schema.certified]
                    new_facts = symbolic_eff(
                        AbstractBelief(known), AbstractEffect(subbed_certified), store
                    )

                    if AbstractBelief(known) != new_facts:
                        new_type_dict = copy.deepcopy(type_dict)
                        new_type_dict[stream_schema.output_type] = [new_s_arg]
                        stream = Stream(stream_schema.name, stream_arg_tuple, new_s_arg)
                        new_stream_plan = (
                            new_type_dict,
                            new_facts.items,
                            stream_plan + [stream],
                            new_subbed_preconditions,
                        )
                        if eval_expr(
                            substitute(And(new_subbed_preconditions), action_arg_map),
                            {},
                            new_facts.items,
                            store.type_dict,
                        ):
                            return new_stream_plan[2]
                        else:
                            open_stream_plans.append(new_stream_plan)

    return None


def execute_stream_plan(
    stream_plan: List[Stream], spec: ProblemSpec, store: AliasStore
) -> AliasStore:
    """Execute a stream plan by sampling from the stream samplers."""
    new_arg_map = {}
    for stream in stream_plan:
        ss = spec.get_stream_schema(stream.name)
        input_map = {k: v for k, v in zip(ss.inputs, stream.inputs)}

        # Replace the input placeholders with the recently generated objects
        for k, v in input_map.items():
            if v in new_arg_map:
                input_map[k] = new_arg_map[v]

        sample_inputs = [input_map[k] for k in ss.inputs]
        assert None not in store.get_all(sample_inputs)
        output = ss.sample_fn(sample_inputs, store)
        output_sym = store.add_typed(output, ss.output_type)
        new_arg_map[stream.output] = output_sym
        subbed_cert = [
            substitute(cert, input_map | {ss.output: output_sym}) for cert in ss.certified
        ]
        store.certified += subbed_cert
    return store


def progressive_widening(
    a: Action, ab: AbstractBelief, spec: ProblemSpec, alpha: float, k: float, store: AliasStore
) -> AliasStore:
    action_schema = spec.get_action_schema(a.name)
    cont_stream_plans = {}
    # Split up the action input arguments into discrete and continuous depending on if they are generated by a stream
    discrete_args = [
        arg
        for (arg, t) in zip(action_schema.inputs, action_schema.input_types)
        if t not in spec.continuous_types
    ]
    continuous_args = [arg for arg in action_schema.inputs if arg not in discrete_args]

    discrete_component = tuple(
        [a.args[action_schema.inputs.index(discrete_arg)] for discrete_arg in discrete_args]
    )
    d_action = Action(a.name, discrete_component)

    store.sample_counts[d_action] = store.get_sample_count(d_action) + 1

    if len(continuous_args) > 0:
        if k * (store.get_sample_count(d_action) ** alpha) >= store.get_branching_factor(d_action):
            logging.debug(
                "Progressive widening on action {}, {}>{}".format(
                    d_action,
                    k * (store.get_sample_count(d_action) ** alpha),
                    store.branching_factor[d_action],
                )
            )

            store.branching_factor[d_action] += 1
            for continuous_arg in continuous_args:
                stream_plan = get_stream_plan(a, ab, continuous_arg, spec, store)
                if stream_plan is not None:
                    cont_stream_plans[continuous_arg] = stream_plan
            if len(cont_stream_plans) > 0:
                # Choose a random continuous argument to widen and query the streams to generate this new continuous value
                selected_stream_plan = random.choice(list(cont_stream_plans.values()))
                store = execute_stream_plan(selected_stream_plan, spec, store)

    return store


def policy_search(
    b0: Belief,
    spec: ProblemSpec,
    F: AbstractTransitionModel,
    R: AbstractRewardModel,
    belief_map: Dict[AbstractBelief, List[Belief]],
    store: AliasStore,
    config: Dict[str, Any],
    save_dir: str,
    last_action: Action = None,
    llm_generator: Any = None,
    last_observation: Optional[Dict[str, Any]] = None,
) -> Tuple[AbstractTransitionModel, AbstractRewardModel, Dict[AbstractBelief, List[Belief]], bool, Dict[str, str]]:
    """Sample trajectories according to the provided abstract policy."""

    a_b0 = b0.abstract(store)

    default_cost = int(
        -np.log(beta_quantile(0, 0, F.total_count()) * config["gamma"]) * 1 / (1 - config["gamma"])
    )

    cost_modifiers = compute_cost_modifiers(
        spec, F, config["learning_strategy"], config["gamma"], store
    )

    os.makedirs(save_dir, exist_ok=True)

    (domain_file, problem_file) = spec.save_pddl(
        a_b0, default_cost=default_cost, folder=save_dir, store=store
    )
    pddl_files = {
        "domain_file": domain_file,
        "problem_file": problem_file,
    }

    # Choose planning backend: LLM or symK
    use_llm = llm_generator is not None and config.get("use_llm_planner", False)

    if use_llm:
        logging.info("[PolicySearch] Using LLM for plan generation")
        obs_dict = last_observation if last_observation is not None else {}
        plans_as_actions = llm_generator.generate_plans(
            current_abstract_belief=a_b0,
            last_observation=obs_dict,
            reward=R.reward.get(a_b0, 0.0),
        )
        if len(plans_as_actions) == 0:
            logging.warning("[PolicySearch] LLM generated 0 plans, fallback to symK")
            use_llm = False
        else:
            rollouts = [plan_to_rollout(spec, plan, a_b0, store) for plan in plans_as_actions]
            logging.info(f"[PolicySearch] LLM generated {len(rollouts)} rollouts")

    if not use_llm:
        logging.info("[PolicySearch] Using symK for plan generation")
        output_sas_file = symk_translate(domain_file, problem_file)

        output_sas_file = inject_action_costs(
            output_sas_file,
            a_b0,
            action_costs=cost_modifiers,
            store=store,
        )
        plans = symk_search(output_sas_file, config)

        if len(plans) == 0:
            return F, R, belief_map, False, pddl_files
        else:
            rollouts = [plan_to_rollout(spec, plan, a_b0, store) for plan in plans]

    newly_explored_edges = set()
    for step in range(config["batch_size"]):
        rollout = random.choice(rollouts)
        if len(rollout.transitions) == 0:
            continue

        # Sort transitions by num collected samples
        filtered_transitions = [t for t in rollout.transitions if t[1] in belief_map]
        assert len(filtered_transitions) > 0

        max_info_transition = sorted(
            filtered_transitions,
            key=lambda t: sum(F.get_transition(t[0], t[1]).ab_counts.values()),
        )[0]
        mi_a, mi_ab, mi_ab_p = max_info_transition
        action_schema = spec.get_action_schema(mi_a.name)
        sel_b = None
        if any([o is None for o in store.get_all(mi_a.args)]):
            logging.debug(
                "Step {} Skipping Action {} w/ Objects {} and Outcome {}".format(
                    step, mi_a, store.get_all(mi_a.args), mi_ab_p
                )
            )
            ab_p_belief_set = AbstractBeliefSet(ab_counts={mi_ab: 1e6}, belief_map={mi_ab: []})
        else:
            logging.debug("Step {} Sampling Action {} w/ Outcome {}".format(step, mi_a, mi_ab_p))
            logging.debug(
                "From abstract belief: {} with {} beliefs".format(mi_ab, len(belief_map[mi_ab]))
            )

            sel_b = random.choice(belief_map[mi_ab])
            ab_p_belief_set = action_schema.effects_fn(mi_a, sel_b, store)

        if not config["flat_sample"]:
            # Progressive widening
            store = progressive_widening(mi_a, mi_ab, spec, config["pwa"], config["pwk"], store)

        for ab_p, belief_set in ab_p_belief_set.belief_map.items():
            if ab_p == mi_ab_p:
                ab_p_belief_set.outcome_successes[mi_ab_p] = ab_p_belief_set.ab_counts[ab_p]

            belief_map[ab_p] += belief_set
            R.reward[ab_p] = spec.get_reward(ab_p, store)

        ab_p_belief_set.outcome_attempts[mi_ab_p] = ab_p_belief_set.total_count()
        ab_p_belief_set.sampling_attempts[mi_ab_p] = 1

        # Track newly explored edges for visualization
        newly_explored_edges.add((mi_a, mi_ab, mi_ab_p))

        F.update(mi_a, mi_ab, ab_p_belief_set)

    # # Visualize the exploration MDP with the last action taken
    # mdp_viz_file = os.path.join(save_dir, "exploration_mdp.png")
    # visualize_exploration_mdp(
    #     F, R, cost_modifiers, mdp_viz_file, a_b0, config["learning_strategy"], last_action, newly_explored_edges
    # )

    return F, R, belief_map, True, pddl_files
