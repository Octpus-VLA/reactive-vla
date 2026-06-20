# CLAUDE.md

This file provides project guidance for AI coding agents working in this repository.

## Response Rules

- Use Japanese for project-facing explanations by default unless the user asks for another language.
- When referring to code, include clickable file paths with line numbers when possible.
- Distinguish clearly between implemented behavior, design notes, and future work.
- Keep documentation concise and avoid presenting unvalidated experiments as completed functionality.

## Research Goal

The project studies reactive VLA control for dynamic pick tasks, especially grasping a cube moving on a conveyor belt and placing it into a box. The near-term direction is to combine SmolVLA/LeRobot action chunking with supervisor-triggered and eventually predictive replanning.

## Current Reactivity Terms

- Tier 1: queue-based replan from the existing action queue threshold.
- Tier 2: event-triggered early replan from a camera supervisor.
- Tier 3: predictive/adaptive replan using cube position, velocity, predicted grasp timing, and dynamic effective horizon.

These Tier names are project design terms, not upstream LeRobot API names.

## Repository Notes

- Parent repository changes should stay in docs, wrappers, jobs, and project configuration unless the task explicitly targets LeRobot internals.
- LeRobot internals live in `third_party/lerobot` and must be committed inside that submodule before updating the parent submodule pointer.
- Default behavior should remain non-disruptive: experimental supervisor or detector features should be disabled unless explicitly configured.
