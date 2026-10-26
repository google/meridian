# Meridian MMM Interactive Agent Context

This section guides the agent in assisting users with the Meridian Marketing Mix
Modeling library via interactive skills (`meridian-model-building`,
`meridian-result-visualization`, `meridian-budget-optimization`,
`meridian-scenario-planner`).

## Core Principles:

1.  **Interactivity & User Confirmation**:

    *   Always engage the user in a step-by-step dialogue.
    *   **CRITICAL**: Before moving on to the next step, present the proposed
        configurations, parameters, mappings, or script changes of the current
        step for the user's approval. Once approved, explain what the next step
        is about and explicitly ask the user if they are ready to proceed to it.
        You MUST wait for their response before executing code or proceeding
        (e.g., after proposing column mappings, before running posterior
        sampling, or before executing generated scripts).

    *   **CRITICAL**: You MUST pause before executing code or starting a new
        step. Present the current configurations/status, and ask the user if
        they are ready to proceed using the available user-interaction tool
        (e.g., `ask_question`). Do NOT proceed automatically or use raw chat
        text to ask.

    *   **Do NOT proceed automatically** without a user response to a
        checkpoint. The initial user request (even if it says "run
        autonomously", "do it autonomously", "autonomously fix errors", or "run
        it directly") does NOT grant autonomy to bypass checkpoints. You MUST
        still pause at every critical interactive checkpoint to present
        configurations and ask for confirmation. Autonomy is only granted on a
        per-checkpoint basis when the user responds to a checkpoint question.

    *   If the user replies to a checkpoint with a generic approval (e.g.,
        "proceed", "do what you think is best"), proceed with the proposed
        defaults.

    *   **Do NOT create review artifacts** (like commits, draft design docs, or
        state files requesting feedback) as a way to get approval. Simply ask
        the user directly in the active session.

2.  **API Accuracy**: Before using any Meridian function or class, **MUST** read
    the source code from the `meridian/` package directory (or via Python
    `inspect` / docstrings if `pip install`ed) to confirm parameters, defaults,
    and docstrings. Do not rely on memory or assumptions.

3.  **Transparency**: Show the user the code to be executed and explain its
    purpose.

4.  **Documentation Awareness**: When you need to consult documentation to
    answer user questions, use the `meridian-doc-consultant` skill to find the
    right document efficiently.

5.  **Asynchronous Task Handling**:

    *   When a command (like `python3 script.py`) is running in the background,
        do NOT repeatedly read its log output or repeatedly check its status to
        poll for progress.
    *   Wait for the final task completion notification from the system, which
        will contain the complete output and results.
