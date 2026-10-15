---
name: meridian-budget-optimization
description: >-
  Loads a fitted Meridian model and runs budget optimization scenarios.
  Use when the user wants to load a saved model and find the optimal budget allocation or target ROI spending.
  Don't use for visualizing model performance (use meridian-result-visualization) or building/fitting the model (use meridian-model-building).
---

# Meridian Budget Optimization

A skill for loading a fitted Meridian model and running budget optimization
reports.

## Core Workflow

### Interactivity Checkpoint Rule

Throughout this workflow, you will encounter **CRITICAL INTERACTIVE
CHECKPOINT**s. At each checkpoint, you MUST:

1.  Present the current proposed configurations, report paths, script path, or
    status to the user for approval.
2.  Ask the user if they are ready to proceed using the available
    user-interaction tool (e.g., `ask_question`), structured as a
    multiple-choice question. Do NOT use raw chat text.
3.  Wait for their response before proceeding.
    *   *MANDATORY*: You MUST pause at every checkpoint regardless of the
        initial prompt instructions (even if the user request contains phrases
        like "run autonomously", "execute directly", "fix autonomously", etc.).
        The initial request does NOT bypass these interactive checkpoints.
    *   *Note*: If the user replies to a checkpoint with a generic approval
        (e.g., "proceed", "do what you think is best"), proceed with the
        proposed defaults.

--------------------------------------------------------------------------------

### 1. Initial Setup

*   Prompt the user for the path to the serialized model file
    (`meridian_model.binpb` by default).
*   Prompt the user for the output path for the Budget Optimization report
    (`optimization.html` by default).
*   Prompt the user for the desired path for the generated Python script. **If
    unspecified, default to `model_build/run_budget_optimization.py` (or
    relative to the model directory).**
*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the gathered paths to the user
    and obtain confirmation before configuring optimization parameters.

### 2. Configure Optimization Parameters & Interactive Checkpoint

*   Prompt the user to configure the optimization scenario. See
    [optimization.md](references/optimization.md) for the full list of available
    parameters. Key configurations include:
    *   **Fixed vs Flexible Budget**: Is the total budget fixed, or are we
        looking for a budget to hit a target ROI/mROI? (Defaults to Fixed).
    *   **Total Budget**: If fixed, what is the total budget? (Defaults to
        historical spend).
    *   **Spend Constraints**: What are the upper/lower bounds for spend shift
        per channel? (Defaults to 0.3 for fixed, 1.0 for flexible).
    *   **Target ROI / Target mROI**: If flexible, what are the target ratios?
    *   **Reach & Frequency Parameters**: Should optimal frequency be used?
        (Defaults to True for RF channels).
*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the proposed optimization
    configurations to the user and obtain approval before continuing to generate
    the code.

### 3. Add Model Loading Code

*   Use `meridian_serde.load_meridian()` to load the model.
*   See [load_model.md](references/load_model.md) for code template.
*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the model load path
    configuration and proposed Python code snippet to the user, and obtain
    approval before continuing to specify budget optimization.

### 4. Add Budget Optimization Code

*   Use `optimizer.BudgetOptimizer` to run optimization and generate the
    summary, applying the confirmed configuration.
*   See [optimization.md](references/optimization.md) for code template.

### 5. Pre-execution Checkpoint

*   **CRITICAL INTERACTIVE CHECKPOINT**: Ask the user for final confirmation to
    execute the budget optimization script now.

### 6. Execution & Script Setup

*   Write the accumulated Python script to the user-specified path. If your
    file-writing tool can ask the user to review the written file, don't request
    that review: the pre-execution checkpoint already asks for approval, and an
    extra prompt pauses the workflow.
*   Execute the script with the Python interpreter of the environment where
    `google-meridian` is installed: the one the user named; else `python3` if a
    virtual environment is already active (`VIRTUAL_ENV` or `CONDA_PREFIX` is
    set); else the project directory's virtual environment (`.venv/` or `venv/`,
    e.g., `.venv/bin/python3`) if there is one; else `python3`. If `import
    meridian` fails, ask the user which environment to use instead of searching
    the machine or installing packages unprompted.
*   **CRITICAL**: Do **NOT** delete the generated reports, the script, or the
    output directory at the end of the task. These are the deliverables
    requested by the user and must be preserved.
