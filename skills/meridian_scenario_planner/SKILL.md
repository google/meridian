---
name: meridian-scenario-planner
description: >-
  Generates Scenario Planner data from a fitted Meridian model and exports it for Looker Studio dashboard creation via Colab handoff.
  Use when the user wants to generate scenario planning data, budget grids, or Looker Studio dashboards from a saved model.
  Don't use for visualizing model results (use meridian-result-visualization) or running standard budget optimization (use meridian-budget-optimization).
---

# Meridian Scenario Planner Generation

This skill guides the agent to generate Scenario Planner data, serialize it as a
proto file for Colab handoff (zero GCP setup required), and provide the link to
the Meridian Looker Studio Scenario Planner Colab notebook.

## Prerequisites

*   The agent must have access to a fitted Meridian model (serialized as
    `meridian_model.binpb`).
*   > [!IMPORTANT] `meridian_model.binpb` is a **binary file**. Do NOT try to
    read its content directly using file viewing tools or grep, as this will
    produce invalid UTF-8 errors. Always use `meridian_serde.load_meridian()` in
    a Python script to load it.

## Core Workflow

### Interactivity Checkpoint Rule

Throughout this workflow, you will encounter **CRITICAL INTERACTIVE
CHECKPOINT**s. At each checkpoint, you MUST:

1.  Present the current proposed configurations, paths, or status to the user
    for approval.
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
*   Prompt the user for the output proto path (default:
    `model_build/scenario_planner_data.binpb`).
*   Prompt the user for the desired path for the generated Python script. **If
    unspecified, default to `model_build/run_scenario_planner.py` (or relative
    to the model directory).**
*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the gathered paths to the user
    and obtain confirmation before configuring spec parameters.

### 2. Configure Scenario Planner Spec & Interactive Checkpoint

*   Prompt the user for the following Scenario Planner spec configurations (or
    confirm defaults):
    *   `optimization_name` (String, e.g., "Scenario Planner")
    *   `include_non_paid_channels` (Boolean, default: True)
    *   Time breakdown: `yearly` (default: False), `quarterly` (default: True),
        `monthly` (default: False)
    *   `min_spend_shift_ratio` (Float 0-1, default: 1.0)
    *   `max_spend_shift_ratio` (Float > 0, default: 1.0)
    *   `use_optimal_frequency` (Boolean, default: True)
    *   `max_frequency` (Float > 0, default: 10.0)
*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the proposed Scenario Planner
    spec configurations to the user and obtain approval before continuing to
    load the model.

### 3. Add Model Loading Code

*   Use `meridian_serde.load_meridian()` to load the model.
*   See [script_template.md](references/script_template.md) for code template.
*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the model load path
    configuration and proposed Python code snippet to the user, and obtain
    approval before continuing to data generation.

### 4. Add Scenario Planner Data Generation & Colab Handoff Code

*   Create specs for `ModelFitSpec`, `MarketingAnalysisSpec`, and
    `BudgetOptimizationSpec` using the user-provided configurations.
*   Use `mmm_ui_gen.create_mmm_ui_data_proto()` to create the proto, including
    the requested time breakdown generators.
*   Serialize and save the proto to disk (e.g.
    `model_build/scenario_planner_data.binpb`).
*   Instruct the user to upload the saved file to the Meridian Scenario Planner
    Colab notebook:
    `https://colab.research.google.com/github/google/meridian/blob/main/demo/Meridian_Scenario_Planner_Beta.ipynb`
*   See [script_template.md](references/script_template.md) for code template.
*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the proposed scenario planner
    spec configuration and export code snippet to the user, and obtain approval
    before continuing to execution.

### 5. Pre-execution Checkpoint

*   **CRITICAL INTERACTIVE CHECKPOINT**: Ask the user for final confirmation to
    execute the scenario planner generation script now.

### 6. Execution & Script Setup

*   Write the accumulated Python script to the user-specified path. When writing
    the file using `write_to_file`, explicitly set
    `ArtifactMetadata.RequestFeedback=false` to avoid pausing execution.
*   You can use the full script template in
    [script_template.md](references/script_template.md) as a guide.
*   Execute the script using Python: prefer the active virtual environment if
    available (e.g. `.venv/bin/python3` or
    `/tmp/meridian_eval_cache/bin/python3`, otherwise `python3`).
*   When executing the script, if `run_command` runs as a background task,
    simply end the turn and wait for the completion notification.
*   **CRITICAL**: Do **NOT** delete the generated script, output proto files, or
    deliverables at the end of the task. These are the deliverables requested by
    the user and must be preserved.
