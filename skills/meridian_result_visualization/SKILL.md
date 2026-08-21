---
name: meridian-result-visualization
description: >-
  Loads a fitted Meridian model and generates standard visualization reports:
  Model Results Summary.
  Use when the user wants to load a saved model and view results (model performance, fit, health).
  Don't use for running budget optimization (use meridian-budget-optimization) or building/fitting the model (use meridian-model-building).
---

# Meridian Result Visualization

A skill for loading a fitted Meridian model and generating standard
visualization reports.

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
*   Prompt the user for output paths for:
    *   Model Results Summary report (`results_summary.html` by default).
*   Prompt the user for the desired path for the generated Python script.
*   Prompt the user for any optional configuration for the reports (e.g., date
    ranges for results summary).
*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the gathered paths and
    configurations to the user and obtain confirmation before loading the model.

### 2. Add Model Loading Code

*   Use `meridian_serde.load_meridian()` to load the model.
*   See [load_model.md](references/load_model.md) for code template.
*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the model load path
    configuration and proposed Python code snippet to the user, and obtain
    approval before continuing to specify health checks.

### 3. Add Post-Modeling Health Checks Code (Optional)

*   Use `reviewer.ModelReviewer` to run health checks and save to HTML.
*   *Small / Test Models*: If a model has fewer than 2 MCMC draws (e.g. test or
    mock models where R-hat computation requires >= 2 samples), catch
    `ValueError` or skip R-hat calculation gracefully so health check reports
    generate cleanly.
*   See [health_check.md](references/health_check.md) for code template.
*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the proposed health check
    report path and code snippet to the user, and obtain approval before
    continuing to results summary configuration.

### 4. Add Model Results Summary Code

*   Use `summarizer.Summarizer` to generate the HTML results summary, applying
    any user-specified configuration (e.g., date ranges).
    *   *Date Range Auto-Clipping*: If applying a user-specified date range
        falls outside the model's time coordinates, clip or adjust the date
        range to match the model's actual coordinates.
    *   *Tip for errors*: If you encounter errors regarding `sample_prior`,
        bypass or fix the requirement (for example, by setting
        `sample_prior=False` or passing the required parameters) to ensure
        successful generation.
*   See [results_summary.md](references/results_summary.md) for code template.
*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the proposed date ranges,
    output path configuration, and code snippet for the results summary to the
    user, and obtain approval before proceeding to script generation and
    execution.

### 5. Pre-execution Checkpoint

*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the final report path and
    script path to the user, and ask for final confirmation to execute the
    results generation script now.

### 6. Execution & Script Setup

*   Write the accumulated Python script to the user-specified path. When writing
    the file using `write_to_file`, explicitly set
    `ArtifactMetadata.RequestFeedback=false` to avoid pausing execution.
*   Execute the script using Python: prefer the active virtual environment if
    available (e.g. `.venv/bin/python3` or
    `/tmp/meridian_eval_cache/bin/python3`, otherwise `python3`).
*   **CRITICAL**: Do **NOT** delete the generated reports, the script, or the
    output directory at the end of the task. These are the deliverables
    requested by the user and must be preserved.
