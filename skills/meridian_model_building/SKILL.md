---
name: meridian-model-building
description: >-
  Guides users through building a Meridian Marketing Mix Modeling (MMM) model. Use when a user wants to load data, map columns, configure ModelSpec, run Exploratory Data Analysis (EDA), fit a model, and save the model. Don't use for visualizing results or creating a scenario planner.
---

# Meridian Model Building Skill

This skill guides the user through the process of creating a Meridian model,
accumulating the code into a Python script.

## Core Workflow

### Interactivity Checkpoint Rule

Throughout this workflow, you will encounter **CRITICAL INTERACTIVE
CHECKPOINT**s. At each checkpoint, you MUST:

1.  Present the current proposed configurations, parameters, mappings, script
    path, or status to the user for approval.
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

*   Prompt the user for the input CSV file path, the desired path for the
    generated Python script, the EDA HTML report output path, and the saved
    model path (`meridian_model.binpb` by default). **If the user does not
    specify output paths, default to `model_build/` in the active project
    directory (or relative to the input data directory) for the script and all
    outputs (`meridian_model.binpb`, `eda.html`).**
*   **CRITICAL INTERACTIVE CHECKPOINT**: Present the gathered paths to the user
    and obtain confirmation before proceeding to data loading.

### 2. Add Data Loading & Column Mapping Code

*   **Target Module:** `meridian.data.data_frame_input_data_builder`
*   **Action:**
    *   **Check CSV Format**: Before loading data, verify if the CSV data is in
        the right format. Consult the `meridian-doc-consultant` skill or check
        the documentation map in
        `skills/meridian_doc_consultant/references/documentation_map.md` under
        "Data Preparation & Loading" to find specific guides (like
        `load-geo-data-without-rf.md`,
        `load-geo-data-with-organic-and-non-media.md` based on the **columns
        observed in the data**) to understand the expected columns and data
        types. Consult `references/csv_format_reference.md` for details on
        expected row/column structure and data quality guardrails. If the format
        is incorrect or missing required columns, **attempt to autonomously
        convert the dataset to the expected format for the user** (e.g.,
        renaming columns, restructuring) unless you are uncertain and need user
        input.
    *   Read the header row of the provided CSV using Python to get the column
        names.
    *   Propose heuristic mappings based on column keywords (e.g., 'sales' ->
        `kpi_col`, 'spend' -> `media_spend_cols`) and infer the `kpi_type`
        ('revenue' or 'non_revenue') based on the columns (e.g., 'revenue' or
        'sales' implying 'revenue', and 'conversions' or 'leads' implying
        'non_revenue').
    *   **Robust Mapping**: If the user prompt specifies mapping a column name
        that does not exist in the CSV, do not assume it is a literal name if it
        looks like a description (e.g., 'media_impressions' vs
        'ChannelX_impression'). Use heuristics to find matching columns and
        proceed.
    *   Present the proposed mapping to the user.
    *   **CRITICAL INTERACTIVE CHECKPOINT**: Present the proposed column
        mappings to the user and obtain approval before continuing to model
        configuration.
    *   Accumulate the data loading code using
        `meridian.data.data_frame_input_data_builder.DataFrameInputDataBuilder`
        and its `with_*` methods (e.g. `with_kpi`, `with_media`). See
        [data_builder_template.md](references/data_builder_template.md).

### 3. Add Model Configuration Code

*   **Target Modules:** `meridian.model.spec`, `meridian.model.model`
*   **Action:**
    *   Read the `ModelSpec` and `PriorDistribution` definitions in
        `meridian.model.spec`.
    *   Guide the user through configuration, prompting for relevant values
        while explaining their purpose based on the source code docstrings.
    *   **CRITICAL INTERACTIVE CHECKPOINT**: Present the proposed model
        specification parameters to the user and obtain approval before
        continuing.
    *   Accumulate the code to initialize `meridian.model.spec.ModelSpec` and
        `meridian.model.model.Meridian`. See
        [model_spec_template.md](references/model_spec_template.md).
    *   Accumulate code: `mmm.sample_prior()`

### 4. Add Exploratory Data Analysis (EDA) Code

*   **Target Module:** `meridian.model.eda.meridian_eda`
*   **Action:**
    *   Read `meridian_eda.py` or module docstrings to confirm the
        `generate_and_save_report` method.
    *   Accumulate code to initialize `meridian_eda.MeridianEDA` and call
        `generate_and_save_report(filepath)` using the user's specified path.
    *   **CRITICAL INTERACTIVE CHECKPOINT**: Present the EDA output path
        configuration and obtain approval before proceeding to the model fitting
        step.

### 5. Add Model Fitting Code

*   **Target Module:** `meridian.model.model`
*   **Action:**
    *   Read the `sample_posterior` method in `meridian.model.model` to
        understand its parameters.
    *   Prompt the user for MCMC parameters: `n_chains`, `n_adapt`, `n_burnin`,
        `n_keep`.
    *   **CRITICAL INTERACTIVE CHECKPOINT**: Present the MCMC parameters to the
        user and obtain approval before proceeding to compile the model fitting
        code.
    *   Accumulate code: `mmm.sample_posterior(...)`

### 6. Add Model Saving Code

*   **Target Module:** `meridian.schema.serde.meridian_serde`
*   **Action:**
    *   Generate code to save the model using `meridian_serde.save_meridian()`
        to the user-specified path (or the default). See
        [script_template.md](references/script_template.md).
    *   **Default Filename**: The default filename for the saved model is
        `meridian_model.binpb` (in the `model_build/` directory). Use this
        filename if the user does not specify a model filename, even if the
        script file is named differently.
    *   **Skip Sampling Handling**: If the user requests to skip fitting or
        posterior sampling, still include the model saving step
        (`meridian_serde.save_meridian(mmm, save_path)`) using the initialized
        `Meridian` model object so the output model file is always created.
    *   **WARNING:** Do NOT use the deprecated `meridian.model.model.save_mmm`
        function. Use `meridian_serde.save_meridian` exclusively.
    *   **CRITICAL INTERACTIVE CHECKPOINT**: Present the model save path and
        filename to the user and obtain approval before proceeding to script
        execution.

### 7. Execution & Script Setup

*   **Action:**
    *   Write the accumulated Python script to the user-specified path. When
        writing the file using `write_to_file`, explicitly set
        `ArtifactMetadata.RequestFeedback=false` to avoid pausing execution.
    *   **CRITICAL INTERACTIVE CHECKPOINT**: Ask the user for final confirmation
        to execute the model building script now.
    *   **Artifact Preservation**: When completing a task that requires
        generating outputs (like scripts, models, or reports), do **NOT** delete
        these generated artifacts at the end of your turn. They are the
        deliverables requested by the user. Only clean up truly temporary
        scratch files if necessary.
        *   **Path Handling for Outputs**: In generated scripts, construct
            output file paths using `os.environ.get("BUILD_WORKSPACE_DIRECTORY",
            ".")` so files land in the source workspace during script execution and
            in the current directory during standalone OSS Python execution.
    *   **Execute the Script**:
        *   Always run the script from the workspace root directory (keep `Cwd`
            as the workspace root, do not set `Cwd` to a subdirectory).
        *   Use Python: prefer the active virtual environment if available (e.g.
            `.venv/bin/python3` or `/tmp/meridian_eval_cache/bin/python3`,
            otherwise `python3`).
        *   Example command: `/tmp/meridian_eval_cache/bin/python3
            model_build/my_model.py`
    *   **Handling Long Runs**: If the command is sent to the background due to
        execution time, wait for the background task to complete and check the
        final output to catch runtime errors.
    *   **Differentiated Error Handling**:
        *   If it's a **Syntax Error** or **Import Error**, read the relevant
            source code to understand the correct usage or interface.
        *   If it's a **ValueError** or parameter constraint violation (e.g.,
            `knots` too large), check the **docstring** of the class/function or
            consult the `meridian-doc-consultant` skill to find valid values in
            the documentation.
        *   *Autonomy*: If a fix requires changing configuration, prompt the
            user for confirmation. If the user response grants autonomy, proceed
            to fix it.

### 8. Conclusion

*   **Action:**
    *   Confirm execution success and artifact generation.
