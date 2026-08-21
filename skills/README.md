# Meridian Agent Skills

A collection of interactive skills for AI coding assistants to guide users
through building, visualizing, optimizing, and scenario planning with the
[Meridian](https://github.com/google/meridian) Marketing Mix Modeling (MMM)
framework.

These skills are designed for interactive AI coding environments, providing
step-by-step guidance, smart defaults, and reproducible Python script generation
across the entire MMM lifecycle.

## Installation & Integration

### Option 1: Standard Skills CLI (`npx skills`) — Recommended

Install Meridian skills directly into your workspace:

```bash
# Install all Meridian skills in your project
npx skills add google/meridian

# Or install a specific skill (e.g., model building)
npx skills add google/meridian --skill meridian-model-building

# Or install globally across all projects
npx skills add google/meridian --global --yes
```

### Option 2: Antigravity IDE & CLI (`agy`)

-   **In-Repo Direct Discovery**: When developing inside a clone of the
    `google/meridian` repository, Antigravity IDE automatically discovers and
    indexes all skills via `.agents/skills.json`.
-   **Plugin Installation**:

    ```bash
    agy plugin install https://github.com/google/meridian/tree/main/skills
    ```

### Option 3: Compatible AI Assistants

Any AI assistant supporting workspace skill manifests or directory-based
instructions can load these skills by referencing the `skills/` directory.

--------------------------------------------------------------------------------

## Available Skills

Skill                               | Directory                                                         | Description
:---------------------------------- | :---------------------------------------------------------------- | :----------
**`meridian-doc-consultant`**       | [`meridian_doc_consultant`](meridian_doc_consultant/)             | Answers conceptual, parameter, and best practice questions about Meridian MMM using relevant documentation.
**`meridian-model-building`**       | [`meridian_model_building`](meridian_model_building/)             | Guides users through loading data, column mapping, ModelSpec configuration, EDA, model fitting, and saving.
**`meridian-result-visualization`** | [`meridian_result_visualization`](meridian_result_visualization/) | Loads fitted Meridian models and generates Model Results Summary and diagnostics visualizations.
**`meridian-budget-optimization`**  | [`meridian_budget_optimization`](meridian_budget_optimization/)   | Loads fitted models to compute optimal budget allocation and target ROI spending scenarios.
**`meridian-scenario-planner`**     | [`meridian_scenario_planner`](meridian_scenario_planner/)         | Generates Scenario Planner data, uploads to Google Sheets, and provides dashboard URLs.

--------------------------------------------------------------------------------

## Key Features

-   **Progressive Disclosure**: The AI agent only indexes the skill's name and
    summary initially. Full instructions and templates are loaded on-demand when
    the skill is activated.
-   **Interactive Checkpoints**: The agent pauses before critical decisions
    (column mapping, prior distributions, script execution) to present proposals
    and confirm user approval.
-   **Reproducible Python Scripts**: Generated workflows produce clean,
    standalone Python scripts runnable via `python3` against your active Python
    environment.

--------------------------------------------------------------------------------

## Contributing & Support

-   **Bug Reports & Feedback**: Please file issues or feature requests in the
    [GitHub Issue Tracker](https://github.com/google/meridian/issues).
-   **License**: Apache 2.0 (see
    [LICENSE](https://github.com/google/meridian/blob/main/LICENSE) for
    details).
