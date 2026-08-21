# Meridian Documentation Map

This document maps Meridian concepts and topics to their corresponding
documentation files. Use this map to find the relevant file to consult when
answering user questions.

## Documentation Map

<!-- TODO: Refine the description summary for each doc below. -->

*   **General Introduction & Basics:**

    *   `docs/basics/meridian-introduction.md` (Overview of Meridian as an
        open-source Bayesian MMM framework. It answers core questions about ROI,
        response curves, and budget allocation. Key features include causal
        inference focus, use of priors for expertise, control variables like
        Google Query Volume, geo-level data advantages, and reach/frequency
        support. Outlines the end-to-end workflow: pre-modeling, modeling, and
        post-modeling.)
    *   `docs/basics/glossary.md` (Defines key terms. Notable definitions
        include: Baseline (counterfactual scenario), Calibration (using
        experiments for priors), Contribution (incremental outcome %),
        Confounding vs Predictor variables, CPIK, Effect Window, Flighting
        pattern (spend allocation), Lagged effect (adstock), mROI, Saturation
        (no specific threshold), and SUTVA assumptions for non-geo hierarchies.)
    *   `docs/faqs.md` (Answers common questions. Key points: Data privacy
        (Google only sees data requested from its platform), GeoX integration,
        LightweightMMM migration data needs, GQV scope (indexed only),
        time-varying effects via data partitioning (warns against
        over-partitioning), lack of synergy support, ROI over time options, and
        strong recommendation against non-geo hierarchies. Emphasizes that
        Meridian is for causal inference, not prediction.)

*   **Data Preparation & Loading:**

    *   `docs/pre-modeling/collect-data.md` (Guides on gathering historical data
        for media, spend, controls, KPI, and population. Recommends weekly
        granularity and a minimum of 2-3 years of data. Advises limiting the
        number of media channels to ensure sufficient variation and volume,
        suggesting grouping smaller channels.)
    *   `docs/pre-modeling/amount-data-needed.md` (Explains data sufficiency
        based on 'data points per effect'. National models need more years of
        data compared to geo models. Discourages campaign-level data due to
        parameter inflation and risk of losing Adstock memory. Recommends
        tighter, regularizing priors when data is insufficient in geo models.)
    *   `docs/advanced-modeling/input-data.md` (Defines mathematical notation
        and dimensions for model inputs (KPI, controls, media, etc.). Explains
        internal linear scaling functions applied by Meridian, including geo
        population scaling and standardization to mean zero and standard
        deviation one.)
    *   `docs/user-guide/supported-data-types-formats.md` (Lists supported data
        types (Geo-level with/without R&F, organic and non-media,
        National-level) and formats (CSV, Xarray, Numpy, Pandas).)
    *   `docs/user-guide/load-geo-data-without-rf.md` (Detailed guide with code
        examples for loading geo-level data without reach and frequency using
        `CsvDataLoader`, `XrDatasetDataLoader`, `NDArrayInputDataBuilder`, or
        `DataFrameInputDataBuilder`.)
    *   `docs/user-guide/load-geo-data-with-rf.md` (Detailed guide with code
        examples for loading geo-level data with reach and frequency, extending
        the standard loading to include reach, frequency, and RF spend.)
    *   `docs/user-guide/load-national-data.md` (Detailed guide with code
        examples for loading national-level data, similar to geo-level loading
        but removing the geographic dimension.)
    *   `docs/user-guide/load-geo-data-with-organic-and-non-media.md` (Detailed
        guide with code examples for loading geo-level data that includes
        organic media (no cost) and non-media treatments (e.g., promotions).)

*   **Exploratory Data Analysis (EDA):**

    *   `docs/pre-modeling/perform-eda.md` (Detailed guide on performing
        Exploratory Data Analysis (EDA) using Meridian's EDA package. Explains
        categories of checks: Spend share, Individual variables (variation,
        outliers), Population scaling, Relationships (VIF, correlation heatmap),
        and Prior specifications.)

*   **Model Configuration (ModelSpec, Priors):**

    *   `docs/advanced-modeling/model-spec.md` (Full mathematical equation of
        the Meridian model. Covers extensions like reach/frequency, time-varying
        intercepts ($\mu_t$), and organic media. Defines Adstock and Hill
        functions, and parameter distributions.)
    *   `docs/advanced-modeling/intro-priors.md` (Introduction to Bayesian
        priors in Meridian. Explains how priors allow injecting business
        knowledge to stabilize results and ground them in reality. Supports
        priors on ROI, mROI, and contribution percentage using distributions
        like Normal, Log-Normal and Half-Normal.)
    *   `docs/advanced-modeling/default-prior-distributions.md` (Lists default
        prior distributions for all model parameters (e.g., `knot_values`,
        `tau_g`, `roi_m`, `contribution_m`). Explains rationales, noting which
        are uninformative and which are regularizing (e.g., `Beta(1,99)` for
        contribution).)
    *   `docs/advanced-modeling/how-to-choose-treatment-prior-types.md` (Guides
        on choosing treatment prior types (ROI, mROI, Contribution, Coefficient)
        for paid, organic, and non-media treatments. ROI is default for paid
        media, Contribution for organic/non-media. Explains induced priors and
        budget regularization using mROI priors.)
    *   `docs/advanced-modeling/roi-priors-and-calibration.md` (Explains
        calibration of treatment priors using experiment results and domain
        knowledge. Discusses translating experiment point estimates and standard
        errors into priors, and relevance considerations (timing, duration).
        Emphasizes priors as regularization.)
    *   `docs/advanced-modeling/set-custom-priors-past-experiments.md` (Code
        examples for setting custom ROI priors using past experiments. Details
        helper functions like `lognormal_dist_from_mean_std` and
        `lognormal_dist_from_range`, and lists nuances to consider when using
        experiment results.)
    *   `docs/advanced-modeling/media-saturation-lagging.md` (Explains how
        Meridian models lagged effects (Adstock) and saturation effects (Hill
        function). Details the mathematical formulas, the `hill_before_adstock`
        option, and notes that default priors assume a concave shape to
        facilitate budget optimization.)
    *   `docs/advanced-modeling/set-adstock-decay-spec-parameter.md` (Explains
        how to configure `adstock_decay_spec` (geometric or binomial). Geometric
        is best for short-lived effects, binomial for long-lived effects.
        Discusses customizing the alpha prior to control decay rate.)
    *   `docs/advanced-modeling/set-max-lag-parameter.md` (Explains tradeoffs
        and practical advice for setting `max_lag` ($L$). Recommendations: 2-10
        periods for geometric decay, 4-20 periods for binomial decay or
        combinations.)
    *   `docs/advanced-modeling/setting-knots.md` (Explains modeling time
        effects using knots (splines). Discusses bias-variance tradeoff.
        Recommends `knots = n_times` for geo models and `1` knot for national
        models. Details the Automatic Knot Selection (AKS) feature.)
    *   `docs/advanced-modeling/control-variables.md` (Explains the role of
        control variables (Confounding and Predictor). Advises on selecting
        controls based on marketing decisions. Covers query volume as a control,
        lagged controls, and population scaling.)
    *   `docs/advanced-modeling/organic-and-non-media-variables.md` (Defines
        Organic media (no cost, has Adstock/Hill effects) and Non-media
        treatments (intervenable like price, no Adstock/Hill effects). Explains
        how to decide between these and controls.)
    *   `docs/advanced-modeling/reach-frequency.md` (Explains modeling channels
        using reach and frequency data. Media effect is calculated by applying
        Hill function to frequency and multiplying by linear reach, followed by
        Adstock.)

*   **Model Fitting & Execution:**

    *   `docs/user-guide/run-model.md` (Instructions for running the model
        (sampling prior and posterior). Details MCMC parameters (`n_chains`,
        `n_adapt`, `n_burnin`, `n_keep`) and mentions the No-U-Turn (NUTS)
        sampler.)
    *   `docs/advanced-modeling/using-jax.md` (Explains how to enable and use
        the JAX backend for performance and memory efficiency. Details API
        differences like using `tfp.substrates.jax.distributions` and requiring
        explicit seeds.)

*   **Post-Modeling Analysis & Visualization:**

    *   `docs/post-modeling/intro.md` (Directory for post-modeling section.
        Lists pages covering health checks, health score, model fit,
        ROI/mROI/Response curves, baseline, visualizations, and optimizations.)
    *   `docs/post-modeling/health-checks.md` (Details model health checks via
        `reviewer.ModelReviewer(mmm).run()`. Covers Convergence (R-hat),
        Negative Baseline, Bayesian PPP, Goodness-of-fit, Prior Posterior Shift,
        and ROI Consistency.)
    *   `docs/post-modeling/health-score.md` (Explains the model health score
        (0-100) combining 6 checks. Convergence is a gate. Explains weighting
        (Negative Baseline 30%, PPP 30%, etc.) and transformations.)
    *   `docs/post-modeling/model-fit.md` (Discusses assessing model fit,
        emphasizing causal inference over prediction. Details how to access
        posterior draws and lists `Analyzer` methods.)
    *   `docs/post-modeling/roi-mroi-response-curves.md` (Explains Incremental
        Outcome, ROI, mROI, and Response Curves. Defines them mathematically
        using potential outcomes. ROI is for past performance, response curves
        for optimization, mROI for saturation.)
    *   `docs/post-modeling/baseline.md` (Explains assessing the baseline.
        Negative baseline indicates error. Details how to assess
        probabilistically and mitigate by adjusting priors or improving
        controls/knots.)
    *   `docs/post-modeling/interpret-visualizations.md` (Guides on interpreting
        visualizations. Covers Model fit charts, Channel contribution charts,
        ROI charts, Response curves, Adstock decay curves, and Hill saturation
        curves.)
    *   `docs/user-guide/generate-model-results-output.md` (Instructions for
        generating HTML report or summary table using `Summarizer` and
        `MediaSummary`.)
    *   `docs/user-guide/plot-media-visualizations.md` (Guide for plotting
        customized media visualizations using `MediaSummary` and `MediaEffects`.
        Covers Area, Bump, Waterfall, Pie, Spend vs Contribution, ROI, ROI vs
        Effectiveness, ROI vs mROI, Response curves, Adstock, and Hill curves.)

*   **Budget Optimization:**

    *   `docs/user-guide/optimization-overview.md` (Directory for budget
        optimization section. Lists scenarios: Fixed budget, Flexible budget
        (minimum ROI or marginal ROI), and frequency optimization.)
    *   `docs/post-modeling/optimization-without-reach-frequency.md`
        (Mathematical explanation of budget optimization for channels without
        reach and frequency data. Assumes fixed flighting patterns. Covers fixed
        and flexible budget optimization.)
    *   `docs/post-modeling/optimization-with-reach-frequency.md` (Mathematical
        explanation of budget optimization for channels with reach and frequency
        data. Covers optimizing target frequency and budget allocation under
        various constraints.)
    *   `docs/post-modeling/interpret-optimizations.md` (Guides on interpreting
        budget optimization results. Covers Fixed budget, Flexible budget with
        target ROI/mROI scenarios. Explains visualizations like Optimized spend
        change, allocation pie chart, and optimal frequency.)
    *   `docs/user-guide/generate-optimization-results-output.md` (Instructions
        for generating optimization HTML report or summaries.)
    *   `docs/user-guide/plot-optimization-visualizations.md` (Guide for
        plotting customized optimization visualizations using
        `OptimizationResults`. Covers Spend delta, Outcome delta, Budget
        allocation, and Response curves.)

*   **Scenario Planning:**

    *   `docs/scenario-planning/meridian-scenario-planner.md` (Overview of the
        Meridian Scenario Planner (Looker Studio tool). Covers features like
        interactive budget optimization and data security recommendations.)
    *   `docs/scenario-planning/faqs.md` (FAQs for Scenario Planner. Covers
        'Community Visualization Disabled' and 'Data Set Configuration Error'.)
    *   `docs/post-modeling/scenario-planning-and-future-budget-optimization.md`
        (Advanced guide on scenario planning using `new_data`. Explains how to
        override assumptions about cost, revenue per KPI, and flighting
        patterns.)

*   **Model Saving/Loading:**

    *   `docs/user-guide/saving-model-object.md` (Explains how to save and load
        the model object. Deprecates Python `pickle` in favor of the Meridian
        `serde` package, supporting binary and text protobuf formats.)
    *   `docs/user-guide/mmm-unified-schema.md` (Explains the MMM Unified Schema
        (Protocol Buffers). Decouples training from consumption. Covers Model
        Core, Model Fit, Marketing Analysis, and Optimization.)

*   **Causal Inference Concepts:**

    *   `docs/causal-inference/intro.md` (Introduction to causal inference in
        Meridian. Lists key assumptions required for valid causal estimates.)
    *   `docs/causal-inference/about-mmm-causal-inference-methodology.md`
        (Discusses MMM as a causal inference methodology using observational
        data. Compares it to experiments and discusses untestable and testable
        assumptions.)

*   **Debugging:**

    *   `docs/post-modeling/model-debugging.md` (Comprehensive guide for
        debugging model issues. Covers MCMC convergence, posterior same as
        prior, ROI differences based on priors, GPU OOM errors, high organic
        media contribution, collinearity, and negative R-squared.)

TREAT THE ABOVE MAPPING AS A GUIDE, NOT AN EXHAUSTIVE LIST.
