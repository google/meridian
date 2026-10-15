# Meridian Documentation Map

This document maps Meridian concepts and topics to their corresponding pages on
`https://developers.google.com/meridian/`. Use this map to find the relevant
page to consult when answering user questions. To read a page, fetch its
Markdown version by appending `.md.txt` to the URL listed below.

## Documentation Map

<!-- TODO: Refine the description summary for each doc below. -->

*   **General Introduction & Basics:**

    *   `https://developers.google.com/meridian/docs/basics/meridian-introduction`
        (Overview of Meridian as an open-source Bayesian MMM framework. It
        answers core questions about ROI, response curves, and budget
        allocation. Key features include causal inference focus, use of priors
        for expertise, control variables like Google Query Volume, geo-level
        data advantages, and reach/frequency support. Outlines the end-to-end
        workflow: pre-modeling, modeling, and post-modeling.)
    *   `https://developers.google.com/meridian/docs/basics/glossary` (Defines
        key terms. Notable definitions include: Baseline (counterfactual
        scenario), Calibration (using experiments for priors), Contribution
        (incremental outcome %), Confounding vs Predictor variables, CPIK,
        Effect Window, Flighting pattern (spend allocation), Lagged effect
        (adstock), mROI, Saturation (no specific threshold), and SUTVA
        assumptions for non-geo hierarchies.)
    *   `https://developers.google.com/meridian/docs/faqs` (Answers common
        questions. Key points: Data privacy (Google only sees data requested
        from its platform), GeoX integration, LightweightMMM migration data
        needs, GQV scope (indexed only), time-varying effects via data
        partitioning (warns against over-partitioning), lack of synergy support,
        ROI over time options, and strong recommendation against non-geo
        hierarchies. Emphasizes that Meridian is for causal inference, not
        prediction.)

*   **Data Preparation & Loading:**

    *   `https://developers.google.com/meridian/docs/pre-modeling/collect-data`
        (Guides on gathering historical data for media, spend, controls, KPI,
        and population. Recommends weekly granularity and a minimum of 2-3 years
        of data. Advises limiting the number of media channels to ensure
        sufficient variation and volume, suggesting grouping smaller channels.)
    *   `https://developers.google.com/meridian/docs/pre-modeling/amount-data-needed`
        (Explains data sufficiency based on 'data points per effect'. National
        models need more years of data compared to geo models. Discourages
        campaign-level data due to parameter inflation and risk of losing
        Adstock memory. Recommends tighter, regularizing priors when data is
        insufficient in geo models.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/input-data`
        (Defines mathematical notation and dimensions for model inputs (KPI,
        controls, media, etc.). Explains internal linear scaling functions
        applied by Meridian, including geo population scaling and
        standardization to mean zero and standard deviation one.)
    *   `https://developers.google.com/meridian/docs/user-guide/supported-data-types-formats`
        (Lists supported data types (Geo-level with/without R&F, organic and
        non-media, National-level) and formats (CSV, Xarray, Numpy, Pandas).)
    *   `https://developers.google.com/meridian/docs/user-guide/load-geo-data-without-rf`
        (Detailed guide with code examples for loading geo-level data without
        reach and frequency using `CsvDataLoader`, `XrDatasetDataLoader`,
        `NDArrayInputDataBuilder`, or `DataFrameInputDataBuilder`.)
    *   `https://developers.google.com/meridian/docs/user-guide/load-geo-data-with-rf`
        (Detailed guide with code examples for loading geo-level data with reach
        and frequency, extending the standard loading to include reach,
        frequency, and RF spend.)
    *   `https://developers.google.com/meridian/docs/user-guide/load-national-data`
        (Detailed guide with code examples for loading national-level data,
        similar to geo-level loading but removing the geographic dimension.)
    *   `https://developers.google.com/meridian/docs/user-guide/load-geo-data-with-organic-and-non-media`
        (Detailed guide with code examples for loading geo-level data that
        includes organic media (no cost) and non-media treatments (e.g.,
        promotions).)

*   **Exploratory Data Analysis (EDA):**

    *   `https://developers.google.com/meridian/docs/pre-modeling/perform-eda`
        (Detailed guide on performing Exploratory Data Analysis (EDA) using
        Meridian's EDA package. Explains categories of checks: Spend share,
        Individual variables (variation, outliers), Population scaling,
        Relationships (VIF, correlation heatmap), and Prior specifications.)

*   **Model Configuration (ModelSpec, Priors):**

    *   `https://developers.google.com/meridian/docs/advanced-modeling/model-spec`
        (Full mathematical equation of the Meridian model. Covers extensions
        like reach/frequency, time-varying intercepts ($\mu_t$), and organic
        media. Defines Adstock and Hill functions, and parameter distributions.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/intro-priors`
        (Introduction to Bayesian priors in Meridian. Explains how priors allow
        injecting business knowledge to stabilize results and ground them in
        reality. Supports priors on ROI, mROI, and contribution percentage using
        distributions like Normal, Log-Normal and Half-Normal.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/default-prior-distributions`
        (Lists default prior distributions for all model parameters (e.g.,
        `knot_values`, `tau_g`, `roi_m`, `contribution_m`). Explains rationales,
        noting which are uninformative and which are regularizing (e.g.,
        `Beta(1,99)` for contribution).)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/how-to-choose-treatment-prior-types`
        (Guides on choosing treatment prior types (ROI, mROI, Contribution,
        Coefficient) for paid, organic, and non-media treatments. ROI is default
        for paid media, Contribution for organic/non-media. Explains induced
        priors and budget regularization using mROI priors.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/roi-priors-and-calibration`
        (Explains calibration of treatment priors using experiment results and
        domain knowledge. Discusses translating experiment point estimates and
        standard errors into priors, and relevance considerations (timing,
        duration). Emphasizes priors as regularization.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/set-custom-priors-past-experiments`
        (Code examples for setting custom ROI priors using past experiments.
        Details helper functions like `lognormal_dist_from_mean_std` and
        `lognormal_dist_from_range`, and lists nuances to consider when using
        experiment results.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/media-saturation-lagging`
        (Explains how Meridian models lagged effects (Adstock) and saturation
        effects (Hill function). Details the mathematical formulas, the
        `hill_before_adstock` option, and notes that default priors assume a
        concave shape to facilitate budget optimization.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/set-adstock-decay-spec-parameter`
        (Explains how to configure `adstock_decay_spec` (geometric or binomial).
        Geometric is best for short-lived effects, binomial for long-lived
        effects. Discusses customizing the alpha prior to control decay rate.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/set-max-lag-parameter`
        (Explains tradeoffs and practical advice for setting `max_lag` ($L$).
        Recommendations: 2-10 periods for geometric decay, 4-20 periods for
        binomial decay or combinations.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/setting-knots`
        (Explains modeling time effects using knots (splines). Discusses
        bias-variance tradeoff. Recommends `knots = n_times` for geo models and
        `1` knot for national models. Details the Automatic Knot Selection (AKS)
        feature.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/control-variables`
        (Explains the role of control variables (Confounding and Predictor).
        Advises on selecting controls based on marketing decisions. Covers query
        volume as a control, lagged controls, and population scaling.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/organic-and-non-media-variables`
        (Defines Organic media (no cost, has Adstock/Hill effects) and Non-media
        treatments (intervenable like price, no Adstock/Hill effects). Explains
        how to decide between these and controls.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/reach-frequency`
        (Explains modeling channels using reach and frequency data. Media effect
        is calculated by applying Hill function to frequency and multiplying by
        linear reach, followed by Adstock.)

*   **Model Fitting & Execution:**

    *   `https://developers.google.com/meridian/docs/user-guide/run-model`
        (Instructions for running the model (sampling prior and posterior).
        Details MCMC parameters (`n_chains`, `n_adapt`, `n_burnin`, `n_keep`)
        and mentions the No-U-Turn (NUTS) sampler.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/using-jax`
        (Explains how to enable and use the JAX backend for performance and
        memory efficiency. Details API differences like using
        `tfp.substrates.jax.distributions` and requiring explicit seeds.)

*   **Post-Modeling Analysis & Visualization:**

    *   `https://developers.google.com/meridian/docs/post-modeling/intro`
        (Directory for post-modeling section. Lists pages covering health
        checks, health score, model fit, ROI/mROI/Response curves, baseline,
        visualizations, and optimizations.)
    *   `https://developers.google.com/meridian/docs/post-modeling/health-checks`
        (Details model health checks via `reviewer.ModelReviewer(mmm).run()`.
        Covers Convergence (R-hat), Negative Baseline, Bayesian PPP,
        Goodness-of-fit, Prior Posterior Shift, and ROI Consistency.)
    *   `https://developers.google.com/meridian/docs/post-modeling/health-score`
        (Explains the model health score (0-100) combining 6 checks. Convergence
        is a gate. Explains weighting (Negative Baseline 30%, PPP 30%, etc.) and
        transformations.)
    *   `https://developers.google.com/meridian/docs/post-modeling/model-fit`
        (Discusses assessing model fit, emphasizing causal inference over
        prediction. Details how to access posterior draws and lists `Analyzer`
        methods.)
    *   `https://developers.google.com/meridian/docs/post-modeling/roi-mroi-response-curves`
        (Explains Incremental Outcome, ROI, mROI, and Response Curves. Defines
        them mathematically using potential outcomes. ROI is for past
        performance, response curves for optimization, mROI for saturation.)
    *   `https://developers.google.com/meridian/docs/post-modeling/baseline`
        (Explains assessing the baseline. Negative baseline indicates error.
        Details how to assess probabilistically and mitigate by adjusting priors
        or improving controls/knots.)
    *   `https://developers.google.com/meridian/docs/post-modeling/interpret-visualizations`
        (Guides on interpreting visualizations. Covers Model fit charts, Channel
        contribution charts, ROI charts, Response curves, Adstock decay curves,
        and Hill saturation curves.)
    *   `https://developers.google.com/meridian/docs/user-guide/generate-model-results-output`
        (Instructions for generating HTML report or summary table using
        `Summarizer` and `MediaSummary`.)
    *   `https://developers.google.com/meridian/docs/user-guide/plot-media-visualizations`
        (Guide for plotting customized media visualizations using `MediaSummary`
        and `MediaEffects`. Covers Area, Bump, Waterfall, Pie, Spend vs
        Contribution, ROI, ROI vs Effectiveness, ROI vs mROI, Response curves,
        Adstock, and Hill curves.)

*   **Budget Optimization:**

    *   `https://developers.google.com/meridian/docs/user-guide/optimization-overview`
        (Directory for budget optimization section. Lists scenarios: Fixed
        budget, Flexible budget (minimum ROI or marginal ROI), and frequency
        optimization.)
    *   `https://developers.google.com/meridian/docs/post-modeling/optimization-without-reach-frequency`
        (Mathematical explanation of budget optimization for channels without
        reach and frequency data. Assumes fixed flighting patterns. Covers fixed
        and flexible budget optimization.)
    *   `https://developers.google.com/meridian/docs/post-modeling/optimization-with-reach-frequency`
        (Mathematical explanation of budget optimization for channels with reach
        and frequency data. Covers optimizing target frequency and budget
        allocation under various constraints.)
    *   `https://developers.google.com/meridian/docs/post-modeling/interpret-optimizations`
        (Guides on interpreting budget optimization results. Covers Fixed
        budget, Flexible budget with target ROI/mROI scenarios. Explains
        visualizations like Optimized spend change, allocation pie chart, and
        optimal frequency.)
    *   `https://developers.google.com/meridian/docs/user-guide/generate-optimization-results-output`
        (Instructions for generating optimization HTML report or summaries.)
    *   `https://developers.google.com/meridian/docs/user-guide/plot-optimization-visualizations`
        (Guide for plotting customized optimization visualizations using
        `OptimizationResults`. Covers Spend delta, Outcome delta, Budget
        allocation, and Response curves.)

*   **Scenario Planning:**

    *   `https://developers.google.com/meridian/docs/scenario-planning/meridian-scenario-planner`
        (Overview of the Meridian Scenario Planner (Looker Studio tool). Covers
        features like interactive budget optimization and data security
        recommendations.)
    *   `https://developers.google.com/meridian/docs/scenario-planning/faqs`
        (FAQs for Scenario Planner. Covers 'Community Visualization Disabled'
        and 'Data Set Configuration Error'.)
    *   `https://developers.google.com/meridian/docs/post-modeling/scenario-planning-and-future-budget-optimization`
        (Advanced guide on scenario planning using `new_data`. Explains how to
        override assumptions about cost, revenue per KPI, and flighting
        patterns.)

*   **Model Saving/Loading:**

    *   `https://developers.google.com/meridian/docs/user-guide/saving-model-object`
        (Explains how to save and load the model object. Deprecates Python
        `pickle` in favor of the Meridian `serde` package, supporting binary and
        text protobuf formats.)
    *   `https://developers.google.com/meridian/docs/user-guide/mmm-unified-schema`
        (Explains the MMM Unified Schema (Protocol Buffers). Decouples training
        from consumption. Covers Model Core, Model Fit, Marketing Analysis, and
        Optimization.)

*   **Causal Inference Concepts:**

    *   `https://developers.google.com/meridian/docs/causal-inference/intro`
        (Introduction to causal inference in Meridian. Lists key assumptions
        required for valid causal estimates.)
    *   `https://developers.google.com/meridian/docs/causal-inference/about-mmm-causal-inference-methodology`
        (Discusses MMM as a causal inference methodology using observational
        data. Compares it to experiments and discusses untestable and testable
        assumptions.)

*   **Debugging:**

    *   `https://developers.google.com/meridian/docs/post-modeling/model-debugging`
        (Comprehensive guide for debugging model issues. Covers MCMC
        convergence, posterior same as prior, ROI differences based on priors,
        GPU OOM errors, high organic media contribution, collinearity, and
        negative R-squared.)

TREAT THE ABOVE MAPPING AS A GUIDE, NOT AN EXHAUSTIVE LIST.
