# Meridian Documentation Map

This document maps Meridian concepts and topics to their corresponding pages on
`https://developers.google.com/meridian/`. Use this map to find the relevant
page to consult when answering user questions. To read a page, fetch its
Markdown version by appending `.md.txt` to the URL listed below.

The map covers the Meridian MMM guides (`/meridian/docs/...`) and the Meridian
GeoX geo-experiment guides (`/meridian/geox/...`). GeoX results feed Meridian
through `CalibrationBuilder`, so questions about calibrating priors with
experiments usually span both sections.

## Documentation Map

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
        scenario), Brand equity variable and brand marketing (upper-funnel),
        Calibration (using experiments for priors), Contribution (incremental
        outcome %), Confounding vs Predictor vs Mediator variables (mediators
        excluded or handled via two-stage Full-Funnel MMM), CPIK, Effect Window,
        Flighting pattern (spend allocation), Lagged effect (adstock), mROI,
        Saturation (no specific threshold), Time effects (time-varying
        intercepts), and Treatment variables split into paid media, organic
        media, and non-media treatments (linear, no adstock).)
    *   `https://developers.google.com/meridian/docs/basics/math-notation`
        (Reference tables for the mathematical symbols used across Meridian
        equations. Covers base variables (KPI y, controls z[C], paid media x[M],
        reach r[RF], frequency f[RF], organic x[OM], non-media x[N], population
        p, unit values u), data-state markers (double-dot raw, dagger
        population-scaled, transformation function L), indexes (g, t, i, G, T,
        N_C, N_M, N_RF, N_OM, N_ORF, N_N), learned parameters (tau_g geo
        intercepts, mu_t time effects, beta media effects, gamma, sigma, eta,
        xi, alpha adstock decay, ec half-saturation, slope where slope above 1
        gives an S-curve), hyperparameters (max lag L, knots K, knot locations
        s_k), knot interpolation weights w(t), logic symbols (pipe, indicator,
        tilde, braces, for all), and causal and optimization notation (potential
        outcomes, counterfactual x(1) and x(0), budget b_i, spend scaling omega,
        optimal frequency f*, MCMC draw (j)).)
    *   `https://developers.google.com/meridian/docs/faqs` (Answers common
        questions. Key points: Data privacy (Google only sees data requested
        from its platform), GeoX integration, LightweightMMM migration data
        needs, GQV scope (indexed only), time-varying effects via data
        partitioning (warns against over-partitioning), lack of synergy support,
        ROI over time options, and strong recommendation against non-geo
        hierarchies. Emphasizes that Meridian is for causal inference, not
        prediction.)
    *   `https://developers.google.com/meridian/docs/migrate` (Guides migration
        from LightweightMMM to Meridian, its official successor; migration means
        installing Meridian and loading data as a new user, since input data is
        the same. Feature comparison table: Meridian uses TensorFlow Probability
        instead of Numpyro, adds built-in experiment calibration, reach and
        frequency modeling, ROI model formulation, GQV confounder support, knots
        for trend and seasonality instead of linear plus sinusoidal terms, and
        automatic input scaling. Meridian only supports Hill-Adstock
        (configurable order, except R&F which is Hill then Adstock), adds
        `eta_m` for media coefficient geo variation (normal or log-normal),
        hierarchical controls with `xi_c`, beta or ROI priors, and a baseline
        from geo plus time fixed effects. Expect somewhat longer MCMC sampling
        time.)
    *   `https://developers.google.com/meridian/docs/user-guide/installing`
        (Installation instructions for Meridian. Requires Python 3.11 or 3.12;
        recommends at least 1 GPU (tested on V100 and T4 with 16 GB RAM).
        Install via `pip install --upgrade 'google-meridian[and-cuda]'` for
        Linux GPU (needs CUDA toolchain), `google-meridian` for CPU or macOS (no
        macOS GPU support), or from GitHub for the latest version, then verify
        with `meridian.__version__`. Notes that the NUTS MCMC sampler is compute
        intensive so GPUs are recommended, that JAX is the default backend, and
        that the legacy TensorFlow backend is deprecated. Points to the Getting
        started colab.)
    *   `https://developers.google.com/meridian/docs/user-guide/agent-skills`
        (Describes Meridian's five Agent Skills (open Agent Skills
        specification) for AI coding assistants: `meridian-doc-consultant`,
        `meridian-model-building` (CSV loading, column mapping,
        `ModelSpec`/priors, EDA, MCMC fitting, `.binpb` save),
        `meridian-result-visualization` (`Reviewer` health checks, `Summarizer`
        report), `meridian-budget-optimization`, and `meridian-scenario-planner`
        (Looker Studio handoff). Covers installation via `npx skills add
        google/meridian`, in-repository discovery through `.agents/skills.json`
        and `skills/`, manual copy into `.agents/skills/`, and the
        human-in-the-loop checkpoint design that pauses before column mappings,
        prior settings, long MCMC runs, and script execution.)

*   **Data Preparation & Loading:**

    *   `https://developers.google.com/meridian/docs/pre-modeling/intro`
        (Directory for the pre-modeling section covering data gathering,
        cleaning, and organization before building an MMM. Links to pages on
        collecting and organizing data (media spend, KPIs, control variables,
        geographic/time/channel granularity), amount of data needed (national vs
        geo-level requirements), geo-level modeling advantages and geo
        selection, using Google MMM Data Platform (Google Search trends, YouTube
        metrics, access, delivery, data schemas), and performing exploratory
        data analysis (EDA HTML report, missing data, correlations, custom
        configurations).)
    *   `https://developers.google.com/meridian/docs/pre-modeling/collect-data`
        (Guides on gathering historical data for media, spend, controls, KPI,
        revenue per KPI, population, reach and frequency, organic media, and
        non-media treatments. Requires KPI and media metrics to be summable
        (totals, not rates like CTR) and complete without missing values:
        zero-fill inactive media, interpolate or forward-fill KPI and controls.
        Explains aggregating campaign-level platform exports into channel-level
        data. Recommends geo-level, weekly granularity, a minimum of 2 years
        (geo) or 3 years (national or monthly) of data, combining low-spend
        channels, and sourcing Google data and GQV via MMM Data Platform.)
    *   `https://developers.google.com/meridian/docs/pre-modeling/amount-data-needed`
        (Explains data sufficiency based on data points per parameter. National
        models count media channels, controls, and knots per parameter and need
        more years of data or reduced scope (combine channels, fewer knots, drop
        non-confounder controls). Geo models are assessed via strict no-pooling
        versus lenient perfect-pooling views; the EDA data-to-parameter ratio
        uses the lenient count, with actual partial pooling governed by `eta_m`
        and `xi_c`. Recommends regularizing hierarchical variance priors like
        `HalfNormal(0.1)` when data is scarce. Discourages campaign-level data
        due to lost Adstock memory.)
    *   `https://developers.google.com/meridian/docs/pre-modeling/geo-selection-national-data`
        (Compares geo-level and national-level data and recommends geo-level
        modeling when possible. Lists geo-model advantages: larger effective
        sample size, tighter credible intervals, better trend and seasonality
        estimates, support for more `knots`, greater spend variability for
        estimating Hill saturation, and reduced omitted variable bias. Geo
        selection guidance: drop the smallest geos by total KPI, model the top
        50-100 US DMAs by population, and consider `unique_sigma_for_each_geo`
        tradeoffs (convergence difficulty, checking population vs `sigma`
        pattern). Discusses caveats of aggregating small geos and recommends
        imputing national-level media to geos by population share when only some
        channels have geo data.)
    *   `https://developers.google.com/meridian/docs/pre-modeling/using-mmm-data-platform`
        (Guide to Google MMM Data Platform for sourcing Google Query Volume
        (GQV, indexed, brand and generic keywords, usable as a control or
        Full-Funnel mediator) and YouTube reach and frequency data (Google Ads
        and DV360, de-duped). Covers granting access (Owner, Reader, Writer
        roles), configuring direct delivery to Amazon S3, Google Cloud (grant
        `mmm-robots@google.com` Storage Object Admin), or SFTP, and requesting
        data (one country per request, brand names, campaign source, daily or
        weekly granularity, up to 5 years history, 3-year R&F window, recurring
        deliveries, up to three weeks for delivery). Documents data feed schemas
        for Google Ads, DV360, R&F (`Reach`, `AverageFrequency`, `Population`,
        rolling window metrics), and GQV (`IndexedQueryVolume`, `QueryLabel`).)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/input-data`
        (Defines mathematical notation and dimensions for model inputs: KPI,
        controls, paid media, reach and frequency, organic media, organic R&F,
        and non-media treatments, including media data for lagged periods before
        the first KPI time. Explains internal linear scaling functions: KPI
        divided by geo population then standardized to mean zero and standard
        deviation one; media, reach, and organic media population-scaled then
        divided by the median non-zero value so `ec_m` of one means
        half-saturation at the median; controls and non-media treatments
        standardized, with optional `population_scaled_controls` and
        `population_scaled_non_media_channels`.)
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
    *   `https://developers.google.com/meridian/docs/user-guide/filter-selected-times-geos`
        (Explains scoping analysis to subsets of geos or time periods using
        `selected_geos` and `selected_times` in `Analyzer` and `Visualizer`
        methods (for example `summary_metrics`, `incremental_outcome`,
        `ModelFit.plot_model_fit`), and `start_date`/`end_date` for
        `BudgetOptimizer.optimize()`. Covers `aggregate_times` and
        `aggregate_geos` booleans to segment or aggregate results, and
        `show_geo_level` and `include_baseline` for model fit plots. Warns that
        values passed must exactly match the geo and time coordinates in
        `InputData`. Includes example scenarios for Q4 metrics, per-geo weekly
        incremental outcome, filtered model fit plots, and time-windowed
        optimization.)

*   **Exploratory Data Analysis (EDA):**

    *   `https://developers.google.com/meridian/docs/pre-modeling/perform-eda`
        (Detailed guide on Meridian's EDA package: instantiate `MeridianEDA` and
        call `generate_and_save_report` for an HTML report with `FAIL` (blocks
        posterior sampling), `REVIEW`, and `INFO` severity levels. Covers five
        categories: spend share and data-to-parameter ratio, individual
        variables (lack of variation, outliers, boxplots), population scaling
        correlations, relationships (correlation heatmap, pairwise correlation,
        VIF, collinearity with geo and time effects), and prior specifications
        (prior probability of negative baseline, experiment calibration summary,
        calibrated prior quality). Customizable thresholds via `EDASpec`,
        `VIFSpec`, `PairwiseCorrSpec`, `AggregationConfig`.)

*   **Model Configuration (ModelSpec, Priors):**

    *   `https://developers.google.com/meridian/docs/advanced-modeling/intro`
        (Landing page and table of contents for the Applied Modeling section,
        with one-paragraph descriptions linking to each sub-page. Groups topics
        into The Meridian Model (input data, control variables, organic and
        non-media treatments, paid search, holdout observations, model spec,
        prevent negative baseline, reach and frequency, saturation and lagging,
        full-funnel MMM), Time Based Parameters (knots, `max_lag`,
        `adstock_decay_spec`), Backend Configuration (JAX), Geo versus National
        modeling, and Priors (intro, treatment prior types,
        ROI/mROI/contribution parameterizations, default priors, non-revenue
        KPI, calibration, Calibration Builder, combining distribution
        families).)
    *   `https://developers.google.com/meridian/docs/user-guide/configure-model`
        (Walks through the default `ModelSpec` and how to customize it. Covers
        setting ROI priors via `PriorDistribution` with `roi_m` and `roi_rf`
        (`LogNormal(0.2, 0.9)` default), channel-ordered priors through argument
        builders, `holdout` with `HoldoutSpec`, `RandomHoldoutSpec`,
        `DateRange`, and `GeoHoldoutSpec`, `knots` (integer, list, or `None`),
        `enable_aks`, hybrid knot selection via
        `knots.AKS.automatic_knot_selection()` with `required_knots`,
        `excluded_knots`, and `restrict_to_right_of_required_knots` for
        refreshes, `media_prior_type='coefficient'` to disable ROI calibration,
        `roi_calibration` and `rf_roi_calibration` with `CalibrationSpec` and
        `ChannelCalibrationSpec` (not generally recommended), plus
        `baseline_geo`, `hill_before_adstock`, `max_lag` (default 8),
        `media_effects_dist`, `population_scaled_controls`,
        `unique_sigma_for_each_geo`, `adstock_decay_spec`, and
        `allows_negative_aggregate_baseline`.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/model-spec`
        (Full mathematical equation of the Meridian model. Covers extensions
        like reach/frequency, time-varying intercepts ($\mu_t$), and organic
        media. Defines Adstock and Hill functions, and parameter distributions.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/national-models`
        (Explains that the national-level model is a special case of the
        geo-level model with a single geo and no separate model class; geo-level
        modeling is recommended when possible. Lists parameter restrictions
        automatically enforced for national data: the geo-variation parameters
        eta for media, organic media, RF, and organic RF are set to zero, xi for
        controls is zero so coefficients (beta, gamma) are shared, and
        `unique_sigma_for_each_geo = False`. Notes that population scaling has
        no effect in the national-level model.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/intro-priors`
        (Introduction to Bayesian priors in Meridian. Explains how priors allow
        injecting business knowledge (experiments, benchmarks, prior MMMs,
        expertise) to stabilize results and ground them in reality. Supports
        priors on ROI, mROI, and contribution percentage using distributions
        like Normal, Log-Normal, Beta, and Half-Normal, where standard deviation
        encodes confidence. Code examples build `PriorDistribution` with `roi_m`
        via `lognormal_dist_from_mean_std` and `lognormal_dist_from_range`, and
        plot prior samples. Frames informative priors as regularization and
        advises against flat, non-informative priors, using defaults when no
        information exists.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/default-prior-distributions`
        (Lists default prior distributions for all model parameters (e.g.,
        `knot_values`, `tau_g`, `roi_m`, `contribution_m`). Explains rationales,
        noting which are uninformative and which are regularizing (e.g.,
        `Beta(1,99)` for contribution).)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/how-to-choose-treatment-prior-types`
        (Guides on choosing and configuring treatment prior types (ROI, mROI,
        Contribution, Coefficient) for paid, organic, and non-media treatments.
        ROI is default for paid media, Contribution for organic/non-media;
        coefficient priors rarely recommended. Covers non-revenue outcome
        exceptions (total paid media contribution default, no mROI default),
        `ModelSpec` arguments `media_prior_type`, `rf_prior_type`,
        `organic_media_prior_type`, `non_media_treatments_prior_type`, and which
        `PriorDistribution` parameters (`roi_m`, `mroi_m`, `contribution_m`,
        `beta_m`) are used versus ignored. Explains induced priors and channel
        parity, inspecting them via `Analyzer` with `use_posterior=False`,
        customizing `contribution_n` with TruncatedNormal or scaled Beta, and
        budget regularization using mROI priors with `roi_calibration`.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/roi-mroi-contribution-parameterizations`
        (Mathematical derivation of how Meridian re-parameterizes media
        coefficients beta so that ROI, marginal ROI (mROI), or contribution
        proportion of each channel is a model parameter, enabling priors from
        incrementality experiments or benchmarks and equal regularization across
        channels. Shows the one-to-one mapping between beta_m and ROI_m via
        HillAdstock-transformed media, depending on `media_effects_dist`
        (`LogNormal` restricts ROI to positive values; `Normal` allows any real
        value). mROI uses a 1% spend increase (0.01 times cost); contribution
        uses total observed outcome as denominator instead of spend. Covers
        reach and frequency channels and the `roi_calibration` and
        `rf_roi_calibration` arguments for restricting the ROI numerator to a
        subset of time periods, while recommending priors over all periods.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi-default`
        (Explains Meridian's default behavior when the KPI is not revenue.
        Recommends supplying `revenue_per_kpi` (even an estimate) so incremental
        outcome is monetary and default ROI priors remain unitless and
        reasonable. When `revenue_per_kpi` is unavailable and `media_prior_type`
        and `rf_prior_type` are `'roi'`, Meridian automatically replaces the
        default ROI prior with a common total paid media contribution prior
        implying total paid media contribution to the KPI has prior mean 40% and
        standard deviation 20%, with ROI interpreted as incremental KPI units
        per spend unit. An explicitly set ROI prior overrides this default.
        Distinguishes this from channel-level contribution priors and points to
        custom prior options such as free app installs with unknown lifetime
        value.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/unknown-revenue-kpi-custom`
        (Presents three options for custom priors when the KPI is not revenue
        and `revenue_per_kpi` is not passed to `InputData`. A custom total paid
        media contribution prior derives a common LogNormal ROI prior on `roi_m`
        or `roi_rf` from a desired prior mean and standard deviation of total
        media contribution (code example with 50% mean, 15% sd using total `kpi`
        and per-channel `cost`). A custom incremental KPI per cost (IKPC) prior
        sets `roi_m`/`roi_rf` directly, but only if specified for every channel
        since defaults are not scale-aware. A channel-level contribution prior
        sets `media_prior_type='contribution'` and
        `rf_prior_type='contribution'` in `ModelSpec` with Beta priors on
        `contribution_m` and `contribution_rf`, e.g. Beta(1, 99) for a 1% mean.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/roi-priors-and-calibration`
        (Explains calibration of treatment priors using incrementality
        experiment results and domain knowledge, treating priors as guardrails
        rather than flat priors. Discusses translating experiments into priors,
        relevance considerations (recency, duration, media granularity,
        population differences, complexity, estimand mismatch), and automated
        adjustments via `CalibrationBuilder` with Meridian GeoX integration.
        Shows constructing lognormal ROI priors with
        `lognormal_dist_from_mean_std` and `lognormal_dist_from_range` in
        `PriorDistribution`, setting standard deviation by confidence, plotting
        priors, using the `roi_calibration` period in `ModelSpec`, and checking
        prior-posterior shifts.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/set-custom-priors-past-experiments`
        (Explains automated prior calibration with `CalibrationBuilder`,
        translating incrementality experiment results into `roi_m` and `roi_rf`
        priors. Covers registering Meridian GeoX `AnalysisResult` via
        `with_meridian_geox_experiment_result` or generic experiments via
        `with_incrementality_experiment_result`, automatic spend, recency, and
        duration adjustments, merging multiple experiments with Bayesian
        updating and an optional `baseline_prior`, and fitting LogNormal, Gamma,
        or Normal distributions by minimizing KL divergence. Output is a
        `CalibratedPriors` dataclass passed to `PriorDistribution` and
        `ModelSpec`.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/custom-prior-combine-distribution-families`
        (Shows how to use
        `prior_distribution.IndependentMultivariateDistribution` to combine
        distributions from different families into a single prior, for example
        `LogNormal` ROI priors for three channels and a `HalfNormal` prior for a
        fourth, then pass it as `roi_m` in `PriorDistribution` and
        `ModelSpec(prior=...)`. Uses `tfp_jax.distributions` with float64
        arrays. Warns of slightly longer runtimes because tensors are split and
        delegated to child distributions, and suggests first considering varying
        parameters within one family or switching families entirely.)
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
        (Explains modeling time effects using knots splines and the
        bias-variance tradeoff. Defaults: `knots = n_times` for geo models, `1`
        for national models. Details Automatic Knot Selection via
        `enable_aks=True` in `ModelSpec`, reviewing
        `mmm.knot_info.knot_locations`, tuning
        `knots.AKS.automatic_knot_selection` with `base_penalty`,
        `min_internal_knots`, `max_internal_knots`, and hybrid selection with
        `required_knots`, `excluded_knots`, and
        `restrict_to_right_of_required_knots` for model refresh. Covers
        alternatives like binary indicators and periodic functions as controls,
        and disabling geo variation via the `xi_c` prior.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/control-variables`
        (Explains control variables as confounders or predictors, and why
        mediators must be excluded. Advises selecting controls based on what
        drives marketing budget decisions, with guidance for full-funnel MMM
        Stage 1 and Stage 2 models. Covers extracting control coefficient
        samples from `inference_data.prior.gamma_c` and `posterior.gamma_c`,
        query volume as confounder versus mediator, lagged controls via backdoor
        criteria, population scaling with `population_scaled_controls` and
        `population_scaled_non_media_channels`, and why controls receive no
        causal effect or baseline breakdown.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/organic-and-non-media-variables`
        (Defines Organic media (no cost, has Adstock/Hill effects) and Non-media
        treatments (intervenable like price, no Adstock/Hill effects). Explains
        how to decide between these and controls.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/paid-search-modeling`
        (Best practices for modeling paid search, where Google Query Volume
        (GQV) is a key confounder because query volume drives ad volume;
        omitting it overestimates paid search effects. Recommends including GQV
        as a control variable scaled by population via
        `population_scaled_controls`, separating brand-specific and generic paid
        search as distinct media channels with matching brand and generic GQV
        controls, and modeling brand campaigns with clicks and generic campaigns
        with impressions. Discusses clicks vs impressions trade-offs, selection
        bias from demand-driven spend, sourcing GQV from the MMM Data Platform,
        and alternatives for non-Google search engines (assess GQV as proxy via
        correlation plots or omit the channel).)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/reach-frequency`
        (Explains modeling channels using reach and frequency data. Media effect
        is calculated by applying Hill function to frequency and multiplying by
        linear reach, followed by Adstock.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/holdout-observations`
        (Describes the `holdout` argument of `ModelSpec`, a `HoldoutSpec` that
        can be a `RandomHoldoutSpec` (same number of random time periods per
        geo), a sequence of `DateRange` (same dates in every geo), or a sequence
        of `GeoHoldoutSpec` (given dates in given geos). Holdout KPI data is
        ignored during MCMC training but media data is still used for adstock.
        The main use is out-of-sample goodness of fit (R-squared) to compare
        model specifications such as prior strengths. Recommends a holdout
        balanced across geos and time periods; no holdout is set by default.
        Warns against holding out large contiguous end-of-window chunks for
        forecasting, since knots need nearby data.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/prevent-negative-baseline`
        (Explains `allows_negative_aggregate_baseline=False` in `ModelSpec`
        (default `True`), which constrains the baseline summed across all geos
        and time periods to be non-negative on every prior and posterior draw,
        so `Analyzer.negative_baseline_probability()` is `0.0` and the
        `ModelReviewer` Negative Baseline check passes; individual geo or time
        baselines can still be negative. Works by reparameterizing `knot_values`
        with an orthogonal Householder change of basis (`rotated_knot_0`,
        `rotated_knot_rest`) and placing a `TruncatedNormal` with dynamic lower
        bound on one coordinate, with no NUTS slowdown and no tuning. Requires
        `prior.knot_values` to be `Normal(0, scale)` with common scale (default
        5.0). Overly optimistic ROI, mROI, or contribution priors can surface as
        elevated R-hat, Bayesian PPP failure, or lower R-squared; recalibrate
        priors rather than allowing a negative baseline. Includes a mathematical
        appendix.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/full-funnel`
        (Explains full-funnel MMM, a two-stage approach that resolves the
        mediator-confounder dilemma for brand equity variables such as Branded
        Google Query Volume (bGQV): Stage 1 models brand marketing (TV, video,
        PR) driving brand equity; Stage 2 models brand equity plus all marketing
        driving the KPI. Including bGQV as a control under-credits brand
        channels, excluding it over-credits lower-funnel channels like paid
        search. The indirect effect is propagated into ROI, mROI, response
        curves and budget optimization using `AnalyzerFullFunnel`. Covers when
        to prefer single-stage MMM, compounded adstock decay across `max_lag` of
        both stages, and the generic framework for any brand health metric.)

*   **Model Fitting & Execution:**

    *   `https://developers.google.com/meridian/docs/user-guide/run-model`
        (Instructions for running the model: define `ModelSpec`, then call
        `sample_prior` and `sample_posterior` using the No-U-Turn NUTS sampler
        on the default JAX backend. Details MCMC parameters `n_chains`,
        `n_adapt`, `n_burnin`, `n_keep`. Describes optional posterior thinning
        via `posterior_thinning()` with `sampling_rate` recommended 0.15 or
        `n_draws`, `preserve_original`, and `restore_full_posterior()` for
        exploratory iteration only, warning not to use thinned posteriors for
        final deliverables.)
    *   `https://developers.google.com/meridian/docs/advanced-modeling/using-jax`
        (Explains that JAX is the default backend since Meridian 2.0, with the
        TensorFlow backend deprecated but selectable via
        `MERIDIAN_BACKEND='tensorflow'`. Covers 64-bit precision default and
        switching to float32 with `MERIDIAN_ENABLE_JAX_X64='False'`, strict type
        consistency requiring `np.float64` parameters in `PriorDistribution`,
        using `tfp_jax.distributions` from
        `tensorflow_probability.substrates.jax`, the explicit `seed` requirement
        in `sample_posterior()`, numerical differences across backends, and
        performance gains enabling larger `unrolled_leapfrog_steps` and
        `n_adapt`.)

*   **Post-Modeling Analysis & Visualization:**

    *   `https://developers.google.com/meridian/docs/post-modeling/intro`
        (Directory for post-modeling section. Lists pages covering model health
        checks, channel calibration recommendation and score, model health
        score, model fit, baseline, incremental outcome, ROI, mROI, and response
        curves, interpreting visualizations, interpreting optimizations,
        refreshing the model, model debugging, optimization with and without
        reach and frequency, and scenario planning with future budget
        optimization.)
    *   `https://developers.google.com/meridian/docs/post-modeling/health-checks`
        (Details model health checks via `reviewer.ModelReviewer(mmm).run()`.
        Covers Convergence (R-hat), Negative Baseline, Bayesian PPP,
        Goodness-of-fit, Prior Posterior Shift, and ROI Consistency.)
    *   `https://developers.google.com/meridian/docs/post-modeling/health-score`
        (Explains the model health score, 0-100, combining six health checks
        plus the model-level composite calibration score. Convergence is a gate;
        non-converged models score zero. Scores of 90 or above suggest the model
        is usable, 70 or below warrants investigation. Details component
        transformations: binary Bayesian PPP, linear interpolation for negative
        baseline, sigmoid for R-squared, power transform of channel failure rate
        for prior-posterior shift and ROI consistency. Weights: Negative
        Baseline 22.5%, PPP 22.5%, Composite Calibration 22.5%, Prior Posterior
        Shift 12.5%, ROI Consistency 12.5%, Goodness-of-fit 7.5%, with dynamic
        normalization when checks are skipped.)
    *   `https://developers.google.com/meridian/docs/user-guide/model-diagnostics`
        (Covers assessing convergence, prior vs posterior, and model fit after
        fitting. Uses `visualizer.ModelDiagnostics(mmm).plot_rhat_boxplot()` for
        R-hat (values near 1.0 indicate convergence; R-hat < 1.2 is a reasonable
        threshold; non-convergence suggests misspecification or insufficient
        `n_adapt + n_burnin`), `az.plot_trace` on `inference_data` for trace and
        density plots, and `plot_prior_and_posterior_distribution()` (default
        ROI, or pass parameters like `'beta_m'`); low-spend channels often show
        posterior equal to prior, so drop or combine them. Model fit uses
        `predictive_accuracy_table()` for R-squared, MAPE, and wMAPE (split by
        Train/Test if `holdout` is set) and
        `visualizer.ModelFit(mmm).plot_model_fit()` with `n_top_largest_geos`,
        `show_geo_level`, `include_baseline`, and `include_ci` for expected
        versus actual plots.)
    *   `https://developers.google.com/meridian/docs/post-modeling/model-fit`
        (Discusses assessing model fit, emphasizing causal inference over
        prediction. Details how to access posterior draws and lists `Analyzer`
        methods.)
    *   `https://developers.google.com/meridian/docs/post-modeling/roi-mroi-response-curves`
        (Explains Incremental Outcome, ROI, mROI, and Response Curves. Defines
        them mathematically using potential outcomes. ROI is for past
        performance, response curves for optimization, mROI for saturation.)
    *   `https://developers.google.com/meridian/docs/post-modeling/baseline`
        (Explains assessing the baseline, the expected outcome with treatments
        at baseline values. A negative baseline indicates statistical error;
        assess probabilistically with `Analyzer.negative_baseline_probability`.
        Distinguishes negative baseline from contribution exceeding 100% of
        observed outcome. Mitigation: compute prior contribution probabilities,
        set custom `contribution` or ROI priors, add quality controls,
        population scale via `population_scaled_controls`, or adjust `knots`.
        Describes preventing a negative aggregate baseline by setting
        `allows_negative_aggregate_baseline=False` in `ModelSpec`, which
        reparameterizes `knot_values`. Also covers checking trend and
        seasonality.)
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
    *   `https://developers.google.com/meridian/docs/post-modeling/channel-recommendation`
        (Explains how Meridian flags channels to prioritize for incrementality
        experiments (for example Meridian GeoX) via the model health report from
        `reviewer.ModelReviewer(mmm).run()` and `output_model_health_card`.
        Channels with composite calibration score below 67.5 are recommended.
        Four spend-weighted criteria: implausibly high ROI (threshold 20),
        implausibly low ROI (0.5), high variance ROI (80% posterior CI width
        relative to default `LogNormal(0.2, 0.9)` prior, threshold 1.0), and
        potential bias (max absolute correlation with controls below 0.1).
        Composite score weights implausible ROI 0.5, high variance 0.25,
        potential bias 0.25 using normalized bowl and half-bowl formulas.
        Model-level score averages channel scores; use `CalibratedDistribution`
        with `is_calibrated` to mark channels calibrated without
        `CalibrationBuilder`.)
    *   `https://developers.google.com/meridian/docs/post-modeling/refreshing-model`
        (Best practices for refreshing a Meridian model with new data.
        Recommends appending new data to older data and refitting, deciding
        whether to drop the oldest data to stay within the common 2-3 year
        window (a bias-variance tradeoff since media effectiveness is not
        time-varying). Notes MMM estimates have high variance, so small amounts
        of new data can shift results; priors informed by past MMM results are
        allowed but effectively count old data twice. Keep hyperparameters like
        `knots` consistent across refreshes using AKS hybrid knot selection.
        Modeling new data disjointly is not recommended because it ignores
        lagged effects (include `max_lag` periods of prior media data) and
        independent parametric priors cannot capture the joint posterior.)

*   **Budget Optimization:**

    *   `https://developers.google.com/meridian/docs/user-guide/optimization-overview`
        (Directory for budget optimization section. Lists scenarios: Fixed
        budget, Flexible budget (minimum ROI or marginal ROI), and frequency
        optimization.)
    *   `https://developers.google.com/meridian/docs/user-guide/budget-optimization-scenarios`
        (Explains the three `BudgetOptimizer.optimize()` scenarios with code:
        fixed budget maximizing ROI or incremental KPI (default), flexible
        budget with `target_roi` (constraint at total marketing level), and
        flexible budget with `target_mroi` (applied per channel). Documents key
        parameters `start_date`, `end_date`, `budget`, `pct_of_spend` (must sum
        to 1, ordered as `InputData.media + InputData.reach`), `fixed_budget`,
        `spend_constraint_lower`, and `spend_constraint_upper`. Defaults are 0.3
        for fixed budget (70%-130% of spend) and 1 for flexible budget
        (0%-200%). Uses `get_paid_channels_argument_builder()` for channel-keyed
        arguments.)
    *   `https://developers.google.com/meridian/docs/post-modeling/optimization-without-reach-frequency`
        (Mathematical explanation of budget optimization for channels without
        reach and frequency data. Assumes fixed flighting patterns. Covers fixed
        and flexible budget optimization.)
    *   `https://developers.google.com/meridian/docs/post-modeling/optimization-with-reach-frequency`
        (Mathematical explanation of budget optimization for channels with reach
        and frequency data. Covers optimizing target frequency and budget
        allocation under various constraints.)
    *   `https://developers.google.com/meridian/docs/user-guide/optimize-frequency`
        (Explains optimal frequency for reach and frequency channels: the
        average frequency that maximizes expected ROI, computed numerically and
        constrained to be the same across all geos and times. Only available for
        channels with both reach and frequency inputs. Plot with
        `visualizer.ReachAndFrequency(mmm).plot_optimal_frequency()`. Budget
        optimization uses the optimal frequency by default (recommended); set
        `use_optimal_frequency=False` in `BudgetOptimizer.optimize()` to use
        historical frequency instead.)
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
        (Overview of the Meridian Scenario Planner, an Open Beta Looker Studio
        dashboard generated from a trained model via Colab. Covers key features,
        GPU-accelerated scenario planning by precomputing a response grid with
        `WeeklyOptimizationGrid.create` and `to_optimization_grid` passed to
        `optimize(optimization_grid=...)`, model fit and ROI visualizations,
        interactive fixed and flexible budget optimization configuration
        including `min_spend_shift_ratio` and `max_spend_shift_ratio` channel
        spend constraints, interpreting optimization results with response
        curves and reach and frequency charts, and data security recommendations
        for temporary versus persisted dashboard links.)
    *   `https://developers.google.com/meridian/docs/scenario-planning/faqs`
        (FAQs and troubleshooting for the Scenario Planner. Covers the
        'Community Visualization Disabled' chart error fixed by saving the
        report and allowing community visualizations, the 'Data Set
        Configuration Error' on the Plan page caused by an infeasible scenario
        setup, and the Colab getting stuck or running slowly at the 'Generate
        dashboard' step, mitigated by reducing time granularity or limiting the
        reach and frequency range, with unresolved issues reported on the GitHub
        Issues page.)
    *   `https://developers.google.com/meridian/docs/post-modeling/scenario-planning-and-future-budget-optimization`
        (Advanced guide on scenario planning and future budget optimization
        using the `new_data` argument with `DataTensors`. Explains how metric
        definitions, not parameter estimation, change when overriding cost per
        media unit, revenue per KPI, flighting patterns, and time windows for
        `incremental_outcome`, `roi`, `response_curves`, and
        `BudgetOptimizer.optimize`. Covers `selected_times` and
        `media_selected_times`, the `create_optimization_tensors` helper with
        `cpmu`, `budget` and `pct_of_spend` overrides, `start_date` and
        `end_date`, four code examples, and why Meridian forecasts incremental
        outcome rather than total outcome.)

*   **Model Saving/Loading:**

    *   `https://developers.google.com/meridian/docs/user-guide/saving-model-object`
        (Explains how to save and load the model object. Deprecates Python
        `pickle` in favor of the Meridian `serde` package, supporting binary and
        text protobuf formats.)
    *   `https://developers.google.com/meridian/docs/user-guide/mmm-unified-schema`
        (Explains the MMM Unified Schema (Protocol Buffers). Decouples training
        from consumption. Covers Model Core, Model Fit, Marketing Analysis, and
        Optimization.)

*   **Causal Inference & Bayesian Modeling Theory:**

    *   `https://developers.google.com/meridian/docs/causal-inference/intro`
        (Landing page for the Bayesian Modeling and Causal Inference Theory
        section. Introduces MMM as a causal problem and links to pages on the
        rationale for causal inference and Bayesian modeling, a primer on
        Bayesian inference with priors and posteriors, MMM as a causal inference
        methodology, required assumptions such as conditional exchangeability,
        the causal graph, estimating incremental outcome using regression, the
        extension to reach and frequency models, and a reference list of
        academic papers.)
    *   `https://developers.google.com/meridian/docs/causal-inference/rationale-for-causal-inference-and-bayesian-modeling`
        (Explains why Meridian adopts a causal inference perspective: ROI,
        response curves, and budget optimization are inherently counterfactual
        quantities, so causal estimands and transparent assumptions (such as the
        causal DAG) are required even though Meridian is a regression model.
        Lists three reasons for the Bayesian approach: priors provide intuitive
        regularization needed because MMM has many highly correlated variables
        and complex adstock and diminishing returns effects; the model can be
        reparameterized in terms of ROI, enabling custom ROI priors informed by
        experiments; and nonlinear media transformation parameters cannot be
        estimated by linear mixed models, so MCMC sampling is used.)
    *   `https://developers.google.com/meridian/docs/causal-inference/bayesian-inference`
        (Explains Bayes' theorem (posterior proportional to likelihood times
        prior), and how Meridian's hierarchical regression likelihood,
        informative vs uninformative priors, and joint posterior over all
        parameters (including Adstock and Hill parameters, ROI, mROI, response
        curves) fit together. Describes MCMC sampling with the No U-Turn Sampler
        (NUTS) from TensorFlow Probability, running multiple independent chains
        to assess convergence. Recommends checking R-hat values for all
        parameters, targeting R-hat below 1.2 (not strict); slightly higher
        values usually converge with longer chains, while values of 2.0 or more
        may require adjusting the model.)
    *   `https://developers.google.com/meridian/docs/causal-inference/about-mmm-causal-inference-methodology`
        (Discusses MMM as a causal inference methodology using observational
        data. Compares it to experiments and discusses untestable and testable
        assumptions.)
    *   `https://developers.google.com/meridian/docs/causal-inference/required-assumptions`
        (States the two key causal inference assumptions that let a regression
        model estimate potential outcomes: conditional exchangeability
        (potential outcomes are independent of the advertiser's historical media
        execution given controls) and consistency (the observed KPI equals the
        potential outcome of the realized scenario). Consistency holds unless
        the counterfactual is poorly defined. Exchangeability holds if all
        confounding variables, anything causally affecting both treatment
        (budget level, allocation across channels, geos, or time) and outcome,
        are measured and included in the control array; there is no statistical
        test for this, but it holds under a causal graph satisfying Pearl's
        backdoor criterion.)
    *   `https://developers.google.com/meridian/docs/causal-inference/causal-graph`
        (Describes Meridian's causal DAG and Pearl's backdoor criterion: control
        for all confounders, never for mediators. Treatments T (paid media,
        organic media, non-media) affect KPI K, controls C affect both; adstock
        gives paid and organic media lagged effects across time periods but
        non-media treatments have none. Concludes the regression should
        condition on current and lagged media up to the maximum lag, plus
        current-period non-media and control variables only. Warns that arrows
        T1 to C2 or T2 from K1 make a variable both mediator and confounder,
        which a single regression cannot handle; lagged controls increase
        parameter count. Introduces the full-funnel two-stage DAG with Brand
        marketing B, Other marketing O, brand equity G (such as branded GQV),
        and confounder nodes V, Z, W.)
    *   `https://developers.google.com/meridian/docs/causal-inference/estimating-incremental-outcome`
        (Derives mathematically how, under exchangeability and consistency, the
        expected potential outcome of any counterfactual treatment scenario
        (media, organic media, non-media) equals a conditional expectation
        estimable by regression. Because the KPI scaling transformation L is
        linear, it passes outside the expectation, so incremental outcome
        between scenarios x(1) and x(0) is a difference of regression
        predictions summed over geos and time. Writes out the Meridian model
        expectation with time effects mu_t, geo effects tau_g, control and
        non-media gamma coefficients, and HillAdstock terms with alpha, ec, and
        slope. Notes incremental outcome, ROI, mROI, and response curves all
        have posterior distributions sampled via MCMC.)
    *   `https://developers.google.com/meridian/docs/causal-inference/models-reach-freq`
        (Extends potential outcome, incremental outcome, and ROI definitions to
        channels with reach and frequency data. Incremental outcome of an R&F
        channel is defined by setting its reach to zero everywhere as the
        counterfactual (frequency values are irrelevant when reach is zero);
        Cost is the sum of reach times unit cost. Defines two response curve
        types: a reach response curve scaling reach by omega while holding
        frequency at historical values, and a frequency response curve scaling
        frequency while holding reach fixed. Warns that for small omega the
        counterfactual average frequency can fall below one, an implausible
        value, so interpret such response curves with caution.)
    *   `https://developers.google.com/meridian/docs/causal-inference/reference-list`
        (Bibliography of academic and Google research papers underpinning
        Meridian. Includes Pearl's Causality, Hernan and Robins' Causal
        Inference: What If, Gelman et al. Bayesian Data Analysis, Gelman-Rubin
        and Brooks-Gelman MCMC convergence diagnostics (R-hat), Bayesian
        R-squared, spline regression and knot selection, and TensorFlow
        Probability. Google MMM papers cover carryover and shape effects (Jin et
        al. 2017), geo-level Bayesian hierarchical MMM (Sun et al. 2017),
        category data, paid search bias correction (Chen et al. 2018), MMM
        calibration with Bayesian priors (Zhang et al. 2023), reach and
        frequency hierarchical MMM (Zhang et al. 2023), and time-varying
        coefficient models.)

*   **Debugging:**

    *   `https://developers.google.com/meridian/docs/post-modeling/model-debugging`
        (Comprehensive guide for debugging model issues. Covers MCMC
        convergence, posterior same as prior, ROI differences based on priors,
        GPU OOM errors, high organic media contribution, collinearity, and
        negative R-squared.)

*   **Meridian GeoX (Geo Experiments):**

    *   `https://developers.google.com/meridian/geox/intro-to-geox` (Overview of
        Meridian GeoX, an open-source geo-experimentation framework for
        measuring marketing incrementality. Core benefits: a single library for
        study design and incrementality analysis across methodologies, support
        for holdback, go-dark, and heavy-up designs, multi-cell execution with a
        shared control group, flexible constraints such as excluding geos, and
        Meridian MMM integration for calibration priors. Outlines the user
        journey: optional MMM-suggested experiment, pretest data collection
        (daily geo-level KPI and spend), study design, implementation, results
        data collection with optional cooldown, counterfactual analysis via
        time-based regression, and optional MMM calibration.)
    *   `https://developers.google.com/meridian/geox/glossary` (Defines key GeoX
        terms: Alpha (significance level, Type I error), Conversion (primary
        KPI), Cooldown (post-test delayed impact period), Cost per incremental
        conversion (CpIC = 1/ICPD, used only for holdback budget projection,
        ignored for go-dark and heavy-up), Experiment budget, Geo unit
        (recommends DMA for US, GMA for non-US, minimizing spillover across
        commuting corridors), Control, Excluded, and Treatment geos, Incremental
        conversion, ICPD (equivalent to iROAS for revenue), Minimum detectable
        effect (MDE), Power, Pretest, and Test periods.)
    *   `https://developers.google.com/meridian/geox/faqs` (Answers common GeoX
        questions. Covers GeoX versus user-based lift and legacy TBRMM (GeoX
        uses out-of-sample validation and placebo testing, hence higher budget
        and MDE recommendations, plus `compare_designs`), data rules (no
        negative net revenue, daily data only, single `conversions` KPI, spend
        mandatory for go-dark and heavy-up, preserve ZIP leading zeros), forcing
        geos with `included_control_geos` and `excluded_geos` in `Constraints`
        (force-to-treatment unsupported), minimum 10 geos and random sampling
        for 10-20 geos, `cost_per_incremental_conversion` and `budget_pct`,
        design viability (`min_r2` default 0.8, A/A p-value), campaign
        duplication for capped budgets, 1-2 week cooldown, deactivating
        user-level lift studies, and no significance test between treatment
        arms.)
    *   `https://developers.google.com/meridian/geox/install-geox` (Installation
        instructions for the `meridian-geox` library. Requires Python >= 3.10
        standalone or Python >= 3.11 alongside Meridian MMM. Relies on JAX; CPU
        JAX installs automatically, with GPU setup recommended for heavy
        simulations. Install via `pip install --upgrade meridian-geox`, install
        both libraries with `google-meridian[meridian-geox]`, or install the
        development version from the GitHub repository. Points to
        getting-started Colabs with sample data in the meridian-geox GitHub
        repo.)
    *   `https://developers.google.com/meridian/geox/intro-to-design`
        (Introduces the GeoX design phase, where a design is a blueprint
        assigning geos to treatment and control groups while projecting the
        required budget and minimum detectable effect (MDE). Serves as a summary
        index linking to related pages: Types of experiments (holdback, go-dark,
        heavy-up, multi-cell), Prepare your pretest data (historical geo-level
        response and spend series), Design methodology (geo assignment rule and
        design selection process), and Find your design (running design
        algorithms based on randomization and power analysis).)
    *   `https://developers.google.com/meridian/geox/types-of-experiments`
        (Describes GeoX experiment designs and intervention types. Single-cell
        design (one treatment versus control) is recommended for standard
        incrementality testing because it needs fewer geos and lower budget for
        a low MDE. Multi-cell design compares multiple treatment groups against
        a common control for budget level tests, cross-publisher comparison, or
        optimization experiments, but requires significantly more geos and
        budget. Intervention types: Holdback (new spend versus zero, for net-new
        channels), Go-dark (spend shut off versus BAU, to defend existing
        budgets), and Heavy-up (increased spend +X% versus BAU, for marginal
        returns or when spend is too low to power go-dark).)
    *   `https://developers.google.com/meridian/geox/prepare-your-pretest-data`
        (Specifies pretest data requirements for GeoX design: daily geo-level
        time series of conversions and spend as a Pandas DataFrame with `date`,
        `location`, `conversions`, and optional `spend` columns (`spend_cell_1`,
        `spend_cell_2` for multi-cell). Spend is mandatory for go-dark and
        heavy-up, not for holdback. Requires at least 3*N days of history where
        N is the test duration, or a year or more for seasonal businesses.
        Conversion metric must be absolute (not ROAS ratios), raw and
        unattributed, non-negative (avoid net revenue). Weekly data unsupported;
        for volatile data extend duration, use shallower KPIs, or raise
        `max_conversions_percent`. Only one primary KPI per run; secondary KPIs
        analyzed post-test via `analyze`. Mentions Google Ads API and MMM Data
        Platform for spend pulls.)
    *   `https://developers.google.com/meridian/geox/design-methodology`
        (Explains how GeoX generates and selects designs. Stratified sampling
        partitions geos into strata via k-means clustering (default four
        clusters) on KPI features including mean, coefficient of variation,
        slope, lag-1 autocorrelation, and dynamic time warping, then randomizes
        assignment within each cluster; this extends to multi-cell designs.
        Random sampling is supported but only recommended for small geo counts.
        The design search pipeline evaluates out-of-sample R-squared fit, A/A
        robustness via placebo simulation p-values, statistical power as MDE
        lift (SE_lift times (Z_{1-alpha/2} + Z_{1-beta}) divided by baseline
        control conversion, two-sided by default), and budget, splitting pretest
        data into training, validation, and final selection periods across six
        steps.)
    *   `https://developers.google.com/meridian/geox/config-your-preferred-design`
        (Step-by-step guide to configuring a GeoX design with
        `geox.DesignConfig` (`experiment_duration`, `cell_count`,
        `experiment_types`, `methodology` such as `Methodology.TBR`,
        `geo_assignment_rule`, `design_output_count`, optional `alpha` default
        0.10, `power` default 80%, `cost_per_incremental_conversion` for
        holdback only, `min_r2` default 0.8), `geox.Constraints`
        (`budget_constraint` with `geox.Budget`, `excluded_geos`,
        `excluded_dates`, `budget_percent`, `max_conversions_percent` default
        30%), and `geox.run_design()` returning designs ranked by MDE. Covers
        evaluating R-squared >= 0.8 and A/A p-value > alpha, troubleshooting A/A
        failures by excluding anomalies or extending history from 3N to 6N,
        `geox.plot_design()`, `export_to_json()` and `load_from_json()`, and
        multi-cell budget-neutral designs with `GO_DARK` and `HEAVY_UP` cells
        using `budget_pct`.)
    *   `https://developers.google.com/meridian/geox/compare-methodologies`
        (Shows how to use `geox.compare_designs()` to compare multiple design
        configurations side-by-side on the same pretest dataset, such as
        time-based regression (TBR) paired with `GeoAssignmentRule.RANDOM`
        versus `GeoAssignmentRule.STRATIFIED_SAMPLING`. Takes a list of
        (`DesignConfig`, `Constraints`) tuples and returns a `design_metrics`
        table for evaluating which approach yields higher R-squared or lower
        minimum detectable effect (MDE). Notes that stratified sampling is
        generally recommended because it pairs well with TBR and yields
        lower-variance designs.)
    *   `https://developers.google.com/meridian/geox/troubleshooting`
        (Troubleshooting guide for GeoX design outputs centered on minimum
        detectable effect (MDE). Explains verifying MDE realism by comparing
        design-implied CpIC (`design_implied_cpic`, budget divided by minimum
        incremental conversions) against theoretical CpIC (CPA divided by
        elasticity). MDE is independent of budget and determined by historical
        volatility and geo matching; budget only shifts the design-implied CpIC.
        When no design fits, increase budget in `geox.Constraints()`, lengthen
        `experiment_duration`, raise `max_conversions_percent`, or use shallower
        KPIs; avoid raising campaign budgets mid-test as it breaks the control
        BAU assumption. For seasonal peaks, prefer heavy-up (+20% to +50%) or
        partial go-dark (go-dim), start tests early and end late, use at least
        one year of pretest data, and placebo-validate on the prior year's peak
        (A/A p-value >= alpha, R-squared >= 0.8).)
    *   `https://developers.google.com/meridian/geox/implementation-of-geo-testing`
        (Guides executing a GeoX design in ad platforms like Google Ads.
        Outlines a 7-14 week timeline: pretest planning (up to 2 weeks), active
        test (3-10 weeks), and a recommended 2+ week cool-down. Covers key
        specifications: aligning campaigns with the MMM channel to calibrate,
        using one primary raw (not attributed) KPI, geo unit bias-variance
        tradeoff, and covering at least one purchase cycle. Describes heavy-up,
        go-dark, and holdback targeting strategies plus multi-cell setups. Best
        practices: lock geo allocation, isolate from other lift studies, balance
        local marketing, synchronize platforms, use 'Physical Presence' and
        positive (not negative) targeting.)
    *   `https://developers.google.com/meridian/geox/intro-to-analysis`
        (Overview of the GeoX analysis module for measuring incremental campaign
        impact. Analysis requires pretest and test period cost and conversion
        time series plus provenance information (design configuration,
        methodology, geo assignment) from the design phase. Counterfactual
        modeling, primarily time-based regression (TBR), estimates what would
        have happened without the intervention. Design-aware inference provides
        precise p-values without inflating type-I error while maximizing power.
        Outputs include lift, percentage lift, confidence intervals, p-values,
        and iCPD (equivalent to iROAS with revenue), which can be exported as
        custom priors for Meridian MMM calibration.)
    *   `https://developers.google.com/meridian/geox/prepare-your-analysis-data`
        (Step-by-step code workflow for running a GeoX analysis. Load the saved
        design with `geox.Design.load_from_json()` to reuse the exact treatment
        and control geo assignments. Prepare conversion and spend data in the
        same format as pretest data but covering both pretest and test periods.
        Define `geox.AnalysisConfig` with `design`, `analysis_start_date`, and
        `analysis_end_date`; the date range must encompass the full test period
        and `analysis_end_date` can be extended to include a cooldown. Call
        `geox.analyze(analysis_data, analysis_config)` to get point estimates
        and confidence intervals via `analysis_result.results`.)
    *   `https://developers.google.com/meridian/geox/counterfactual-modeling`
        (Explains how GeoX uses counterfactual or potential outcome modeling to
        estimate the average treatment effect on treated (ATT), defined as
        E[Y(1) - Y(0) | D=1]. Only the factual Y(1) is observed in treated geos
        during the test; the counterfactual Y(0) is predicted from control geos
        under the business-as-usual (BAU) assumption. Describes the machine
        learning loss function minimizing the L2 norm of pretest residuals
        between a functional f of treated geos (average conversion) and a
        learned functional g of control geos, optionally with L1 or L2
        regularization. Time-based regression (TBR) is the supported method,
        where g is a linear transformation.)
    *   `https://developers.google.com/meridian/geox/time-based-regression`
        (Details the time-based regression (TBR) methodology GeoX uses to
        measure causal impact. A linear model is trained during the pretest
        period on average KPI time series of aggregated treatment and control
        geos, then predicts the counterfactual during the test period.
        Mathematical formulation: group aggregation, OLS fit of treatment
        average = alpha + beta * control average with a non-negativity
        constraint (beta >= 0), no distributional assumptions on residuals
        (uncertainty via non-parametric placebo inference), daily pointwise
        lift, and cumulative incremental conversions scaled by the number of
        treatment geos. For go-dark experiments the effect is negated; heavy-up
        and holdback use it directly.)
    *   `https://developers.google.com/meridian/geox/robust-inference` (Explains
        design-aware inference for determining whether observed lift is
        statistically significant. GeoX geo assignment comes from constrained
        optimization, so standard t-tests, sliding-window tests, and
        uniform-shuffle placebo tests produce 'bad splits' that inflate type-I
        error and reduce power. Design-aware inference draws placebo assignments
        from the restricted design space with identical objective functions,
        budget limits, and constraints, tightening the null distribution and
        lowering the minimum detectable effect (MDE). Four implementation steps:
        formalize the assignment mechanism P(Z|X), generate design-aware placebo
        vectors, compute the empirical null distribution of placebo test
        statistics, and calculate exact p-values (optionally studentized by
        pretest RMSE) and confidence intervals by p-value inversion.)
    *   `https://developers.google.com/meridian/geox/analysis-outputs`
        (Describes the numerical metrics and charts returned after a GeoX
        analysis. `geox.analyze` returns an `AnalysisResult` whose `results` map
        each treatment cell to an `AnalysisMetrics` object containing `lift`
        (absolute incremental conversions), `percent_lift` (relative to
        counterfactual baseline), and `icpd` (incremental conversion per dollar,
        equivalent to iROAS with revenue), each as an `Estimate` with
        `point_estimate`, `lower_bound`, `upper_bound`, and `p_value` from
        design-aware placebo inference. `geox.plot_analysis(analysis_result)`
        produces four charts per cell: observed versus counterfactual daily
        conversions, pointwise differences, cumulative lift, and cumulative
        iCPD. Pretest series should closely align, with deviation appearing in
        the test period.)
    *   `https://developers.google.com/meridian/geox/data-validation-and-quality-checks`
        (Explains GeoX built-in data checks: validation checks raise
        `ValueError` and halt, while quality checks log warnings and return a
        `QualityCheckResult`. Thresholds (30% missing days, 500 geos, 50% zero
        conversions) are hardcoded. Baseline validations require `date`,
        `location`, `conversions` columns, non-negative `spend`, positive total
        conversions, and daily (not weekly) granularity. `geox.run_design()`
        requires pretest length at least 3x `experiment_duration`, at least `2 *
        (cell_count + 1)` geos, `alpha` and `power` in (0,1),
        `max_conversions_percent` below 0.5, `spend` or `spend_cell_N` columns
        for `GO_DARK` and `HEAVY_UP`, and positive
        `cost_per_incremental_conversion` for `HOLDBACK`. `geox.analyze()`
        enforces geo set consistency and pretest duration. `QualityCheckConfig`
        controls `exclude_geos_no_response` and `exclude_outlier_dates`.)
    *   `https://developers.google.com/meridian/geox/api-reference` (Complete
        API reference for the `meridian_geox` package. Design module:
        `DesignConfig` (`experiment_duration`, `experiment_types`
        HOLDBACK/GO_DARK/HEAVY_UP, `methodology` TBR, `geo_assignment_rule`
        RANDOM/STRATIFIED_SAMPLING, `cell_count`, `alpha` default 0.1, `power`
        0.8, `test_type`, `cost_per_incremental_conversion`, advanced
        `n_candidates`, `min_r2`, `num_strata`), `Budget` (`budget`,
        `budget_pct`), `Constraints` (`included_control_geos`, `excluded_geos`,
        `excluded_dates`, `budget_constraint`, `max_conversions_percent` 0.3),
        `DesignSet` metrics (`mde`, `mde_abs`, `r2`, `p_value (AA)`,
        `design_implied_cpic`), `Design`, `PerCellDesign`, `run_design()`,
        `compare_designs()`, `concat_design_reports()`, `plot_design()`.
        Analysis module: `AnalysisConfig` (`pretest_end_date`,
        `n_placebo_candidates`, `n_top_placebos`, `min_placebo_r2`),
        `AnalysisResult`, `AnalysisMetrics`, `Estimate`, `DescriptiveMetrics`
        (`estimated_bau_spend`), `analyze()`, `plot_analysis()`. Data quality
        module: `QualityCheckConfig`, `QualityCheckResult`,
        `check_design_data_quality()`, `check_analysis_data_quality()`.)

TREAT THE ABOVE MAPPING AS A GUIDE, NOT AN EXHAUSTIVE LIST.
