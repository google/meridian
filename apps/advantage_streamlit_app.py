"""Streamlit UI tailored for Advantage Medical Professionals MMM workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import importlib.util
import textwrap

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

if importlib.util.find_spec("meridian") is None:
  st.set_page_config(page_title="Advantage Medical MMM", page_icon="🩺", layout="wide")
  st.error(
      "The Meridian Python package is missing. Install it before launching this app.",
  )
  st.markdown(
      textwrap.dedent(
          """
          ### Install Meridian

          Choose the command that matches your environment and rerun the app:

          * **CPU-only / macOS**
            ```bash
            pip install --upgrade 'google-meridian'
            ```
          * **Linux with CUDA GPU**
            ```bash
            pip install --upgrade 'google-meridian[and-cuda]'
            ```
          * **Latest development build**
            ```bash
            pip install --upgrade git+https://github.com/google/meridian.git
            ```

          After Meridian is available, install the UI extras from this repo:
          ```bash
          pip install -e .[ui]
          ```
          """
      )
  )
  st.stop()

from meridian import constants
from meridian.analysis import analyzer, summarizer
from meridian.data import data_frame_input_data_builder
from meridian.model import model, prior_distribution, spec


BRAND_PRIMARY = "#1b6ca8"
BRAND_ACCENT = "#00a8b5"
BRAND_DARK = "#0b2545"
SAMPLE_PATH = Path(__file__).parent / "sample_data" / "advantage_marketing_sample.csv"


@dataclass
class ModelingConfig:
  kpi_col: str
  kpi_type: str
  time_col: str
  geo_col: str | None
  population_col: str | None
  revenue_per_kpi_col: str | None
  control_cols: list[str]
  media_cols: list[str]
  media_channel_labels: list[str]
  media_prior_type: str
  max_lag: int
  n_chains: int
  n_adapt: int
  n_burnin: int
  n_keep: int
  seed: int | None


def _inject_branding() -> None:
  st.set_page_config(
      page_title="Advantage Medical MMM",
      page_icon="🩺",
      layout="wide",
  )
  st.markdown(
      f"""
      <style>
      :root {{
        --adv-primary: {BRAND_PRIMARY};
        --adv-accent: {BRAND_ACCENT};
        --adv-dark: {BRAND_DARK};
      }}
      .stApp {{
        background: linear-gradient(180deg, rgba(11,37,69,0.08) 0%, white 45%);
        color: #0f1b2b;
      }}
      .adv-header {{
        background: linear-gradient(120deg, var(--adv-primary), var(--adv-accent));
        padding: 1.6rem 2.4rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 16px 28px rgba(27,108,168,0.25);
      }}
      .adv-header h1 {{
        font-size: 2.4rem;
        margin-bottom: 0.25rem;
      }}
      .adv-header p {{
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
      }}
      .block-container {{
        padding-top: 1.5rem;
        max-width: 1200px;
      }}
      .css-1aumxhk, .css-18ni7ap {{
        font-family: "Helvetica Neue", "Inter", sans-serif;
      }}
      </style>
      """,
      unsafe_allow_html=True,
  )


def _load_sample_data() -> pd.DataFrame:
  df = pd.read_csv(SAMPLE_PATH)
  df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
  return df


def _load_uploaded_data(upload) -> pd.DataFrame:
  df = pd.read_csv(upload)
  return df


def _tensor_to_numpy(tensor) -> np.ndarray:
  if tensor is None:
    raise ValueError("Expected tensor-like value, received None.")
  if hasattr(tensor, "numpy"):
    return tensor.numpy()
  return np.asarray(tensor)


def _ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
  df = df.copy()
  df[time_col] = pd.to_datetime(df[time_col])
  return df


def _prepare_population_dataframe(
    df: pd.DataFrame, config: ModelingConfig
) -> tuple[pd.DataFrame, str]:
  if config.population_col:
    pop_df = (
        df[[config.geo_col, config.population_col]]
        if config.geo_col
        else df[[config.population_col]]
    ).copy()
    pop_df[config.population_col] = pd.to_numeric(
        pop_df[config.population_col], errors="coerce"
    )
    if config.geo_col:
      pop_df = (
          pop_df.dropna(subset=[config.population_col])
          .groupby(config.geo_col, as_index=False)[config.population_col]
          .mean()
      )
      return pop_df, config.geo_col
    geo_col = "geo"
    pop_df[geo_col] = "national"
    return pop_df[[geo_col, config.population_col]], geo_col

  population_col = "population"
  if config.geo_col:
    pop_df = df[[config.geo_col]].drop_duplicates().copy()
    pop_df[population_col] = 1.0
    return pop_df[[config.geo_col, population_col]], config.geo_col
  pop_df = pd.DataFrame({"geo": ["national"], population_col: [1.0]})
  return pop_df, "geo"


def _build_input_data(df: pd.DataFrame, config: ModelingConfig):
  builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
      kpi_type=config.kpi_type,
  )
  working_df = _ensure_datetime(df, config.time_col)

  builder.with_kpi(
      working_df,
      kpi_col=config.kpi_col,
      time_col=config.time_col,
      geo_col=config.geo_col,
  )

  population_df, population_geo_col = _prepare_population_dataframe(
      working_df, config
  )
  builder.with_population(
      population_df,
      population_col=config.population_col or "population",
      geo_col=population_geo_col,
  )

  if config.control_cols:
    builder.with_controls(
        working_df,
        control_cols=config.control_cols,
        time_col=config.time_col,
        geo_col=config.geo_col,
    )

  if config.revenue_per_kpi_col:
    builder.with_revenue_per_kpi(
        working_df,
        revenue_per_kpi_col=config.revenue_per_kpi_col,
        time_col=config.time_col,
        geo_col=config.geo_col,
    )

  builder.with_media(
      working_df,
      media_cols=config.media_cols,
      media_spend_cols=config.media_cols,
      media_channels=config.media_channel_labels,
      time_col=config.time_col,
      geo_col=config.geo_col,
  )
  return builder.build()


def _flatten_last_axis(values: np.ndarray) -> np.ndarray:
  if values.ndim == 1:
    return values
  return values.reshape(-1, values.shape[-1])


def _channel_metrics(
    mmm: model.Meridian,
    adv_analyzer: analyzer.Analyzer,
    use_kpi: bool,
) -> pd.DataFrame:
  channel_names: list[str] = []
  if mmm.input_data.media_channel is not None:
    channel_names.extend(mmm.input_data.media_channel.values.tolist())
  if not channel_names:
    return pd.DataFrame()

  roi_tensor = adv_analyzer.roi(use_kpi=use_kpi)
  inc_tensor = adv_analyzer.incremental_outcome(use_kpi=use_kpi)

  roi_values = _flatten_last_axis(_tensor_to_numpy(roi_tensor))
  inc_values = _flatten_last_axis(_tensor_to_numpy(inc_tensor))

  roi_mean = roi_values.mean(axis=0)
  roi_ci_low = np.percentile(roi_values, 5, axis=0)
  roi_ci_high = np.percentile(roi_values, 95, axis=0)
  incremental_mean = inc_values.mean(axis=0)

  total_incremental = incremental_mean.sum()
  share = (
      incremental_mean / total_incremental
      if total_incremental
      else np.full_like(incremental_mean, np.nan)
  )

  return pd.DataFrame(
      {
          "channel": channel_names,
          "mean_roi": roi_mean,
          "roi_p05": roi_ci_low,
          "roi_p95": roi_ci_high,
          "incremental_outcome": incremental_mean,
          "contribution_share": share,
      }
  )


def _plot_roi_summary(metrics: pd.DataFrame) -> alt.Chart:
  return (
      alt.Chart(metrics)
      .mark_bar(color=BRAND_PRIMARY)
      .encode(
          x=alt.X("channel", sort="-y", title="Channel"),
          y=alt.Y("mean_roi", title="Mean ROI"),
          tooltip=[
              alt.Tooltip("mean_roi", format=".2f", title="Mean ROI"),
              alt.Tooltip("roi_p05", format=".2f", title="ROI 5th pct"),
              alt.Tooltip("roi_p95", format=".2f", title="ROI 95th pct"),
          ],
      )
      .properties(height=360)
  )


def _plot_contribution(metrics: pd.DataFrame) -> alt.Chart:
  return (
      alt.Chart(metrics)
      .mark_arc(innerRadius=60)
      .encode(
          theta=alt.Theta("contribution_share", stack=True),
          color=alt.Color("channel", legend=alt.Legend(title="Channel")),
          tooltip=[
              alt.Tooltip("channel"),
              alt.Tooltip("contribution_share", format=".1%", title="Share"),
          ],
      )
      .properties(height=320)
  )


def _scale_tensor(values: np.ndarray, scale_vector: np.ndarray) -> np.ndarray:
  reshape_dims = (1,) * (values.ndim - 1) + (scale_vector.shape[0],)
  return values * scale_vector.reshape(reshape_dims)


def _scenario_simulation(
    mmm: model.Meridian,
    adv_analyzer: analyzer.Analyzer,
    scales: Mapping[str, float],
    base_incremental: np.ndarray,
    use_kpi: bool,
) -> pd.DataFrame:
  channel_names = list(scales.keys())
  scale_vector = np.array([scales[name] for name in channel_names], dtype=float)

  if mmm.media_tensors.media is None:
    raise ValueError("No media channels available for simulation.")

  media_tensor = _tensor_to_numpy(mmm.media_tensors.media)
  spend_tensor = _tensor_to_numpy(mmm.media_tensors.media_spend)

  scaled_media = _scale_tensor(media_tensor, scale_vector)
  scaled_spend = _scale_tensor(spend_tensor, scale_vector)

  new_data = analyzer.DataTensors(
      media=scaled_media,
      media_spend=scaled_spend,
  )
  scenario_tensor = adv_analyzer.incremental_outcome(
      new_data=new_data, use_kpi=use_kpi
  )
  scenario_values = _flatten_last_axis(_tensor_to_numpy(scenario_tensor))
  scenario_mean = scenario_values.mean(axis=0)

  delta = scenario_mean - base_incremental
  summary = pd.DataFrame(
      {
          "channel": channel_names,
          "spend_multiplier": scale_vector,
          "scenario_incremental": scenario_mean,
          "baseline_incremental": base_incremental,
          "delta": delta,
      }
  )
  baseline = summary["baseline_incremental"].to_numpy()
  scenario = summary["scenario_incremental"].to_numpy()
  delta_pct = np.divide(
      scenario,
      baseline,
      out=np.full_like(scenario, np.nan),
      where=baseline != 0,
  ) - 1.0
  summary["delta_pct"] = delta_pct
  return summary


def _sidebar_configuration(df: pd.DataFrame) -> ModelingConfig:
  st.sidebar.header("Configuration")
  columns = df.columns.tolist()

  default_time = "date" if "date" in columns else columns[0]
  time_col = st.sidebar.selectbox("Time column", columns, index=columns.index(default_time))

  geo_options = ["<None>"] + columns
  geo_selection = st.sidebar.selectbox("Geo column", geo_options, index=0)
  geo_col = None if geo_selection == "<None>" else geo_selection

  kpi_col = st.sidebar.selectbox("Outcome (KPI) column", columns, index=columns.index("kpi_leads") if "kpi_leads" in columns else 0)
  kpi_type_label = st.sidebar.radio(
      "KPI type",
      ["Revenue", "Non-revenue"],
      index=0,
      help="Use revenue when KPI already reflects monetary outcome.",
  )
  kpi_type = constants.REVENUE if kpi_type_label == "Revenue" else constants.NON_REVENUE

  population_options = ["<Use 1.0>"] + columns
  population_selection = st.sidebar.selectbox(
      "Population column",
      population_options,
      index=population_options.index("population") if "population" in columns else 0,
  )
  population_col = (
      None if population_selection == "<Use 1.0>" else population_selection
  )

  revenue_options = ["<None>"] + columns
  revenue_selection = st.sidebar.selectbox(
      "Revenue per KPI column",
      revenue_options,
      index=revenue_options.index("revenue_per_lead") if "revenue_per_lead" in columns else 0,
  )
  revenue_per_kpi_col = (
      None if revenue_selection == "<None>" else revenue_selection
  )

  control_cols = st.sidebar.multiselect(
      "Control features",
      columns,
      default=["seasonality_index"] if "seasonality_index" in columns else [],
  )

  numeric_candidates = [c for c in columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {kpi_col}]
  default_media = [
      c
      for c in ["search_spend", "social_spend", "referral_spend", "brand_spend"]
      if c in numeric_candidates
  ]
  media_cols = st.sidebar.multiselect(
      "Paid media or investment columns",
      numeric_candidates,
      default=default_media,
      help="Selected columns are treated as both execution and spend values.",
  )

  media_labels = [
      st.sidebar.text_input(
          f"Label for {col}",
          value=col.replace("_", " ").title(),
          key=f"label_{col}",
      )
      for col in media_cols
  ]

  st.sidebar.subheader("Modeling controls")
  media_prior_type = st.sidebar.selectbox(
      "Paid media prior type",
      [
          constants.TREATMENT_PRIOR_TYPE_ROI,
          constants.TREATMENT_PRIOR_TYPE_MROI,
          constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION,
      ],
      index=0,
  )
  max_lag = st.sidebar.slider("Max lag (weeks)", min_value=0, max_value=16, value=8)

  n_chains = st.sidebar.slider("MCMC chains", min_value=1, max_value=4, value=1)
  n_adapt = st.sidebar.slider("Adapt draws", min_value=50, max_value=800, value=200, step=50)
  n_burnin = st.sidebar.slider("Burn-in draws", min_value=50, max_value=1000, value=200, step=50)
  n_keep = st.sidebar.slider("Posterior draws", min_value=50, max_value=1000, value=300, step=50)
  seed_input = st.sidebar.text_input("Random seed", value="2024")
  seed = int(seed_input) if seed_input.strip() else None

  return ModelingConfig(
      kpi_col=kpi_col,
      kpi_type=kpi_type,
      time_col=time_col,
      geo_col=geo_col,
      population_col=population_col,
      revenue_per_kpi_col=revenue_per_kpi_col,
      control_cols=control_cols,
      media_cols=media_cols,
      media_channel_labels=media_labels,
      media_prior_type=media_prior_type,
      max_lag=max_lag,
      n_chains=n_chains,
      n_adapt=n_adapt,
      n_burnin=n_burnin,
      n_keep=n_keep,
      seed=seed,
  )


def main() -> None:
  _inject_branding()

  st.markdown(
      """
      <div class="adv-header">
        <h1>Advantage Medical Professionals Marketing Mix Lab</h1>
        <p>Turn staffing insights into action with Meridian's probabilistic MMM engine tailored for Advantage Medical Professionals.</p>
      </div>
      """,
      unsafe_allow_html=True,
  )

  st.markdown(
      """
      Use the configuration sidebar to map Advantage Medical Professionals data,
      tune Bayesian priors, and simulate future investment strategies. The app
      ships with a curated weekly sample dataset that mirrors Advantage's
      recruiting funnel to help the team experiment immediately.
      """
  )

  data_source = st.radio(
      "Select data source",
      ["Sample Advantage dataset", "Upload CSV"],
      horizontal=True,
  )
  if data_source == "Upload CSV":
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
      st.info("Upload a CSV to continue or switch to the sample dataset.")
      return
    df = _load_uploaded_data(uploaded)
  else:
    df = _load_sample_data()

  st.write("### Data preview")
  st.dataframe(df.head(20))

  if df.empty:
    st.warning("The provided dataset is empty. Please upload a valid CSV.")
    return

  config = _sidebar_configuration(df)
  if not config.media_cols:
    st.warning(
        "Select at least one media column to build an MMM. Configure this in the sidebar."
    )
    return

  run_model = st.button("Run Advantage MMM", type="primary")
  if not run_model:
    st.stop()

  try:
    with st.spinner("Building Meridian input data and sampling posterior..."):
      input_data = _build_input_data(df, config)
      priors = prior_distribution.PriorDistribution()
      model_spec = spec.ModelSpec(
          prior=priors,
          media_prior_type=config.media_prior_type,
          max_lag=config.max_lag,
      )
      mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
      if config.n_adapt:
        mmm.sample_prior(n_draws=config.n_keep, seed=config.seed)
      mmm.sample_posterior(
          n_chains=config.n_chains,
          n_adapt=config.n_adapt,
          n_burnin=config.n_burnin,
          n_keep=config.n_keep,
          seed=config.seed,
      )
  except Exception as exc:  # pylint: disable=broad-except
    st.error(
        "Meridian encountered an error while fitting the model. "
        "Please inspect your data selections and try again."
    )
    st.exception(exc)
    return

  use_kpi_metric = (
      config.kpi_type == constants.NON_REVENUE
      and not config.revenue_per_kpi_col
  )
  adv_analyzer = analyzer.Analyzer(mmm)
  metrics = _channel_metrics(mmm, adv_analyzer, use_kpi_metric)

  st.success("Model training complete! Explore the insights below.")

  if not metrics.empty:
    st.write("### Channel ROI and incremental impact")
    roi_columns = {
        "mean_roi": "{:.2f}",
        "roi_p05": "{:.2f}",
        "roi_p95": "{:.2f}",
        "incremental_outcome": "{:.2f}",
        "contribution_share": "{:.1%}",
    }
    st.dataframe(metrics.style.format(roi_columns))

    col1, col2 = st.columns(2)
    with col1:
      st.altair_chart(_plot_roi_summary(metrics), use_container_width=True)
    with col2:
      st.altair_chart(_plot_contribution(metrics), use_container_width=True)
  else:
    st.info("No paid media channels detected; skipping ROI summary.")

  summarizer_instance = summarizer.Summarizer(mmm)
  summary_html = summarizer_instance._gen_model_results_summary(  # pylint: disable=protected-access
      use_kpi=use_kpi_metric
  )
  st.write("### Advantage Medical results digest")
  st.components.v1.html(summary_html, height=1200, scrolling=True)

  if metrics.empty:
    return

  st.write("### Scenario sandbox")
  scenario_scales = {}
  for channel in metrics["channel"]:
    scenario_scales[channel] = st.slider(
        f"{channel} spend multiplier",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.05,
        help="Scale both spend and delivery for the channel to test alternative budgets.",
    )

  if st.button("Evaluate scenario impact"):
    scenario_summary = _scenario_simulation(
        mmm,
        adv_analyzer,
        scenario_scales,
        metrics["incremental_outcome"].to_numpy(),
        use_kpi_metric,
    )
    total_baseline = scenario_summary["baseline_incremental"].sum()
    total_scenario = scenario_summary["scenario_incremental"].sum()
    total_delta = total_scenario - total_baseline

    st.write("#### Scenario results")
    st.metric(
        "Total incremental outcome",
        f"{total_scenario:,.2f}",
        delta=f"{total_delta:,.2f}",
    )
    st.dataframe(
        scenario_summary[[
            "channel",
            "spend_multiplier",
            "baseline_incremental",
            "scenario_incremental",
            "delta",
            "delta_pct",
        ]].style.format(
            {
                "spend_multiplier": "{:.2f}x",
                "baseline_incremental": "{:.2f}",
                "scenario_incremental": "{:.2f}",
                "delta": "{:.2f}",
                "delta_pct": "{:.1%}",
            }
        )
    )


if __name__ == "__main__":
  main()
