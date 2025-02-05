# Copyright 2024 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualization module that creates analytical plots for the Meridian model."""

from collections.abc import Sequence
import functools
import altair as alt
from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import formatter
from meridian.analysis import summary_text
from meridian.model import model
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr


__all__ = [
    'ModelDiagnostics',
    'ModelFit',
    'ReachAndFrequency',
]


# Disable max row limitations in Altair.
alt.data_transformers.disable_max_rows()


class ModelDiagnostics:
  """Generates model diagnostics plots from the Meridian model fitting."""

  def __init__(self, meridian: model.Meridian):
    self._meridian = meridian
    self._analyzer = analyzer.Analyzer(meridian)

  @functools.lru_cache(maxsize=128)
  def _predictive_accuracy_dataset(
      self,
      selected_geos: frozenset[str] | None = None,
      selected_times: frozenset[str] | None = None,
      batch_size: int = c.DEFAULT_BATCH_SIZE,
  ) -> xr.Dataset:
    """Displays the predictive accuracy Dataset.

    Args:
      selected_geos: Optional list of a subset of geo dimensions to include. By
        default, all geos are included. Geos should match the geo dimension
        names from meridian.InputData. Only set one of `selected_geos` and
        `n_top_largest_geos`.
      selected_times: Optional list of a subset of time dimensions to include.
        By default, all times are included. Times should match the time
        dimensions from meridian.InputData.
      batch_size: Integer representing max draws per chain in each batch. The
        calculation is run in batches to avoid memory exhaustion. If a memory
        error occurs, try reducing `batch_size`. The calculation will generally
        be faster with larger `batch_size` values.

    Returns:
      A Dataset displaying the predictive accuracy metrics.
    """
    selected_geos_list = list(sorted(selected_geos)) if selected_geos else None
    selected_times_list = (
        list(sorted(selected_times)) if selected_times else None
    )
    return self._analyzer.predictive_accuracy(
        selected_geos=selected_geos_list,
        selected_times=selected_times_list,
        batch_size=batch_size,
    )

  def predictive_accuracy_table(
      self,
      selected_geos: Sequence[str] | None = None,
      selected_times: Sequence[str] | None = None,
      column_var: str | None = None,
      batch_size: int = c.DEFAULT_BATCH_SIZE,
  ) -> pd.DataFrame:
    """Displays the predictive accuracy of the DataFrame.

    Args:
      selected_geos: Optional list of a subset of geo dimensions to include. By
        default, all geos are included. Geos should match the geo dimension
        names from `meridian.InputData`. Set either `selected_geos` or
        `n_top_largest_geos`, do not set both.
      selected_times: Optional list of a subset of time dimensions to include.
        By default, all times are included. Times must match the time dimensions
        from `meridian.InputData`.
      column_var: Optional string that indicates whether to pivot the table by
        `metric`, `geo_granularity` or `evaluation_set`. By default,
        `column_var=None` indicates that the `metric`, `geo_granularity` and
        `value` (along with `evaluation_set` when `holdout_id` isn't `None`)
        columns are displayed in the returning unpivoted DataFrame.
      batch_size: Integer representing the number of maximum draws per chain in
        each batch. The calculation is run in batches to avoid memory
        exhaustion. If a memory error occurs, try reducing `batch_size`. The
        calculation will generally be faster with larger `batch_size` values.

    Returns:
      A DataFrame containing the computed `R_Squared`, `MAPE` and `wMAPE`
      values. If `holdout_id` exists, the data is split into `Train`, `Test`,
      and `All Data` subsections, and `evaluation_set` is included as a column
      in the transformation from Dataset to DataFrame.
    """
    selected_geos_frozenset = (
        frozenset(selected_geos) if selected_geos else None
    )
    selected_times_frozenset = (
        frozenset(selected_times) if selected_times else None
    )
    predictive_accuracy_dataset = self._predictive_accuracy_dataset(
        selected_geos_frozenset, selected_times_frozenset, batch_size
    )
    df = predictive_accuracy_dataset.to_dataframe().reset_index()
    if not column_var:
      return df
    coords = list(predictive_accuracy_dataset.coords)
    if column_var not in coords:
      raise ValueError(
          f'The DataFrame cannot be pivoted by {column_var} as it does not'
          ' exist in the DataFrame.'
      )
    indices = coords.copy()
    indices.remove(column_var)
    return (
        df.pivot(
            index=indices,
            columns=column_var,
            values=c.VALUE,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

  def plot_prior_and_posterior_distribution(
      self,
      parameter: str = 'roi_m',
      num_geos: int = 3,
      selected_times: list[str] | None = None,
  ) -> alt.Chart | alt.FacetChart:
    """Plots prior and posterior distributions for a model parameter.

    Args:
      parameter: Model parameter name to plot. By default, the ROI parameter is
        shown if a name is not specified.
      num_geos: Number of largest geos by population to show in the plots for
        the geo-level parameters. By default, only the top three largest geos
        are shown.
      selected_times: List of specific time periods to plot for time-level
        parameters. These times must match the time periods from the data. By
        default, the first three time periods are plotted.

    Returns:
      An Altair plot showing the parameter distributions.

    Raises:
      NotFittedModelError: The model hasn't been fitted.
      ValueError: A `parameter` is not a Meridian model parameter.
    """
    if not (
        hasattr(self._meridian.inference_data, c.PRIOR)
        and hasattr(self._meridian.inference_data, c.POSTERIOR)
    ):
      raise model.NotFittedModelError(
          'Plotting prior and posterior distributions requires fitting the'
          ' model.'
      )

    # Check if the selected parameter is part of Meridian's model parameters.
    if (
        parameter
        not in self._meridian.inference_data.posterior.data_vars.keys()
    ):
      raise ValueError(
          f"The selected param '{parameter}' does not exist in Meridian's model"
          ' parameters.'
      )

    if selected_times:
      param_data = self._meridian.inference_data.posterior[parameter]
      if not (hasattr(param_data, c.TIME)):
        raise ValueError(
            '`selected_times` can only be used if the parameter has a time'
            f" dimension. The selected param '{parameter}' does not have a time"
            ' dimension.'
        )
      if any(time not in param_data.time for time in selected_times):
        raise ValueError(
            'The selected times must match the time dimensions in the Meridian'
            ' model.'
        )

    prior_dat = self._meridian.inference_data.prior[parameter]
    posterior_dat = self._meridian.inference_data.posterior[parameter]
    prior_df = (
        prior_dat.to_dataframe().reset_index().drop(columns=[c.CHAIN, c.DRAW])
    )
    posterior_df = (
        posterior_dat.to_dataframe()
        .reset_index()
        .drop(columns=[c.CHAIN, c.DRAW])
    )

    # Tag the data before combining.
    prior_df[c.DISTRIBUTION] = c.PRIOR
    posterior_df[c.DISTRIBUTION] = c.POSTERIOR
    prior_posterior_df = pd.concat([prior_df, posterior_df])

    if c.GEO in prior_posterior_df.columns:
      top_geos = self._meridian.input_data.get_n_top_largest_geos(num_geos)
      prior_posterior_df = prior_posterior_df[
          prior_posterior_df[c.GEO].isin(top_geos)
      ]

    if c.TIME in prior_posterior_df.columns:
      default_num_times = 3
      times = (
          selected_times
          if selected_times
          else prior_dat[c.TIME][:default_num_times].values
      )
      prior_posterior_df = prior_posterior_df[
          prior_posterior_df[c.TIME].isin(times)
      ]

    groupby = posterior_df.columns.tolist()
    groupby.remove(parameter)
    plot = (
        alt.Chart(prior_posterior_df)
        .transform_density(
            parameter, groupby=groupby, as_=[parameter, 'density']
        )
        .mark_area(opacity=0.7)
        .encode(
            x=f'{parameter}:Q',
            y='density:Q',
            color=f'{c.DISTRIBUTION}:N',
        )
    )

    # Create faceted charts for corresponding parameter dimensions. The model
    # only groups by 2 dimensions at most.
    if len(groupby) == 2:
      plot = plot.facet(groupby[0], columns=3).resolve_scale(x=c.INDEPENDENT)
    elif len(groupby) == 3:
      plot = plot.facet(column=groupby[0], row=groupby[1]).resolve_scale(
          x=c.INDEPENDENT
      )

    return plot.properties(
        title=formatter.custom_title_params(
            summary_text.PRIOR_POSTERIOR_DIST_CHART_TITLE
        )
    ).configure_axis(**formatter.TEXT_CONFIG)

  def plot_rhat_boxplot(self) -> alt.Chart:
    """Plots the R-hat box plot.

    Visual summary of the Gelman & Rubin (1992) potential scale reduction for
    chain convergence, commonly referred to as R-hat. It is a convergence
    diagnostic measure that measures the degree to which variance (of the means)
    between chains exceeds what you would expect if the chains were identically
    distributed. Values close to 1.0 indicate convergence. R-hat < 1.2 indicates
    approximate convergence and is a reasonable threshold for many problems
    (Brooks & Gelman, 1998).

    There is a single R-hat value for each model parameter. The box plot
    summarizes the distribution of R-hat values across indices. For example, the
    box corresponding to `beta_gm` summarizes the distribution of R-hat values
    across both the geo index `g` and the channel index `m`.

    The R-hat is not defined for any parameters that have deterministic priors,
    so these parameters are not shown on the boxplot.

    References:
      Andrew Gelman and Donald B. Rubin. Inference from Iterative Simulation
        Using Multiple Sequences. Statistical Science, 7(4):457-472, 1992.
      Stephen P. Brooks and Andrew Gelman. General Methods for Monitoring
        Convergence of Iterative Simulatio

    Returns:
      An Altair plot showing the R-hat boxplot per parameter.

    Raises:
      NotFittedModelError: The model hasn't been fitted.
      MCMCSamplingError: The MCMC sampling did not converge.
    """
    if not hasattr(self._meridian.inference_data, c.POSTERIOR):
      raise model.NotFittedModelError(
          'Plotting the r-hat values requires fitting the model.'
      )

    rhat = pd.DataFrame()
    mcmc_states = {
        k: v.values
        for k, v in self._meridian.inference_data.posterior.data_vars.items()
    }
    for k, v in tfp.mcmc.potential_scale_reduction(
        {k: tf.einsum('ij...->ji...', v) for k, v in mcmc_states.items()}
    ).items():
      rhat_temp = v.numpy().flatten()
      rhat = pd.concat([
          rhat,
          pd.DataFrame({
              c.PARAMETER: np.repeat(k, len(rhat_temp)),
              c.RHAT: rhat_temp,
          }),
      ])

    # If the MCMC sampling fails, the r-hat value calculated will be very large.
    if (rhat[c.RHAT] > 1e10).any():
      max_rhat = max(rhat[c.RHAT])
      raise model.MCMCSamplingError(
          f'MCMC sampling failed with a maximum R-hat value of {max_rhat}.'
      )

    # Drop any parameters with a deterministic prior, such as slope_m, which
    # will produce an NaN r-hat value.
    rhat = rhat.dropna(subset=[c.RHAT])

    boxplot = (
        alt.Chart(rhat)
        .mark_boxplot(median={'color': c.BLUE_300}, outliers={'filled': True})
        .encode(
            x=alt.X(c.PARAMETER, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(c.RHAT, scale=alt.Scale(zero=False)),
        )
    )
    rhat_reference_line = (
        alt.Chart()
        .mark_rule(color=c.RED_600, opacity=0.8)
        .encode(y=alt.datum(1))
    )
    return (
        (boxplot + rhat_reference_line)
        .properties(
            title=formatter.custom_title_params(summary_text.RHAT_BOXPLOT_TITLE)
        )
        .configure_axis(**formatter.TEXT_CONFIG)
    )


class ModelFit:
  """Generates model fit plots from the Meridian model fitting.

  Calculates the expected versus actual outcome with the confidence level over
  time, and plots graphs to compare the values.
  """

  def __init__(
      self,
      meridian: model.Meridian,
      confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL,
  ):
    """Initializes the dataset based on the model and confidence level.

    Args:
      meridian: Media mix model with the raw data from the model fitting.
      confidence_level: Confidence level for expected outcome credible intervals
        represented as a value between zero and one. Default is `0.9`.
    """
    self._meridian = meridian
    self._analyzer = analyzer.Analyzer(meridian)
    self._model_fit_data = self._analyzer.expected_vs_actual_data(
        confidence_level=confidence_level
    )

  @property
  def model_fit_data(self) -> xr.Dataset:
    """Dataset holding the expected, actual, and baseline outcome over time.

    The dataset contains the following:

    - **Coordinates:** `geo`, `time`, `metric` (`mean`, `ci_hi`, `ci_lo`)
    - **Data variables:** `expected`, `baseline`, `actual` (outcome)
    """
    return self._model_fit_data

  def update_confidence_level(self, confidence_level: float):
    self._model_fit_data = self._analyzer.expected_vs_actual_data(
        confidence_level=confidence_level
    )

  def plot_model_fit(
      self,
      selected_times: Sequence[str] | None = None,
      selected_geos: Sequence[str] | None = None,
      n_top_largest_geos: int | None = None,
      show_geo_level: bool = False,
      include_baseline: bool = True,
      include_ci: bool = True,
  ) -> alt.Chart:
    """Plots the expected versus actual outcome over time.

    Args:
      selected_times: Optional list of a subset of time dimensions to include.
        By default, all times are included. Times should match the time
        dimensions from `meridian.InputData`.
      selected_geos: Optional list of a subset of geo dimensions to include. By
        default, all geos are included. Geos should match the geo dimension
        names from `meridian.InputData`. Set either `selected_geos` or
        `n_top_largest_geos`, do not set both.
      n_top_largest_geos: Optional number of largest geos by population to
        include. By default, all geos are included. Set either `selected_geos`
        or `n_top_largest_geos`, do not set both.
      show_geo_level: If `True`, plots at the geo-level instead of one national
        level plot. Only available if `selected_geos` or `n_top_largest_geos` is
        provided.
      include_baseline: If `True`, shows the expected baseline outcome without
        any media execution.
      include_ci: If `True`, shows the credible intervals for the expected
        outcome.

    Returns:
      An Altair plot showing the model fit.
    """
    outcome = (
        c.REVENUE
        if self._meridian.input_data.revenue_per_kpi is not None
        else c.KPI.upper()
    )
    self._validate_times_to_plot(selected_times)
    self._validate_geos_to_plot(
        selected_geos, n_top_largest_geos, show_geo_level
    )

    if n_top_largest_geos:
      selected_geos = self._meridian.input_data.get_n_top_largest_geos(
          n_top_largest_geos
      )

    model_fit_df = self._transform_data_to_dataframe(
        selected_times, selected_geos, show_geo_level, include_baseline
    )

    # Specify custom colors to use to plot each metric category.
    domain = [c.EXPECTED, c.ACTUAL]
    colors = [c.BLUE_600, c.GREEN_300]
    if include_baseline:
      domain.append(c.BASELINE)
      colors.append(c.YELLOW_600)

    title = summary_text.EXPECTED_ACTUAL_OUTCOME_CHART_TITLE.format(
        outcome=outcome
    )
    if self._meridian.input_data.revenue_per_kpi is not None:
      y_axis_label = summary_text.REVENUE_LABEL
    else:
      y_axis_label = summary_text.KPI_LABEL
    plot = (
        alt.Chart(model_fit_df)
        .mark_line()
        .encode(
            x=alt.X(
                f'{c.TIME}:T',
                title='Time period',
                axis=alt.Axis(
                    format='%Y %b',
                    grid=False,
                    tickCount=8,
                    domainColor=c.GREY_300,
                ),
            ),
            y=alt.Y(
                f'{c.MEAN}:Q',
                title=y_axis_label,
                axis=alt.Axis(
                    ticks=False,
                    domain=False,
                    tickCount=5,
                    labelPadding=c.PADDING_10,
                    labelExpr=formatter.compact_number_expr(),
                    **formatter.Y_AXIS_TITLE_CONFIG,
                ),
            ),
            color=alt.Color(
                'type:N', scale=alt.Scale(domain=domain, range=colors)
            ),
        )
    )

    if include_ci:
      # Only add a confidence interval area for the modeled data.
      confidence_band = (
          alt.Chart(model_fit_df)
          .mark_area(opacity=0.3)
          .encode(
              x=f'{c.TIME}:T',
              y=f'{c.CI_HI}:Q',
              y2=f'{c.CI_LO}:Q',
              color=alt.Color(
                  'type:N',
                  scale=alt.Scale(domain=[domain[0]], range=[colors[0]]),
                  legend=None,
              ),
          )
      )
      plot = (plot + confidence_band).resolve_scale(color=c.INDEPENDENT)

    if show_geo_level:
      plot = plot.facet(column=alt.Column(f'{c.GEO}:O', sort=selected_geos))

    return plot.configure_axis(**formatter.TEXT_CONFIG).properties(
        title=formatter.custom_title_params(title)
    )

  def _validate_times_to_plot(
      self, selected_times: Sequence[str] | None = None
  ):
    """Validates the time dimensions."""
    time_dims = self.model_fit_data.time
    if selected_times and any(time not in time_dims for time in selected_times):
      raise ValueError(
          '`selected_times` should match the time dimensions from '
          'meridian.InputData.'
      )

  def _validate_geos_to_plot(
      self,
      selected_geos: Sequence[str] | None = None,
      n_top_largest_geos: int | None = None,
      show_geo_level: bool = False,
  ):
    """Validates the parameters related to the geo-level filtering for plotting.

    Args:
      selected_geos: Optional list of a subset of geo dimensions to include.
        Geos should match the geo dimension names from meridian.InputData. Only
        one of `selected_geos` and `n_top_largest_geos` can be specified.
      n_top_largest_geos: Optional number of largest geos by population to
        include. Only one of `selected_geos` and `n_top_largest_geos` can be
        specified.
      show_geo_level: Whether to plot at the geo-level. Only available if
        `selected_geos` or `n_top_largest_geos` is provided.
    """
    if show_geo_level and not selected_geos and not n_top_largest_geos:
      raise ValueError(
          'Geo-level plotting is only available when `selected_geos` or'
          ' `n_top_largest_geos` is specified.'
      )
    if selected_geos and n_top_largest_geos:
      raise ValueError(
          'Only one of `selected_geos` and `n_top_largest_geos` can be'
          ' specified.'
      )
    analyzed_geos = self.model_fit_data.geo
    if selected_geos:
      if any(geo not in analyzed_geos for geo in selected_geos):
        raise ValueError(
            '`selected_geos` should match the geo dimension names from '
            'meridian.InputData.'
        )
    if n_top_largest_geos:
      if n_top_largest_geos > len(analyzed_geos):
        raise ValueError(
            '`n_top_largest_geos` should be less than or equal to the total'
            f' number of geos: {len(analyzed_geos)}.'
        )

  def _transform_data_to_dataframe(
      self,
      selected_times: Sequence[str] | None = None,
      selected_geos: Sequence[str] | None = None,
      show_geo_level: bool = False,
      include_baseline: bool = True,
  ) -> pd.DataFrame:
    """Transforms the model fit data to a dataframe modified for plotting.

    Args:
      selected_times: Optional list of a subset of time dimensions to filter by.
      selected_geos: Optional list of a subset of geo dimensions to filter by.
      show_geo_level: If False, aggregates the geo-level data to national level.
      include_baseline: If False, removes the expected baseline outcome without
        any media execution.

    Returns:
      A data frame filtered based on the specifications.
    """
    times = selected_times or self.model_fit_data.time
    geos = selected_geos or self.model_fit_data.geo
    model_fit_df = (
        self.model_fit_data.sel(time=times, geo=geos)
        .to_dataframe()
        .reset_index()
        .melt(
            id_vars=[c.GEO, c.TIME, c.METRIC],
            value_vars=[
                c.EXPECTED,
                c.BASELINE,
                c.ACTUAL,
            ],
            var_name=c.TYPE,
            value_name=c.OUTCOME,
        )
        .pivot(
            index=[c.GEO, c.TIME, c.TYPE],
            columns=c.METRIC,
            values=c.OUTCOME,
        )
        .reset_index()
    )

    # Aggregate the data at the national level.
    if not show_geo_level:
      model_fit_df = model_fit_df.groupby([c.TIME, c.TYPE], as_index=False).sum(
          numeric_only=True
      )

    if not include_baseline:
      model_fit_df = model_fit_df[model_fit_df[c.TYPE] != c.BASELINE]

    return model_fit_df


class ReachAndFrequency:
  """Generates reach and frequency plots for the Meridian model.

  Plots the ROI by frequency for reach and frequency (RF) channels.
  """

  def __init__(
      self,
      meridian: model.Meridian,
      selected_times: Sequence[str] | None = None,
      use_kpi: bool | None = None,
  ):
    """Initializes the reach and frequency dataset for the model data.

    Args:
      meridian: Media mix model with the raw data from the model fitting.
      selected_times: Optional list containing a subset of times to include. By
        default, all time periods are included.
      use_kpi: If `True`, KPI is used instead of revenue.
    """
    self._meridian = meridian
    self._analyzer = analyzer.Analyzer(meridian)
    self._selected_times = selected_times
    # TODO Adapt the mechanisms to choose between KPI and REVENUE
    # from Analyzer.
    if use_kpi is None:
      self._use_kpi = (
          meridian.input_data.kpi_type == c.NON_REVENUE
          and meridian.input_data.revenue_per_kpi is None
      )
    else:
      self._use_kpi = use_kpi
    self._optimal_frequency_data = self._analyzer.optimal_freq(
        selected_times=selected_times,
        use_kpi=self._use_kpi,
    )

  @property
  def optimal_frequency_data(self) -> xr.Dataset:
    """Dataset holding the calculated optimal reach and frequency metrics.

    The dataset contains the following:

    * Coordinates: `frequency`, `rf_channel`, `metric` (`mean`, `ci_hi`,
      `ci_lo`)
    * Data variables: `roi`, `optimal_frequency`
    """
    return self._optimal_frequency_data

  def update_optimal_reach_and_frequency_selected_times(
      self, selected_times: Sequence[str] | None = None
  ):
    """Updates the selected times for optimal reach and frequency data.

    Args:
      selected_times: Optional list containing a subset of times to include. By
        default, all time periods are included.
    """
    self._selected_times = selected_times
    self._optimal_frequency_data = self._analyzer.optimal_freq(
        selected_times=selected_times,
        use_kpi=self._use_kpi,
    )

  def _transform_optimal_frequency_metrics(
      self, selected_channels: Sequence[str] | None = None
  ) -> pd.DataFrame | None:
    """Transforms the RF metrics for the optimal frequency plot.

    Args:
      selected_channels: Optional list of channels to include. If None, all RF
        channels are included.

    Returns:
      A DataFrame containing the weekly average frequency, mean ROI, and
      singularly valued optimal frequency per given channel.
    """
    selected_channels = (
        selected_channels
        if selected_channels
        else self.optimal_frequency_data.rf_channel.values
    )

    performance_by_frequency_df = (
        self.optimal_frequency_data[[c.ROI]]
        .sel(rf_channel=selected_channels)
        .drop_sel(metric=c.MEDIAN)
        .to_dataframe()
        .reset_index()
        .pivot(
            index=[c.RF_CHANNEL, c.FREQUENCY],
            columns=c.METRIC,
            values=c.ROI,
        )
        .reset_index()
        .rename(columns={c.MEAN: c.ROI})
    )
    optimal_freq_df = (
        self.optimal_frequency_data[[c.OPTIMAL_FREQUENCY]]
        .sel(rf_channel=selected_channels)
        .to_dataframe()
        .reset_index()
    )
    return performance_by_frequency_df.merge(optimal_freq_df, on=c.RF_CHANNEL)

  def plot_optimal_frequency(
      self,
      selected_channels: list[str] | None = None,
  ):
    """Plots the optimal frequency curves for selected channels.

    Args:
      selected_channels: Optional list of RF channels to plot.

    Returns:
      A faceted Altair plot showing a curve of the optimal frequency for the
      RF channels.
    """
    rf_channels = self.optimal_frequency_data.rf_channel.values
    if selected_channels and not all(
        item in rf_channels for item in selected_channels
    ):
      raise ValueError(
          'Channels specified are not in the list of all RF channels.'
      )

    optimal_frequency_df = self._transform_optimal_frequency_metrics(
        selected_channels
    )
    color_scale = alt.Scale(
        domain=[
            summary_text.OPTIMAL_FREQ_LABEL,
            summary_text.EXPECTED_ROI_LABEL,
        ],
        range=[c.BLUE_600, c.RED_600],
    )

    base = alt.Chart().transform_calculate(
        optimal_freq=f"'{summary_text.OPTIMAL_FREQ_LABEL}'",
        expected_roi=f"'{summary_text.EXPECTED_ROI_LABEL}'",
    )

    line = base.mark_line(strokeWidth=4).encode(
        x=alt.X(c.FREQUENCY, title='Weekly Average Frequency'),
        y=alt.Y(
            c.ROI,
            title=summary_text.ROI_LABEL,
            axis=alt.Axis(
                **formatter.Y_AXIS_TITLE_CONFIG,
            ),
        ),
        color=alt.Color('expected_roi:N', scale=color_scale, title=''),
    )

    vertical_optimal_freq = base.mark_rule(
        strokeWidth=3, strokeDash=[6, 6], color=c.BLUE_400
    ).encode(
        x=f'{c.OPTIMAL_FREQUENCY}:Q',
        color=alt.Color(f'{c.OPTIMAL_FREQ}:N', scale=color_scale, title=''),
    )

    label_text = vertical_optimal_freq.mark_text(
        align='left',
        dx=5,
        dy=-5,
        fontSize=c.AXIS_FONT_SIZE,
        font=c.FONT_ROBOTO,
        fontWeight='lighter',
    ).encode(
        text=alt.value(summary_text.OPTIMAL_FREQ_LABEL),
        color=alt.value(c.BLACK_100),
    )

    label_freq = vertical_optimal_freq.mark_text(
        align='left',
        dx=110,
        dy=-5,
        fontSize=c.AXIS_FONT_SIZE,
        font=c.FONT_ROBOTO,
        fontWeight='lighter',
    ).encode(
        text=alt.Text(f'{c.OPTIMAL_FREQUENCY}:Q', format='.2f'),
        color=alt.value(c.BLACK_100),
    )

    return (
        alt.layer(
            line,
            vertical_optimal_freq,
            label_text,
            label_freq,
            data=optimal_frequency_df,
        )
        .facet(column=alt.Column(f'{c.RF_CHANNEL}:N', title=None))
        .properties(
            title=formatter.custom_title_params(
                summary_text.OPTIMAL_FREQUENCY_CHART_TITLE.format(
                    metric=summary_text.ROI_LABEL
                )
            )
        )
        .resolve_scale(x=c.INDEPENDENT, y=c.INDEPENDENT)
        .configure_axis(**formatter.TEXT_CONFIG)
    )


class MediaEffects:
  """Generates media effects plots for the Meridian model.

  Plots incremental outcome and effectiveness for all channels.
  """

  def __init__(
      self,
      meridian: model.Meridian,
      by_reach: bool = True,
  ):
    """Initializes the Media Effects based on the model data and params.

    Args:
      meridian: Media mix model with the raw data from the model fitting.
      by_reach: For the channel w/ reach and frequency, return the response
        curves by reach given fixed frequency if true; return the response
        curves by frequency given fixed reach if false.
    """
    self._meridian = meridian
    self._analyzer = analyzer.Analyzer(meridian)
    self._by_reach = by_reach

  @functools.lru_cache(maxsize=128)
  def response_curves_data(
      self,
      confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL,
      selected_times: frozenset[str] | None = None,
      by_reach: bool = True,
  ) -> xr.Dataset:
    """Dataset holding the calculated response curves data.

    The dataset contains the following:

    - **Coordinates:** `media`, `metric` (`mean`, `ci_hi`, `ci_lo`),
      `spend_multiplier`
    - **Data variables:** `spend`, `incremental_outcome`, `roi`

    Args:
      confidence_level: Confidence level for modeled response credible
        intervals, represented as a value between zero and one. Default is 0.9.
      selected_times: Optional list of a subset of time dimensions to include.
        By default, all times are included. Times should match the time
        dimensions from `meridian.InputData`.
      by_reach: For the channel w/ reach and frequency, return the response
        curves by reach given fixed frequency if true; return the response
        curves by frequency given fixed reach if false.

    Returns:
      A Dataset displaying the response curves data.
    """
    selected_times_list = list(selected_times) if selected_times else None
    return self._analyzer.response_curves(
        spend_multipliers=list(np.arange(0, 2.2, c.RESPONSE_CURVE_STEP_SIZE)),
        confidence_level=confidence_level,
        selected_times=selected_times_list,
        by_reach=by_reach,
    )

  @functools.lru_cache(maxsize=128)
  def adstock_decay_dataframe(
      self, confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL
  ) -> pd.DataFrame:
    """A DataFrame holding the calculated Adstock decay metrics.

    The DataFrame contains the following columns:

      `time_units`, `channel`, `distribution`, `mean`, `ci_lo`, `ci_hi`.
    Args:
      confidence_level: Confidence level for modeled adstock decay credible
        intervals, represented as a value between zero and one. Default is 0.9.

    Returns:
      A DataFrame displaying the adstock decay metrics.
    """
    return self._analyzer.adstock_decay(confidence_level=confidence_level)

  @functools.lru_cache(maxsize=128)
  def hill_curves_dataframe(
      self, confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL
  ) -> pd.DataFrame:
    """A DataFrame holding the calculated Hill curve metrics.

    The DataFrame contains the following columns:
      `channel`, `media_units`, `ci_hi`, `ci_lo`, `mean`, `channel_type`,
      `scaled_count_histogram`, `start_interval_histogram`,
      `end_interval_histogram`.
    Args:
      confidence_level: Confidence level for modeled hill curves credible
        intervals, represented as a value between zero and one. Default is 0.9.

    Returns:
      Hill curves `pd.DataFrame` with columns:

      *   `channel`: `media` or `rf` channel name.
      *   `media_units`: Media (for `media` channels) or average frequency (for
          `rf` channels) units.
      *   `distribution`: Indication of `posterior` or `prior` draw.
      *   `ci_hi`: Upper bound of the credible interval of the value of the Hill
          function.
      *   `ci_lo`: Lower bound of the credible interval of the value of the Hill
          function.
      *   `mean`: Point-wise mean of the value of the Hill function per draw.
      *   `channel_type`: Indication of a `media` or `rf` channel.
      *   `scaled_count_histogram`: Scaled count of media units or average
          frequencies within the bin.
      *   `count_histogram`: Count value of media units or average
          frequencies within the bin.
      *   `start_interval_histogram`: Media unit or average frequency starting
          point for a histogram bin.
      *   `end_interval_histogram`: Media unit or average frequency ending point
          for a histogram bin.
    """
    return self._analyzer.hill_curves(confidence_level=confidence_level)

  def plot_response_curves(
      self,
      confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL,
      selected_times: frozenset[str] | None = None,
      by_reach: bool = True,
      plot_separately: bool = True,
      include_ci: bool = True,
      num_channels_displayed: int | None = None,
  ) -> alt.Chart:
    """Plots the response curves for each channel.

    To avoid congestion when the channels are plotted in the same graph, we cap
    the number of channels that can be displayed visually on the graph to 7
    channels maximum. If the num_channels_displayed is greater than the total
    number of channels in the dataset, the total number of channels in the
    dataset is displayed.

    Args:
      confidence_level: Confidence level to update to for the response curve
        credible intervals, represented as a value between zero and one.
      selected_times: Optional list containing a subset of times to include. By
        default, all time periods are included.
      by_reach: For the channel w/ reach and frequency, return the response
        curves by reach given fixed frequency if true; return the response
        curves by frequency given fixed reach if false.
      plot_separately: If `True`, the plots are faceted. If `False`, the plots
        are layered to create one plot with all of the channels.
      include_ci: If `True`, plots the credible interval. Defaults to `True`.
      num_channels_displayed: Number of channels to show on the layered plot. If
        plotting a faceted chart, this value is ignored.

    Returns:
      An Altair plot showing the response curves per channel.
    """

    total_num_channels = len(self._meridian.input_data.get_all_channels())
    if plot_separately:
      title = summary_text.RESPONSE_CURVES_CHART_TITLE.format(top_channels='')
      num_channels_displayed = total_num_channels
    else:
      max_num_channels = min(total_num_channels, 10)
      if num_channels_displayed is None:
        if total_num_channels >= 7:
          num_channels_displayed = 7  # default value to display
        else:
          num_channels_displayed = max_num_channels

      if num_channels_displayed > max_num_channels:
        num_channels_displayed = max_num_channels
      if num_channels_displayed < 1:
        num_channels_displayed = 1
      title = summary_text.RESPONSE_CURVES_CHART_TITLE.format(
          top_channels=f'(top {num_channels_displayed})'
      )

    response_curves_df = self._transform_response_curve_metrics(
        num_channels_displayed,
        confidence_level=confidence_level,
        selected_times=selected_times,
        by_reach=by_reach,
    )
    if self._meridian.input_data.revenue_per_kpi is not None:
      y_axis_label = summary_text.INC_OUTCOME_LABEL
    else:
      y_axis_label = summary_text.INC_KPI_LABEL
    base = (
        alt.Chart(response_curves_df)
        .transform_calculate(
            spend_level=(
                'datum.spend_multiplier >= 1.0 ? "Above current spend" : "Below'
                ' current spend"'
            )
        )
        .encode(
            x=alt.X(
                f'{c.SPEND}:Q',
                title=summary_text.SPEND_LABEL,
                axis=alt.Axis(
                    labelExpr=formatter.compact_number_expr(),
                    **formatter.AXIS_CONFIG,
                ),
            ),
            y=alt.Y(
                f'{c.MEAN}:Q',
                title=y_axis_label,
                axis=alt.Axis(
                    labelExpr=formatter.compact_number_expr(),
                    **formatter.Y_AXIS_TITLE_CONFIG,
                ),
            ),
            color=f'{c.CHANNEL}:N',
        )
    )

    line = base.mark_line().encode(
        strokeDash=alt.StrokeDash(
            f'{c.SPEND_LEVEL}:N',
            sort='descending',
            legend=alt.Legend(title=None),
        )
    )

    historic_spend_point = (
        base.mark_point(filled=True, size=c.POINT_SIZE, opacity=1)
        .encode(
            tooltip=[c.SPEND, c.MEAN],
            shape=alt.Shape(
                f'{c.CURRENT_SPEND}:N', legend=alt.Legend(title=None)
            ),
        )
        .transform_filter(alt.datum.spend_multiplier == 1.0)
    )
    if plot_separately:
      define_color = alt.Color(f'{c.CHANNEL}:N', legend=None)
    else:
      define_color = alt.Color(f'{c.CHANNEL}:N')

    band = base.mark_area(opacity=0.5).encode(
        x=f'{c.SPEND}:Q',
        y=f'{c.CI_LO}:Q',
        y2=f'{c.CI_HI}:Q',
        color=define_color,
    )

    if include_ci:
      plot = alt.layer(line, historic_spend_point, band)
    else:
      plot = alt.layer(line, historic_spend_point)
    if plot_separately:
      plot = plot.facet(c.CHANNEL, columns=3).resolve_scale(
          x=c.INDEPENDENT, y=c.INDEPENDENT
      )

    return plot.properties(
        title=formatter.custom_title_params(title)
    ).configure_axis(**formatter.TEXT_CONFIG)

  def plot_adstock_decay(
      self,
      confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL,
      include_ci: bool = True,
  ):
    """Plots the Adstock decay for each channel.

    Args:
      confidence_level: Confidence level to update to for the adstock decay
        credible intervals, represented as a value between zero and one.
      include_ci: If `True`, plots the credible interval. Defaults to `True`.

    Returns:
      An Altair plot showing the Adstock decay prior and posterior per media.
    """
    dataframe = self.adstock_decay_dataframe(confidence_level=confidence_level)
    base = alt.Chart(dataframe)

    scaled_confidence_level = int(confidence_level * 100)

    posterior_label = f'posterior ({scaled_confidence_level}% CI)'
    prior_label = f'prior ({scaled_confidence_level}% CI)'

    color_scale = alt.Scale(
        domain=[c.PRIOR, c.POSTERIOR],
        range=[c.RED_600, c.BLUE_700],
    )

    prior_posterior_line = base.mark_line().encode(
        x=alt.X(f'{c.TIME_UNITS}:Q', title='Time Units'),
        y=alt.Y(f'{c.MEAN}:Q', title='Effect'),
        color=alt.Color(
            c.DISTRIBUTION,
            scale=color_scale,
            legend=alt.Legend(
                title='',
                labelExpr=(
                    f'datum.value === "posterior" ? "{posterior_label}" :'
                    f' "{prior_label}"'
                ),
            ),
        ),
    )

    prior_posterior_band = base.mark_area(opacity=0.2).encode(
        x=f'{c.TIME_UNITS}:Q',
        y=f'{c.CI_LO}:Q',
        y2=f'{c.CI_HI}:Q',
        color=alt.Color(c.DISTRIBUTION, scale=color_scale),
    )

    discrete_value_points = (
        base.mark_circle(filled=True, size=c.POINT_SIZE, opacity=1)
        .encode(
            tooltip=[c.TIME_UNITS, c.MEAN],
            x=alt.X(f'{c.TIME_UNITS}:Q'),
            y=alt.Y(f'{c.MEAN}:Q'),
            color=alt.Color(c.DISTRIBUTION, scale=color_scale),
        )
        .transform_filter(alt.datum.is_int_time_unit)
    )

    if include_ci:
      plot = alt.layer(
          prior_posterior_line,
          prior_posterior_band,
          discrete_value_points,
      )
    else:
      plot = alt.layer(prior_posterior_line, discrete_value_points)

    return (
        plot.facet(c.CHANNEL, columns=3)
        .properties(
            title=formatter.custom_title_params(
                summary_text.ADSTOCK_DECAY_CHART_TITLE
            )
        )
        .configure_axis(**formatter.TEXT_CONFIG)
        .resolve_scale(x=c.INDEPENDENT, y=c.INDEPENDENT)
    )

  def plot_hill_curves(
      self,
      confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL,
      include_prior: bool = True,
      include_ci: bool = True,
  ) -> alt.Chart | list[alt.Chart]:
    """Plots the Hill curves for each channel.

    Args:
      confidence_level: Confidence level to update to for the hill curves
        credible intervals, represented as a value between zero and one.
      include_prior: If `True`, plots contain both the prior and posterior.
        Defaults to `True`.
      include_ci: If `True`, plots the credible interval. Defaults to `True`.

    Returns:
      A faceted Altair plot showing the histogram, prior and posterior lines,
      and bands for the Hill saturation curves. When there are both media and
      RF channels, a list of 2 faceted Altair plots are returned: one
      for the media channels and another for the RF channels.
    """
    hill_curves_dataframe = self.hill_curves_dataframe(
        confidence_level=confidence_level
    )
    channel_types = list(set(hill_curves_dataframe[c.CHANNEL_TYPE]))
    plot_media, plot_rf = None, None

    if c.MEDIA in channel_types:
      media_df = hill_curves_dataframe[
          hill_curves_dataframe[c.CHANNEL_TYPE] == c.MEDIA
      ]
      plot_media = self._plot_hill_curves_helper(
          media_df, include_prior, include_ci
      )

    if c.RF in channel_types:
      rf_df = hill_curves_dataframe[
          hill_curves_dataframe[c.CHANNEL_TYPE] == c.RF
      ]
      plot_rf = self._plot_hill_curves_helper(rf_df, include_prior, include_ci)

    if plot_media and plot_rf:
      return [plot_media, plot_rf]
    elif plot_media:
      return plot_media
    else:
      return plot_rf

  def _plot_hill_curves_helper(
      self,
      df_channel_type: pd.DataFrame,
      include_prior: bool = True,
      include_ci: bool = True,
  ) -> alt.Chart:
    """Plots Hill Curves with unique x-axis label based on channel type.

    Args:
      df_channel_type: Pandas DataFrame either used to plot channels or R+F
        channels.
      include_prior: If True, plots contain both the prior and posterior.
        Defaults to True.
      include_ci: If True, plots the credible interval. Defaults to True.

    Returns:
      A faceted Altair plot showing the histogram and prior+posterior lines and
      bands for the Hill curves.
    """
    if c.MEDIA in list(df_channel_type[c.CHANNEL_TYPE]):
      x_axis_title = 'Media Units per Capita'
      shaded_area_title = summary_text.HILL_SHADED_REGION_MEDIA_LABEL
    else:
      x_axis_title = 'Average Frequency'
      shaded_area_title = summary_text.HILL_SHADED_REGION_RF_LABEL
    domain_list = [
        c.POSTERIOR,
        c.PRIOR,
        shaded_area_title,
    ]
    range_list = [c.BLUE_700, c.RED_600, c.GREY_600]
    if not include_prior:
      df_channel_type = df_channel_type[
          df_channel_type[c.DISTRIBUTION] == c.POSTERIOR
      ]
      domain_list = [
          c.POSTERIOR,
          shaded_area_title,
      ]
      range_list = [c.BLUE_700, c.GREY_600]

    base = alt.Chart(df_channel_type)
    color_scale = alt.Scale(
        domain=domain_list,
        range=range_list,
    )
    prior_posterior_line = base.mark_line().encode(
        x=alt.X(
            f'{c.MEDIA_UNITS}:Q',
            title=x_axis_title,
            scale=alt.Scale(nice=False),
        ),
        y=alt.Y(f'{c.MEAN}:Q', title=summary_text.HILL_Y_AXIS_LABEL),
        color=alt.Color(f'{c.DISTRIBUTION}:N', scale=color_scale),
    )
    prior_posterior_band = base.mark_area(opacity=0.3).encode(
        x=f'{c.MEDIA_UNITS}:Q',
        y=f'{c.CI_LO}:Q',
        y2=f'{c.CI_HI}:Q',
        color=alt.Color(f'{c.DISTRIBUTION}:N', scale=color_scale),
    )
    histogram = base.mark_bar(color=c.GREY_600, opacity=0.4).encode(
        x=f'{c.START_INTERVAL_HISTOGRAM}:Q',
        x2=f'{c.END_INTERVAL_HISTOGRAM}:Q',
        y=alt.Y(f'{c.SCALED_COUNT_HISTOGRAM}:Q'),
    )
    if include_ci:
      plot = alt.layer(
          histogram,
          prior_posterior_line,
          prior_posterior_band,
          data=df_channel_type,
      )
    else:
      plot = alt.layer(
          histogram,
          prior_posterior_line,
          data=df_channel_type,
      )
    return (
        plot.facet(f'{c.CHANNEL}:N', columns=3)
        .properties(
            title=formatter.custom_title_params(
                summary_text.HILL_SATURATION_CHART_TITLE
            )
        )
        .configure_axis(**formatter.TEXT_CONFIG)
        .configure_legend(labelLimit=0)
        .resolve_scale(x=c.INDEPENDENT, y=c.INDEPENDENT)
    )

  def _transform_response_curve_metrics(
      self,
      num_channels: int | None = None,
      selected_times: frozenset[str] | None = None,
      confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL,
      by_reach: bool = True,
  ) -> pd.DataFrame:
    """Returns DataFrame with top channels by spend for the layered plot.

    Args:
      num_channels: Optional number of top channels by spend to include. By
        default, all channels are included.
      selected_times: Optional list of a subset of time dimensions to include.
        By default, all times are included. Times should match the time
        dimensions from `meridian.InputData`.
      confidence_level: Confidence level to update to for the response curve
        credible intervals, represented as a value between zero and one.
      by_reach: For the channel w/ reach and frequency, return the response
        curves by reach given fixed frequency if true; return the response
        curves by frequency given fixed reach if false.

    Returns:
      A DataFrame containing the top chosen channels
      num_channels, ordered by the spend, with the columns being
      channel, spend, spend_multiplier, ci_hi, ci_lo and incremental_outcome
    """
    data = self.response_curves_data(
        confidence_level=confidence_level,
        selected_times=selected_times,
        by_reach=by_reach,
    )
    list_sorted_channels_cost = list(
        data.sel(spend_multiplier=1)
        .sortby(c.SPEND, ascending=False)[c.CHANNEL]
        .values
    )

    df = (
        data[[c.SPEND, c.INCREMENTAL_OUTCOME]]
        .sel(channel=list_sorted_channels_cost[:num_channels])
        .to_dataframe()
        .reset_index()
        .pivot(
            index=[
                c.CHANNEL,
                c.SPEND,
                c.SPEND_MULTIPLIER,
            ],
            columns=c.METRIC,
            values=c.INCREMENTAL_OUTCOME,
        )
        .reset_index()
    )
    df[c.CURRENT_SPEND] = np.where(
        df[c.SPEND_MULTIPLIER] == 1.0,
        summary_text.CURRENT_SPEND_LABEL,
        pd.NA,
    )
    return df


class MediaSummary:
  """Generates media summary metrics plots for the Meridian model.

  Calculates the mean and credible intervals (CI) for each channel's outcome
  metrics (incremental outcome, contribution, ROI, mROI, effectiveness) and
  media summary metrics (impressions, spend). Metrics are calculated at the
  national-level. These are used for various plots displaying these metrics.
  """

  def __init__(
      self,
      meridian: model.Meridian,
      confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL,
      selected_times: Sequence[str] | None = None,
      marginal_roi_by_reach: bool = True,
      non_media_baseline_values: Sequence[str | float] | None = None,
  ):
    """Initializes the media summary metrics based on the model data and params.

    Args:
      meridian: Media mix model with the raw data from the model fitting.
      confidence_level: Confidence level for media summary metrics credible
        intervals, represented as a value between zero and one.
      selected_times: Optional list containing a subset of times to include. By
        default, all time periods are included.
      marginal_roi_by_reach: Boolean. Marginal ROI is defined as the return on
        the next dollar spent.  If this argument is True, then we assume the
        next dollar spent only impacts reach, holding frequency constant.  If
        this argument is False, we assume the next dollar spent only impacts
        frequency, holding reach constant.
      non_media_baseline_values: Optional list of shape (n_non_media_channels,).
        Each element is either a float (which means that the fixed value will be
        used as baseline for the given channel) or one of the strings "min" or
        "max" (which mean that the global minimum or maximum value will be used
        as baseline for the values of the given non_media treatment channel). If
        None, the minimum value is used as baseline for each non_media treatment
        channel.
    """
    self._meridian = meridian
    self._analyzer = analyzer.Analyzer(meridian)
    self._confidence_level = confidence_level
    self._selected_times = selected_times
    self._marginal_roi_by_reach = marginal_roi_by_reach
    self._non_media_baseline_values = non_media_baseline_values

  @functools.cached_property
  def paid_summary_metrics(self) -> xr.Dataset:
    """Dataset holding the calculated summary metrics for the paid channels.

    The dataset contains the following:

    - **Coordinates:** `channel`, `metric` (`mean`, `median`, `ci_lo`, `ci_hi`),
      `distribution` (`prior`, `posterior`)
    - **Data variables:** `impressions`, `pct_of_impressions`, `spend`,
      `pct_of_spend`, `CPM`, `incremental_outcome`, `pct_of_contribution`,
      `roi`,
      `effectiveness`, `mroi`.
    """
    return self._analyzer.summary_metrics(
        selected_times=self._selected_times,
        marginal_roi_by_reach=self._marginal_roi_by_reach,
        use_kpi=self._meridian.input_data.revenue_per_kpi is None,
        confidence_level=self._confidence_level,
        include_non_paid_channels=False,
    )

  @functools.cached_property
  def all_summary_metrics(self) -> xr.Dataset:
    """Dataset holding the calculated summary metrics for all channels.

    The dataset contains the following:

    - **Coordinates:** `channel`, `metric` (`mean`, `median`, `ci_lo`, `ci_hi`),
      `distribution` (`prior`, `posterior`)
    - **Data variables:** `incremental_outcome`, `pct_of_contribution`,
      `effectiveness`.
    """
    return self._analyzer.summary_metrics(
        selected_times=self._selected_times,
        use_kpi=self._meridian.input_data.revenue_per_kpi is None,
        confidence_level=self._confidence_level,
        include_non_paid_channels=True,
        non_media_baseline_values=self._non_media_baseline_values,
    )

  def summary_table(
      self,
      include_prior: bool = True,
      include_posterior: bool = True,
      include_non_paid_channels: bool = False,
  ) -> pd.DataFrame:
    """Returns a formatted dataframe table of the summary metrics.

    Mean and credible interval summary metrics are formatted as text.

    Args:
      include_prior: If True, prior distribution summary metrics are included.
        One of `include_prior` and `include_posterior` must be True.
      include_posterior: If True, posterior distribution summary metrics are
        included. One of `include_prior` and `include_posterior` must be True.
      include_non_paid_channels: Boolean. If `True`, non-paid channels (organic
        media, organic reach and frequency, and non-media treatments) are
        included in the summary but only the metrics independent of spend are
        reported. If `False`, only the paid channels (media, reach and
        frequency) are included but the summary contains also the metrics
        dependent on spend. Default: `False`.

    Returns:
      pandas.DataFrame of formatted summary metrics.
    """
    if not (include_posterior or include_prior):
      raise ValueError(
          'At least one of `include_posterior` or `include_prior` must be True.'
      )

    use_revenue = self._meridian.input_data.revenue_per_kpi is not None
    distribution = [c.PRIOR] * include_prior + [c.POSTERIOR] * include_posterior

    percentage_metrics = [
        c.PCT_OF_CONTRIBUTION,
    ]
    if include_non_paid_channels:
      monetary_metrics = [c.INCREMENTAL_OUTCOME] * use_revenue
      summary_metrics = self.all_summary_metrics
      columns_rename_dict = {
          c.PCT_OF_CONTRIBUTION: summary_text.PCT_CONTRIBUTION_COL,
          c.INCREMENTAL_OUTCOME: (
              summary_text.INC_OUTCOME_COL
              if use_revenue
              else summary_text.INC_KPI_COL
          ),
      }
      df = (
          summary_metrics.sel(distribution=distribution)
          .drop_sel(metric=c.MEDIAN)
          .to_dataframe()
          .rename({c.MEAN: 'central_tendency'})
      )
    else:  # not include_non_paid_channels
      percentage_metrics.extend([
          c.PCT_OF_IMPRESSIONS,
          c.PCT_OF_SPEND,
      ])
      monetary_metrics = [c.CPM, c.CPIK] + [
          c.SPEND,
          c.INCREMENTAL_OUTCOME,
      ] * use_revenue
      summary_metrics = self.paid_summary_metrics
      columns_rename_dict = {
          c.PCT_OF_IMPRESSIONS: summary_text.PCT_IMPRESSIONS_COL,
          c.PCT_OF_SPEND: summary_text.PCT_SPEND_COL,
          c.PCT_OF_CONTRIBUTION: summary_text.PCT_CONTRIBUTION_COL,
          c.INCREMENTAL_OUTCOME: (
              summary_text.INC_OUTCOME_COL
              if use_revenue
              else summary_text.INC_KPI_COL
          ),
      }
      # Format CPIK to use median instead of mean.
      df_mean = (
          summary_metrics.drop_vars([c.CPIK])
          .sel(distribution=distribution)
          .drop_sel(metric=c.MEDIAN)
          .to_dataframe()
          .rename({c.MEAN: 'central_tendency'})
      )
      df_median = (
          summary_metrics[c.CPIK]
          .sel(distribution=distribution)
          .drop_sel(metric=c.MEAN)
          .to_dataframe()
          .rename({c.MEDIAN: 'central_tendency'})
      )
      df = pd.concat([df_mean, df_median], axis=1)

    data_vars = summary_metrics.data_vars
    digits = {k: 1 if min(abs(df[k])) < 1 else 0 for k in list(data_vars)}
    digits[c.EFFECTIVENESS] = 2
    for k, v in digits.items():
      df[k] = df[k].apply(f'{{0:,.{v}f}}'.format)

    # Format percentages.
    for k in percentage_metrics:
      df[k] = df[k].astype(str) + '%'

    # Format monetary values.
    for k in monetary_metrics:
      if k in df.columns:
        df[k] = '$' + df[k].astype(str)

    # Format the model result data variables as central_tendency (ci_lo, ci_hi).
    index_vars = [c.CHANNEL, c.DISTRIBUTION]
    input_data = [k for k, v in data_vars.items() if len(v.shape) == 1]
    return (
        df.groupby(index_vars + input_data, sort=False)
        .aggregate(lambda g: f'{g[0]} ({g[1]}, {g[2]})')
        .reset_index()
        .rename(columns=columns_rename_dict)
    )

  def update_summary_metrics(
      self,
      confidence_level: float | None = None,
      selected_times: Sequence[str] | None = None,
      marginal_roi_by_reach: bool = True,
      non_media_baseline_values: Sequence[str | float] | None = None,
  ):
    """Runs the computation for the media summary metrics with new parameters.

    Args:
      confidence_level: Confidence level to update to for the media summary
        metrics credible intervals, represented as a value between zero and one.
        If `None`, the current confidence level is used.
      selected_times: Optional list containing a subset of times to include. If
        `None`, all time periods are included.
      marginal_roi_by_reach: Boolean. Marginal ROI is defined as the return on
        the next dollar spent.  If `True`, then the assumptions is the next
        dollar spent only impacts reach, holding frequency constant. If `False`,
        the assumption is the next dollar spent only impacts frequency, holding
        reach constant.
      non_media_baseline_values: Optional list of shape (n_non_media_channels,).
        Each element is either a float (which means that the fixed value will be
        used as baseline for the given channel) or one of the strings "min" or
        "max" (which mean that the global minimum or maximum value will be used
        as baseline for the values of the given non_media treatment channel). If
        None, the minimum value is used as baseline for each non_media treatment
        channel.
    """
    self._confidence_level = confidence_level or self._confidence_level
    self._selected_times = selected_times
    self._marginal_roi_by_reach = marginal_roi_by_reach
    self._non_media_baseline_values = non_media_baseline_values

  def plot_contribution_waterfall_chart(self) -> alt.Chart:
    """Plots a waterfall chart of the contribution share per channel.

    Returns:
      An Altair plot showing the contributions per channel.
    """
    outcome = (
        c.REVENUE.title()
        if self._meridian.input_data.revenue_per_kpi is not None
        else c.KPI.upper()
    )
    outcome_df = self._transform_contribution_metrics(include_non_paid=True)
    pct = c.PCT_OF_CONTRIBUTION
    value = c.INCREMENTAL_OUTCOME
    outcome_df['outcome_text'] = outcome_df.apply(
        lambda x: formatter.format_number_text(x[pct], x[value]),
        axis=1,
    )
    outcome_df[c.CHANNEL] = outcome_df[c.CHANNEL].str.upper()

    num_channels = len(outcome_df[c.CHANNEL])

    base = (
        alt.Chart(outcome_df)
        .transform_window(
            sum_outcome=f'sum({c.PCT_OF_CONTRIBUTION})',
            kwargs=f'lead({c.CHANNEL})',
        )
        .transform_calculate(
            prev_sum=f'datum.sum_outcome - datum.{c.PCT_OF_CONTRIBUTION}',
            text_x=(
                'datum.incremental_outcome < 0 ? datum.prev_sum :'
                ' datum.sum_outcome'
            ),
        )
        .encode(
            y=alt.Y(
                f'{c.CHANNEL}:N',
                axis=alt.Axis(
                    ticks=False, labelPadding=c.PADDING_10, domain=False
                ),
                title=None,
                sort=None,
                scale=alt.Scale(paddingOuter=c.SCALED_PADDING),
            )
        )
    )
    x_axis_label = f'% {outcome}'
    bar = base.mark_bar(size=c.BAR_SIZE).encode(
        x=alt.X(
            'prev_sum:Q',
            title=x_axis_label,
            axis=alt.Axis(
                ticks=False,
                tickCount=5,
                format='%',
                domain=False,
                labelPadding=c.PADDING_10,
            ),
        ),
        x2='sum_outcome:Q',
        color=alt.condition(
            alt.datum.channel == c.BASELINE.upper(),
            alt.value(c.YELLOW_600),
            alt.value(c.BLUE_700),
        ),
    )
    text = base.mark_text(
        align='left',
        dx=c.PADDING_10,
        fontSize=c.TEXT_FONT_SIZE,
        color=c.GREY_700,
    ).encode(
        text='outcome_text',
        x='text_x:Q',
    )
    return (
        (bar + text)
        .properties(
            title=formatter.custom_title_params(
                summary_text.CHANNEL_DRIVERS_CHART_TITLE
            ),
            height=c.BAR_SIZE * num_channels
            + c.BAR_SIZE * 2 * c.SCALED_PADDING,
            width=500,
        )
        .configure_axis(titlePadding=c.PADDING_10, **formatter.TEXT_CONFIG)
        .configure_view(strokeOpacity=0)
    )

  def plot_contribution_pie_chart(self) -> alt.Chart:
    """Plots a pie chart of the total contributions from channels.

    Returns:
      An Altair plot showing the contributions for all channels.
    """
    outcome_df = self._transform_contribution_metrics(
        [c.ALL_CHANNELS], include_non_paid=True
    )

    domain = [c.BASELINE, c.ALL_CHANNELS]
    colors = [c.YELLOW_600, c.BLUE_700]
    base = alt.Chart(outcome_df).encode(
        alt.Theta(f'{c.PCT_OF_CONTRIBUTION}:Q', stack=True),
        alt.Color(
            f'{c.CHANNEL}:N',
            scale=alt.Scale(domain=domain, range=colors),
            legend=alt.Legend(
                orient='none',
                direction='horizontal',
                legendX=130,
                legendY=320,
                labelFontSize=c.AXIS_FONT_SIZE,
                labelFont=c.FONT_ROBOTO,
                title=None,
            ),
        ),
    )
    pie = base.mark_arc(outerRadius=150, innerRadius=70)
    text = base.mark_text(
        radius=110,
        fill='white',
        size=c.TITLE_FONT_SIZE,
        font=c.FONT_ROBOTO,
    ).encode(text=alt.Text(f'{c.PCT_OF_CONTRIBUTION}:Q', format='.0%'))
    return (
        alt.layer(pie, text, data=outcome_df)
        .configure_view(stroke=None)
        .properties(
            title=formatter.custom_title_params(
                summary_text.CONTRIBUTION_CHART_TITLE
            )
        )
    )

  def plot_spend_vs_contribution(self) -> alt.Chart:
    """Plots a bar chart comparing spend versus contribution shares per channel.

    This compares the spend and contribution percentages for each channel, and
    the ROI per channel. The contribution percentages are out of the total
    media-driven outcome amount.

    Returns:
      An Altair plot showing the spend versus outcome percentages per channel.
    """
    outcome = (
        c.REVENUE
        if self._meridian.input_data.revenue_per_kpi is not None
        else c.KPI.upper()
    )
    df = self._transform_contribution_spend_metrics()
    domain = [
        f'% {outcome.title() if outcome == c.REVENUE else outcome}',
        '% Spend',
    ]
    title = summary_text.SPEND_OUTCOME_CHART_TITLE.format(outcome=outcome)
    colors = [c.BLUE_400, c.BLUE_200]
    domain.append('Return on Investment')
    colors.append(c.GREEN_700)
    spend_outcome = (
        alt.Chart()
        .mark_bar(cornerRadiusEnd=2, tooltip=True)
        .encode(
            tooltip=alt.Tooltip([f'{c.PCT}:Q'], format='.1%'),
            x=alt.X(
                'label:N',
                axis=alt.Axis(title=None, labels=False, ticks=False),
                scale=alt.Scale(paddingOuter=0.5),
            ),
            y=alt.Y(
                f'{c.PCT}:Q',
                axis=alt.Axis(
                    format='%', tickCount=2, **formatter.Y_AXIS_TITLE_CONFIG
                ),
            ),
            color=alt.Color(
                'label:N',
                scale=alt.Scale(
                    domain=domain,
                    range=colors,
                ),
                legend=alt.Legend(
                    orient='bottom',
                    title=None,
                    columnPadding=c.PADDING_20,
                    rowPadding=c.PADDING_10,
                ),
            ),
        )
    )
    roi_marker = (
        alt.Chart()
        .mark_tick(
            color=c.GREEN_700,
            thickness=4,
            cornerRadius=c.CORNER_RADIUS,
            size=c.PADDING_20,
            tooltip=True,
        )
        .encode(
            tooltip=alt.Tooltip([f'{c.ROI}:Q'], format='.2f'),
            y=alt.Y(f'{c.ROI_SCALED}:Q', title='%'),
        )
    )
    roi_text = (
        alt.Chart()
        .mark_text(
            dy=-15,
            fontSize=c.AXIS_FONT_SIZE,
            color=c.GREY_900,
        )
        .encode(
            text=alt.Text(f'{c.ROI}:Q', format='.1f'),
            y=f'{c.ROI_SCALED}:Q',
        )
    )
    layer = alt.layer(spend_outcome, roi_marker, roi_text, data=df)

    # To group the outcome and spend bar plot with the ROI markers, facet the
    # layered plot by channel. This creates separate plots per channel.
    # To appear as 1 plot, remove the spacing between the facets and remove the
    # border outlines.
    return (
        layer.facet(
            column=alt.Column(
                f'{c.CHANNEL}:N',
            ),
            spacing=-1,  # Combine the facets to appear as 1 unfaceted plot.
        )
        .properties(title=formatter.custom_title_params(title))
        .configure_header(
            title=None,
            labelOrient='bottom',
            labelAngle=-45,
            labelAlign='right',
            labelBaseline='middle',
        )
        .configure_axis(titlePadding=c.PADDING_10, **formatter.TEXT_CONFIG)
        .configure_view(strokeOpacity=0)  # Remove facet outlines.
    )

  def plot_roi_bar_chart(self, include_ci: bool = True) -> alt.Chart:
    """Plots the ROI bar chart for each channel.

    Args:
      include_ci: If `True`, plots the credible interval. Defaults to `True`.

    Returns:
      An Altair plot showing the ROI per channel.
    """
    if include_ci:
      ci = int(self._confidence_level * 100)
      title = summary_text.ROI_CHANNEL_CHART_TITLE_FORMAT.format(
          ci=f'with {ci}% credible interval'
      )
    else:
      title = summary_text.ROI_CHANNEL_CHART_TITLE_FORMAT.format(ci='')
    return self._plot_metric_bar_chart(
        c.ROI, summary_text.ROI_LABEL, title, include_ci=include_ci
    )

  def plot_cpik(self, include_ci: bool = True) -> alt.Chart:
    """Plots the CPIK bar chart for each channel.

    Args:
      include_ci: If `True`, plots the credible interval. Defaults to `True`.

    Returns:
      An Altair plot showing the CPIK per channel.
    """
    if include_ci:
      ci = int(self._confidence_level * 100)
      title = summary_text.CPIK_CHANNEL_CHART_TITLE_FORMAT.format(
          ci=f'with {ci}% credible interval'
      )
    else:
      title = summary_text.CPIK_CHANNEL_CHART_TITLE_FORMAT.format(ci='')
    return self._plot_metric_bar_chart(
        c.CPIK, summary_text.CPIK_LABEL, title, include_ci=include_ci
    )

  def plot_roi_vs_effectiveness(
      self,
      selected_channels: Sequence[str] | None = None,
      disable_size: bool = False,
  ) -> alt.Chart:
    """Plots the ROI versus effectiveness bubble chart.

    This chart compares the ROI, effectiveness, and spend for each media
    channel. Spend is depicted by the pixel area of the bubble.

    Args:
      selected_channels: List of channels to include. If `None`, all of the
        media channels are shown in the plot.
      disable_size: If `True`, disables the different sizing of the bubbles and
        plots each channel uniformly. Defaults to `False`.

    Returns:
      An Altair plot showing the ROI and effectiveness per channel.
    """
    return self._plot_roi_bubble_chart(
        metric=c.EFFECTIVENESS,
        metric_title='Effectiveness',
        title=summary_text.ROI_EFFECTIVENESS_CHART_TITLE,
        selected_channels=selected_channels,
        disable_size=disable_size,
    )

  def plot_roi_vs_mroi(
      self,
      selected_channels: Sequence[str] | None = None,
      disable_size: bool = False,
      equal_axes: bool = False,
  ) -> alt.Chart:
    """Plots the ROI versus mROI bubble chart.

    This chart compares the ROI, mROI, and spend for each channel. Spend is
    depicted by the pixel area of the bubble.

    Args:
      selected_channels: List of channels to include. If `None`, all of the
        media channels are shown in the plot.
      disable_size: If `True`, disables the different sizing of the bubbles and
        plots each channel uniformly. Defaults to `False`.
      equal_axes: If `True`, plots the X and Y axes with equal scale. Defaults
        to `False`.

    Returns:
      An Altair plot showing the ROI and mROI per channel.
    """
    return self._plot_roi_bubble_chart(
        metric=c.MROI,
        metric_title='Marginal ROI',
        title=summary_text.ROI_MARGINAL_CHART_TITLE,
        selected_channels=selected_channels,
        disable_size=disable_size,
        equal_axes=equal_axes,
    )

  def _plot_roi_bubble_chart(
      self,
      metric: str,
      metric_title: str,
      title: str,
      selected_channels: Sequence[str] | None = None,
      disable_size: bool = False,
      equal_axes: bool = False,
  ) -> alt.Chart:
    """Plots a bubble chart comparing ROI to another metric of choice.

    This chart compares the ROI, spend, and another metric, either effectiveness
    or mROI, for each channel where the spend is depicted by the pixel
    area of the bubble and each bubble represents a channel.

    Args:
      metric: Name of the metric in the media summary metrics dataset to compare
        against ROI.
      metric_title: The label to show for this metric on the y-axis of the plot.
      title: Title of the bubble chart.
      selected_channels: List of channels to include. If None, all media
        channels will be shown in the plot.
      disable_size: If True, disables the differing size of the bubbles and
        plots each channel uniformly. Defaults to False.
      equal_axes: If True, plots the X and Y axes with equal scale. Defaults to
        False.

    Returns:
      An Altair bubble plot showing the ROI, spend, and another metric.
    """
    if selected_channels:
      channels = self.paid_summary_metrics.channel
      if any(channel not in channels for channel in selected_channels):
        raise ValueError(
            '`selected_channels` should match the channel dimension names from '
            'meridian.InputData'
        )

    plot_df = self._transform_media_metrics_for_roi_bubble_plot(
        metric, selected_channels
    )

    axes_scale = alt.Scale()
    if equal_axes:
      max_roi = max(plot_df.roi.max(), plot_df.mroi.max())
      axes_scale = alt.Scale(domain=(0, max_roi), nice=True)

    plot = (
        alt.Chart(plot_df)
        .mark_circle(tooltip=True, size=c.POINT_SIZE)
        .encode(
            x=alt.X(c.ROI, title='ROI', scale=axes_scale),
            y=alt.Y(
                metric,
                title=metric_title,
                scale=axes_scale,
                axis=alt.Axis(**formatter.Y_AXIS_TITLE_CONFIG),
            ),
            color=alt.Color(
                f'{c.CHANNEL}:N',
                legend=alt.Legend(
                    orient='bottom',
                    title=None,
                    columns=7,
                    columnPadding=20,
                    rowPadding=10,
                ),
            ),
        )
        .configure_axis(
            gridDash=[3, 2],
            titlePadding=c.PADDING_10,
            **formatter.TEXT_CONFIG,
        )
    )
    if not disable_size:
      plot = plot.encode(
          size=alt.Size(
              c.SPEND, scale=alt.Scale(range=[100, 5000]), legend=None
          )
      )
    return plot.properties(title=formatter.custom_title_params(title))

  def _plot_metric_bar_chart(
      self, metric: str, metric_label: str, title: str, include_ci: bool = True
  ) -> alt.Chart:
    """Plots a bar chart showing the specified metric for each channel.

    Args:
      metric: A summary metric to plot.
      metric_label: The label to use to identify the metric on the plot axis.
      title: The title of the plot.
      include_ci: If `True`, plots the credible interval. Defaults to `True`.

    Returns:
      An Altair plot showing the specified metric per channel.
    """
    df = self._summary_metric_to_df(metric)
    base = (
        alt.Chart(df)
        .mark_bar(
            size=c.BAR_SIZE, cornerRadiusEnd=c.CORNER_RADIUS, color=c.BLUE_600
        )
        .encode(
            x=alt.X(
                f'{c.CHANNEL}:N',
                title='Channel',
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y(
                f'{metric}:Q',
                axis=alt.Axis(gridDash=[3, 2], **formatter.Y_AXIS_TITLE_CONFIG),
                title=metric_label,
            ),
        )
        .properties(width=alt.Step(80))
    )
    metric_text = base.mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        fontSize=c.AXIS_FONT_SIZE,
        color=c.GREY_900,
    ).encode(text=alt.Text(f'{metric}:Q', format='.2f'))

    bar_width = 2
    if include_ci:
      error_bar = (
          alt.Chart(df)
          .mark_errorbar(ticks=True, color=c.BLUE_300)
          .encode(
              alt.X(f'{c.CHANNEL}:N'),
              alt.Y(f'{c.CI_HI}:Q', title=metric_label),
              alt.Y2(f'{c.CI_LO}:Q'),
              strokeWidth=alt.value(bar_width),
          )
      )
      mean_dot = (
          alt.Chart(df)
          .mark_point(filled=True, color=c.BLUE_300, tooltip=True)
          .encode(alt.X(f'{c.CHANNEL}:N'), alt.Y(f'{metric}:Q'))
      )
      plot = base + error_bar + mean_dot + metric_text
    else:
      plot = base + metric_text

    return (
        plot.configure_tick(bandSize=10, thickness=bar_width)
        .properties(title=formatter.custom_title_params(title))
        .configure_axis(titlePadding=c.PADDING_10, **formatter.TEXT_CONFIG)
    )

  def _transform_media_metrics_for_roi_bubble_plot(
      self, metric: str, selected_channels: Sequence[str] | None = None
  ) -> pd.DataFrame:
    """Transforms the metrics specifically for plotting the bubble plots.

    This dataframe is for comparing ROI, spend, and the specified metric to be
    used in plotting the bubble comparison plots.

    Args:
      metric: Name of the metric in the summary metrics dataset to compare
        against ROI.
      selected_channels: Optional list of a subset of channels to filter by.

    Returns:
      A dataframe filtered based on the specifications.
    """
    metrics_df = self._summary_metrics_to_mean_df(
        metrics=[c.ROI, metric], selected_channels=selected_channels
    )
    spend_df = self.paid_summary_metrics[c.SPEND].to_dataframe().reset_index()
    return metrics_df.merge(spend_df, on=c.CHANNEL)

  def _transform_contribution_metrics(
      self,
      selected_channels: Sequence[str] | None = None,
      include_non_paid: bool = False,
  ) -> pd.DataFrame:
    """Transforms the media metrics for the contribution plot.

    This adds the calculations for incremental outcome and percentages of the
    expected outcome for the baseline, where there is no media effects. It also
    transforms the percentages to values between 0 and 1 for Altair to format
    when plotting.

    Args:
      selected_channels: Optional list of a subset of channels to filter by.
      include_non_paid: If `True`, includes the organic media, organic RF and
        non-media channels in the contribution plot. Defaults to `False`.

    Returns:
      A dataframe with contributions per channel.
    """
    total_media_criteria = {
        c.DISTRIBUTION: c.POSTERIOR,
        c.METRIC: c.MEAN,
        c.CHANNEL: c.ALL_CHANNELS,
    }
    summary_metrics = (
        self.all_summary_metrics
        if include_non_paid
        else self.paid_summary_metrics
    )
    total_media_outcome = (
        summary_metrics[c.INCREMENTAL_OUTCOME].sel(total_media_criteria).item()
    )
    total_media_pct = (
        summary_metrics[c.PCT_OF_CONTRIBUTION].sel(total_media_criteria).item()
        / 100
    )
    total_outcome = total_media_outcome / total_media_pct
    baseline_pct = 1 - total_media_pct
    baseline_outcome = total_outcome * baseline_pct

    baseline_df = pd.DataFrame(
        {
            c.CHANNEL: c.BASELINE,
            c.INCREMENTAL_OUTCOME: baseline_outcome,
            c.PCT_OF_CONTRIBUTION: baseline_pct,
        },
        index=[0],
    )
    outcome_df = self._summary_metrics_to_mean_df(
        metrics=[
            c.INCREMENTAL_OUTCOME,
            c.PCT_OF_CONTRIBUTION,
        ],
        selected_channels=selected_channels,
        include_non_paid=include_non_paid,
    )
    # Convert to percentage values between 0-1.
    outcome_df[c.PCT_OF_CONTRIBUTION] = outcome_df[c.PCT_OF_CONTRIBUTION].div(
        100
    )
    outcome_df = pd.concat([baseline_df, outcome_df]).reset_index(drop=True)
    outcome_df.sort_values(
        by=c.INCREMENTAL_OUTCOME, ascending=False, inplace=True
    )
    return outcome_df

  def _transform_contribution_spend_metrics(self) -> pd.DataFrame:
    """Transforms the media metrics for the spend vs contribution plot.

    The dataframe holds the percentages spent on each channel and then
    percentage of incremental outcome attributed to each channel out of the
    total media-driven outcome. It combines these percentages with the ROI per
    channel and scales the ROI to fit the percentage data.

    Returns:
      A dataframe of spend and outcome percentages and ROI per channel.
    """
    if self._meridian.input_data.revenue_per_kpi is not None:
      outcome = summary_text.REVENUE_LABEL
    else:
      outcome = summary_text.KPI_LABEL
    total_media_outcome = (
        self.paid_summary_metrics[c.INCREMENTAL_OUTCOME]
        .sel(
            distribution=c.POSTERIOR,
            metric=c.MEAN,
            channel=[c.ALL_CHANNELS],
        )
        .item()
    )
    outcome_pct_df = self._summary_metrics_to_mean_df(
        metrics=[c.INCREMENTAL_OUTCOME]
    )
    outcome_pct_df[c.PCT] = outcome_pct_df[c.INCREMENTAL_OUTCOME].div(
        total_media_outcome
    )
    outcome_pct_df.drop(columns=[c.INCREMENTAL_OUTCOME], inplace=True)
    outcome_pct_df['label'] = f'% {outcome}'
    spend_pct_df = (
        self.paid_summary_metrics[c.PCT_OF_SPEND]
        .drop_sel(channel=[c.ALL_CHANNELS])
        .to_dataframe()
        .reset_index()
    )
    spend_pct_df.rename(columns={c.PCT_OF_SPEND: c.PCT}, inplace=True)
    spend_pct_df[c.PCT] = spend_pct_df[c.PCT].div(100)
    spend_pct_df['label'] = '% Spend'

    pct_df = pd.concat([outcome_pct_df, spend_pct_df])
    roi_df = self._summary_metrics_to_mean_df(metrics=[c.ROI])
    plot_df = pct_df.merge(roi_df, on=c.CHANNEL)
    scale_factor = plot_df[c.PCT].max() / plot_df[c.ROI].max()
    plot_df[c.ROI_SCALED] = plot_df[c.ROI] * scale_factor

    return plot_df

  def _summary_metrics_to_mean_df(
      self,
      metrics: Sequence[str],
      selected_channels: Sequence[str] | None = None,
      include_non_paid: bool = False,
  ) -> pd.DataFrame:
    """Transforms the summary metrics to a dataframe of mean values.

    The dataframe has the selected metrics as the columns as well as the media.
    The metrics values are the posterior mean values for each of the selected
    channels.

    Args:
      metrics: A list of the metrics to include in the dataframe.
      selected_channels: List of channels to include. If None, all media
        channels will be included.
      include_non_paid: If `True`, includes the organic media, organic RF and
        non-media channels in the dataframe. Defaults to `False`.

    Returns:
      A dataframe of posterior mean values for the selected metrics and media.
    """
    summary_metrics = (
        self.all_summary_metrics
        if include_non_paid
        else self.paid_summary_metrics
    )
    metrics_dataset = summary_metrics[metrics].sel(
        distribution=c.POSTERIOR, metric=c.MEAN
    )
    if selected_channels:
      metrics_dataset = metrics_dataset.sel(channel=selected_channels)
    else:
      metrics_dataset = metrics_dataset.drop_sel(channel=c.ALL_CHANNELS)
    return (
        metrics_dataset.to_dataframe()
        .drop(columns=[c.METRIC, c.DISTRIBUTION])
        .reset_index()
    )

  def _summary_metric_to_df(self, metric: str) -> pd.DataFrame:
    """Transforms a summary metric to a pivoted dataframe.

    The dataframe includes the posterior data for the selected metric and its
    credible interval per channel.

    Args:
      metric: The summary metric to include in the dataframe.

    Returns:
      A dataframe of the posterior values for the selected metric.
    """
    # Format CPIK to use median instead of mean.
    central_tendency = c.MEDIAN if metric == c.CPIK else c.MEAN
    unused_central_tendency = c.MEAN if metric == c.CPIK else c.MEDIAN
    return (
        self.paid_summary_metrics[metric]
        .sel(distribution=c.POSTERIOR)
        .drop_sel(
            channel=c.ALL_CHANNELS,
            metric=unused_central_tendency,
        )
        .to_dataframe()
        .reset_index()
        .pivot(
            index=c.CHANNEL,
            columns=c.METRIC,
            values=metric,
        )
        .reset_index()
        .rename(columns={central_tendency: metric})
    )
