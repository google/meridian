# Copyright 2025 The Meridian Authors.
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

import os

from absl.testing import absltest
from absl.testing import parameterized
import altair as alt
import arviz as az
from meridian import constants as c
from meridian.analysis import analyzer
from meridian.analysis import formatter
from meridian.analysis import summary_text
from meridian.analysis import test_utils
from meridian.analysis import visualizer
from meridian.data import input_data
from meridian.data import test_utils as data_test_utils
from meridian.model import model
import numpy as np
import xarray as xr

mock = absltest.mock

_TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model", "test_data"
)

# Data dimensions for sample input.
_N_CHAINS = 2
_N_KEEP = 10
_N_DRAWS = 10
_N_GEOS = 5
_N_TIMES = 49
_N_MEDIA_TIMES = 52
_N_CONTROLS = 2
_N_MEDIA_CHANNELS = 3
_N_RF_CHANNELS = 2


class ModelDiagnosticsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(ModelDiagnosticsTest, cls).setUpClass()
    cls.input_data = mock.create_autospec(input_data.InputData, instance=True)
    cls.meridian = mock.create_autospec(
        model.Meridian, instance=True, input_data=cls.input_data
    )
    inference_data = az.InferenceData()
    inference_data.prior = xr.open_dataset(
        os.path.join(_TEST_DATA_DIR, "sample_prior_media_and_rf.nc")
    )
    inference_data.posterior = xr.open_dataset(
        os.path.join(_TEST_DATA_DIR, "sample_posterior_media_and_rf.nc")
    )
    type(cls.meridian).inference_data = mock.PropertyMock(
        return_value=inference_data
    )
    cls.mock_analyzer_method = cls.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            "predictive_accuracy",
        )
    )
    cls.model_diagnostics = visualizer.ModelDiagnostics(cls.meridian)

  def test_predictive_accuracy_called_correctly(self):
    self.model_diagnostics.predictive_accuracy_table()
    self.mock_analyzer_method.assert_called_once()

  def test_predictive_accuracy_selected_geos_times_called_correctly(self):
    self.model_diagnostics.predictive_accuracy_table(
        selected_geos=["geo 1", "geo 2"],
        selected_times=["2021-02-22", "2021-03-01"],
    )
    self.mock_analyzer_method.assert_called_with(
        selected_geos=["geo 1", "geo 2"],
        selected_times=["2021-02-22", "2021-03-01"],
        batch_size=100,
    )

  @parameterized.named_parameters(
      ("pivot_none", None, [c.METRIC, c.GEO_GRANULARITY, c.VALUE]),
      (
          "pivot_geo_granularity",
          c.GEO_GRANULARITY,
          [c.METRIC, c.GEO, c.NATIONAL],
      ),
      (
          "pivot_metric",
          c.METRIC,
          [c.GEO_GRANULARITY, c.MAPE, c.R_SQUARED, c.WMAPE],
      ),
  )
  def test_transform_predictive_accuracy_no_holdout_id(
      self, column_var, expected_columns
  ):
    mock_analyzer_method_no_holdout = self.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            "predictive_accuracy",
        )
    )
    mock_analyzer_method_no_holdout.return_value = (
        test_utils.generate_predictive_accuracy_data(holdout_id=False)
    )
    model_diagnostics = visualizer.ModelDiagnostics(self.meridian)
    df = model_diagnostics.predictive_accuracy_table(
        column_var=column_var,
    )
    self.assertEqual(list(df.columns), expected_columns)

  @parameterized.named_parameters(
      (
          "pivot_none",
          None,
          [c.METRIC, c.GEO_GRANULARITY, c.EVALUATION_SET_VAR, c.VALUE],
      ),
      (
          "pivot_geo_granularity",
          c.GEO_GRANULARITY,
          [c.METRIC, c.EVALUATION_SET_VAR, c.GEO, c.NATIONAL],
      ),
      (
          "pivot_metric",
          c.METRIC,
          [
              c.GEO_GRANULARITY,
              c.EVALUATION_SET_VAR,
              c.MAPE,
              c.R_SQUARED,
              c.WMAPE,
          ],
      ),
      (
          "pivot_evaluation_set",
          c.EVALUATION_SET_VAR,
          [c.METRIC, c.GEO_GRANULARITY, c.ALL_DATA, c.TEST, c.TRAIN],
      ),
  )
  def test_transform_predictive_accuracy_holdout_id(
      self, column_var, expected_columns
  ):
    mock_analyzer_method_with_holdout = self.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            "predictive_accuracy",
        )
    )
    mock_analyzer_method_with_holdout.return_value = (
        test_utils.generate_predictive_accuracy_data(holdout_id=True)
    )
    meridian = mock.create_autospec(
        model.Meridian, instance=True, input_data=self.input_data
    )
    model_diagnostics = visualizer.ModelDiagnostics(meridian)
    df = model_diagnostics.predictive_accuracy_table(
        column_var=column_var,
    )
    self.assertEqual(list(df.columns), expected_columns)

  def test_transform_predictive_accuracy_incorrect_column_var(self):
    incorrect_var = c.MU_T
    with self.assertRaisesRegex(
        ValueError,
        f"The DataFrame cannot be pivoted by {incorrect_var} as it does not"
        " exist in the DataFrame.",
    ):
      self.model_diagnostics.predictive_accuracy_table(
          column_var=c.MU_T,
      )

  def test_distribution_pre_fitting_raises_exception(self):
    not_fitted_mmm = mock.create_autospec(model.Meridian, instance=True)
    type(not_fitted_mmm).inference_data = mock.PropertyMock(
        return_value=az.InferenceData()
    )
    not_fitted_model_diagnostics = visualizer.ModelDiagnostics(not_fitted_mmm)
    with self.assertRaisesRegex(
        model.NotFittedModelError,
        "Plotting prior and posterior distributions requires fitting the model",
    ):
      not_fitted_model_diagnostics.plot_prior_and_posterior_distribution()

  def test_distribution_incorrect_parameter_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "The selected param 'incorrect_parameter' does not exist in Meridian's"
        " model parameters.",
    ):
      self.model_diagnostics.plot_prior_and_posterior_distribution(
          "incorrect_parameter"
      )

  def test_distribution_selected_times_no_time_dim_param_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "`selected_times` can only be used if the parameter has a time"
        " dimension. The selected param 'tau_g' does not have a time"
        " dimension.",
    ):
      self.model_diagnostics.plot_prior_and_posterior_distribution(
          c.TAU_G, selected_times=["2021-02-22", "2021-03-01"]
      )

  def test_distribution_incorrect_selected_times_raises_exception(self):
    with self.assertRaisesRegex(
        ValueError,
        "The selected times must match the time dimensions in the Meridian"
        " model.",
    ):
      self.model_diagnostics.plot_prior_and_posterior_distribution(
          c.MU_T, selected_times=["Jan 1, 2023"]
      )

  def test_distribution_correct_config(self):
    plot = self.model_diagnostics.plot_prior_and_posterior_distribution()
    self.assertEqual(
        plot.title.text, summary_text.PRIOR_POSTERIOR_DIST_CHART_TITLE
    )
    self.assertEqual(plot.config.axis.to_dict(), formatter.TEXT_CONFIG)

  def test_distribution_media_dim_plots_media_faceted_chart(self):
    plot = self.model_diagnostics.plot_prior_and_posterior_distribution()

    self.assertEqual(
        list(plot.data.columns),
        [c.MEDIA_CHANNEL, c.ROI_M, c.DISTRIBUTION],
    )
    self.assertIsInstance(plot, alt.FacetChart)
    self.assertIsInstance(plot.facet, alt.Facet)
    self.assertEqual(plot.facet.shorthand, c.MEDIA_CHANNEL)
    self.assertEqual(plot.columns, 3)

  def test_distribution_control_dim_plots_control_faceted_chart(self):
    plot = self.model_diagnostics.plot_prior_and_posterior_distribution(
        c.GAMMA_C
    )

    self.assertEqual(
        list(plot.data.columns),
        [c.CONTROL_VARIABLE, c.GAMMA_C, c.DISTRIBUTION],
    )
    self.assertIsInstance(plot, alt.FacetChart)
    self.assertIsInstance(plot.facet, alt.Facet)
    self.assertEqual(plot.facet.shorthand, c.CONTROL_VARIABLE)

  def test_distribution_media_geo_dims_plots_2_level_faceted_chart(self):
    plot = self.model_diagnostics.plot_prior_and_posterior_distribution(
        c.BETA_GM
    )

    self.assertEqual(
        list(plot.data.columns),
        [
            c.GEO,
            c.MEDIA_CHANNEL,
            c.BETA_GM,
            c.DISTRIBUTION,
        ],
    )
    self.assertIsInstance(plot, alt.FacetChart)
    self.assertIsInstance(plot.facet, alt.FacetMapping)
    self.assertEqual(plot.facet.column, c.GEO)
    self.assertEqual(plot.facet.row, c.MEDIA_CHANNEL)

  def test_geo_distribution_calls_default_top_geos(self):
    plot = self.model_diagnostics.plot_prior_and_posterior_distribution(c.TAU_G)

    self.assertEqual(
        list(plot.data.columns),
        [c.GEO, c.TAU_G, c.DISTRIBUTION],
    )
    self.input_data.get_n_top_largest_geos.assert_called_with(3)

  def test_distribution_plots_top_num_geos(self):
    plot = self.model_diagnostics.plot_prior_and_posterior_distribution(
        c.BETA_GM, num_geos=4
    )

    self.assertEqual(
        list(plot.data.columns),
        [
            c.GEO,
            c.MEDIA_CHANNEL,
            c.BETA_GM,
            c.DISTRIBUTION,
        ],
    )
    self.input_data.get_n_top_largest_geos.assert_called_with(4)

  def test_distribution_default_time_dim(self):
    plot = self.model_diagnostics.plot_prior_and_posterior_distribution(c.MU_T)

    self.assertEqual(
        list(plot.data.columns),
        [c.TIME, c.MU_T, c.DISTRIBUTION],
    )
    self.assertEqual(plot.data[c.TIME].nunique(), 3)

  def test_distribution_selected_time_dim(self):
    plot = self.model_diagnostics.plot_prior_and_posterior_distribution(
        c.MU_T, selected_times=["2021-02-22", "2021-03-01"]
    )

    self.assertEqual(
        list(plot.data.columns),
        [c.TIME, c.MU_T, c.DISTRIBUTION],
    )
    self.assertEqual(plot.data[c.TIME].nunique(), 2)

  def test_plot_rhat_boxplot_pre_fitting_raises_exception(self):
    not_fitted_mmm = mock.create_autospec(model.Meridian, instance=True)
    type(not_fitted_mmm).inference_data = mock.PropertyMock(
        return_value=az.InferenceData()
    )
    not_fitted_model_diagnostics = visualizer.ModelDiagnostics(not_fitted_mmm)
    with self.assertRaisesRegex(
        model.NotFittedModelError,
        "Plotting the r-hat values requires fitting the model.",
    ):
      not_fitted_model_diagnostics.plot_rhat_boxplot()

  def test_plot_rhat_boxplot_mcmc_failed_raises_exception(self):
    # Simulate alpha_m posterior data to not converge. When the MCMC sampling
    # does not converge, there will be a constant value within each chain, but
    # each chain will have a different constant value.
    chain = [0, 1]
    draw = [0, 1, 2, 3, 4]
    channels = ["channel_0", "channel_1"]
    alpha_m = [[0.1, 0.2], [0.3, 0.4]]
    da = (
        xr.DataArray(
            data=alpha_m,
            coords={c.CHAIN: chain, c.MEDIA_CHANNEL: channels},
            name=c.ALPHA_M,
        )
        .expand_dims({c.DRAW: len(draw)})
        .assign_coords({c.DRAW: draw})
        .transpose(c.CHAIN, c.DRAW, c.MEDIA_CHANNEL)
    )
    ds = xr.Dataset()
    ds[c.ALPHA_M] = da

    inference_data = az.InferenceData()
    inference_data.posterior = ds
    meridian = mock.create_autospec(
        model.Meridian, instance=True, input_data=self.input_data
    )
    type(meridian).inference_data = mock.PropertyMock(
        return_value=inference_data
    )
    model_diagnostics = visualizer.ModelDiagnostics(meridian)

    with self.assertRaisesRegex(
        model.MCMCSamplingError,
        "MCMC sampling failed with a maximum R-hat value of inf",
    ):
      model_diagnostics.plot_rhat_boxplot()

  def test_plot_rhat_boxplot_correct_data(self):
    plot = self.model_diagnostics.plot_rhat_boxplot()
    plot_data = plot.layer[0].data
    self.assertSetEqual(
        set(plot_data[c.PARAMETER].unique()),
        # slope_m has a deterministic prior.
        set(
            c.COMMON_PARAMETER_NAMES
            + c.MEDIA_PARAMETER_NAMES
            + c.RF_PARAMETER_NAMES
        )
        - {c.SLOPE_M},
    )

  def test_plot_rhat_boxplot_correct_marks(self):
    plot = self.model_diagnostics.plot_rhat_boxplot()
    boxplot_mark = plot.layer[0].mark
    line_mark = plot.layer[1].mark
    self.assertEqual(boxplot_mark.type, "boxplot")
    self.assertTrue(boxplot_mark.outliers["filled"])
    self.assertEqual(boxplot_mark.median["color"], c.BLUE_300)
    self.assertEqual(line_mark.type, "rule")
    self.assertEqual(line_mark.color, c.RED_600)
    self.assertEqual(line_mark.opacity, 0.8)

  def test_plot_rhat_boxplot_correct_encoding(self):
    plot = self.model_diagnostics.plot_rhat_boxplot()
    boxplot_encoding = plot.layer[0].encoding
    line_encoding = plot.layer[1].encoding
    self.assertEqual(line_encoding.y.datum, 1)
    self.assertEqual(boxplot_encoding.x.shorthand, c.PARAMETER)
    self.assertEqual(boxplot_encoding.y.shorthand, c.RHAT)
    self.assertEqual(boxplot_encoding.x["axis"]["labelAngle"], -45)
    self.assertFalse(boxplot_encoding.y["scale"]["zero"])

  def test_plot_rhat_boxplot_correct_config(self):
    plot = self.model_diagnostics.plot_rhat_boxplot()
    self.assertEqual(plot.title.text, summary_text.RHAT_BOXPLOT_TITLE)
    self.assertEqual(plot.config.axis.to_dict(), formatter.TEXT_CONFIG)


class ModelFitTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(ModelFitTest, cls).setUpClass()
    n_times = 149
    n_geos = 10
    revenue_per_kpi = data_test_utils.constant_revenue_per_kpi(
        n_geos=n_geos, n_times=n_times, value=2.2
    )
    cls.input_data_1 = mock.create_autospec(input_data.InputData, instance=True)
    cls.input_data_2 = mock.create_autospec(input_data.InputData, instance=True)
    cls.input_data_3 = mock.create_autospec(input_data.InputData, instance=True)

    largest_geos = cls.input_data_1.get_n_top_largest_geos
    largest_geos.side_effect = lambda n: [f"geo {i}" for i in range(n)]

    # Testing three scenarios.
    # Label = `Revenue`.
    meridian_revenue = mock.create_autospec(
        model.Meridian,
        instance=True,
        input_data=cls.input_data_1,
    )
    meridian_revenue.input_data.kpi_type = c.REVENUE
    meridian_revenue.input_data.revenue_per_kpi = revenue_per_kpi
    # Label = `KPI`.
    meridian_kpi = mock.create_autospec(
        model.Meridian,
        instance=True,
        input_data=cls.input_data_2,
    )
    meridian_kpi.input_data.kpi_type = c.NON_REVENUE
    meridian_kpi.input_data.revenue_per_kpi = None
    # Label = `Revenue`.
    meridian_revenue_2 = mock.create_autospec(
        model.Meridian,
        instance=True,
        input_data=cls.input_data_3,
    )
    meridian_revenue_2.input_data.kpi_type = c.NON_REVENUE
    meridian_revenue_2.input_data.revenue_per_kpi = revenue_per_kpi
    mock_model_fit_data = test_utils.generate_model_fit_data()
    cls.mock_analyzer_method = cls.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            "expected_vs_actual_data",
            return_value=mock_model_fit_data,
        )
    )
    cls.model_fit_kpi_type_revenue = visualizer.ModelFit(meridian_revenue)
    cls.model_fit_kpi_type_kpi = visualizer.ModelFit(meridian_kpi)
    cls.model_fit_kpi_type_revenue_2 = visualizer.ModelFit(meridian_revenue_2)

  def test_model_fit_plot_different_scenarios_labels_correct(self):
    # Verifies each of the three kpi_type and revenue_per_kpi scenarios have
    # the correct labels associated with them.
    plot_revenue_revenue_label: alt.Chart = (
        self.model_fit_kpi_type_revenue.plot_model_fit()
    )
    plot_non_revenue_kpi_label: alt.Chart = (
        self.model_fit_kpi_type_kpi.plot_model_fit()
    )
    plot_non_revenue_revenue_label = (
        self.model_fit_kpi_type_revenue_2.plot_model_fit()
    )
    self.assertEqual(
        plot_revenue_revenue_label.layer[0].encoding.y["title"],
        summary_text.REVENUE_LABEL,
    )
    self.assertEqual(
        plot_non_revenue_kpi_label.layer[0].encoding.y["title"],
        summary_text.KPI_LABEL,
    )
    self.assertEqual(
        plot_non_revenue_revenue_label.layer[0].encoding.y["title"],
        summary_text.REVENUE_LABEL,
    )

  def test_model_fit_incorrect_selected_times_raises_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "`selected_times` should match the time dimensions from"
        " meridian.InputData.",
    ):
      self.model_fit_kpi_type_revenue.plot_model_fit(
          selected_times=["2000-01-01"]
      )

  def test_model_fit_show_geo_no_geo_specified_raises_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "Geo-level plotting is only available when `selected_geos` or"
        " `n_top_largest_geos` is specified.",
    ):
      self.model_fit_kpi_type_revenue.plot_model_fit(show_geo_level=True)

  def test_model_fit_selected_geo_and_n_top_largest_geos_raises_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "Only one of `selected_geos` and `n_top_largest_geos` can be"
        " specified.",
    ):
      self.model_fit_kpi_type_revenue.plot_model_fit(
          selected_geos=["geo 1", "geo 2"], n_top_largest_geos=3
      )

  def test_model_fit_selected_geo_not_in_geo_dims(self):
    with self.assertRaisesRegex(
        ValueError,
        "`selected_geos` should match the geo dimension names from"
        " meridian.InputData.",
    ):
      self.model_fit_kpi_type_revenue.plot_model_fit(
          selected_geos=["nonexistent geo"]
      )

  def test_model_fit_n_top_largest_geos_greater_than_total_num_geos(self):
    with self.assertRaisesRegex(
        ValueError,
        "`n_top_largest_geos` should be less than or equal to the total number"
        " of geos: 5.",
    ):
      self.model_fit_kpi_type_revenue.plot_model_fit(n_top_largest_geos=100)

  def test_model_fit_update_ci(self):
    self.assertEqual(
        self.model_fit_kpi_type_revenue.model_fit_data.confidence_level,
        c.DEFAULT_CONFIDENCE_LEVEL,
    )
    self.model_fit_kpi_type_revenue.update_confidence_level(0.8)
    self.mock_analyzer_method.assert_called_with(confidence_level=0.8)

  def test_model_fit_plots_selected_times(self):
    times = ["2023-01-01", "2023-01-08", "2023-01-15"]
    plot = self.model_fit_kpi_type_revenue.plot_model_fit(selected_times=times)
    self.assertListEqual(list(plot.data.time.unique()), times)

  def test_model_fit_national_level_aggregates_all_geos(self):
    geo = ["geo 1", "geo 2", "geo 3", "geo 4"]
    time = ["2023-01-01", "2023-01-08"]
    actual = [[1, 2], [3, 4], [5, 6], [7, 8]]
    model_fit_data = test_utils.generate_model_fit_data(
        geo=geo, time=time, actual=actual
    )

    with mock.patch.object(
        visualizer.ModelFit,
        "model_fit_data",
        new=property(lambda unused_self: model_fit_data),
    ):
      plot = self.model_fit_kpi_type_revenue.plot_model_fit()

    plot_actual_value = plot.data.loc[
        (plot.data[c.TIME] == "2023-01-01") & (plot.data[c.TYPE] == c.ACTUAL)
    ][c.MEAN][0]
    self.assertEqual(plot_actual_value, 16)
    self.assertEqual(
        list(plot.data.columns),
        [c.TIME, c.TYPE, c.CI_HI, c.CI_LO, c.MEAN, c.MEDIAN],
    )

  def test_model_fit_national_level_aggregates_selected_geos(self):
    geo = ["geo 1", "geo 2", "geo 3", "geo 4"]
    time = ["2023-01-01", "2023-01-08"]
    actual = [[1, 2], [3, 4], [5, 6], [7, 8]]
    model_fit_data = test_utils.generate_model_fit_data(
        geo=geo, time=time, actual=actual
    )

    with mock.patch.object(
        visualizer.ModelFit,
        "model_fit_data",
        new=property(lambda unused_self: model_fit_data),
    ):
      plot = self.model_fit_kpi_type_revenue.plot_model_fit(
          selected_geos=["geo 1", "geo 3"]
      )

    plot_actual_value = plot.data.loc[
        (plot.data[c.TIME] == "2023-01-01") & (plot.data["type"] == c.ACTUAL)
    ][c.MEAN][0]
    self.assertEqual(plot_actual_value, 6)

  def test_model_fit_geo_level_plots_selected_geos(self):
    plot = self.model_fit_kpi_type_revenue.plot_model_fit(
        selected_geos=["geo 1", "geo 3"], show_geo_level=True
    )

    self.assertContainsSubset([c.GEO], plot.data.columns.tolist())
    self.assertIsInstance(plot, alt.FacetChart)
    self.assertListEqual(plot.facet.column["sort"], ["geo 1", "geo 3"])

  def test_model_fit_geo_level_plots_n_largest_geos(self):
    plot = self.model_fit_kpi_type_revenue.plot_model_fit(
        n_top_largest_geos=3, show_geo_level=True
    )

    self.input_data_1.get_n_top_largest_geos.assert_called_with(3)
    self.assertContainsSubset([c.GEO], plot.data.columns.tolist())
    self.assertIsInstance(plot, alt.FacetChart)
    self.assertListEqual(plot.facet.column["sort"], ["geo 0", "geo 1", "geo 2"])

  def test_model_fit_plots_baseline(self):
    plot = self.model_fit_kpi_type_revenue.plot_model_fit(include_baseline=True)

    self.assertListEqual(
        plot.data.type.unique().tolist(),
        [c.ACTUAL, c.BASELINE, c.EXPECTED],
    )
    self.assertEqual(
        plot.layer[0].encoding.color["scale"]["domain"],
        [c.EXPECTED, c.ACTUAL, c.BASELINE],
    )
    self.assertLen(plot.layer[0].encoding.color["scale"]["range"], 3)

  def test_model_fit_plots_no_baseline(self):
    plot = self.model_fit_kpi_type_revenue.plot_model_fit(
        include_baseline=False
    )

    self.assertListEqual(
        plot.data.type.unique().tolist(), [c.ACTUAL, c.EXPECTED]
    )
    self.assertEqual(
        plot.layer[0].encoding.color["scale"]["domain"],
        [c.EXPECTED, c.ACTUAL],
    )

  def test_model_fit_plots_expected_ci(self):
    plot = self.model_fit_kpi_type_revenue.plot_model_fit(include_ci=True)

    self.assertIsInstance(plot, alt.LayerChart)
    self.assertEqual(
        plot.layer[1].encoding.color["scale"]["domain"], [c.EXPECTED]
    )
    self.assertEqual(plot.layer[1].encoding.y.shorthand, f"{c.CI_HI}:Q")
    self.assertEqual(plot.layer[1].encoding.y2.shorthand, f"{c.CI_LO}:Q")

  def test_model_fit_plots_no_ci(self):
    plot = self.model_fit_kpi_type_revenue.plot_model_fit(include_ci=False)
    self.assertIsInstance(plot, alt.Chart)
    self.assertEqual(plot.encoding.x.shorthand, f"{c.TIME}:T")
    self.assertEqual(plot.encoding.y.shorthand, f"{c.MEAN}:Q")

  def test_model_fit_axis_encoding(self):
    plot = self.model_fit_kpi_type_revenue.plot_model_fit()
    self.assertEqual(
        plot.layer[0].encoding.x["axis"].to_dict(),
        {
            "domainColor": c.GREY_300,
            "grid": False,
            "tickCount": 8,
            "format": "%Y %b",
        },
    )
    self.assertEqual(
        plot.layer[0].encoding.y["axis"].to_dict(),
        {
            "domain": False,
            "labelExpr": formatter.compact_number_expr(),
            "labelPadding": c.PADDING_10,
            "tickCount": 5,
            "ticks": False,
        }
        | formatter.Y_AXIS_TITLE_CONFIG,
    )

  def test_model_fit_correct_config(self):
    plot = self.model_fit_kpi_type_revenue.plot_model_fit()
    self.assertEqual(plot.config.axis.to_dict(), formatter.TEXT_CONFIG)


class ReachAndFrequencyTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(ReachAndFrequencyTest, cls).setUpClass()
    cls.input_data = mock.create_autospec(input_data.InputData, instance=True)
    cls.meridian = mock.create_autospec(
        model.Meridian, instance=True, input_data=cls.input_data
    )
    cls.mock_optimal_frequency_data = (
        test_utils.generate_optimal_frequency_data()
    )
    cls.mock_optimal_freq_method = cls.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            "optimal_freq",
            return_value=cls.mock_optimal_frequency_data,
        )
    )
    cls.reach_and_frequency = visualizer.ReachAndFrequency(
        cls.meridian, use_kpi=False
    )

  def test_reach_and_frequency_plot_optimal_freq_update_selected_times(self):
    times1 = ["2023-01-01", "2023-04-21"]
    reach_and_frequency = visualizer.ReachAndFrequency(
        self.meridian, selected_times=times1, use_kpi=False
    )
    self.mock_optimal_freq_method.assert_called_with(
        selected_times=times1, use_kpi=False
    )
    reach_and_frequency.update_optimal_reach_and_frequency_selected_times(
        selected_times=None
    )
    self.mock_optimal_freq_method.assert_called_with(
        selected_times=None, use_kpi=False
    )
    times2 = ["2023-02-01", "2023-06-30"]
    reach_and_frequency.update_optimal_reach_and_frequency_selected_times(
        selected_times=times2
    )
    self.mock_optimal_freq_method.assert_called_with(
        selected_times=times2, use_kpi=False
    )

  def test_reach_and_frequency_plot_optimal_freq_correct_line(self):
    plot = self.reach_and_frequency.plot_optimal_frequency()

    layer = plot.spec.layer
    line_layer = layer[0]

    line_encoding = line_layer.encoding
    line_mark = line_layer.mark

    self.assertEqual(line_encoding.x.shorthand, c.FREQUENCY)
    self.assertEqual(line_encoding.x["title"], "Weekly Average Frequency")
    self.assertEqual(line_encoding.y.shorthand, c.ROI)
    self.assertEqual(line_encoding.y["title"], summary_text.ROI_LABEL)

    self.assertEqual(
        line_encoding.color["scale"]["domain"],
        [summary_text.OPTIMAL_FREQ_LABEL, summary_text.EXPECTED_ROI_LABEL],
    )
    self.assertEqual(
        line_encoding.color["scale"]["range"], [c.BLUE_600, c.RED_600]
    )

    self.assertEqual(line_mark.type, "line")
    self.assertEqual(line_mark.strokeWidth, 4)

  def test_reach_and_frequency_plot_optimal_freq_correct_vertical_line(self):
    plot = self.reach_and_frequency.plot_optimal_frequency()
    layer = plot.spec.layer
    vertical_line_layer = layer[1]

    vertical_line_encoding = vertical_line_layer.encoding
    vertical_line_mark = vertical_line_layer.mark

    self.assertEqual(
        vertical_line_encoding.x.shorthand, f"{c.OPTIMAL_FREQUENCY}:Q"
    )

    self.assertEqual(vertical_line_mark.strokeDash, [6, 6])
    self.assertEqual(vertical_line_mark.strokeWidth, 3)
    self.assertEqual(vertical_line_mark.type, "rule")

  def test_reach_and_frequency_plot_optimal_freq_label_text_encoding(self):
    plot = self.reach_and_frequency.plot_optimal_frequency()
    layer = plot.spec.layer
    label_text_layer = layer[2]

    label_text_encoding = label_text_layer.encoding
    label_text_mark = label_text_layer.mark

    self.assertEqual(
        label_text_encoding.x.shorthand, f"{c.OPTIMAL_FREQUENCY}:Q"
    )
    self.assertEqual(label_text_mark.dx, 5)

  def test_reach_and_frequency_plot_optimal_freq_label_freq_encoding(self):
    plot = self.reach_and_frequency.plot_optimal_frequency()
    layer = plot.spec.layer
    label_value_layer = layer[3]

    label_value_encoding = label_value_layer.encoding
    label_value_mark = label_value_layer.mark

    self.assertEqual(
        label_value_encoding.text.shorthand, f"{c.OPTIMAL_FREQUENCY}:Q"
    )
    self.assertEqual(label_value_encoding.text["format"], ".2f")
    self.assertEqual(
        label_value_encoding.x.shorthand, f"{c.OPTIMAL_FREQUENCY}:Q"
    )
    self.assertEqual(label_value_mark.dx, 110)

  @parameterized.named_parameters(
      ("label_text_layer", 2),
      ("label_value_layer", 3),
  )
  def test_reach_and_frequency_plot_optimal_freq_correct_label_marks(
      self, layer_index
  ):
    plot = self.reach_and_frequency.plot_optimal_frequency()
    layer = plot.spec.layer
    label_freq_layer = layer[layer_index]
    label_freq_mark = label_freq_layer.mark

    self.assertEqual(label_freq_mark.align, "left")
    self.assertEqual(label_freq_mark.dy, -5)
    self.assertEqual(label_freq_mark.font, c.FONT_ROBOTO)
    self.assertEqual(label_freq_mark.fontSize, 12)
    self.assertEqual(label_freq_mark.fontWeight, "lighter")
    self.assertEqual(label_freq_mark.type, "text")

  def test_reach_and_frequency_plot_optimal_freq_correct_specified_channels(
      self,
  ):
    plot = self.reach_and_frequency.plot_optimal_frequency(
        ["rf_channel_0", "rf_channel_3"]
    )
    plot_data = plot.data
    channels = plot_data[c.RF_CHANNEL].unique().tolist()
    num_channels = len(channels)
    self.assertIn("rf_channel_0", channels)
    self.assertIn("rf_channel_3", channels)
    self.assertEqual(num_channels, 2)

  def test_reach_and_frequency_plot_optimal_freq_correct_wrong_channel(self):
    with self.assertRaisesRegex(
        ValueError,
        """Channels specified are not in the list of all RF channels""",
    ):
      plot = self.reach_and_frequency.plot_optimal_frequency(["rf_channel 8"])
      self.assertIsNone(plot)

  def test_reach_and_frequency_plot_optimal_freq_correct_layers(self):
    plot = self.reach_and_frequency.plot_optimal_frequency()

    self.assertIsInstance(plot, alt.FacetChart)
    self.assertEqual(plot.facet.column.shorthand, f"{c.RF_CHANNEL}:N")
    self.assertIsNone(plot.facet.column["title"])
    self.assertLen(plot.spec.layer, 4)

  def test_reach_and_frequency_plot_optimal_freq_properties(self):
    plot_facet_by_channel = self.reach_and_frequency.plot_optimal_frequency()

    self.assertEqual(plot_facet_by_channel.resolve.scale.x, c.INDEPENDENT)
    self.assertEqual(plot_facet_by_channel.resolve.scale.y, c.INDEPENDENT)
    self.assertEqual(
        plot_facet_by_channel.config.axis.to_dict(), formatter.TEXT_CONFIG
    )
    self.assertEqual(
        plot_facet_by_channel.title.text,
        summary_text.OPTIMAL_FREQUENCY_CHART_TITLE.format(
            metric=summary_text.ROI_LABEL
        ),
    )

  def test_reach_and_frequency_plot_optimal_freq_correct_data(self):
    plot = self.reach_and_frequency.plot_optimal_frequency()
    df = plot.data

    num_channels = len(set(df.rf_channel))

    self.assertEqual(
        list(df.columns),
        [
            c.RF_CHANNEL,
            c.FREQUENCY,
            c.CI_HI,
            c.CI_LO,
            c.ROI,
            c.OPTIMAL_FREQUENCY,
        ],
    )

    self.assertGreater(num_channels, 0)
    self.assertTrue(frequency > 0 for frequency in df.frequency)
    self.assertTrue(roi > 0 for roi in df.roi)
    self.assertTrue(opt_freq > 0 for opt_freq in df.optimal_frequency)


class MediaEffectsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(MediaEffectsTest, cls).setUpClass()
    n_times = 149
    n_geos = 10
    revenue_per_kpi = data_test_utils.constant_revenue_per_kpi(
        n_geos=n_geos, n_times=n_times, value=2.2
    )
    cls.input_data_1 = data_test_utils.sample_input_data_revenue(
        n_geos=_N_GEOS,
        n_times=_N_TIMES,
        n_media_times=_N_MEDIA_TIMES,
        n_controls=_N_CONTROLS,
        n_media_channels=_N_MEDIA_CHANNELS,
        n_rf_channels=_N_RF_CHANNELS,
        seed=0,
    )
    cls.input_data_2 = (
        data_test_utils.sample_input_data_non_revenue_no_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=_N_CONTROLS,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            seed=0,
        )
    )
    cls.input_data_3 = (
        data_test_utils.sample_input_data_non_revenue_revenue_per_kpi(
            n_geos=_N_GEOS,
            n_times=_N_TIMES,
            n_media_times=_N_MEDIA_TIMES,
            n_controls=_N_CONTROLS,
            n_media_channels=_N_MEDIA_CHANNELS,
            n_rf_channels=_N_RF_CHANNELS,
            seed=0,
        )
    )
    # Testing three scenarios.
    # Label = `Revenue`.
    cls.meridian_revenue = mock.create_autospec(
        model.Meridian,
        instance=True,
        input_data=cls.input_data_1,
    )
    cls.meridian_revenue.input_data.kpi_type = c.REVENUE
    cls.meridian_revenue.input_data.revenue_per_kpi = revenue_per_kpi
    # Label = `KPI`.
    meridian_kpi = mock.create_autospec(
        model.Meridian,
        instance=True,
        input_data=cls.input_data_2,
    )
    meridian_kpi.input_data.kpi_type = c.NON_REVENUE
    meridian_kpi.input_data.revenue_per_kpi = None
    # Label = `Revenue`.
    meridian_revenue_2 = mock.create_autospec(
        model.Meridian,
        instance=True,
        input_data=cls.input_data_3,
    )
    meridian_revenue_2.input_data.kpi_type = c.NON_REVENUE
    meridian_revenue_2.input_data.revenue_per_kpi = revenue_per_kpi
    cls.mock_response_curves_data = test_utils.generate_response_curve_data()
    cls.mock_response_curves_method = cls.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            "response_curves",
            return_value=cls.mock_response_curves_data,
        )
    )
    cls.mock_adstock_decay_dataframe = test_utils.generate_adstock_decay_data()
    cls.mock_adstock_decay_method = cls.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            "adstock_decay",
            return_value=cls.mock_adstock_decay_dataframe,
        )
    )
    cls.mock_hill_curves_dataframe = test_utils.generate_hill_curves_dataframe()
    cls.mock_hill_curves_method = cls.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            "hill_curves",
            return_value=cls.mock_hill_curves_dataframe,
        )
    )
    cls.media_effects_kpi_type_revenue = visualizer.MediaEffects(
        cls.meridian_revenue
    )
    cls.media_effects_kpi_type_kpi = visualizer.MediaEffects(meridian_kpi)
    cls.media_effects_kpi_type_revenue_2 = visualizer.MediaEffects(
        meridian_revenue_2
    )

  def test_media_effects_plot_different_scenarios_labels_correct(self):
    # Verifies each of the three kpi_type and revenue_per_kpi scenarios have
    # the correct labels associated with them.
    plot_revenue_revenue_label = (
        self.media_effects_kpi_type_revenue.plot_response_curves(
            plot_separately=False
        )
    )
    plot_non_revenue_kpi_label = (
        self.media_effects_kpi_type_kpi.plot_response_curves(
            plot_separately=False
        )
    )
    plot_non_revenue_revenue_label = (
        self.media_effects_kpi_type_revenue_2.plot_response_curves(
            plot_separately=False
        )
    )
    self.assertEqual(
        plot_revenue_revenue_label.layer[0].encoding.y["title"],
        summary_text.INC_OUTCOME_LABEL,
    )
    self.assertEqual(
        plot_non_revenue_kpi_label.layer[0].encoding.y["title"],
        summary_text.INC_KPI_LABEL,
    )
    self.assertEqual(
        plot_non_revenue_revenue_label.layer[0].encoding.y["title"],
        summary_text.INC_OUTCOME_LABEL,
    )

  def test_media_effects_caching_done_correctly(self):
    self.media_effects_kpi_type_revenue.response_curves_data(
        selected_times=frozenset(["2021-02-22", "2021-03-01"])
    )
    self.media_effects_kpi_type_revenue.response_curves_data(
        selected_times=frozenset(["2021-02-22", "2021-03-01"])
    )  # Should be retrieved from cache instead of calling Analyzer's method.
    self.mock_response_curves_method.assert_called_once()
    self.media_effects_kpi_type_revenue.adstock_decay_dataframe(
        confidence_level=0.95
    )
    self.media_effects_kpi_type_revenue.adstock_decay_dataframe(
        confidence_level=0.95
    )
    self.mock_adstock_decay_method.assert_called_once()
    self.media_effects_kpi_type_revenue.hill_curves_dataframe(
        confidence_level=0.85
    )
    self.media_effects_kpi_type_revenue.hill_curves_dataframe(
        confidence_level=0.85
    )
    self.mock_hill_curves_method.assert_called_once()

  def test_media_effects_plot_response_curves_plot_include_ci(self):
    plot = self.media_effects_kpi_type_revenue.plot_response_curves(
        plot_separately=False
    )
    band_mark = plot.layer[2].mark
    band_encoding = plot.layer[2].encoding
    self.assertEqual(band_mark.type, "area")
    self.assertEqual(band_mark.opacity, 0.5)
    self.assertEqual(band_encoding.color.shorthand, f"{c.CHANNEL}:N")
    self.assertEqual(band_encoding.x.shorthand, f"{c.SPEND}:Q")
    self.assertEqual(band_encoding.y.shorthand, f"{c.CI_LO}:Q")
    self.assertEqual(band_encoding.y2.shorthand, f"{c.CI_HI}:Q")

  def test_media_effects_plot_response_curves_no_ci(self):
    plot_no_band = self.media_effects_kpi_type_revenue.plot_response_curves(
        plot_separately=False, include_ci=False
    )
    self.assertEqual(plot_no_band.layer[0].encoding.x.shorthand, f"{c.SPEND}:Q")
    self.assertEqual(plot_no_band.layer[0].encoding.y.shorthand, f"{c.MEAN}:Q")
    self.assertLen(plot_no_band.layer, 2)

  def test_media_effects_plot_response_curves_plot_correct_layers(self):
    plot_facet_by_channel = (
        self.media_effects_kpi_type_revenue.plot_response_curves(
            plot_separately=True, num_channels_displayed=5
        )
    )

    self.assertIsInstance(plot_facet_by_channel, alt.FacetChart)
    self.assertIsInstance(plot_facet_by_channel.facet, alt.Facet)
    self.assertLen(plot_facet_by_channel.spec.layer, 3)

  def test_media_effects_plot_response_curves_plot_facet_properties(self):
    plot_facet_by_channel = (
        self.media_effects_kpi_type_revenue.plot_response_curves(
            plot_separately=True, num_channels_displayed=5
        )
    )

    self.assertEqual(plot_facet_by_channel.columns, 3)
    self.assertEqual(plot_facet_by_channel.resolve.scale.x, c.INDEPENDENT)
    self.assertEqual(plot_facet_by_channel.resolve.scale.y, c.INDEPENDENT)

  def test_media_effects_plot_response_curves_solid_striked_line(self):
    plot = self.media_effects_kpi_type_revenue.plot_response_curves(
        plot_separately=False
    )
    layer = plot.layer
    solid_line_layer = layer[0]
    strike_line_layer = layer[1]

    solid_line_encoding = solid_line_layer.encoding
    strike_line_encoding = strike_line_layer.encoding
    self.assertEqual(solid_line_encoding.color.shorthand, f"{c.CHANNEL}:N")
    self.assertEqual(solid_line_encoding.x.shorthand, f"{c.SPEND}:Q")
    self.assertEqual(solid_line_encoding.y.shorthand, f"{c.MEAN}:Q")
    self.assertEqual(strike_line_encoding.color.shorthand, f"{c.CHANNEL}:N")
    self.assertEqual(strike_line_encoding.x.shorthand, f"{c.SPEND}:Q")
    self.assertEqual(strike_line_encoding.y.shorthand, f"{c.MEAN}:Q")

  def test_media_effects_plot_response_curves_points_filled(self):
    plot = self.media_effects_kpi_type_revenue.plot_response_curves(
        plot_separately=False
    )
    layer = plot.layer
    point_encoding = layer[2].encoding
    self.assertEqual(point_encoding.x.shorthand, f"{c.SPEND}:Q")
    self.assertEqual(point_encoding.y.shorthand, f"{c.CI_LO}:Q")
    self.assertEqual(point_encoding.y2.shorthand, f"{c.CI_HI}:Q")
    point_mark = layer[2].mark
    self.assertEqual(point_mark.type, "area")
    self.assertEqual(point_mark.opacity, 0.5)

  def test_media_effects_plot_response_curves_correct_data(self):
    plot = self.media_effects_kpi_type_revenue.plot_response_curves(
        num_channels_displayed=5
    )
    df = plot.data
    self.assertEqual(
        list(df.columns),
        [
            c.CHANNEL,
            c.SPEND,
            c.SPEND_MULTIPLIER,
            c.CI_HI,
            c.CI_LO,
            c.MEAN,
            c.CURRENT_SPEND,
        ],
    )
    self.assertFalse(any(pct > 2.0 or pct < 0.0 for pct in df.spend_multiplier))
    self.assertTrue(spend > 0 for spend in df.spend)

  def test_media_effects_plot_response_curves_customize_channels(self):
    plot_2_layered = self.media_effects_kpi_type_revenue.plot_response_curves(
        plot_separately=False, include_ci=True, num_channels_displayed=2
    )
    plot_3_layered = self.media_effects_kpi_type_revenue.plot_response_curves(
        plot_separately=False, include_ci=True, num_channels_displayed=3
    )
    plot_5_layered = self.media_effects_kpi_type_revenue.plot_response_curves(
        plot_separately=False, include_ci=True, num_channels_displayed=5
    )
    plot_no_num_channel_arg = (
        self.media_effects_kpi_type_revenue.plot_response_curves(
            plot_separately=False, include_ci=True
        )
    )
    self.assertLen(set(plot_2_layered.data[c.CHANNEL]), 2)
    self.assertSetEqual(
        set(plot_2_layered.data[c.CHANNEL]),
        {"channel 0", "channel 3"},
    )
    self.assertEqual(
        plot_2_layered.title.text,
        summary_text.RESPONSE_CURVES_CHART_TITLE.format(top_channels="(top 2)"),
    )
    self.assertLen(set(plot_3_layered.data[c.CHANNEL]), 3)
    self.assertSetEqual(
        set(plot_3_layered.data[c.CHANNEL]),
        {"channel 4", "channel 3", "channel 0"},
    )
    self.assertEqual(
        plot_3_layered.title.text,
        summary_text.RESPONSE_CURVES_CHART_TITLE.format(top_channels="(top 3)"),
    )
    self.assertLen(set(plot_5_layered.data[c.CHANNEL]), 5)
    self.assertEqual(
        plot_5_layered.title.text,
        summary_text.RESPONSE_CURVES_CHART_TITLE.format(top_channels="(top 5)"),
    )
    self.assertLen(set(plot_no_num_channel_arg.data[c.CHANNEL]), 5)
    self.assertEqual(
        plot_no_num_channel_arg.title.text,
        summary_text.RESPONSE_CURVES_CHART_TITLE.format(top_channels="(top 5)"),
    )

  def test_media_effects_plot_response_curves_channels_exceeded_truncated(self):
    plot = self.media_effects_kpi_type_revenue.plot_response_curves(
        plot_separately=False, include_ci=True, num_channels_displayed=20
    )
    self.assertLen(set(plot.data[c.CHANNEL]), 5)

  def test_media_effects_plot_response_curves_axis_configs(self):
    plot = self.media_effects_kpi_type_revenue.plot_response_curves(
        plot_separately=False
    )
    self.assertEqual(plot.config["axis"].to_dict(), formatter.TEXT_CONFIG)
    self.assertEqual(
        plot.layer[0].encoding.x["axis"].to_dict(),
        {"labelExpr": formatter.compact_number_expr()} | formatter.AXIS_CONFIG,
    )
    self.assertEqual(
        plot.layer[0].encoding.y["axis"].to_dict(),
        {"labelExpr": formatter.compact_number_expr()}
        | formatter.Y_AXIS_TITLE_CONFIG,
    )

  def test_media_effects_plot_adstock_decay_plot_include_ci(self):
    plot = self.media_effects_kpi_type_revenue.plot_adstock_decay()

    band_mark = plot.spec.layer[1].mark
    band_encoding = plot.spec.layer[1].encoding

    self.assertEqual(band_mark.type, "area")
    self.assertEqual(band_mark.opacity, 0.2)
    self.assertEqual(band_encoding.color.shorthand, c.DISTRIBUTION)
    self.assertEqual(band_encoding.x.shorthand, f"{c.TIME_UNITS}:Q")
    self.assertEqual(band_encoding.y.shorthand, f"{c.CI_LO}:Q")
    self.assertEqual(band_encoding.y2.shorthand, f"{c.CI_HI}:Q")

  def test_media_effects_plot_adstock_decay_no_ci(self):
    plot_no_band = self.media_effects_kpi_type_revenue.plot_adstock_decay(
        include_ci=False
    )
    self.assertEqual(
        plot_no_band.spec.layer[0].encoding.x.shorthand, f"{c.TIME_UNITS}:Q"
    )
    self.assertEqual(
        plot_no_band.spec.layer[0].encoding.y.shorthand, f"{c.MEAN}:Q"
    )
    self.assertLen(plot_no_band.spec.layer, 2)

  def test_media_effects_plot_adstock_decay_plot_posterior_prior_line(self):
    plot = self.media_effects_kpi_type_revenue.plot_adstock_decay()

    layer = plot.spec.layer
    posterior_or_prior_line_layer = layer[0]
    line_encoding = posterior_or_prior_line_layer.encoding

    self.assertEqual(line_encoding.color.shorthand, c.DISTRIBUTION)
    self.assertEqual(
        line_encoding.color["legend"]["labelExpr"],
        'datum.value === "posterior" ? "posterior (90% CI)" : "prior (90% CI)"',
    )
    self.assertEqual(line_encoding.x.shorthand, f"{c.TIME_UNITS}:Q")
    self.assertEqual(line_encoding.y.shorthand, f"{c.MEAN}:Q")
    self.assertEqual(
        line_encoding.color["scale"]["domain"],
        [c.PRIOR, c.POSTERIOR],
    )

    self.assertEqual(
        line_encoding.color["scale"]["range"], [c.RED_600, c.BLUE_700]
    )

  def test_media_effects_plot_adstock_decay_plot_correct_properties(self):
    plot_facet_by_channel = (
        self.media_effects_kpi_type_revenue.plot_adstock_decay()
    )
    self.assertIsInstance(plot_facet_by_channel, alt.FacetChart)
    self.assertIsInstance(plot_facet_by_channel.facet, alt.Facet)
    self.assertLen(plot_facet_by_channel.spec.layer, 3)
    self.assertEqual(plot_facet_by_channel.columns, 3)
    self.assertEqual(plot_facet_by_channel.resolve.scale.x, c.INDEPENDENT)
    self.assertEqual(plot_facet_by_channel.resolve.scale.y, c.INDEPENDENT)
    self.assertEqual(
        plot_facet_by_channel.title.text, summary_text.ADSTOCK_DECAY_CHART_TITLE
    )

  def test_media_effects_plot_adstock_decay_plot_discrete_value_points(self):
    plot = self.media_effects_kpi_type_revenue.plot_adstock_decay()
    layer = plot.spec.layer
    discrete_value_points_layer = layer[2]
    discrete_value_points_encoding = discrete_value_points_layer.encoding
    discrete_value_points_mark = discrete_value_points_layer.mark

    self.assertEqual(
        discrete_value_points_encoding.color.shorthand, c.DISTRIBUTION
    )
    self.assertEqual(
        discrete_value_points_encoding.x.shorthand, f"{c.TIME_UNITS}:Q"
    )
    self.assertEqual(discrete_value_points_encoding.y.shorthand, f"{c.MEAN}:Q")
    self.assertEqual(
        discrete_value_points_encoding.color["scale"]["domain"],
        [c.PRIOR, c.POSTERIOR],
    )
    self.assertEqual(
        discrete_value_points_encoding.color["scale"]["range"],
        [c.RED_600, c.BLUE_700],
    )

    self.assertEqual(discrete_value_points_mark.filled, True)
    self.assertEqual(discrete_value_points_mark.opacity, 1)
    self.assertEqual(discrete_value_points_mark.size, 80)
    self.assertEqual(discrete_value_points_mark.type, "circle")

  def test_media_effects_plot_adstock_decay_correct_data(self):
    plot = self.media_effects_kpi_type_revenue.plot_adstock_decay()
    df = plot.data
    self.assertEqual(
        list(df.columns),
        [
            c.TIME_UNITS,
            c.CHANNEL,
            c.DISTRIBUTION,
            c.MEAN,
            c.CI_LO,
            c.CI_HI,
        ],
    )
    self.assertTrue(
        all(time_unit >= 0 and time_unit < 21 for time_unit in df.time_units)
    )
    self.assertTrue(ci_hi > 0 for ci_hi in df.ci_hi)
    self.assertTrue(ci_lo > 0 for ci_lo in df.ci_lo)

  @parameterized.named_parameters(
      (
          "media_channel",
          c.MEDIA,
          summary_text.HILL_X_AXIS_MEDIA_LABEL,
          summary_text.HILL_SHADED_REGION_MEDIA_LABEL,
      ),
      (
          "rf_channel",
          c.RF,
          summary_text.HILL_X_AXIS_RF_LABEL,
          summary_text.HILL_SHADED_REGION_RF_LABEL,
      ),
      (
          "organic_media_channel",
          c.ORGANIC_MEDIA,
          summary_text.HILL_X_AXIS_MEDIA_LABEL,
          summary_text.HILL_SHADED_REGION_MEDIA_LABEL,
      ),
      (
          "organic_rf_channel",
          c.ORGANIC_RF,
          summary_text.HILL_X_AXIS_RF_LABEL,
          summary_text.HILL_SHADED_REGION_RF_LABEL,
      ),
  )
  def test_media_effects_plot_hill_curves_channel_properties(
      self, channel_type, expected_x_label, expected_shaded_label
  ):
    plots = self.media_effects_kpi_type_revenue.plot_hill_curves()
    self.assertIn(channel_type, plots)
    plot = plots[channel_type]

    self.assertIsInstance(plot, alt.FacetChart)
    self.assertIsInstance(plot.facet, alt.Facet)
    self.assertLen(plot.spec.layer, 3)
    self.assertEqual(plot.columns, 3)

    band_layer = plot.spec.layer[2]
    self.assertEqual(band_layer.mark.type, "area")
    self.assertEqual(band_layer.mark.opacity, 0.3)
    self.assertEqual(
        band_layer.encoding.color["scale"]["domain"],
        [
            c.POSTERIOR,
            c.PRIOR,
            expected_shaded_label,
        ],
    )

    line_layer = plot.spec.layer[1]
    self.assertEqual(line_layer.encoding.x["title"], expected_x_label)

  def test_media_effects_plot_hill_curves_no_ci(self):
    plot_media = self.media_effects_kpi_type_revenue.plot_hill_curves(
        include_ci=False
    )[c.MEDIA]
    facet_chart_layer = plot_media.spec.layer

    self.assertLen(facet_chart_layer, 2)
    self.assertEqual(
        facet_chart_layer[1].encoding.x.shorthand,
        f"{c.MEDIA_UNITS}:Q",
    )
    self.assertEqual(facet_chart_layer[1].encoding.y.shorthand, f"{c.MEAN}:Q")

  def test_media_effects_plot_hill_curves_no_prior(self):
    plot_media = self.media_effects_kpi_type_revenue.plot_hill_curves(
        include_prior=False
    )[c.MEDIA]
    no_prior_data = plot_media.data
    self.assertNotIn(
        c.PRIOR,
        no_prior_data.distribution,
    )

  def test_media_effects_plot_hill_curves_plot_posterior_prior_lines(self):
    plot_media = self.media_effects_kpi_type_revenue.plot_hill_curves()[c.MEDIA]
    facet_chart_layer = plot_media.spec.layer
    posterior_or_prior_line_layer = facet_chart_layer[1]

    line_encoding = posterior_or_prior_line_layer.encoding
    self.assertEqual(posterior_or_prior_line_layer.mark, "line")
    self.assertEqual(line_encoding.color.shorthand, f"{c.DISTRIBUTION}:N")
    self.assertEqual(line_encoding.x.shorthand, f"{c.MEDIA_UNITS}:Q")
    self.assertFalse(line_encoding.x["scale"]["nice"])
    self.assertEqual(line_encoding.y.shorthand, f"{c.MEAN}:Q")
    self.assertEqual(
        line_encoding.color["scale"]["domain"],
        [c.POSTERIOR, c.PRIOR, summary_text.HILL_SHADED_REGION_MEDIA_LABEL],
    )
    self.assertEqual(
        line_encoding.color["scale"]["range"],
        [c.BLUE_700, c.RED_600, c.GREY_600],
    )

  def test_media_effects_plot_hill_curves_histogram(self):
    plot_media = self.media_effects_kpi_type_revenue.plot_hill_curves()[c.MEDIA]
    facet_chart_layer = plot_media.spec.layer
    histogram_layer = facet_chart_layer[0]

    histogram_encoding = histogram_layer.encoding
    histogram_mark = histogram_layer.mark

    self.assertEqual(
        histogram_encoding.x.shorthand,
        f"{c.START_INTERVAL_HISTOGRAM}:Q",
    )
    self.assertEqual(
        histogram_encoding.x2.shorthand, f"{c.END_INTERVAL_HISTOGRAM}:Q"
    )
    self.assertEqual(
        histogram_encoding.y.shorthand,
        f"{c.SCALED_COUNT_HISTOGRAM}:Q",
    )
    self.assertEqual(histogram_mark.color, c.GREY_600)
    self.assertEqual(histogram_mark.type, "rect")
    self.assertEqual(histogram_mark.opacity, 0.4)

  def test_media_effects_plot_hill_curves_plot_correct_properties(self):
    plot_media = self.media_effects_kpi_type_revenue.plot_hill_curves()[c.MEDIA]

    self.assertIsInstance(plot_media, alt.FacetChart)
    self.assertIsInstance(plot_media.facet, alt.Facet)
    self.assertLen(plot_media.spec.layer, 3)
    self.assertEqual(plot_media.columns, 3)
    self.assertEqual(plot_media.resolve.scale.x, c.INDEPENDENT)
    self.assertEqual(plot_media.resolve.scale.y, c.INDEPENDENT)
    self.assertEqual(
        plot_media.title.text,
        summary_text.HILL_SATURATION_CHART_TITLE,
    )
    self.assertEqual(plot_media.config.legend.labelLimit, 0)

  def test_media_effects_plot_hill_curves_correct_data(self):
    plot_media = self.media_effects_kpi_type_revenue.plot_hill_curves()[c.MEDIA]
    df = plot_media.data
    self.assertEqual(
        list(df.columns),
        [
            c.CHANNEL,
            c.MEDIA_UNITS,
            c.DISTRIBUTION,
            c.CI_HI,
            c.CI_LO,
            c.MEAN,
            c.CHANNEL_TYPE,
            c.SCALED_COUNT_HISTOGRAM,
            c.START_INTERVAL_HISTOGRAM,
            c.END_INTERVAL_HISTOGRAM,
        ],
    )


class MediaSummaryTest(parameterized.TestCase):

  def setUp(self):
    super(MediaSummaryTest, self).setUp()
    n_times = 149
    n_geos = 10
    revenue_per_kpi = data_test_utils.constant_revenue_per_kpi(
        n_geos=n_geos, n_times=n_times, value=2.2
    )
    self.input_data_1 = mock.create_autospec(
        input_data.InputData, instance=True
    )
    self.input_data_2 = mock.create_autospec(
        input_data.InputData, instance=True
    )
    self.input_data_3 = mock.create_autospec(
        input_data.InputData, instance=True
    )

    # Testing three scenarios.
    # Label = `Revenue`.
    self.meridian_revenue = mock.create_autospec(
        model.Meridian,
        instance=True,
        input_data=self.input_data_1,
    )
    self.meridian_revenue.input_data.kpi_type = c.REVENUE
    self.meridian_revenue.input_data.revenue_per_kpi = revenue_per_kpi
    # Label = `KPI`.
    meridian_kpi = mock.create_autospec(
        model.Meridian,
        instance=True,
        input_data=self.input_data_2,
    )
    meridian_kpi.input_data.kpi_type = c.NON_REVENUE
    meridian_kpi.input_data.revenue_per_kpi = None
    # Label = `Revenue`.
    meridian_revenue_2 = mock.create_autospec(
        model.Meridian,
        instance=True,
        input_data=self.input_data_3,
    )
    meridian_revenue_2.input_data.kpi_type = c.NON_REVENUE
    meridian_revenue_2.input_data.revenue_per_kpi = revenue_per_kpi
    self.mock_paid_media_metrics = test_utils.generate_paid_summary_metrics()
    self.mock_all_summary_metrics = test_utils.generate_all_summary_metrics()

    def mock_summary_metrics(**kwargs):
      if kwargs.get("include_non_paid_channels", False):
        return self.mock_all_summary_metrics
      else:
        return self.mock_paid_media_metrics

    self.mock_analyzer_summary_metrics = self.enter_context(
        mock.patch.object(
            analyzer.Analyzer,
            "summary_metrics",
            side_effect=mock_summary_metrics,
        )
    )
    self.media_summary_revenue = visualizer.MediaSummary(self.meridian_revenue)
    self.media_summary_kpi = visualizer.MediaSummary(meridian_kpi)
    self.media_summary_revenue_2 = visualizer.MediaSummary(meridian_revenue_2)

  def test_media_summary_init_non_media_baseline_values(self):
    non_media_baseline_values = [2, 3]
    media_summary = visualizer.MediaSummary(
        self.meridian_revenue,
        non_media_baseline_values=non_media_baseline_values,
    )
    self.assertEqual(
        media_summary._non_media_baseline_values,
        non_media_baseline_values,
    )

  def test_media_summary_get_all_summary_metrics_calls_analyzer_correctly(self):
    non_media_baseline_values = [1.0, 3.0]
    media_summary = visualizer.MediaSummary(
        self.meridian_revenue,
        non_media_baseline_values=non_media_baseline_values,
    )
    _ = media_summary.get_all_summary_metrics()
    self.mock_analyzer_summary_metrics.assert_called_with(
        selected_times=None,
        use_kpi=False,
        confidence_level=0.9,
        include_non_paid_channels=True,
        non_media_baseline_values=non_media_baseline_values,
        aggregate_times=True,
    )

  def test_media_summary_update_non_media_baseline_values(self):
    media_summary = visualizer.MediaSummary(self.meridian_revenue)
    non_media_baseline_values = [1.0, 0.5]
    media_summary.update_summary_metrics(
        non_media_baseline_values=non_media_baseline_values
    )
    self.assertEqual(
        media_summary._non_media_baseline_values,
        non_media_baseline_values,
    )

  def test_media_summary_plot_contribution_waterfall_different_scenarios(self):
    # Verifies each of the three kpi_type and revenue_per_kpi scenarios have
    # the correct labels associated with them.
    plot_revenue_revenue_label = (
        self.media_summary_revenue.plot_contribution_waterfall_chart()
    )
    plot_non_revenue_kpi_label = (
        self.media_summary_kpi.plot_contribution_waterfall_chart()
    )
    plot_non_revenue_revenue_label = (
        self.media_summary_revenue_2.plot_contribution_waterfall_chart()
    )
    self.assertEqual(
        plot_revenue_revenue_label.layer[0].encoding.x["title"], "% Revenue"
    )
    self.assertEqual(
        plot_non_revenue_kpi_label.layer[0].encoding.x["title"], "% KPI"
    )
    self.assertEqual(
        plot_non_revenue_revenue_label.layer[0].encoding.x["title"], "% Revenue"
    )

  def test_media_summary_plot_spend_vs_contribution_different_scenarios(self):
    plot_revenue = self.media_summary_revenue.plot_spend_vs_contribution()
    plot_kpi = self.media_summary_kpi.plot_spend_vs_contribution()
    self.assertEqual(
        plot_revenue.title.text,
        summary_text.SPEND_OUTCOME_CHART_TITLE.format(outcome=c.REVENUE),
    )
    self.assertEqual(
        plot_kpi.title.text,
        summary_text.SPEND_OUTCOME_CHART_TITLE.format(outcome=c.KPI.upper()),
    )
    self.assertEqual(
        plot_revenue.spec.layer[0].encoding.color["scale"]["domain"],
        ["% Revenue", "% Spend", "Return on Investment"],
    )
    self.assertEqual(
        plot_kpi.spec.layer[0].encoding.color["scale"]["domain"],
        ["% KPI", "% Spend", "Return on Investment"],
    )

  def test_media_summary_media_summary_metrics_property(self):
    self.assertEqual(
        self.media_summary_revenue.paid_summary_metrics,
        self.mock_paid_media_metrics,
    )

  def test_media_summary_summary_table_distribution_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "At least one of `include_posterior` or `include_prior` must be True.",
    ):
      self.media_summary_revenue.summary_table(
          include_prior=False, include_posterior=False
      )

  def test_media_summary_summary_table_posterior_only(self):
    df = self.media_summary_revenue.summary_table(include_prior=False)
    self.assertNotIn(c.PRIOR, df[c.DISTRIBUTION])

  def test_media_summary_summary_table_prior_only(self):
    df = self.media_summary_revenue.summary_table(include_posterior=False)
    self.assertNotIn(c.POSTERIOR, df[c.DISTRIBUTION])

  def test_media_summary_summary_table_revenue(self):
    df = self.media_summary_revenue.summary_table()
    self.assertListEqual(
        list(df.columns),
        [
            c.CHANNEL,
            c.DISTRIBUTION,
            c.IMPRESSIONS,
            summary_text.PCT_IMPRESSIONS_COL,
            c.SPEND,
            summary_text.PCT_SPEND_COL,
            c.CPM,
            summary_text.INC_OUTCOME_COL,
            summary_text.PCT_CONTRIBUTION_COL,
            c.ROI,
            c.EFFECTIVENESS,
            c.MROI,
            c.CPIK,
        ],
    )

  def test_media_summary_summary_table_kpi(self):
    media_summary_metrics = (
        test_utils.generate_paid_summary_metrics().drop_vars([c.ROI, c.MROI])
    )
    with mock.patch.object(
        visualizer.MediaSummary, "get_paid_summary_metrics"
    ) as mock_paid_metrics:
      mock_paid_metrics.return_value = media_summary_metrics
      df = self.media_summary_kpi.summary_table()
    self.assertListEqual(
        list(df.columns),
        [
            c.CHANNEL,
            c.DISTRIBUTION,
            c.IMPRESSIONS,
            summary_text.PCT_IMPRESSIONS_COL,
            c.SPEND,
            summary_text.PCT_SPEND_COL,
            c.CPM,
            summary_text.INC_KPI_COL,
            summary_text.PCT_CONTRIBUTION_COL,
            c.EFFECTIVENESS,
            c.CPIK,
        ],
    )

  def test_media_summary_summary_table_format_pct(self):
    df = self.media_summary_revenue.summary_table()
    self.assertTrue(
        all("%" in pct for pct in df[summary_text.PCT_IMPRESSIONS_COL])
    )
    self.assertTrue(all("%" in pct for pct in df[summary_text.PCT_SPEND_COL]))
    self.assertTrue(
        all("%" in pct for pct in df[summary_text.PCT_CONTRIBUTION_COL])
    )

  def test_media_summary_summary_table_format_monetary(self):
    df = self.media_summary_revenue.summary_table()
    self.assertTrue(all("$" in dollar for dollar in df[c.SPEND]))
    self.assertTrue(all("$" in pct for pct in df[summary_text.INC_OUTCOME_COL]))

  def test_media_summary_summary_table_format_credible_interval(self):
    df = self.media_summary_revenue.summary_table()
    inc_rev = self.mock_paid_media_metrics.incremental_outcome
    expected_mean = inc_rev.sel(metric="mean", distribution="prior").values[0]
    expected_ci_lo = inc_rev.sel(metric="ci_lo", distribution="prior").values[0]
    expected_ci_hi = inc_rev.sel(metric="ci_hi", distribution="prior").values[0]
    mean = "${0:,.0f}".format(expected_mean)
    ci_lo = "${0:,.0f}".format(expected_ci_lo)
    ci_hi = "${0:,.0f}".format(expected_ci_hi)
    actual_inc_rev = df[summary_text.INC_OUTCOME_COL][0]
    self.assertEqual(actual_inc_rev, f"{mean} ({ci_lo}, {ci_hi})")
    self.assertContainsInOrder(
        ["(", ",", ")"], df[summary_text.PCT_CONTRIBUTION_COL][0]
    )
    self.assertContainsInOrder(["(", ",", ")"], df[c.ROI][0])
    self.assertContainsInOrder(["(", ",", ")"], df[c.EFFECTIVENESS][0])
    self.assertContainsInOrder(["(", ",", ")"], df[c.MROI][0])

  def test_media_summary_update_ci(self):
    self.assertEqual(
        self.media_summary_revenue.get_paid_summary_metrics().confidence_level,
        c.DEFAULT_CONFIDENCE_LEVEL,
    )
    self.media_summary_revenue.update_summary_metrics(
        confidence_level=0.8, marginal_roi_by_reach=False
    )

  def test_media_summary_update_selected_times(self):
    times = ["2023-01-01", "2023-01-08", "2023-01-15"]
    self.assertEqual(
        self.media_summary_revenue.get_paid_summary_metrics().confidence_level,
        0.9,
    )
    self.media_summary_revenue.update_summary_metrics(
        selected_times=times, marginal_roi_by_reach=False
    )

  def test_media_summary_plot_roi_correct_columns(self):
    plot = self.media_summary_revenue.plot_roi_bar_chart()
    df = plot.data

    self.assertEqual(
        list(df.columns),
        [
            c.CHANNEL,
            c.CI_HI,
            c.CI_LO,
            c.ROI,
        ],
    )

  def test_media_summary_plot_roi_no_ci_plots_bar_chart(self):
    plot = self.media_summary_revenue.plot_roi_bar_chart(include_ci=False)
    self.assertIsInstance(plot, alt.LayerChart)
    self.assertEqual(plot.layer[0].encoding.x.shorthand, f"{c.CHANNEL}:N")
    self.assertEqual(plot.layer[0].encoding.x["axis"]["labelAngle"], -45)
    self.assertEqual(plot.layer[0].encoding.y.shorthand, f"{c.ROI}:Q")

    self.assertEqual(plot.layer[1].encoding.x.shorthand, f"{c.CHANNEL}:N")
    self.assertEqual(plot.layer[1].encoding.y.shorthand, f"{c.ROI}:Q")
    self.assertEqual(plot.layer[1].mark.align, "center")
    self.assertEqual(plot.layer[1].mark.baseline, "bottom")
    self.assertEqual(plot.layer[1].mark.dy, -5)
    self.assertEqual(plot.layer[1].mark.type, "text")

  def test_media_summary_plot_roi_include_ci(self):
    plot = self.media_summary_revenue.plot_roi_bar_chart()
    self.assertIsInstance(plot, alt.LayerChart)
    self.assertLen(plot.layer, 4)
    self.assertEqual(
        plot.title.text,
        summary_text.ROI_CHANNEL_CHART_TITLE_FORMAT.format(
            ci="with 90% credible interval"
        ),
    )
    self.assertEqual(plot.layer[1].encoding.y.shorthand, f"{c.CI_HI}:Q")
    self.assertEqual(plot.layer[1].encoding.y2.shorthand, f"{c.CI_LO}:Q")
    self.assertTrue(plot.layer[1].mark.ticks)
    self.assertTrue(plot.layer[2].mark.tooltip)

    self.assertEqual(plot.layer[3].mark.align, "center")
    self.assertEqual(plot.layer[3].mark.baseline, "bottom")
    self.assertEqual(plot.layer[3].mark.dy, -5)
    self.assertEqual(plot.layer[3].mark.type, "text")

  def test_media_summary_plot_cpik_no_ci_plots_bar_chart(self):
    plot = self.media_summary_kpi.plot_cpik(include_ci=False)
    self.assertIsInstance(plot, alt.LayerChart)
    self.assertEqual(plot.layer[0].encoding.x.shorthand, f"{c.CHANNEL}:N")
    self.assertEqual(plot.layer[0].encoding.x["axis"]["labelAngle"], -45)
    self.assertEqual(plot.layer[0].encoding.y.shorthand, f"{c.CPIK}:Q")

    self.assertEqual(plot.layer[1].encoding.x.shorthand, f"{c.CHANNEL}:N")
    self.assertEqual(plot.layer[1].encoding.y.shorthand, f"{c.CPIK}:Q")
    self.assertEqual(plot.layer[1].mark.align, "center")
    self.assertEqual(plot.layer[1].mark.baseline, "bottom")
    self.assertEqual(plot.layer[1].mark.dy, -5)
    self.assertEqual(plot.layer[1].mark.type, "text")
    self.assertEqual(
        plot.title.text,
        summary_text.CPIK_CHANNEL_CHART_TITLE_FORMAT.format(ci=""),
    )

  def test_media_summary_plot_cpik_include_ci(self):
    plot = self.media_summary_kpi.plot_cpik(include_ci=True)
    self.assertIsInstance(plot, alt.LayerChart)
    self.assertLen(plot.layer, 4)
    self.assertEqual(
        plot.title.text,
        summary_text.CPIK_CHANNEL_CHART_TITLE_FORMAT.format(
            ci="with 90% credible interval"
        ),
    )
    self.assertEqual(plot.layer[1].encoding.y.shorthand, f"{c.CI_HI}:Q")
    self.assertEqual(plot.layer[1].encoding.y2.shorthand, f"{c.CI_LO}:Q")
    self.assertTrue(plot.layer[1].mark.ticks)
    self.assertTrue(plot.layer[2].mark.tooltip)

    self.assertEqual(plot.layer[3].mark.align, "center")
    self.assertEqual(plot.layer[3].mark.baseline, "bottom")
    self.assertEqual(plot.layer[3].mark.dy, -5)
    self.assertEqual(plot.layer[3].mark.type, "text")

  def test_media_summary_plot_channel_contribution_area_chart_correct_data(
      self,
  ):
    summary_metrics = test_utils.generate_all_summary_metrics(
        aggregate_times=False
    )
    with mock.patch.object(
        visualizer.MediaSummary, "get_all_summary_metrics"
    ) as mock_all_metrics:
      mock_all_metrics.return_value = summary_metrics
      plot = self.media_summary_revenue.plot_channel_contribution_area_chart()

    df = plot.data
    self.assertEqual(
        list(df.columns),
        [c.TIME, c.CHANNEL, c.INCREMENTAL_OUTCOME, c.PCT_OF_CONTRIBUTION],
    )
    self.assertIn(c.BASELINE, list(df.channel))
    self.assertEqual(
        df[c.CHANNEL].iloc[0],
        c.BASELINE,
        "Baseline should be last for stacking",
    )

  def test_media_summary_plot_channel_contribution_area_chart_encoding(
      self,
  ):
    summary_metrics = test_utils.generate_all_summary_metrics(
        aggregate_times=False
    )
    with mock.patch.object(
        visualizer.MediaSummary, "get_all_summary_metrics"
    ) as mock_all_metrics:
      mock_all_metrics.return_value = summary_metrics
      plot = self.media_summary_revenue.plot_channel_contribution_area_chart()

    encoding = plot.encoding
    self.assertEqual(encoding.y.shorthand, f"{c.INCREMENTAL_OUTCOME}:Q")
    self.assertEqual(encoding.y["title"], "Revenue")
    self.assertEqual(encoding.color.shorthand, f"{c.CHANNEL}:N")
    self.assertIsNone(encoding.color["legend"]["title"])
    self.assertIsNotNone(encoding.order)
    self.assertEqual(encoding.order["sort"], "descending")
    self.assertIsNotNone(encoding.tooltip)

  def test_plot_channel_contribution_area_chart_encoding_x_axis_quarterly(self):
    summary_metrics = test_utils.generate_all_summary_metrics(
        aggregate_times=False
    )
    with mock.patch.object(
        visualizer.MediaSummary, "get_all_summary_metrics"
    ) as mock_all_metrics:
      mock_all_metrics.return_value = summary_metrics
      plot = self.media_summary_revenue.plot_channel_contribution_area_chart()

    encoding = plot.encoding

    self.assertEqual(encoding.x.shorthand, f"{c.TIME}:T")
    self.assertEqual(encoding.x["title"], "Time period")
    self.assertEqual(encoding.x["axis"]["format"], "%Y %b")
    self.assertFalse(encoding.x["axis"]["grid"])
    self.assertEqual(encoding.x["axis"]["tickCount"], 8)
    self.assertEqual(encoding.x["axis"]["domainColor"], c.GREY_300)

    self.assertIsInstance(encoding.tooltip, list)
    self.assertLen(encoding.tooltip, 3)
    self.assertEqual(encoding.tooltip[0].shorthand, f"{c.TIME}:T")
    self.assertEqual(encoding.tooltip[0]["format"], "%Y-%m-%d")

  def test_plot_channel_contribution_area_chart_encoding_x_axis_weekly(self):
    summary_metrics = test_utils.generate_all_summary_metrics(
        aggregate_times=False
    )
    with mock.patch.object(
        visualizer.MediaSummary, "get_all_summary_metrics"
    ) as mock_all_metrics:
      mock_all_metrics.return_value = summary_metrics
      plot = self.media_summary_revenue.plot_channel_contribution_area_chart(
          time_granularity=c.WEEKLY
      )

    encoding = plot.encoding

    self.assertEqual(encoding.x.shorthand, f"{c.TIME}:T")
    self.assertEqual(encoding.x["title"], "Time period")
    self.assertEqual(encoding.x["axis"]["format"], "%Y-%m-%d")
    self.assertFalse(encoding.x["axis"]["grid"])
    self.assertEqual(encoding.x["axis"]["tickCount"], 8)
    self.assertEqual(encoding.x["axis"]["domainColor"], c.GREY_300)

    self.assertIsInstance(encoding.tooltip, list)
    self.assertLen(encoding.tooltip, 3)
    self.assertEqual(encoding.tooltip[0].shorthand, f"{c.TIME}:T")
    self.assertEqual(encoding.tooltip[0]["format"], "%Y-%m-%d")

  def test_media_summary_plot_channel_contribution_area_chart_y_axis_label(
      self,
  ):
    summary_metrics = test_utils.generate_all_summary_metrics(
        aggregate_times=False
    )
    with mock.patch.object(
        visualizer.MediaSummary, "get_all_summary_metrics"
    ) as mock_all_metrics:
      mock_all_metrics.return_value = summary_metrics
      plot_kpi = self.media_summary_kpi.plot_channel_contribution_area_chart()
      plot_revenue = (
          self.media_summary_revenue.plot_channel_contribution_area_chart()
      )
    self.assertEqual(plot_kpi.encoding.y["title"], "KPI")
    self.assertEqual(plot_revenue.encoding.y["title"], "Revenue")

  def test_media_summary_plot_channel_contribution_area_chart_baseline_min(
      self,
  ):
    summary_metrics = xr.Dataset(
        data_vars={
            c.INCREMENTAL_OUTCOME: (
                [c.TIME, c.CHANNEL, c.METRIC, c.DISTRIBUTION],
                np.array([
                    [[[500]], [[2000]], [[1500]], [[400]]],
                    [[[600]], [[1800]], [[1200]], [[360]]],
                    [[[400]], [[2200]], [[1600]], [[-1]]],
                ]),
            ),
            c.PCT_OF_CONTRIBUTION: (
                [c.TIME, c.CHANNEL, c.METRIC, c.DISTRIBUTION],
                np.array([
                    [[[5]], [[20]], [[15]], [[10]]],
                    [[[6]], [[18]], [[12]], [[9]]],
                    [[[4]], [[22]], [[16]], [[1]]],
                ]),
            ),
            c.EFFECTIVENESS: (
                [c.TIME, c.CHANNEL, c.METRIC, c.DISTRIBUTION],
                np.zeros(shape=(3, 4, 1, 1)),
            ),
        },
        coords={
            c.TIME: ["2023-01-01", "2023-01-08", "2023-01-15"],
            c.CHANNEL: ["1", "2", "3", c.ALL_CHANNELS],
            c.METRIC: [c.MEAN],
            c.DISTRIBUTION: [c.POSTERIOR],
        },
        attrs={c.CONFIDENCE_LEVEL: c.DEFAULT_CONFIDENCE_LEVEL},
    )

    with mock.patch.object(
        visualizer.MediaSummary,
        "get_all_summary_metrics",
        return_value=summary_metrics,
    ) as _:
      plot = self.media_summary_revenue.plot_channel_contribution_area_chart()

    # Calculate the expected minimum y-value by finding the minimum baseline
    # outcome across all time periods.
    total_media_criteria = {
        c.DISTRIBUTION: c.POSTERIOR,
        c.METRIC: c.MEAN,
        c.CHANNEL: c.ALL_CHANNELS,
        c.TIME: summary_metrics.time,
    }
    total_media_outcome = summary_metrics[c.INCREMENTAL_OUTCOME].sel(
        total_media_criteria
    )
    total_media_pct = (
        summary_metrics[c.PCT_OF_CONTRIBUTION].sel(total_media_criteria) / 100
    )
    total_outcome = total_media_outcome / total_media_pct
    baseline_pct = 1 - total_media_pct
    baseline_outcome = total_outcome * baseline_pct
    expected_min_y = baseline_outcome.min()

    self.assertEqual(plot.encoding.y["scale"]["domainMin"], expected_min_y)
    self.assertTrue(plot.encoding.y["scale"]["clamp"])

  def test_media_summary_plot_channel_contribution_bump_chart_encoding(self):
    summary_metrics = test_utils.generate_all_summary_metrics(
        aggregate_times=False
    )

    with mock.patch.object(
        visualizer.MediaSummary, "get_all_summary_metrics"
    ) as mock_all_metrics:
      mock_all_metrics.return_value = summary_metrics
      plot = self.media_summary_revenue.plot_channel_contribution_bump_chart()

    self.assertEqual(plot.mark.type, "line")
    self.assertTrue(plot.mark.point)

    self.assertEqual(plot.encoding.y.shorthand, "rank:Q")
    self.assertEqual(plot.encoding.y["title"], "Contribution Rank")
    self.assertTrue(plot.encoding.y["scale"].reverse)
    self.assertFalse(plot.encoding.y["scale"].zero)
    self.assertEqual(plot.encoding.y["axis"]["tickMinStep"], 1)

    expected_legend_order = [
        "baseline",
        "ch_0",
        "ch_1",
        "ch_2",
        "non_media_0",
        "non_media_1",
        "organic_media_0",
        "organic_media_1",
        "organic_media_2",
        "organic_media_3",
        "organic_rf_ch_0",
        "rf_ch_0",
        "rf_ch_1",
    ]
    self.assertListEqual(
        plot.encoding.color["scale"].domain, expected_legend_order
    )
    self.assertListEqual(plot.encoding.color["sort"], expected_legend_order)

    self.assertEqual(
        plot.title.text, summary_text.CHANNEL_CONTRIB_RANK_CHART_TITLE
    )

  def test_plot_channel_contribution_bump_chart_encoding_x_axis_quarterly(self):
    summary_metrics = test_utils.generate_all_summary_metrics(
        aggregate_times=False
    )

    with mock.patch.object(
        visualizer.MediaSummary, "get_all_summary_metrics"
    ) as mock_all_metrics:
      mock_all_metrics.return_value = summary_metrics
      plot = self.media_summary_revenue.plot_channel_contribution_bump_chart()
    encoding = plot.encoding

    self.assertEqual(encoding.x.shorthand, f"{c.TIME}:T")
    self.assertEqual(encoding.x["title"], "Time period")
    self.assertEqual(encoding.x["axis"].format, "%Y %b")
    self.assertFalse(encoding.x["axis"].grid)

    tooltip_defs = encoding.tooltip
    tooltip_list_dict = [item.to_dict() for item in tooltip_defs]

    self.assertLen(tooltip_list_dict, 4)
    self.assertEqual(tooltip_list_dict[0]["field"], c.TIME)
    self.assertEqual(tooltip_list_dict[0]["type"], "temporal")
    self.assertEqual(tooltip_list_dict[0]["format"], "%Y %b")
    self.assertEqual(tooltip_list_dict[0]["title"], "Quarter")

  def test_plot_channel_contribution_bump_chart_encoding_x_axis_weekly(self):
    summary_metrics = test_utils.generate_all_summary_metrics(
        aggregate_times=False
    )

    with mock.patch.object(
        visualizer.MediaSummary, "get_all_summary_metrics"
    ) as mock_all_metrics:
      mock_all_metrics.return_value = summary_metrics
      plot = self.media_summary_revenue.plot_channel_contribution_bump_chart(
          time_granularity=c.WEEKLY
      )
    encoding = plot.encoding

    self.assertEqual(encoding.x.shorthand, f"{c.TIME}:T")
    self.assertEqual(encoding.x["title"], "Time period")
    self.assertEqual(encoding.x["axis"].format, "%Y-%m-%d")
    self.assertFalse(encoding.x["axis"].grid)

    tooltip_defs = encoding.tooltip
    tooltip_list_dict = [item.to_dict() for item in tooltip_defs]

    self.assertLen(tooltip_list_dict, 4)
    self.assertEqual(tooltip_list_dict[0]["field"], c.TIME)
    self.assertEqual(tooltip_list_dict[0]["type"], "temporal")
    self.assertEqual(tooltip_list_dict[0]["format"], "%Y-%m-%d")
    self.assertEqual(tooltip_list_dict[0]["title"], "Week")

  def test_media_summary_plot_channel_contribution_bump_chart_correct_data(
      self,
  ):
    times = [
        "2023-01-15",
        "2023-03-26",
        "2023-04-09",
        "2023-06-25",
        "2023-07-02",
        "2023-09-24",
    ]
    channels = ["Ch1", "Ch2", "Ch3", c.ALL_CHANNELS]
    metrics = [c.MEAN]
    distributions = [c.POSTERIOR]
    dims = [c.TIME, c.CHANNEL, c.METRIC, c.DISTRIBUTION]

    summary_metrics = xr.Dataset(
        data_vars={
            c.INCREMENTAL_OUTCOME: (
                dims,
                np.array([
                    [[[0]], [[0]], [[0]], [[0]]],  # Time 0
                    [[[1000]], [[800]], [[1200]], [[3000]]],  # Time 1 (Q1 End)
                    [[[0]], [[0]], [[0]], [[0]]],  # Time 2
                    [[[1100]], [[1300]], [[900]], [[3300]]],  # Time 3 (Q2 End)
                    [[[0]], [[0]], [[0]], [[0]]],  # Time 4
                    [[[1500]], [[1400]], [[4600]], [[4500]]],  # Time 5 (Q3 End)
                ]).astype(np.float32),
            ),
            c.PCT_OF_CONTRIBUTION: (
                dims,
                np.array([
                    [[[0]], [[0]], [[0]], [[0]]],
                    [[[0]], [[0]], [[0]], [[50]]],
                    [[[0]], [[0]], [[0]], [[0]]],
                    [[[0]], [[0]], [[0]], [[50]]],
                    [[[0]], [[0]], [[0]], [[0]]],
                    [[[0]], [[0]], [[0]], [[50]]],
                ]),
            ),
            c.EFFECTIVENESS: (
                dims,
                np.zeros(
                    (
                        len(times),
                        len(channels),
                        len(metrics),
                        len(distributions),
                    ),
                ),
            ),
        },
        coords={
            c.TIME: times,
            c.CHANNEL: channels,
            c.METRIC: metrics,
            c.DISTRIBUTION: distributions,
        },
        attrs={c.CONFIDENCE_LEVEL: c.DEFAULT_CONFIDENCE_LEVEL},
    )
    with mock.patch.object(
        visualizer.MediaSummary, "get_all_summary_metrics"
    ) as mock_all_metrics:
      mock_all_metrics.return_value = summary_metrics
      plot = self.media_summary_revenue.plot_channel_contribution_bump_chart()

    df = plot.data

    expected_columns = [
        c.TIME,
        c.CHANNEL,
        c.INCREMENTAL_OUTCOME,
        c.PCT_OF_CONTRIBUTION,
        "rank",
    ]
    self.assertListEqual(list(df.columns), expected_columns)

    q1_date = np.datetime64("2023-03-26")
    q2_date = np.datetime64("2023-06-25")
    q3_date = np.datetime64("2023-09-24")

    # Q1 (2023-03-26): Base(3000), Ch3(1200), Ch1(1000), Ch2(800)
    self.assertEqual(
        df.loc[
            (df[c.TIME] == q1_date) & (df[c.CHANNEL] == c.BASELINE),
            "rank",
        ].item(),
        1.0,
    )
    self.assertEqual(
        df.loc[
            (df[c.TIME] == q1_date) & (df[c.CHANNEL] == "Ch3"), "rank"
        ].item(),
        2.0,
    )
    self.assertEqual(
        df.loc[
            (df[c.TIME] == q1_date) & (df[c.CHANNEL] == "Ch1"), "rank"
        ].item(),
        3.0,
    )
    self.assertEqual(
        df.loc[
            (df[c.TIME] == q1_date) & (df[c.CHANNEL] == "Ch2"), "rank"
        ].item(),
        4.0,
    )

    # Q2 (2023-06-25): Base(3300), Ch2(1300), Ch1(1100), Ch3(900)
    self.assertEqual(
        df.loc[
            (df[c.TIME] == q2_date) & (df[c.CHANNEL] == c.BASELINE),
            "rank",
        ].item(),
        1.0,
    )
    self.assertEqual(
        df.loc[
            (df[c.TIME] == q2_date) & (df[c.CHANNEL] == "Ch2"), "rank"
        ].item(),
        2.0,
    )
    self.assertEqual(
        df.loc[
            (df[c.TIME] == q2_date) & (df[c.CHANNEL] == "Ch1"), "rank"
        ].item(),
        3.0,
    )
    self.assertEqual(
        df.loc[
            (df[c.TIME] == q2_date) & (df[c.CHANNEL] == "Ch3"), "rank"
        ].item(),
        4.0,
    )

    # Q3 (2023-09-24): Ch3(4600), Base(4500), Ch1(1500), Ch2(1400)
    self.assertEqual(
        df.loc[
            (df[c.TIME] == q3_date) & (df[c.CHANNEL] == "Ch3"),
            "rank",
        ].item(),
        1.0,
    )
    self.assertEqual(
        df.loc[
            (df[c.TIME] == q3_date) & (df[c.CHANNEL] == c.BASELINE),
            "rank",
        ].item(),
        2.0,
    )
    self.assertEqual(
        df.loc[
            (df[c.TIME] == q3_date) & (df[c.CHANNEL] == "Ch1"), "rank"
        ].item(),
        3.0,
    )
    self.assertEqual(
        df.loc[
            (df[c.TIME] == q3_date) & (df[c.CHANNEL] == "Ch2"), "rank"
        ].item(),
        4.0,
    )

    # Verify correct channels are present (Baseline + the mocked channels)
    expected_channels_in_plot = {"Ch1", "Ch2", "Ch3", c.BASELINE}
    self.assertSetEqual(set(df[c.CHANNEL].unique()), expected_channels_in_plot)

  def test_media_summary_plot_waterfall_chart_correct_data(self):
    summary_metrics = xr.Dataset(
        data_vars={
            c.INCREMENTAL_OUTCOME: (
                [c.CHANNEL, c.METRIC, c.DISTRIBUTION],
                [[[500]], [[2000]], [[1500]], [[4000]]],
            ),
            c.PCT_OF_CONTRIBUTION: (
                [c.CHANNEL, c.METRIC, c.DISTRIBUTION],
                [[[5]], [[20]], [[15]], [[40]]],
            ),
        },
        coords={
            c.CHANNEL: ["1", "2", "3", c.ALL_CHANNELS],
            c.METRIC: [c.MEAN],
            c.DISTRIBUTION: [c.POSTERIOR],
        },
    )
    with mock.patch.object(
        visualizer.MediaSummary, "get_all_summary_metrics"
    ) as mock_all_metrics:
      mock_all_metrics.return_value = summary_metrics
      plot = self.media_summary_revenue.plot_contribution_waterfall_chart()

    df = plot.data
    self.assertEqual(
        list(df.columns),
        [
            c.CHANNEL,
            c.INCREMENTAL_OUTCOME,
            c.PCT_OF_CONTRIBUTION,
            "outcome_text",
        ],
    )
    self.assertIn(c.BASELINE.upper(), list(df.channel))
    self.assertFalse(any(pct > 1 or pct < 0 for pct in df.pct_of_contribution))
    self.assertEqual(
        df[c.CHANNEL].to_list(),
        [c.BASELINE.upper(), "2", "3", "1"],
    )

    baseline_pct = df.loc[df[c.CHANNEL] == c.BASELINE.upper()][
        c.PCT_OF_CONTRIBUTION
    ].item()
    baseline_outcome = df.loc[df[c.CHANNEL] == c.BASELINE.upper()][
        c.INCREMENTAL_OUTCOME
    ].item()
    baseline_outcome_text = df.loc[df[c.CHANNEL] == c.BASELINE.upper()][
        "outcome_text"
    ].item()
    self.assertEqual(baseline_pct, 0.6)
    self.assertEqual(baseline_outcome, 6000.0)
    self.assertEqual(baseline_outcome_text, "60.0% (6k)")

  @parameterized.named_parameters(
      ("no_suffix", 40, 400, "60.0% (600)"),
      ("k_suffix", 40, 400000, "60.0% (600k)"),
      ("m_suffix", 40, 40000000, "60.0% (60M)"),
      ("b_suffix", 40, 4000000000, "60.0% (6B)"),
      ("beyond_t", 40, 4e15, "60.0% (6000T)"),
  )
  def test_media_summary_plot_waterfall_chart_correct_formatted_text(
      self, pct, outcome, expected_text
  ):
    summary_metrics = test_utils.generate_all_summary_metrics()
    total_media_dict = {
        c.CHANNEL: c.ALL_CHANNELS,
        c.DISTRIBUTION: c.POSTERIOR,
        c.METRIC: c.MEAN,
    }
    summary_metrics[c.INCREMENTAL_OUTCOME].loc[total_media_dict] = outcome
    summary_metrics[c.PCT_OF_CONTRIBUTION].loc[total_media_dict] = pct
    with mock.patch.object(
        visualizer.MediaSummary, "get_all_summary_metrics"
    ) as mock_all_metrics:
      mock_all_metrics.return_value = summary_metrics
      plot = self.media_summary_revenue.plot_contribution_waterfall_chart()
    df = plot.data
    baseline_outcome_text = df.loc[df[c.CHANNEL] == c.BASELINE.upper()][
        "outcome_text"
    ].item()
    self.assertEqual(baseline_outcome_text, expected_text)

  def test_media_summary_plot_waterfall_chart_correct_marks(self):
    plot = self.media_summary_revenue.plot_contribution_waterfall_chart()
    self.assertEqual(plot.layer[0].mark.type, "bar")
    self.assertEqual(plot.layer[1].mark.type, "text")

  def test_media_summary_plot_waterfall_chart_correct_bar_encoding(self):
    plot = self.media_summary_revenue.plot_contribution_waterfall_chart()
    encoding = plot.layer[0].encoding
    self.assertEqual(encoding.x.shorthand, "prev_sum:Q")
    self.assertEqual(encoding.x["title"], "% Revenue")
    self.assertEqual(encoding.x2.shorthand, "sum_outcome:Q")
    self.assertEqual(encoding.y.shorthand, f"{c.CHANNEL}:N")
    self.assertIsNotNone(encoding.color)
    self.assertIsNotNone(encoding.x.axis)
    self.assertFalse(encoding.y["axis"]["domain"])
    self.assertFalse(encoding.y["axis"]["ticks"])

  def test_media_summary_plot_waterfall_chart_correct_properties(self):
    plot = self.media_summary_revenue.plot_contribution_waterfall_chart()
    self.assertEqual(plot.layer[0].mark.size, 42)
    self.assertLen(plot.data.channel, 13)
    self.assertEqual(plot.layer[0].encoding.y["scale"]["paddingOuter"], 0.2)
    expected_height = 42 * 13 + 42 * 2 * 0.2
    self.assertEqual(plot.height, expected_height)
    self.assertEqual(plot.width, 500)
    self.assertEqual(plot.title.text, summary_text.CHANNEL_DRIVERS_CHART_TITLE)

  def test_media_summary_plot_waterfall_chart_correct_text_encoding(self):
    plot = self.media_summary_revenue.plot_contribution_waterfall_chart()
    encoding = plot.layer[1].encoding
    self.assertEqual(encoding.text.shorthand, "outcome_text")
    self.assertEqual(encoding.x.shorthand, "text_x:Q")
    self.assertEqual(encoding.y.shorthand, f"{c.CHANNEL}:N")
    self.assertIsNotNone(encoding.y.axis)
    self.assertIsNone(encoding.y["sort"])
    self.assertIsNone(encoding.y["title"])

  def test_media_summary_plot_waterfall_chart_correct_config(self):
    plot = self.media_summary_revenue.plot_contribution_waterfall_chart()
    config = plot.config.to_dict()
    self.assertEqual(
        config["axis"],
        {
            "labelColor": c.GREY_700,
            "labelFont": c.FONT_ROBOTO,
            "labelFontSize": c.AXIS_FONT_SIZE,
            "titleColor": c.GREY_700,
            "titleFont": c.FONT_ROBOTO,
            "titleFontSize": c.AXIS_FONT_SIZE,
            "titleFontWeight": "normal",
            "titlePadding": c.PADDING_10,
        },
    )
    self.assertEqual(config["view"], {"strokeOpacity": 0})

  def test_media_summary_plot_pie_chart_correct_data(self):
    plot = self.media_summary_revenue.plot_contribution_pie_chart()
    self.assertEqual(sum(plot.data[c.PCT_OF_CONTRIBUTION]), 1)
    self.assertSetEqual(
        set(plot.data[c.CHANNEL]),
        {c.BASELINE, c.ALL_CHANNELS},
    )

  def test_media_summary_plot_pie_chart_correct_mark(self):
    plot = self.media_summary_revenue.plot_contribution_pie_chart()
    self.assertEqual(plot.title.text, summary_text.CONTRIBUTION_CHART_TITLE)
    self.assertEqual(
        plot.layer[0].mark.to_dict(),
        {"innerRadius": 70, "outerRadius": 150, "type": "arc"},
    )
    self.assertEqual(
        plot.layer[1].mark.to_dict(),
        {
            "fill": "white",
            "font": c.FONT_ROBOTO,
            "radius": 110,
            "size": c.TITLE_FONT_SIZE,
            "type": "text",
        },
    )

  def test_media_summary_plot_pie_chart_correct_encoding(self):
    plot = self.media_summary_revenue.plot_contribution_pie_chart()
    arc_encoding = plot.layer[0].encoding.to_dict()
    text_encoding = plot.layer[1].encoding.to_dict()
    text = text_encoding.pop("text")

    self.assertEqual(arc_encoding, text_encoding)
    self.assertEqual(
        arc_encoding,
        {
            "color": {
                "field": c.CHANNEL,
                "legend": {
                    "direction": "horizontal",
                    "labelFont": c.FONT_ROBOTO,
                    "labelFontSize": c.AXIS_FONT_SIZE,
                    "legendX": 130,
                    "legendY": 320,
                    "orient": "none",
                    "title": None,
                },
                "scale": {
                    "domain": [c.BASELINE, c.ALL_CHANNELS],
                    "range": [c.YELLOW_600, c.BLUE_700],
                },
                "type": "nominal",
            },
            "theta": {
                "field": c.PCT_OF_CONTRIBUTION,
                "stack": True,
                "type": "quantitative",
            },
        },
    )
    self.assertEqual(
        text,
        {
            "field": c.PCT_OF_CONTRIBUTION,
            "format": ".0%",
            "type": "quantitative",
        },
    )

  def test_media_summary_plot_spend_vs_contribution_correct_data(self):
    media_summary = visualizer.MediaSummary(self.meridian_revenue)
    media_metrics = test_utils.generate_paid_summary_metrics()
    total_media_dict = {
        c.CHANNEL: c.ALL_CHANNELS,
        c.DISTRIBUTION: c.POSTERIOR,
        c.METRIC: c.MEAN,
    }
    media_metrics[c.INCREMENTAL_OUTCOME].loc[total_media_dict] = 200000
    media_metrics[c.PCT_OF_CONTRIBUTION].loc[total_media_dict] = 60
    with mock.patch.object(
        visualizer.MediaSummary, "get_paid_summary_metrics"
    ) as mock_paid_metrics:
      mock_paid_metrics.return_value = media_metrics
      plot = media_summary.plot_spend_vs_contribution()
    df = plot.data
    self.assertEqual(
        list(df.columns),
        [
            c.CHANNEL,
            c.PCT,
            "label",
            c.ROI,
            "roi_scaled",
        ],
    )
    self.assertNotIn(c.ALL_CHANNELS, list(df.channel))
    self.assertLen(df, len(set(df.channel)) * 2)
    self.assertEqual(
        len(df[df.label == "% Revenue"]), len(df[df.label == "% Spend"])
    )
    self.assertFalse(
        any(scaled_roi > 1 or scaled_roi < 0 for scaled_roi in df.roi_scaled)
    )

  def test_media_summary_plot_spend_vs_contribution_correct_facet_config(self):
    plot = self.media_summary_revenue.plot_spend_vs_contribution()
    self.assertEqual(
        plot.facet.column.to_dict(),
        {
            "field": c.CHANNEL,
            "type": "nominal",
        },
    )
    self.assertEqual(plot.spacing, -1)
    self.assertEqual(plot.config.view.strokeOpacity, 0)
    self.assertEqual(
        plot.config.header.to_dict(),
        {
            "labelAlign": "right",
            "labelAngle": -45,
            "labelOrient": "bottom",
            "title": None,
            "labelBaseline": "middle",
        },
    )
    self.assertEqual(
        plot.title.text,
        summary_text.SPEND_OUTCOME_CHART_TITLE.format(outcome=c.REVENUE),
    )

  def test_media_summary_plot_spend_vs_contribution_roi_marker(self):
    plot = self.media_summary_revenue.plot_spend_vs_contribution()
    self.assertEqual(
        plot.spec.layer[1].encoding.to_dict(),
        {
            "tooltip": [{
                "field": c.ROI,
                "format": ".2f",
                "type": "quantitative",
            }],
            "y": {"field": "roi_scaled", "title": "%", "type": "quantitative"},
        },
    )
    self.assertEqual(
        plot.spec.layer[1].mark.to_dict(),
        {
            "color": c.GREEN_700,
            "cornerRadius": c.CORNER_RADIUS,
            "size": c.PADDING_20,
            "thickness": 4,
            "tooltip": True,
            "type": "tick",
        },
    )

  def test_media_summary_plot_spend_vs_contribution_roi_text(self):
    plot = self.media_summary_revenue.plot_spend_vs_contribution()
    self.assertEqual(
        plot.spec.layer[2].encoding.to_dict(),
        {
            "text": {
                "field": c.ROI,
                "format": ".1f",
                "type": "quantitative",
            },
            "y": {"field": "roi_scaled", "type": "quantitative"},
        },
    )
    self.assertEqual(
        plot.spec.layer[2].mark.to_dict(),
        {
            "color": c.GREY_900,
            "dy": -15,
            "fontSize": c.AXIS_FONT_SIZE,
            "type": "text",
        },
    )

  def test_media_summary_plot_spend_vs_contribution_bar_chart(self):
    plot = self.media_summary_revenue.plot_spend_vs_contribution()
    self.assertEqual(
        plot.spec.layer[0].encoding.to_dict(),
        {
            "color": {
                "field": "label",
                "legend": {
                    "columnPadding": c.PADDING_20,
                    "orient": "bottom",
                    "rowPadding": c.PADDING_10,
                    "title": None,
                },
                "scale": {
                    "domain": ["% Revenue", "% Spend", "Return on Investment"],
                    "range": [
                        c.BLUE_400,
                        c.BLUE_200,
                        c.GREEN_700,
                    ],
                },
                "type": "nominal",
            },
            "tooltip": [{
                "field": c.PCT,
                "format": ".1%",
                "type": "quantitative",
            }],
            "x": {
                "axis": {"labels": False, "ticks": False, "title": None},
                "field": "label",
                "scale": {"paddingOuter": 0.5},
                "type": "nominal",
            },
            "y": {
                "axis": {
                    "format": "%",
                    "tickCount": 2,
                    "titleAlign": "left",
                    "titleAngle": 0,
                    "titleY": -20,
                },
                "field": c.PCT,
                "type": "quantitative",
            },
        },
    )
    self.assertEqual(
        plot.spec.layer[0].mark.to_dict(),
        {
            "cornerRadiusEnd": c.CORNER_RADIUS,
            "tooltip": True,
            "type": "bar",
        },
    )

  def test_media_summary_plot_roi_vs_effectiveness_correct_data(self):
    plot = self.media_summary_revenue.plot_roi_vs_effectiveness()
    self.assertEqual(
        list(plot.data.columns),
        [
            c.CHANNEL,
            c.ROI,
            c.EFFECTIVENESS,
            c.SPEND,
        ],
    )
    self.assertNotIn(c.ALL_CHANNELS, list(plot.data.channel))

  def test_media_summary_plot_roi_vs_effectiveness_correct_mark(self):
    plot = self.media_summary_revenue.plot_roi_vs_effectiveness()
    self.assertEqual(plot.mark.type, "circle")
    self.assertTrue(plot.mark.tooltip)

  def test_media_summary_plot_roi_vs_effectiveness_correct_encoding(self):
    plot = self.media_summary_revenue.plot_roi_vs_effectiveness()
    self.assertEqual(plot.encoding.x.shorthand, c.ROI)
    self.assertEqual(plot.encoding.y.shorthand, c.EFFECTIVENESS)
    self.assertEqual(plot.encoding.size.shorthand, c.SPEND)
    self.assertIsNone(plot.encoding.size["legend"])
    self.assertEqual(plot.encoding.color.shorthand, f"{c.CHANNEL}:N")
    self.assertEqual(plot.encoding.y["axis"]["titleY"], -20)

  def test_media_summary_plot_roi_vs_effectiveness_correct_title(self):
    plot = self.media_summary_revenue.plot_roi_vs_effectiveness()
    self.assertEqual(
        plot.title.text, summary_text.ROI_EFFECTIVENESS_CHART_TITLE
    )
    self.assertEqual(plot.title.anchor, "start")
    self.assertEqual(plot.title.font, "Google Sans Display")
    self.assertEqual(plot.title.fontSize, 18)
    self.assertEqual(plot.title.fontWeight, "normal")
    self.assertEqual(plot.title.offset, 10)

  def test_media_summary_plot_roi_vs_effectiveness_correct_config(self):
    plot = self.media_summary_revenue.plot_roi_vs_effectiveness()
    expected_config = dict(formatter.TEXT_CONFIG) | {
        "gridDash": [3, 2],
        "titlePadding": 10,
    }
    self.assertEqual(plot.config.axis.to_dict(), expected_config)

  def test_media_summary_plot_roi_vs_effectiveness_disable_size(self):
    plot = self.media_summary_revenue.plot_roi_vs_effectiveness(
        disable_size=True
    )
    encoding = plot["encoding"]
    self.assertIsNotNone(encoding)
    self.assertEqual(encoding.size, alt.utils.schemapi.Undefined)

  def test_media_summary_plot_roi_vs_mroi_correct_data(self):
    plot = self.media_summary_revenue.plot_roi_vs_mroi()
    self.assertIn(c.MROI, list(plot.data.columns))

  def test_media_summary_plot_roi_vs_mroi_correct_encoding(self):
    plot = self.media_summary_revenue.plot_roi_vs_mroi()
    self.assertEqual(plot.title.text, summary_text.ROI_MARGINAL_CHART_TITLE)
    self.assertEqual(plot.encoding.y.shorthand, c.MROI)

  def test_media_summary_plot_roi_vs_mroi_selected_channels(self):
    selected_channels = ["ch_0", "ch_2"]
    plot = self.media_summary_revenue.plot_roi_vs_mroi(
        selected_channels=selected_channels
    )
    self.assertEqual(plot.data.channel.unique().tolist(), selected_channels)

  def test_media_summary_plot_roi_vs_mroi_incorrect_selected_channels(self):
    with self.assertRaisesRegex(
        ValueError,
        "`selected_channels` should match the channel dimension names from "
        "meridian.InputData",
    ):
      self.media_summary_revenue.plot_roi_vs_mroi(
          selected_channels=["wrong channel"]
      )

  def test_media_summary_plot_roi_vs_mroi_equal_axes(self):
    plot = self.media_summary_revenue.plot_roi_vs_mroi(equal_axes=True)
    max_value = max(plot.data.roi.max(), plot.data.mroi.max())
    self.assertEqual(plot.encoding.x["scale"], plot.encoding.y["scale"])
    self.assertEqual(plot.encoding.x["scale"]["domain"][1], max_value)


if __name__ == "__main__":
  absltest.main()
