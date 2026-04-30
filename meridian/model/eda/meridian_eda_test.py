# Copyright 2026 The Meridian Authors.
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

from collections.abc import Sequence
import itertools
import os
from xml.etree import ElementTree as ET
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import altair as alt
import arviz as az
from meridian import constants
from meridian.analysis import analyzer as analyzer_module
from meridian.backend import test_utils as backend_test_utils
from meridian.data import input_data
from meridian.data import test_utils as data_test_utils
from meridian.model import context
from meridian.model import model
from meridian.model.eda import constants as eda_constants
from meridian.model.eda import eda_engine
from meridian.model.eda import eda_outcome
from meridian.model.eda import eda_spec
from meridian.model.eda import meridian_eda
from meridian.model.eda import sampling_eda_engine
import numpy as np
import pandas as pd
import xarray as xr

mock = absltest.mock

_N_GEOS = 2
_N_TIMES = 3
_N_MEDIA_CHANNELS = 2
_N_CONTROLS = 2
_N_RF_CHANNELS = 1
_N_NON_MEDIA_CHANNELS = 1
_GEO_NAMES = ('geo_0', 'geo_1')
_MEDIA_CHANNEL_NAMES = ('ch_0', 'ch_1')
_RF_CHANNEL_NAMES = ('rf_ch_0',)
_CONTROL_NAMES = ('control_0', 'control_1')
_NON_MEDIA_CHANNEL_NAMES = ('non_media_0',)

Cause = eda_outcome.FindingCause
Severity = eda_outcome.EDASeverity


def _create_treatments_without_non_media_scaled_ds(
    is_national: bool,
) -> xr.Dataset:
  channels = list(_MEDIA_CHANNEL_NAMES) + list(_RF_CHANNEL_NAMES)
  n_channels = len(channels)

  if is_national:
    data = np.arange(1, _N_TIMES * n_channels + 1, dtype=float).reshape(
        _N_TIMES, n_channels
    )
    return xr.Dataset(
        {
            'media_scaled': (
                [constants.TIME, constants.CHANNEL],
                data,
            ),
        },
        coords={
            constants.TIME: range(_N_TIMES),
            constants.CHANNEL: channels,
        },
    )
  else:
    data = np.arange(
        1, _N_GEOS * _N_TIMES * n_channels + 1, dtype=float
    ).reshape(_N_GEOS, _N_TIMES, n_channels)
    return xr.Dataset(
        {
            'media_scaled': (
                [constants.GEO, constants.TIME, constants.CHANNEL],
                data,
            ),
        },
        coords={
            constants.GEO: list(_GEO_NAMES),
            constants.TIME: range(_N_TIMES),
            constants.CHANNEL: channels,
        },
    )


def _create_controls_and_non_media_scaled_ds(
    is_national: bool,
) -> xr.Dataset:
  controls = list(_CONTROL_NAMES)
  non_media = list(_NON_MEDIA_CHANNEL_NAMES)

  if is_national:
    controls_data = np.arange(
        1, _N_TIMES * len(controls) + 1, dtype=float
    ).reshape(_N_TIMES, len(controls))
    non_media_data = np.arange(
        1, _N_TIMES * len(non_media) + 1, dtype=float
    ).reshape(_N_TIMES, len(non_media))

    return xr.Dataset(
        {
            constants.CONTROLS_SCALED: (
                [constants.TIME, constants.CONTROL_VARIABLE],
                controls_data,
            ),
            constants.NON_MEDIA_TREATMENTS_SCALED: (
                [constants.TIME, constants.NON_MEDIA_CHANNEL],
                non_media_data,
            ),
        },
        coords={
            constants.TIME: range(_N_TIMES),
            constants.CONTROL_VARIABLE: controls,
            constants.NON_MEDIA_CHANNEL: non_media,
        },
    )
  else:
    controls_data = np.arange(
        1, _N_GEOS * _N_TIMES * len(controls) + 1, dtype=float
    ).reshape(_N_GEOS, _N_TIMES, len(controls))
    non_media_data = np.arange(
        1, _N_GEOS * _N_TIMES * len(non_media) + 1, dtype=float
    ).reshape(_N_GEOS, _N_TIMES, len(non_media))

    return xr.Dataset(
        {
            constants.CONTROLS_SCALED: (
                [constants.GEO, constants.TIME, constants.CONTROL_VARIABLE],
                controls_data,
            ),
            constants.NON_MEDIA_TREATMENTS_SCALED: (
                [constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL],
                non_media_data,
            ),
        },
        coords={
            constants.GEO: list(_GEO_NAMES),
            constants.TIME: range(_N_TIMES),
            constants.CONTROL_VARIABLE: controls,
            constants.NON_MEDIA_CHANNEL: non_media,
        },
    )


def _create_generic_finding(
    severity: Severity,
    finding_cause: Cause,
    associated_artifact: eda_outcome.AnalysisArtifact | None = None,
    explanation: str = 'IGNORED',
) -> eda_outcome.EDAFinding:
  """Creates a finding for the KPI Invariability critical check."""
  if severity is Severity.INFO:
    return eda_outcome.EDAFinding(
        finding_cause=Cause.NONE,
        severity=severity,
        associated_artifact=None,
        explanation='Info message',
    )

  return eda_outcome.EDAFinding(
      finding_cause=finding_cause,
      severity=severity,
      associated_artifact=associated_artifact,
      explanation=explanation,
  )


def _create_vif_finding(
    severity: Severity,
    finding_cause: Cause = Cause.MULTICOLLINEARITY,
    level: eda_outcome.AnalysisLevel = eda_outcome.AnalysisLevel.OVERALL,
) -> eda_outcome.EDAFinding:
  base_cols = {
      eda_constants.VIF_COL_NAME: [10.0],
      eda_constants.VARIABLE: [_MEDIA_CHANNEL_NAMES[0]],
  }

  if severity is Severity.ERROR:
    index_cols = [eda_constants.VARIABLE]
    cols = base_cols
  else:
    cols = base_cols | {constants.GEO: [_GEO_NAMES[0]]}
    index_cols = [constants.GEO, eda_constants.VARIABLE]

  return _create_generic_finding(
      finding_cause=finding_cause,
      severity=severity,
      associated_artifact=eda_outcome.VIFArtifact(
          level=level,
          vif_da=mock.create_autospec(
              xr.DataArray, instance=True, spec_set=True
          ),
          outlier_df=pd.DataFrame(cols).set_index(index_cols),
      ),
  )


def _create_pairwise_finding(
    severity: Severity,
    level: eda_outcome.AnalysisLevel = eda_outcome.AnalysisLevel.OVERALL,
) -> eda_outcome.EDAFinding:
  index = pd.MultiIndex.from_tuples(
      [_MEDIA_CHANNEL_NAMES],
      names=[eda_constants.VARIABLE_1, eda_constants.VARIABLE_2],
  )
  return _create_generic_finding(
      finding_cause=Cause.MULTICOLLINEARITY,
      severity=severity,
      associated_artifact=eda_outcome.PairwiseCorrArtifact(
          level=level,
          extreme_corr_threshold=eda_constants.OVERALL_PAIRWISE_CORR_THRESHOLD,
          extreme_corr_var_pairs=pd.DataFrame(index=index),
          corr_matrix=mock.create_autospec(
              xr.DataArray, instance=True, spec_set=True
          ),
      ),
  )


def _create_stdev_finding(
    severity: Severity,
    finding_cause: Cause,
    variable: str,
    level: eda_outcome.AnalysisLevel = eda_outcome.AnalysisLevel.OVERALL,
) -> eda_outcome.EDAFinding:
  if finding_cause is Cause.VARIABILITY:
    std_ds = xr.Dataset({
        eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME: xr.DataArray(
            np.zeros((1, 1)),
            coords={
                eda_constants.VARIABLE: [variable],
                constants.GEO: [_GEO_NAMES[0]],
            },
            dims=[eda_constants.VARIABLE, constants.GEO],
        ),
        eda_constants.STD_WITH_OUTLIERS_VAR_NAME: xr.DataArray(
            np.zeros((1, 1)),
            coords={
                eda_constants.VARIABLE: [variable],
                constants.GEO: [_GEO_NAMES[0]],
            },
            dims=[eda_constants.VARIABLE, constants.GEO],
        ),
    })
    outlier_df = pd.DataFrame()
  elif finding_cause is Cause.OUTLIER:
    std_ds = xr.Dataset()
    outlier_df = pd.DataFrame({
        eda_constants.OUTLIERS_COL_NAME: [1.0],
        eda_constants.ABS_OUTLIERS_COL_NAME: [1.0],
        eda_constants.VARIABLE: [variable],
        constants.GEO: [_GEO_NAMES[0]],
        constants.TIME: [_N_TIMES],
    }).set_index([eda_constants.VARIABLE, constants.GEO, constants.TIME])
  else:
    raise ValueError(f'Unsupported finding cause: {finding_cause}')

  return _create_generic_finding(
      finding_cause=finding_cause,
      severity=severity,
      associated_artifact=eda_outcome.StandardDeviationArtifact(
          level=level,
          std_ds=std_ds,
          outlier_df=outlier_df,
          variable=variable,
      ),
      explanation=f'{variable} {severity.name} {finding_cause.name} message',
  )


def _create_cpmu_finding(
    severity: Severity,
    finding_cause: Cause,
    level: eda_outcome.AnalysisLevel = eda_outcome.AnalysisLevel.OVERALL,
) -> eda_outcome.EDAFinding:
  """Creates a Cost Per Media Unit finding with necessary artifacts.

  Args:
    severity: The severity of the EDA finding.
    finding_cause: The cause of the finding, used to determine which artifact to
      include.
    level: The level of the EDA check.

  Returns:
    An instance of eda_outcome.EDAFinding.
  """

  base_data = {
      constants.CHANNEL: [_MEDIA_CHANNEL_NAMES[0]],
      constants.GEO: [_GEO_NAMES[0]],
      constants.TIME: [0],
      constants.SPEND: [100.0],
  }

  if finding_cause is Cause.OUTLIER:
    outlier_df = pd.DataFrame(
        base_data
        | {
            constants.MEDIA_UNITS: [100.0],
            eda_constants.COST_PER_MEDIA_UNIT: [1.0],
            eda_constants.ABS_COST_PER_MEDIA_UNIT: [1.0],
        }
    ).set_index([constants.CHANNEL, constants.GEO, constants.TIME])
    cost_media_unit_inconsistency_df = pd.DataFrame()
  elif finding_cause is Cause.INCONSISTENT_DATA:
    outlier_df = pd.DataFrame()
    cost_media_unit_inconsistency_df = pd.DataFrame(
        base_data
        | {
            constants.MEDIA_UNITS: [0.0],
        }
    ).set_index([constants.CHANNEL, constants.GEO, constants.TIME])
  else:
    raise ValueError(f'Unsupported finding cause: {finding_cause}')

  return _create_generic_finding(
      finding_cause=finding_cause,
      severity=severity,
      associated_artifact=eda_outcome.CostPerMediaUnitArtifact(
          level=level,
          cost_media_unit_inconsistency_df=cost_media_unit_inconsistency_df,
          cost_per_media_unit_da=mock.create_autospec(
              xr.DataArray, instance=True, spec_set=True
          ),
          outlier_df=outlier_df,
      ),
      explanation=f'CPMU {severity.name} {finding_cause.name} message',
  )


def _create_pop_corr_artifact(
    values: Sequence[float],
    level: eda_outcome.AnalysisLevel = eda_outcome.AnalysisLevel.OVERALL,
) -> eda_outcome.PopulationCorrelationArtifact:
  return eda_outcome.PopulationCorrelationArtifact(
      level=level,
      correlation_ds=xr.Dataset(
          {eda_constants.VALUE: ([constants.CHANNEL], np.array(values))},
          coords={constants.CHANNEL: list(_MEDIA_CHANNEL_NAMES)},
      ),
  )


def _create_prior_artifact(
    values: Sequence[float],
    level: eda_outcome.AnalysisLevel = eda_outcome.AnalysisLevel.OVERALL,
) -> eda_outcome.PriorProbabilityArtifact:
  return eda_outcome.PriorProbabilityArtifact(
      level=level,
      prior_negative_baseline_prob=0.123,
      mean_prior_contribution_da=xr.DataArray(
          values,
          coords={constants.CHANNEL: list(_MEDIA_CHANNEL_NAMES)},
      ),
  )


def _create_eda_outcome(
    check_type: eda_outcome.EDACheckType,
    findings: Sequence[eda_outcome.EDAFinding] | None = None,
    analysis_artifacts: Sequence[eda_outcome.AnalysisArtifact] | None = None,
) -> eda_outcome.EDAOutcome[eda_outcome.AnalysisArtifact]:
  return eda_outcome.EDAOutcome(
      check_type=check_type,
      findings=list(findings) if findings else [],
      analysis_artifacts=list(analysis_artifacts) if analysis_artifacts else [],
  )


class MeridianEdaTestWithMockEngine(backend_test_utils.MeridianTestCase):
  """Tests for MeridianEDA report generation with a mock EDAEngine setup."""

  def setUp(self):
    super().setUp()
    # The create_tempdir() method below internally uses command line flag
    # (--test_tmpdir) and such flags are not marked as parsed by default
    # when running with pytest. Marking as parsed directly here to make the
    # pytest run pass.
    flags.FLAGS.mark_as_parsed()
    self._meridian = mock.create_autospec(
        model.Meridian, instance=True, spec_set=True
    )
    self._mock_input_data = mock.create_autospec(
        input_data.InputData, instance=True, spec_set=True
    )
    self._mock_input_data.geo = _GEO_NAMES
    self._mock_input_data.get_n_top_largest_geos.side_effect = (
        lambda n: _GEO_NAMES[:n]
    )
    self._meridian.input_data = self._mock_input_data

    self._meridian.eda_spec = eda_spec.EDASpec()
    self._meridian.is_national = False
    self._meridian.n_geos = _N_GEOS

    self._meridian.inference_data = mock.create_autospec(
        az.InferenceData, instance=True, spec_set=True
    )
    self._meridian.inference_data.groups.return_value = [constants.PRIOR]
    self._meridian.model_context = mock.create_autospec(
        context.ModelContext, instance=True, spec_set=True
    )

    # Mock Analyzer to avoid testing issues.
    self._mock_analyzer_class = self.enter_context(
        mock.patch.object(
            analyzer_module, 'Analyzer', autospec=True, spec_set=True
        )
    )
    self._mock_analyzer_instance = self._mock_analyzer_class.return_value

    # Mock SamplingEDAEngine
    self._mock_eda_engine = mock.create_autospec(
        sampling_eda_engine.SamplingEDAEngine, instance=True, spec_set=True
    )
    self._mock_eda_engine.spec = self._meridian.eda_spec
    self._mock_sampling_eda_engine_class = self.enter_context(
        mock.patch.object(
            sampling_eda_engine,
            'SamplingEDAEngine',
            autospec=True,
            spec_set=True,
        )
    )
    self._mock_sampling_eda_engine_class.return_value = self._mock_eda_engine

    self._eda = meridian_eda.MeridianEDA(self._meridian)

  def _stub_engine_checks(
      self,
      cpmu_findings: Sequence[eda_outcome.EDAFinding] | None = None,
      vif_findings: Sequence[eda_outcome.EDAFinding] | None = None,
      pairwise_findings: Sequence[eda_outcome.EDAFinding] | None = None,
      kpi_findings: Sequence[eda_outcome.EDAFinding] | None = None,
      stdev_findings: Sequence[eda_outcome.EDAFinding] | None = None,
      pop_raw_media_artifacts: (
          Sequence[eda_outcome.PopulationCorrelationArtifact] | None
      ) = None,
      pop_treatment_artifacts: (
          Sequence[eda_outcome.PopulationCorrelationArtifact] | None
      ) = None,
      prior_artifacts: (
          Sequence[eda_outcome.PriorProbabilityArtifact] | None
      ) = None,
  ) -> None:
    self._mock_eda_engine.check_cost_per_media_unit.return_value = (
        eda_outcome.EDAOutcome(
            check_type=eda_outcome.EDACheckType.COST_PER_MEDIA_UNIT,
            findings=cpmu_findings or [],
            analysis_artifacts=[],
        )
    )

    self._mock_eda_engine.run_all_critical_checks.return_value = (
        eda_outcome.CriticalCheckEDAOutcomes(
            multicollinearity=eda_outcome.EDAOutcome(
                check_type=eda_outcome.EDACheckType.MULTICOLLINEARITY,
                findings=vif_findings or [],
                analysis_artifacts=[],
            ),
            pairwise_correlation=eda_outcome.EDAOutcome(
                check_type=eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
                findings=pairwise_findings or [],
                analysis_artifacts=[],
            ),
            kpi_invariability=eda_outcome.EDAOutcome(
                check_type=eda_outcome.EDACheckType.KPI_INVARIABILITY,
                findings=kpi_findings or [],
                analysis_artifacts=[],
            ),
        )
    )

    variable_geo_time_collinearity_outcome = eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.VARIABLE_GEO_TIME_COLLINEARITY,
        findings=[],
        analysis_artifacts=[
            eda_outcome.VariableGeoTimeCollinearityArtifact(
                level=eda_outcome.AnalysisLevel.OVERALL,
                rsquared_ds=xr.Dataset({
                    eda_constants.RSQUARED_TIME: xr.DataArray(
                        [0.5],
                        coords={constants.CHANNEL: ['ch_0']},
                        name=eda_constants.RSQUARED_TIME,
                    ),
                    eda_constants.RSQUARED_GEO: xr.DataArray(
                        [0.5],
                        coords={constants.CHANNEL: ['ch_0']},
                        name=eda_constants.RSQUARED_GEO,
                    ),
                }),
            )
        ],
    )
    self._mock_eda_engine.check_variable_geo_time_collinearity.return_value = (
        variable_geo_time_collinearity_outcome
    )

    geo_stdev_artifact = eda_outcome.StandardDeviationArtifact(
        variable=constants.TREATMENT_CONTROL_SCALED,
        level=eda_outcome.AnalysisLevel.GEO,
        std_ds=xr.Dataset({
            eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME: xr.DataArray(
                np.zeros((
                    len(_MEDIA_CHANNEL_NAMES)
                    + len(_RF_CHANNEL_NAMES)
                    + len(_CONTROL_NAMES)
                    + len(_NON_MEDIA_CHANNEL_NAMES),
                    len(_GEO_NAMES),
                )),
                coords={
                    eda_constants.VARIABLE: list(
                        itertools.chain(
                            _MEDIA_CHANNEL_NAMES,
                            _RF_CHANNEL_NAMES,
                            _CONTROL_NAMES,
                            _NON_MEDIA_CHANNEL_NAMES,
                        )
                    ),
                    constants.GEO: list(_GEO_NAMES),
                },
                dims=[eda_constants.VARIABLE, constants.GEO],
            ),
            eda_constants.STD_WITH_OUTLIERS_VAR_NAME: xr.DataArray(
                np.zeros((
                    len(_MEDIA_CHANNEL_NAMES)
                    + len(_RF_CHANNEL_NAMES)
                    + len(_CONTROL_NAMES)
                    + len(_NON_MEDIA_CHANNEL_NAMES),
                    len(_GEO_NAMES),
                )),
                coords={
                    eda_constants.VARIABLE: list(
                        itertools.chain(
                            _MEDIA_CHANNEL_NAMES,
                            _RF_CHANNEL_NAMES,
                            _CONTROL_NAMES,
                            _NON_MEDIA_CHANNEL_NAMES,
                        )
                    ),
                    constants.GEO: list(_GEO_NAMES),
                },
                dims=[eda_constants.VARIABLE, constants.GEO],
            ),
        }),
        outlier_df=pd.DataFrame(
            columns=[
                eda_constants.OUTLIERS_COL_NAME,
                eda_constants.ABS_OUTLIERS_COL_NAME,
            ],
            index=pd.MultiIndex.from_arrays(
                [[], [], []],
                names=[
                    eda_constants.VARIABLE,
                    constants.GEO,
                    constants.TIME,
                ],
            ),
        ),
    )
    national_stdev_artifact = eda_outcome.StandardDeviationArtifact(
        variable=constants.NATIONAL_TREATMENT_CONTROL_SCALED,
        level=eda_outcome.AnalysisLevel.NATIONAL,
        std_ds=xr.Dataset({
            eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME: xr.DataArray(
                np.zeros((
                    len(_MEDIA_CHANNEL_NAMES)
                    + len(_RF_CHANNEL_NAMES)
                    + len(_CONTROL_NAMES)
                    + len(_NON_MEDIA_CHANNEL_NAMES),
                    1,
                )),
                coords={
                    eda_constants.VARIABLE: list(
                        itertools.chain(
                            _MEDIA_CHANNEL_NAMES,
                            _RF_CHANNEL_NAMES,
                            _CONTROL_NAMES,
                            _NON_MEDIA_CHANNEL_NAMES,
                        )
                    ),
                    constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
                },
                dims=[eda_constants.VARIABLE, constants.GEO],
            ),
            eda_constants.STD_WITH_OUTLIERS_VAR_NAME: xr.DataArray(
                np.zeros((
                    len(_MEDIA_CHANNEL_NAMES)
                    + len(_RF_CHANNEL_NAMES)
                    + len(_CONTROL_NAMES)
                    + len(_NON_MEDIA_CHANNEL_NAMES),
                    1,
                )),
                coords={
                    eda_constants.VARIABLE: list(
                        itertools.chain(
                            _MEDIA_CHANNEL_NAMES,
                            _RF_CHANNEL_NAMES,
                            _CONTROL_NAMES,
                            _NON_MEDIA_CHANNEL_NAMES,
                        )
                    ),
                    constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
                },
                dims=[eda_constants.VARIABLE, constants.GEO],
            ),
        }),
        outlier_df=pd.DataFrame(
            columns=[
                eda_constants.OUTLIERS_COL_NAME,
                eda_constants.ABS_OUTLIERS_COL_NAME,
            ],
            index=pd.MultiIndex.from_arrays(
                [[], [], []],
                names=[
                    eda_constants.VARIABLE,
                    constants.GEO,
                    constants.TIME,
                ],
            ),
        ),
    )

    self._mock_eda_engine.check_geo_std.return_value = eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.STANDARD_DEVIATION,
        findings=[],
        analysis_artifacts=[geo_stdev_artifact],
    )
    self._mock_eda_engine.check_national_std.return_value = (
        eda_outcome.EDAOutcome(
            check_type=eda_outcome.EDACheckType.STANDARD_DEVIATION,
            findings=[],
            analysis_artifacts=[national_stdev_artifact],
        )
    )

    self._mock_eda_engine.check_std.side_effect = (
        lambda: eda_outcome.EDAOutcome(
            check_type=eda_outcome.EDACheckType.STANDARD_DEVIATION,
            findings=stdev_findings or [],
            analysis_artifacts=[
                national_stdev_artifact
                if self._meridian.is_national
                else geo_stdev_artifact
            ],
        )
    )

    self._mock_eda_engine.check_population_corr_raw_media.return_value = (
        eda_outcome.EDAOutcome(
            check_type=eda_outcome.EDACheckType.POPULATION_CORRELATION,
            findings=[],
            analysis_artifacts=pop_raw_media_artifacts or [],
        )
    )

    pop_treatment_explanation = (
        eda_constants.POPULATION_CORRELATION_SCALED_TREATMENT_CONTROL_INFO
    )
    pop_treatment_findings = [
        eda_outcome.EDAFinding(
            severity=Severity.INFO,
            finding_cause=Cause.NONE,
            explanation=pop_treatment_explanation,
            associated_artifact=next(iter(pop_treatment_artifacts or []), None),
        )
    ]
    self._mock_eda_engine.check_population_corr_scaled_treatment_control.return_value = eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.POPULATION_CORRELATION,
        findings=pop_treatment_findings,
        analysis_artifacts=pop_treatment_artifacts or [],
    )
    self._mock_eda_engine.check_prior_probability.return_value = (
        eda_outcome.EDAOutcome(
            check_type=eda_outcome.EDACheckType.PRIOR_PROBABILITY,
            findings=[],
            analysis_artifacts=prior_artifacts
            or [_create_prior_artifact([1, 2])],
        )
    )

    data_param_artifact = eda_outcome.DataParameterRatioArtifact(
        level=eda_outcome.AnalysisLevel.OVERALL,
        n_geos=2,
        n_times=3,
        n_knots=1,
        n_controls=1,
        n_treatments=1,
    )
    self._mock_eda_engine.check_data_param_ratio.return_value = (
        eda_outcome.EDAOutcome(
            check_type=eda_outcome.EDACheckType.DATA_ADEQUACY,
            findings=[
                eda_outcome.EDAFinding(
                    severity=eda_outcome.EDASeverity.INFO,
                    explanation=eda_constants.DATA_ADEQUACY_INFO,
                    finding_cause=eda_outcome.FindingCause.NONE,
                    associated_artifact=data_param_artifact,
                )
            ],
            analysis_artifacts=[data_param_artifact],
        )
    )

  def _stub_plotters(self) -> None:
    self.enter_context(
        mock.patch.object(self._eda, 'plot_pairwise_correlation', autospec=True)
    ).return_value.to_json.return_value = '{"pairwise": "json"}'
    self.enter_context(
        mock.patch.object(
            self._eda, 'plot_relative_spend_share_barchart', autospec=True
        )
    ).return_value.to_json.return_value = '{"relative_spend_share": "json"}'
    self.enter_context(
        mock.patch.object(
            self._eda,
            'plot_treatments_without_non_media_boxplot',
            autospec=True,
        )
    ).return_value.to_json.return_value = '{"treatments": "json"}'
    self.enter_context(
        mock.patch.object(
            self._eda, 'plot_controls_and_non_media_boxplot', autospec=True
        )
    ).return_value.to_json.return_value = '{"controls_and_non_media": "json"}'
    self.enter_context(
        mock.patch.object(self._eda, 'plot_kpi_boxplot', autospec=True)
    ).return_value.to_json.return_value = '{"kpi": "json"}'
    self.enter_context(
        mock.patch.object(
            self._eda, 'plot_cost_per_media_unit_time_series', autospec=True
        )
    ).return_value.to_json.return_value = '{"cpmu": "json"}'
    self.enter_context(
        mock.patch.object(
            self._eda, 'plot_population_raw_media_correlation', autospec=True
        )
    ).return_value.to_json.return_value = '{"pop_raw": "json"}'
    self.enter_context(
        mock.patch.object(
            self._eda, 'plot_population_treatment_correlation', autospec=True
        )
    ).return_value.to_json.return_value = '{"pop_treatment": "json"}'
    self.enter_context(
        mock.patch.object(self._eda, 'plot_prior_mean', autospec=True)
    ).return_value.to_json.return_value = '{"prior_mean": "json"}'

  # ============================================================================
  # __init__ Tests
  # ============================================================================

  def test_init_eda_engine_with_mock(self):
    with self.subTest(name='analyzer_instantiation'):
      self._mock_analyzer_class.assert_called_once_with(
          model_context=self._meridian.model_context,
          inference_data=self._meridian.inference_data,
      )
    with self.subTest(name='eda_engine_instantiation'):
      self._mock_sampling_eda_engine_class.assert_called_once_with(
          self._mock_analyzer_instance,
          self._meridian.eda_spec,
      )

  def test_init_sample_prior_default_args(self):
    self._meridian.inference_data.groups.return_value = []
    meridian_eda.MeridianEDA(self._meridian)
    self._meridian.sample_prior.assert_called_once_with(
        n_draws=eda_constants.DEFAULT_PRIOR_N_DRAW,
        seed=eda_constants.DEFAULT_PRIOR_SEED,
    )

  def test_init_sample_prior_custom_args(self):
    self._meridian.inference_data.groups.return_value = []
    n_draws_prior = 100
    seed = 42
    meridian_eda.MeridianEDA(
        self._meridian, n_draws_prior=n_draws_prior, seed=seed
    )
    self._meridian.sample_prior.assert_called_once_with(
        n_draws=n_draws_prior,
        seed=seed,
    )

  def test_init_no_sample_prior_called(self):
    self._meridian.inference_data.groups.return_value = [constants.PRIOR]
    meridian_eda.MeridianEDA(self._meridian)
    self._meridian.sample_prior.assert_not_called()

  # ============================================================================
  # Cost Per Media Unit Test
  # ============================================================================
  def test_plot_cost_per_media_unit_time_series_geos(self):
    self._mock_eda_engine.all_spend_ds = xr.Dataset(
        {
            'media_spend': (
                [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
                np.array([
                    [
                        [100.0, 0.0],
                        [200.0, 0.0],
                        [300.0, 0.0],
                    ],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]),
            ),
            'rf_spend': (
                [constants.GEO, constants.TIME, constants.RF_CHANNEL],
                np.array([[[10.0], [10.0], [10.0]], [[0.0], [0.0], [0.0]]]),
            ),
        },
        coords={
            constants.GEO: list(_GEO_NAMES),
            constants.TIME: range(_N_TIMES),
            constants.MEDIA_CHANNEL: list(_MEDIA_CHANNEL_NAMES),
            constants.RF_CHANNEL: list(_RF_CHANNEL_NAMES),
        },
    )

    self._mock_eda_engine.paid_raw_media_units_ds = xr.Dataset(
        {
            'media': (
                [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
                np.array([
                    [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]),
            ),
            'rf_impressions': (
                [constants.GEO, constants.TIME, constants.RF_CHANNEL],
                np.array([[[1.0], [1.0], [1.0]], [[0.0], [0.0], [0.0]]]),
            ),
        },
        coords={
            constants.GEO: list(_GEO_NAMES),
            constants.TIME: range(_N_TIMES),
            constants.MEDIA_CHANNEL: list(_MEDIA_CHANNEL_NAMES),
            constants.RF_CHANNEL: list(_RF_CHANNEL_NAMES),
        },
    )

    mock_artifact = mock.create_autospec(
        eda_outcome.CostPerMediaUnitArtifact, instance=True
    )
    mock_artifact.level = eda_outcome.AnalysisLevel.GEO
    mock_artifact.cost_per_media_unit_da = xr.DataArray(
        np.array([
            [[10.0, np.nan, 10.0], [10.0, np.nan, 10.0], [10.0, np.nan, 10.0]],
            [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ],
        ]),
        coords={
            constants.GEO: list(_GEO_NAMES),
            constants.TIME: range(_N_TIMES),
            constants.CHANNEL: list(
                itertools.chain(_MEDIA_CHANNEL_NAMES, _RF_CHANNEL_NAMES)
            ),
        },
        dims=[constants.GEO, constants.TIME, constants.CHANNEL],
        name=eda_constants.COST_PER_MEDIA_UNIT,
    )
    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.COST_PER_MEDIA_UNIT,
        analysis_artifacts=[mock_artifact],
    )

    self.enter_context(
        mock.patch.object(
            self._mock_eda_engine,
            'check_geo_cost_per_media_unit',
            return_value=mock_outcome,
        )
    )
    plot = self._eda.plot_cost_per_media_unit_time_series(geos=['geo_0'])

    actual_cost = np.concatenate([
        row.vconcat[0].layer[0].data[eda_constants.VALUE]
        for row in plot.vconcat[0].vconcat
    ])
    actual_media_units = np.concatenate([
        row.vconcat[0].layer[1].data[eda_constants.VALUE]
        for row in plot.vconcat[0].vconcat
    ])
    actual_cost_per_media_unit = np.concatenate([
        row.vconcat[1].data[eda_constants.VALUE]
        for row in plot.vconcat[0].vconcat
    ])

    plot_json = plot.to_dict()
    # The structure is:
    # vconcat (main plot)
    #   vconcat (per channel blocks)
    #     vconcat (spend/media_units superimposed + CPMU chart)
    #       vconcat (spend/media_units superimposed)
    #         layer 0 (spend)
    #         layer 1 (media_units)  <- We are accessing this layer's y-axis
    #       vconcat (CPMU chart)
    outer_vconcat = plot_json['vconcat'][0]
    channel_block = outer_vconcat['vconcat'][0]
    superimposed_chart = channel_block['vconcat'][0]
    media_unit_layer = superimposed_chart['layer'][1]
    media_unit_axis = media_unit_layer['encoding']['y']['axis']

    with self.subTest(name='media_units_axis'):
      self.assertEqual(media_unit_axis['orient'], 'right')
      self.assertEqual(media_unit_axis['offset'], 10)

    with self.subTest(name='cost_per_media_unit'):
      np.testing.assert_allclose(
          actual_cost_per_media_unit,
          [10.0, 10.0, 10.0, np.nan, np.nan, np.nan, 10.0, 10.0, 10.0],
      )
    with self.subTest(name='cost'):
      np.testing.assert_allclose(
          actual_cost,
          [100.0, 200.0, 300.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
      )
    with self.subTest(name='media_units'):
      np.testing.assert_allclose(
          actual_media_units,
          [10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
      )

  def test_plot_cost_per_media_unit_time_series_nationalize(self):
    self._mock_eda_engine.national_all_spend_ds = xr.Dataset(
        {
            'media_spend': (
                [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
                np.array([[
                    [100.0, 0.0],
                    [200.0, 0.0],
                    [300.0, 0.0],
                ]]),
            ),
            'rf_spend': (
                [constants.GEO, constants.TIME, constants.RF_CHANNEL],
                np.array([[[10.0], [10.0], [10.0]]]),
            ),
        },
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
            constants.MEDIA_CHANNEL: list(_MEDIA_CHANNEL_NAMES),
            constants.RF_CHANNEL: list(_RF_CHANNEL_NAMES),
        },
    )

    self._mock_eda_engine.national_paid_raw_media_units_ds = xr.Dataset(
        {
            'media': (
                [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
                np.array([[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]]),
            ),
            'rf_impressions': (
                [constants.GEO, constants.TIME, constants.RF_CHANNEL],
                np.array([[[1.0], [1.0], [1.0]]]),
            ),
        },
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
            constants.MEDIA_CHANNEL: list(_MEDIA_CHANNEL_NAMES),
            constants.RF_CHANNEL: list(_RF_CHANNEL_NAMES),
        },
    )

    mock_artifact = mock.create_autospec(
        eda_outcome.CostPerMediaUnitArtifact, instance=True
    )
    mock_artifact.level = eda_outcome.AnalysisLevel.NATIONAL
    mock_artifact.cost_per_media_unit_da = xr.DataArray(
        np.array(
            [[[10.0, np.nan, 10.0], [10.0, np.nan, 10.0], [10.0, np.nan, 10.0]]]
        ),
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
            constants.CHANNEL: list(
                itertools.chain(_MEDIA_CHANNEL_NAMES, _RF_CHANNEL_NAMES)
            ),
        },
        dims=[constants.GEO, constants.TIME, constants.CHANNEL],
        name=eda_constants.COST_PER_MEDIA_UNIT,
    )
    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.COST_PER_MEDIA_UNIT,
        analysis_artifacts=[mock_artifact],
    )

    self.enter_context(
        mock.patch.object(
            self._mock_eda_engine,
            'check_national_cost_per_media_unit',
            return_value=mock_outcome,
        )
    )
    plot = self._eda.plot_cost_per_media_unit_time_series(
        geos=eda_constants.NATIONALIZE
    )

    actual_cost = np.concatenate([
        row.vconcat[0].layer[0].data[eda_constants.VALUE]
        for row in plot.vconcat[0].vconcat
    ])
    actual_media_units = np.concatenate([
        row.vconcat[0].layer[1].data[eda_constants.VALUE]
        for row in plot.vconcat[0].vconcat
    ])
    actual_cost_per_media_unit = np.concatenate([
        row.vconcat[1].data[eda_constants.VALUE]
        for row in plot.vconcat[0].vconcat
    ])

    with self.subTest(name='cost_per_media_unit'):
      np.testing.assert_allclose(
          actual_cost_per_media_unit,
          [10.0, 10.0, 10.0, np.nan, np.nan, np.nan, 10.0, 10.0, 10.0],
      )
    with self.subTest(name='cost'):
      np.testing.assert_allclose(
          actual_cost,
          [100.0, 200.0, 300.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
      )
    with self.subTest(name='media_units'):
      np.testing.assert_allclose(
          actual_media_units, [10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
      )

  def test_plot_cost_per_media_unit_time_series_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    self._mock_eda_engine.national_all_spend_ds = xr.Dataset(
        {
            'media_spend': (
                [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
                np.array([[
                    [100.0, 0.0],
                    [200.0, 0.0],
                    [300.0, 0.0],
                ]]),
            ),
            'rf_spend': (
                [constants.GEO, constants.TIME, constants.RF_CHANNEL],
                np.array([[[10.0], [10.0], [10.0]]]),
            ),
        },
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
            constants.MEDIA_CHANNEL: list(_MEDIA_CHANNEL_NAMES),
            constants.RF_CHANNEL: list(_RF_CHANNEL_NAMES),
        },
    )

    self._mock_eda_engine.national_paid_raw_media_units_ds = xr.Dataset(
        {
            'media': (
                [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
                np.array([[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]]),
            ),
            'rf_impressions': (
                [constants.GEO, constants.TIME, constants.RF_CHANNEL],
                np.array([[[1.0], [1.0], [1.0]]]),
            ),
        },
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
            constants.MEDIA_CHANNEL: list(_MEDIA_CHANNEL_NAMES),
            constants.RF_CHANNEL: list(_RF_CHANNEL_NAMES),
        },
    )

    mock_artifact = mock.create_autospec(
        eda_outcome.CostPerMediaUnitArtifact, instance=True
    )
    mock_artifact.level = eda_outcome.AnalysisLevel.NATIONAL
    mock_artifact.cost_per_media_unit_da = xr.DataArray(
        np.array(
            [[[10.0, np.nan, 10.0], [10.0, np.nan, 10.0], [10.0, np.nan, 10.0]]]
        ),
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
            constants.CHANNEL: list(
                itertools.chain(_MEDIA_CHANNEL_NAMES, _RF_CHANNEL_NAMES)
            ),
        },
        dims=[constants.GEO, constants.TIME, constants.CHANNEL],
        name=eda_constants.COST_PER_MEDIA_UNIT,
    )
    self._mock_eda_engine.check_national_cost_per_media_unit.return_value.get_national_artifacts.return_value = [
        mock_artifact
    ]

    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.COST_PER_MEDIA_UNIT,
        analysis_artifacts=[mock_artifact],
    )

    self.enter_context(
        mock.patch.object(
            self._mock_eda_engine,
            'check_national_cost_per_media_unit',
            return_value=mock_outcome,
        )
    )
    plot = self._eda.plot_cost_per_media_unit_time_series(
        geos=eda_constants.NATIONALIZE
    )

    actual_cost = np.concatenate([
        row.vconcat[0].layer[0].data[eda_constants.VALUE]
        for row in plot.vconcat[0].vconcat
    ])
    actual_media_units = np.concatenate([
        row.vconcat[0].layer[1].data[eda_constants.VALUE]
        for row in plot.vconcat[0].vconcat
    ])
    actual_cost_per_media_unit = np.concatenate([
        row.vconcat[1].data[eda_constants.VALUE]
        for row in plot.vconcat[0].vconcat
    ])

    with self.subTest(name='cost_per_media_unit'):
      np.testing.assert_allclose(
          actual_cost_per_media_unit,
          [10.0, 10.0, 10.0, np.nan, np.nan, np.nan, 10.0, 10.0, 10.0],
      )
    with self.subTest(name='cost'):
      np.testing.assert_allclose(
          actual_cost,
          [100.0, 200.0, 300.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
      )
    with self.subTest(name='media_units'):
      np.testing.assert_allclose(
          actual_media_units, [10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
      )

  # ============================================================================
  # Relative Spend Share Test
  # ============================================================================

  def test_plot_relative_spend_share_barchart_geos(self):
    self._mock_eda_engine.all_spend_ds = xr.Dataset(
        {
            'media_spend': (
                [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
                np.array(
                    [[[5, 10], [5, 10], [0, 10]], [[0, 0], [0, 0], [0, 0]]]
                ),
            ),
            'rf_spend': (
                [constants.GEO, constants.TIME, constants.RF_CHANNEL],
                np.array([[[20], [20], [20]], [[0], [0], [0]]]),
            ),
        },
        coords={
            constants.GEO: list(_GEO_NAMES),
            constants.TIME: range(_N_TIMES),
            constants.MEDIA_CHANNEL: list(_MEDIA_CHANNEL_NAMES),
            constants.RF_CHANNEL: list(_RF_CHANNEL_NAMES),
        },
    )

    plot = self._eda.plot_relative_spend_share_barchart(geos=['geo_0'])

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    np.testing.assert_allclose(actual_values, [0.1, 0.3, 0.6])

  def test_plot_relative_spend_share_barchart_nationalize(self):
    self._mock_eda_engine.national_all_spend_ds = xr.Dataset(
        {
            'media_spend': (
                [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
                np.array([[[10], [10], [10]]]),
            ),
            'rf_spend': (
                [constants.GEO, constants.TIME, constants.RF_CHANNEL],
                np.array([[[20], [20], [30]]]),
            ),
        },
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
            constants.MEDIA_CHANNEL: ['ch_0'],
            constants.RF_CHANNEL: list(_RF_CHANNEL_NAMES),
        },
    )

    plot = self._eda.plot_relative_spend_share_barchart(
        geos=eda_constants.NATIONALIZE
    )

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    np.testing.assert_allclose(actual_values, [0.3, 0.7])

  def test_plot_relative_spend_share_barchart_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    self._mock_eda_engine.national_all_spend_ds = xr.Dataset(
        {
            'media_spend': (
                [constants.GEO, constants.TIME, constants.MEDIA_CHANNEL],
                np.array([[[50], [50], [50]]]),
            ),
            'rf_spend': (
                [constants.GEO, constants.TIME, constants.RF_CHANNEL],
                np.array([[[10], [20], [20]]]),
            ),
        },
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
            constants.MEDIA_CHANNEL: ['ch_0'],
            constants.RF_CHANNEL: list(_RF_CHANNEL_NAMES),
        },
    )

    plot = self._eda.plot_relative_spend_share_barchart()

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    np.testing.assert_allclose(actual_values, [0.25, 0.75])

  # ============================================================================
  # Relative Impression Share Tests
  # ============================================================================

  def test_plot_relative_impression_share_barchart_geos(self):
    ds = xr.Dataset(
        {
            'media_scaled': (
                [constants.GEO, constants.TIME, constants.CHANNEL],
                np.array(
                    [[[50, 0], [0, 25], [0, 25]], [[0, 0], [0, 0], [0, 0]]]
                ),
            ),
        },
        coords={
            constants.GEO: list(_GEO_NAMES),
            constants.TIME: range(_N_TIMES),
            constants.CHANNEL: list(_MEDIA_CHANNEL_NAMES),
        },
    )

    self._mock_eda_engine.treatments_without_non_media_scaled_ds = ds

    plot = self._eda.plot_relative_impression_share_barchart(geos=['geo_0'])

    df = plot.data
    present_vars = df[eda_constants.VARIABLE].unique()
    for var in present_vars:
      self.assertIn(var, _MEDIA_CHANNEL_NAMES)

    actual_values = sorted(df[eda_constants.VALUE].tolist())
    np.testing.assert_allclose(actual_values, [0.5, 0.5])

  def test_plot_relative_impression_share_barchart_nationalize(self):
    ds = xr.Dataset(
        {
            'media_scaled': (
                [constants.GEO, constants.TIME, constants.CHANNEL],
                np.array([[[40, 160], [0, 0], [0, 0]]]),
            ),
        },
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
            constants.CHANNEL: list(_MEDIA_CHANNEL_NAMES),
        },
    )

    self._mock_eda_engine.national_treatments_without_non_media_scaled_ds = ds

    plot = self._eda.plot_relative_impression_share_barchart(
        geos=eda_constants.NATIONALIZE
    )

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    np.testing.assert_allclose(actual_values, [0.2, 0.8])

  def test_plot_relative_impression_share_barchart_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    ds = xr.Dataset(
        {
            'media_scaled': (
                [constants.GEO, constants.TIME, constants.CHANNEL],
                np.array([[[10, 90], [0, 0], [0, 0]]]),
            )
        },
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
            constants.CHANNEL: list(_MEDIA_CHANNEL_NAMES),
        },
    )

    self._mock_eda_engine.national_treatments_without_non_media_scaled_ds = ds

    plot = self._eda.plot_relative_impression_share_barchart()

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    np.testing.assert_allclose(actual_values, [0.1, 0.9])

  def test_plot_relative_impression_share_barchart_top_n_channels(self):
    n_channels = 6
    media_channel_names = [f'ch_{i}' for i in range(n_channels)]
    ds = xr.Dataset(
        {
            'media_scaled': (
                [constants.GEO, constants.TIME, constants.CHANNEL],
                np.array([[
                    [10, 20, 30, 40, 50, 60],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]]),
            ),
        },
        coords={
            constants.GEO: ['geo_0'],
            constants.TIME: range(_N_TIMES),
            constants.CHANNEL: media_channel_names,
        },
    )

    self._mock_eda_engine.treatments_without_non_media_scaled_ds = ds
    self._meridian.input_data.geo = ['geo_0']

    plot = self._eda.plot_relative_impression_share_barchart(
        geos=['geo_0'], n_channels=5, ascending=False
    )

    df = plot.data
    present_vars = df[eda_constants.VARIABLE].tolist()
    self.assertLen(present_vars, 5)
    self.assertEqual(present_vars, ['ch_5', 'ch_4', 'ch_3', 'ch_2', 'ch_1'])
    np.testing.assert_allclose(
        df[eda_constants.VALUE].tolist(),
        [60 / 210, 50 / 210, 40 / 210, 30 / 210, 20 / 210],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='height_capped_at_600',
          n_channels_to_plot=None,
          expected_height=600,
      ),
      dict(
          testcase_name='height_dynamic_below_cap',
          n_channels_to_plot=5,
          expected_height=250,
      ),
      dict(
          testcase_name='height_15_channels',
          n_channels_to_plot=15,
          expected_height=600,
      ),
  )
  def test_plot_barchart_dynamic_height(
      self, n_channels_to_plot, expected_height
  ):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    n_channels_data = 15
    media_channel_names = [f'ch_{i}' for i in range(n_channels_data)]
    ds = xr.Dataset(
        {
            'media_scaled': (
                [constants.GEO, constants.TIME, constants.CHANNEL],
                np.array([[
                    np.arange(n_channels_data) + 1,
                    np.zeros(n_channels_data),
                    np.zeros(n_channels_data),
                ]]),
            ),
        },
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
            constants.CHANNEL: media_channel_names,
        },
    )

    self._mock_eda_engine.national_treatments_without_non_media_scaled_ds = ds

    plot = self._eda.plot_relative_impression_share_barchart(
        n_channels=n_channels_to_plot
    )
    self.assertEqual(plot.vconcat[0].height, expected_height)

  # ============================================================================
  # KPI Tests
  # ============================================================================

  def test_plot_kpi_boxplot_geos(self):
    self._mock_eda_engine.kpi_scaled_da = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6]]),
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: list(_GEO_NAMES),
            constants.TIME: range(_N_TIMES),
        },
        name=constants.KPI,
    )

    plot = self._eda.plot_kpi_boxplot(geos=['geo_0'])

    actual_values = sorted(plot.data[constants.VALUE].tolist())
    self.assertEqual(actual_values, [1, 2, 3])

  def test_plot_kpi_boxplot_nationalize(self):
    self._mock_eda_engine.national_kpi_scaled_da = xr.DataArray(
        np.array([[10, 20, 30]]),
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
        },
        name=constants.KPI,
    )

    plot = self._eda.plot_kpi_boxplot(geos=eda_constants.NATIONALIZE)

    actual_values = sorted(plot.data[constants.VALUE].tolist())
    self.assertEqual(actual_values, [10, 20, 30])

  def test_plot_kpi_boxplot_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    self._mock_eda_engine.national_kpi_scaled_da = xr.DataArray(
        np.array([[100, 200, 300]]),
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: [constants.NATIONAL_MODEL_DEFAULT_GEO_NAME],
            constants.TIME: range(_N_TIMES),
        },
        name=constants.KPI,
    )

    plot = self._eda.plot_kpi_boxplot()

    actual_values = sorted(plot.data[constants.VALUE].tolist())
    self.assertEqual(actual_values, [100, 200, 300])

  # ============================================================================
  # Controls and Non Media Tests (Combined boxplot)
  # ============================================================================

  @parameterized.named_parameters(
      dict(
          testcase_name='geos',
          is_national=False,
          geos=['geo_0'],
          ds_attr_name='controls_and_non_media_scaled_ds',
          dataset=_create_controls_and_non_media_scaled_ds(is_national=False),
          expected_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0],
      ),
      dict(
          testcase_name='nationalize',
          is_national=False,
          geos=eda_constants.NATIONALIZE,
          ds_attr_name='national_controls_and_non_media_scaled_ds',
          dataset=_create_controls_and_non_media_scaled_ds(is_national=True),
          expected_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0],
      ),
      dict(
          testcase_name='national_model',
          is_national=True,
          geos=1,
          ds_attr_name='national_controls_and_non_media_scaled_ds',
          dataset=_create_controls_and_non_media_scaled_ds(is_national=True),
          expected_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0],
      ),
  )
  def test_plot_controls_and_non_media_boxplot(
      self, is_national, geos, ds_attr_name, dataset, expected_values
  ):
    self._meridian.is_national = is_national
    if is_national:
      self._meridian.n_geos = 1
    self._stub_engine_checks()

    setattr(self._mock_eda_engine, ds_attr_name, dataset)

    plot = self._eda.plot_controls_and_non_media_boxplot(geos=geos)

    actual_values = plot.data[eda_constants.VALUE].tolist()
    self.assertCountEqual(actual_values, expected_values)
    actual_channels = set(plot.data[constants.CHANNEL].tolist())
    self.assertIn(constants.CONTROL_VARIABLE, actual_channels)
    self.assertIn(constants.NON_MEDIA_CHANNEL, actual_channels)

  # ============================================================================
  # Treatments Without Non Media Tests (Stacked - Using xr.Dataset)
  # ============================================================================

  @parameterized.named_parameters(
      dict(
          testcase_name='geos',
          is_national=False,
          geos=['geo_0'],
          ds_attr_name='treatments_without_non_media_scaled_ds',
          dataset=_create_treatments_without_non_media_scaled_ds(
              is_national=False
          ),
          expected_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
      ),
      dict(
          testcase_name='nationalize',
          is_national=False,
          geos=eda_constants.NATIONALIZE,
          ds_attr_name='national_treatments_without_non_media_scaled_ds',
          dataset=_create_treatments_without_non_media_scaled_ds(
              is_national=True
          ),
          expected_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
      ),
      dict(
          testcase_name='national_model',
          is_national=True,
          geos=1,
          ds_attr_name='national_treatments_without_non_media_scaled_ds',
          dataset=_create_treatments_without_non_media_scaled_ds(
              is_national=True
          ),
          expected_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
      ),
  )
  def test_plot_treatments_without_non_media_boxplot(
      self, is_national, geos, ds_attr_name, dataset, expected_values
  ):
    self._meridian.is_national = is_national
    if is_national:
      self._meridian.n_geos = 1
    self._stub_engine_checks()

    setattr(self._mock_eda_engine, ds_attr_name, dataset)

    plot = self._eda.plot_treatments_without_non_media_boxplot(geos=geos)

    actual_values = plot.data[eda_constants.VALUE].tolist()
    self.assertCountEqual(actual_values, expected_values)

  # ============================================================================
  # Boxplot Sorting Tests
  # ============================================================================

  @parameterized.named_parameters(
      dict(testcase_name='zero', n_vars=0),
      dict(testcase_name='negative', n_vars=-1),
  )
  def test_boxplot_sorting_config_invalid_n_vars(self, n_vars):
    mock_std_artifact = eda_outcome.StandardDeviationArtifact(
        variable='dummy',
        level=eda_outcome.AnalysisLevel.OVERALL,
        std_ds=xr.Dataset(),
        outlier_df=pd.DataFrame(),
    )

    with self.assertRaisesRegex(
        ValueError, 'n_vars must be a positive integer'
    ):
      meridian_eda.BoxplotSortingConfig(
          n_vars=n_vars, std_artifact=mock_std_artifact
      )

  def _setup_boxplot_sorting_mocks(
      self,
      is_national: bool,
      std_ds_vals: Sequence[float],
      outliers_vals: Sequence[float],
      variables: Sequence[str],
      mock_outcome_property_name: str,
  ):
    self._meridian.is_national = is_national
    if is_national:
      self._meridian.n_geos = 1

    level = (
        eda_outcome.AnalysisLevel.NATIONAL
        if is_national
        else eda_outcome.AnalysisLevel.GEO
    )
    treatment_control_var = (
        constants.NATIONAL_TREATMENT_CONTROL_SCALED
        if is_national
        else constants.TREATMENT_CONTROL_SCALED
    )

    if is_national:
      std_ds = xr.Dataset(
          {
              eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME: (
                  [eda_constants.VARIABLE],
                  std_ds_vals,
              ),
          },
          coords={eda_constants.VARIABLE: variables},
      )
      outlier_df = pd.DataFrame({
          eda_constants.ABS_OUTLIERS_COL_NAME: outliers_vals,
          eda_constants.VARIABLE: variables,
          constants.TIME: [0] * len(variables),
      }).set_index([eda_constants.VARIABLE, constants.TIME])
    else:
      std_ds = xr.Dataset(
          {
              eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME: (
                  [eda_constants.VARIABLE, constants.GEO],
                  np.array(std_ds_vals).reshape(-1, 1),
              ),
          },
          coords={
              eda_constants.VARIABLE: variables,
              constants.GEO: ['geo_0'],
          },
      )
      outlier_df = pd.DataFrame({
          eda_constants.ABS_OUTLIERS_COL_NAME: outliers_vals,
          eda_constants.VARIABLE: variables,
          constants.GEO: ['geo_0'] * len(variables),
          constants.TIME: [0] * len(variables),
      }).set_index([eda_constants.VARIABLE, constants.GEO, constants.TIME])

    std_artifact = eda_outcome.StandardDeviationArtifact(
        variable=treatment_control_var,
        level=level,
        std_ds=std_ds,
        outlier_df=outlier_df,
    )

    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.STANDARD_DEVIATION,
        analysis_artifacts=[std_artifact],
    )

    self.enter_context(
        mock.patch.object(
            meridian_eda.MeridianEDA,
            mock_outcome_property_name,
            new_callable=mock.PropertyMock,
            return_value=mock_outcome,
        )
    )

  def test_plot_treatments_without_non_media_boxplot_missing_artifact(self):
    self._mock_eda_engine.national_treatments_without_non_media_scaled_ds = (
        _create_treatments_without_non_media_scaled_ds(is_national=True)
    )

    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.STANDARD_DEVIATION,
        analysis_artifacts=[],
    )

    self.enter_context(
        mock.patch.object(
            meridian_eda.MeridianEDA,
            'national_stdev_check_outcome',
            new_callable=mock.PropertyMock,
            return_value=mock_outcome,
        )
    )

    with self.assertRaisesRegex(
        ValueError,
        'Could not find StandardDeviationArtifact for',
    ):
      self._eda.plot_treatments_without_non_media_boxplot(
          geos=eda_constants.NATIONALIZE
      )

  def test_plot_controls_and_non_media_boxplot_missing_artifact(self):
    self._mock_eda_engine.national_controls_and_non_media_scaled_ds = (
        _create_controls_and_non_media_scaled_ds(is_national=True)
    )

    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.STANDARD_DEVIATION,
        analysis_artifacts=[],
    )

    self.enter_context(
        mock.patch.object(
            meridian_eda.MeridianEDA,
            'national_stdev_check_outcome',
            new_callable=mock.PropertyMock,
            return_value=mock_outcome,
        )
    )

    with self.assertRaisesRegex(
        ValueError,
        'Could not find StandardDeviationArtifact for',
    ):
      self._eda.plot_controls_and_non_media_boxplot(
          geos=eda_constants.NATIONALIZE
      )

  def test_plot_treatments_without_non_media_boxplot_sorting_missing_vars(self):
    self._setup_boxplot_sorting_mocks(
        is_national=True,
        std_ds_vals=[0.1, 0.2, 0.3],
        outliers_vals=[0.0, 0.0, 0.0],
        variables=list(_MEDIA_CHANNEL_NAMES) + ['c'],
        mock_outcome_property_name='national_stdev_check_outcome',
    )
    self._mock_eda_engine.national_treatments_without_non_media_scaled_ds = (
        _create_treatments_without_non_media_scaled_ds(is_national=True)
    )
    with self.assertRaisesRegex(
        ValueError,
        'The following variables are missing from the std_artifact',
    ):
      self._eda.plot_treatments_without_non_media_boxplot(
          geos=eda_constants.NATIONALIZE, max_vars=2
      )

  def test_plot_controls_and_non_media_boxplot_sorting_missing_vars(self):
    self._setup_boxplot_sorting_mocks(
        is_national=True,
        std_ds_vals=[0.1, 0.2, 0.3],
        outliers_vals=[0.0, 0.0, 0.0],
        variables=list(_CONTROL_NAMES) + ['c'],
        mock_outcome_property_name='national_stdev_check_outcome',
    )
    self._mock_eda_engine.national_controls_and_non_media_scaled_ds = (
        _create_controls_and_non_media_scaled_ds(is_national=True)
    )
    with self.assertRaisesRegex(
        ValueError,
        'The following variables are missing from the std_artifact',
    ):
      self._eda.plot_controls_and_non_media_boxplot(
          geos=eda_constants.NATIONALIZE, max_vars=2
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='treatments_n_vars_none_national',
          plot_function_name='plot_treatments_without_non_media_boxplot',
          is_national=True,
          geos=eda_constants.NATIONALIZE,
          mock_outcome_property_name='national_stdev_check_outcome',
          ds_attr_name='national_treatments_without_non_media_scaled_ds',
          dataset_generator=_create_treatments_without_non_media_scaled_ds,
          max_vars=None,
          std_ds_vals=[0.3, 0.2, 0.1],
          outliers_vals=[1.0, 2.0, 3.0],
          variables=list(_MEDIA_CHANNEL_NAMES) + list(_RF_CHANNEL_NAMES),
          expected_vars=['rf_ch_0', 'ch_1', 'ch_0'],
      ),
      dict(
          testcase_name='treatments_correct_order_national',
          plot_function_name='plot_treatments_without_non_media_boxplot',
          is_national=True,
          geos=eda_constants.NATIONALIZE,
          mock_outcome_property_name='national_stdev_check_outcome',
          ds_attr_name='national_treatments_without_non_media_scaled_ds',
          dataset_generator=_create_treatments_without_non_media_scaled_ds,
          max_vars=2,
          std_ds_vals=[0.3, 0.2, 0.1],
          outliers_vals=[1.0, 2.0, 3.0],
          variables=list(_MEDIA_CHANNEL_NAMES) + list(_RF_CHANNEL_NAMES),
          expected_vars=['rf_ch_0', 'ch_1'],
      ),
      dict(
          testcase_name='treatments_n_vars_exceeds_available_national',
          plot_function_name='plot_treatments_without_non_media_boxplot',
          is_national=True,
          geos=eda_constants.NATIONALIZE,
          mock_outcome_property_name='national_stdev_check_outcome',
          ds_attr_name='national_treatments_without_non_media_scaled_ds',
          dataset_generator=_create_treatments_without_non_media_scaled_ds,
          max_vars=10,
          std_ds_vals=[0.1, 0.2, 0.3],
          outliers_vals=[0.0, 0.0, 0.0],
          variables=list(_MEDIA_CHANNEL_NAMES) + list(_RF_CHANNEL_NAMES),
          expected_vars=['ch_0', 'ch_1', 'rf_ch_0'],
      ),
      dict(
          testcase_name='treatments_n_vars_none_geos',
          plot_function_name='plot_treatments_without_non_media_boxplot',
          is_national=False,
          geos=['geo_0'],
          mock_outcome_property_name='geo_stdev_check_outcome',
          ds_attr_name='treatments_without_non_media_scaled_ds',
          dataset_generator=_create_treatments_without_non_media_scaled_ds,
          max_vars=None,
          std_ds_vals=[0.3, 0.2, 0.1],
          outliers_vals=[1.0, 2.0, 3.0],
          variables=list(_MEDIA_CHANNEL_NAMES) + list(_RF_CHANNEL_NAMES),
          expected_vars=['rf_ch_0', 'ch_1', 'ch_0'],
      ),
      dict(
          testcase_name='treatments_correct_order_geos',
          plot_function_name='plot_treatments_without_non_media_boxplot',
          is_national=False,
          geos=['geo_0'],
          mock_outcome_property_name='geo_stdev_check_outcome',
          ds_attr_name='treatments_without_non_media_scaled_ds',
          dataset_generator=_create_treatments_without_non_media_scaled_ds,
          max_vars=2,
          std_ds_vals=[0.3, 0.2, 0.1],
          outliers_vals=[1.0, 2.0, 3.0],
          variables=list(_MEDIA_CHANNEL_NAMES) + list(_RF_CHANNEL_NAMES),
          expected_vars=['rf_ch_0', 'ch_1'],
      ),
      dict(
          testcase_name='controls_n_vars_none_national',
          plot_function_name='plot_controls_and_non_media_boxplot',
          is_national=True,
          geos=eda_constants.NATIONALIZE,
          mock_outcome_property_name='national_stdev_check_outcome',
          ds_attr_name='national_controls_and_non_media_scaled_ds',
          dataset_generator=_create_controls_and_non_media_scaled_ds,
          max_vars=None,
          std_ds_vals=[0.3, 0.2, 0.1],
          outliers_vals=[1.0, 2.0, 3.0],
          variables=list(_CONTROL_NAMES) + list(_NON_MEDIA_CHANNEL_NAMES),
          expected_vars=['non_media_0', 'control_1', 'control_0'],
      ),
      dict(
          testcase_name='controls_correct_order_national',
          plot_function_name='plot_controls_and_non_media_boxplot',
          is_national=True,
          geos=eda_constants.NATIONALIZE,
          mock_outcome_property_name='national_stdev_check_outcome',
          ds_attr_name='national_controls_and_non_media_scaled_ds',
          dataset_generator=_create_controls_and_non_media_scaled_ds,
          max_vars=2,
          std_ds_vals=[0.3, 0.2, 0.1],
          outliers_vals=[1.0, 2.0, 3.0],
          variables=list(_CONTROL_NAMES) + list(_NON_MEDIA_CHANNEL_NAMES),
          expected_vars=['non_media_0', 'control_1'],
      ),
      dict(
          testcase_name='controls_correct_order_geos',
          plot_function_name='plot_controls_and_non_media_boxplot',
          is_national=False,
          geos=['geo_0'],
          mock_outcome_property_name='geo_stdev_check_outcome',
          ds_attr_name='controls_and_non_media_scaled_ds',
          dataset_generator=_create_controls_and_non_media_scaled_ds,
          max_vars=2,
          std_ds_vals=[0.3, 0.2, 0.1],
          outliers_vals=[1.0, 2.0, 3.0],
          variables=list(_CONTROL_NAMES) + list(_NON_MEDIA_CHANNEL_NAMES),
          expected_vars=['non_media_0', 'control_1'],
      ),
  )
  def test_boxplot_sorting(
      self,
      plot_function_name,
      is_national,
      geos,
      mock_outcome_property_name,
      ds_attr_name,
      dataset_generator,
      max_vars,
      std_ds_vals,
      outliers_vals,
      variables,
      expected_vars,
  ):
    self._setup_boxplot_sorting_mocks(
        is_national=is_national,
        std_ds_vals=std_ds_vals,
        outliers_vals=outliers_vals,
        variables=variables,
        mock_outcome_property_name=mock_outcome_property_name,
    )
    setattr(self._mock_eda_engine, ds_attr_name, dataset_generator(is_national))

    plot_function = getattr(self._eda, plot_function_name)
    plot = plot_function(geos=geos, max_vars=max_vars)

    actual_data = plot.data
    self.assertCountEqual(
        actual_data[eda_constants.VARIABLE].unique(), expected_vars
    )
    plot_dict = plot.to_dict()
    self.assertEqual(
        plot_dict['vconcat'][0]['encoding']['x']['sort'], expected_vars
    )

  # ============================================================================
  # Pairwise Correlation Tests
  # ============================================================================

  def test_plot_pairwise_correlation_geos(self):
    mock_artifact = mock.create_autospec(
        eda_outcome.PairwiseCorrArtifact, instance=True
    )
    mock_artifact.level = eda_outcome.AnalysisLevel.GEO
    mock_artifact.corr_matrix = xr.DataArray(
        np.array([[
            [1.0, 0.8],
            [0.8, 1.0],
        ]]),
        coords={
            constants.GEO: ['geo_0'],
            eda_constants.VARIABLE_1: ['ch_0', 'control_0'],
            eda_constants.VARIABLE_2: ['ch_0', 'control_0'],
        },
        dims=[
            constants.GEO,
            eda_constants.VARIABLE_1,
            eda_constants.VARIABLE_2,
        ],
        name=eda_constants.CORRELATION,
    )
    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
        analysis_artifacts=[mock_artifact],
    )
    mock_outcomes = mock.create_autospec(
        eda_outcome.CriticalCheckEDAOutcomes, instance=True
    )
    mock_outcomes.pairwise_correlation = mock_outcome

    self.enter_context(
        mock.patch.object(
            self._mock_eda_engine,
            'run_all_critical_checks',
            return_value=mock_outcomes,
        )
    )
    plot = self._eda.plot_pairwise_correlation(geos=['geo_0'])

    actual_values = sorted(plot.data[eda_constants.CORRELATION].tolist())
    self.assertEqual(actual_values, [0.8])

  def test_plot_pairwise_correlation_nationalize(self):
    mock_artifact = mock.create_autospec(
        eda_outcome.PairwiseCorrArtifact, instance=True
    )
    mock_artifact.level = eda_outcome.AnalysisLevel.NATIONAL
    mock_artifact.corr_matrix = xr.DataArray(
        np.array([[1.0, -0.2], [-0.2, 1.0]]),
        coords={
            eda_constants.VARIABLE_1: ['ch_0', 'control_0'],
            eda_constants.VARIABLE_2: ['ch_0', 'control_0'],
        },
        dims=[eda_constants.VARIABLE_1, eda_constants.VARIABLE_2],
        name=eda_constants.CORRELATION,
    )
    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
        analysis_artifacts=[mock_artifact],
    )

    self.enter_context(
        mock.patch.object(
            self._mock_eda_engine,
            'check_national_pairwise_corr',
            return_value=mock_outcome,
        )
    )
    plot = self._eda.plot_pairwise_correlation(geos=eda_constants.NATIONALIZE)

    actual_values = sorted(plot.data[eda_constants.CORRELATION].tolist())
    self.assertEqual(actual_values, [-0.2])

  def test_plot_pairwise_correlation_national_model(self):
    self._meridian.is_national = True
    self._meridian.n_geos = 1

    mock_artifact = mock.create_autospec(
        eda_outcome.PairwiseCorrArtifact, instance=True
    )
    mock_artifact.level = eda_outcome.AnalysisLevel.NATIONAL
    mock_artifact.corr_matrix = xr.DataArray(
        np.array([[1.0, 0.99], [0.99, 1.0]]),
        coords={
            eda_constants.VARIABLE_1: ['ch_0', 'control_0'],
            eda_constants.VARIABLE_2: ['ch_0', 'control_0'],
        },
        dims=[eda_constants.VARIABLE_1, eda_constants.VARIABLE_2],
        name=eda_constants.CORRELATION,
    )

    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
        analysis_artifacts=[mock_artifact],
    )
    mock_outcomes = mock.create_autospec(
        eda_outcome.CriticalCheckEDAOutcomes, instance=True
    )
    mock_outcomes.pairwise_correlation = mock_outcome
    self.enter_context(
        mock.patch.object(
            self._mock_eda_engine,
            'run_all_critical_checks',
            return_value=mock_outcomes,
        )
    )

    plot = self._eda.plot_pairwise_correlation()

    actual_values = sorted(plot.data[eda_constants.CORRELATION].tolist())
    self.assertEqual(actual_values, [0.99])

  def _setup_pairwise_correlation_max_vars_mocks(
      self,
      corr_values: Sequence[float],
      all_vars: Sequence[str],
      is_national: bool,
  ):
    self._meridian.is_national = is_national
    self._meridian.n_geos = 1 if is_national else _N_GEOS

    n_vars = len(all_vars)
    corr_matrix = np.eye(n_vars)
    rows, cols = np.tril_indices(n_vars, k=-1)
    corr_matrix[rows, cols] = corr_values
    corr_matrix[cols, rows] = corr_values

    mock_artifact = mock.create_autospec(
        eda_outcome.PairwiseCorrArtifact, instance=True
    )
    mock_artifact.level = (
        eda_outcome.AnalysisLevel.NATIONAL
        if is_national
        else eda_outcome.AnalysisLevel.GEO
    )
    if is_national:
      mock_artifact.corr_matrix = xr.DataArray(
          corr_matrix,
          coords={
              eda_constants.VARIABLE_1: all_vars,
              eda_constants.VARIABLE_2: all_vars,
          },
          dims=[eda_constants.VARIABLE_1, eda_constants.VARIABLE_2],
          name=eda_constants.CORRELATION,
      )
    else:
      mock_artifact.corr_matrix = xr.DataArray(
          np.array([corr_matrix]),
          coords={
              constants.GEO: ['geo_0'],
              eda_constants.VARIABLE_1: all_vars,
              eda_constants.VARIABLE_2: all_vars,
          },
          dims=[
              constants.GEO,
              eda_constants.VARIABLE_1,
              eda_constants.VARIABLE_2,
          ],
          name=eda_constants.CORRELATION,
      )

    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
        analysis_artifacts=[mock_artifact],
    )

    mock_outcomes = mock.create_autospec(
        eda_outcome.CriticalCheckEDAOutcomes, instance=True
    )
    mock_outcomes.pairwise_correlation = mock_outcome
    self._mock_eda_engine.run_all_critical_checks.return_value = mock_outcomes

  def _assert_plot_pairwise_correlation_max_vars(
      self,
      max_vars: int | None,
      expected_vars: Sequence[str],
      expected_title: str,
      geos: list[str] | int,
  ):

    mock_plot_2d_heatmap = self.enter_context(
        mock.patch.object(
            self._eda,
            '_plot_2d_heatmap',
            autospec=True,
            return_value=alt.Chart(),
        )
    )

    self._eda.plot_pairwise_correlation(geos=geos, max_vars=max_vars)

    mock_plot_2d_heatmap.assert_called_once()
    _, actual_title, unique_variables = mock_plot_2d_heatmap.call_args.args

    with self.subTest(name='unique_variables_correct'):
      self.assertCountEqual(list(unique_variables), expected_vars)
    with self.subTest(name='title_correct'):
      self.assertEqual(actual_title.lower(), expected_title.lower())

  @parameterized.named_parameters(
      dict(
          testcase_name='max_vars_less_than_total_national',
          max_vars=2,
          expected_vars=['ch_0', 'ch_1'],
          corr_values=[0.9, 0.0, 0.0],
          all_vars=['ch_0', 'ch_1', 'control_0'],
          expected_title=(
              'Top 2 variables by pairwise correlations among treatments and'
              f' controls for {constants.NATIONAL_MODEL_DEFAULT_GEO_NAME}'
          ),
      ),
      dict(
          testcase_name='max_vars_exceeded_by_next_pair_national',
          max_vars=3,
          # (ch_1, ch_0) = 0.9, (control_1, control_0) = 0.8,
          # (control_0, ch_0) = 0.01
          # (ch_1, ch_0) adds {ch_0, ch_1} (2).
          # (control_1, control_0) adds {control_0, control_1} (4 > 3). STOP.
          # (control_0, ch_0) = 0.01 is NOT evaluated.
          # Expected: {ch_0, ch_1}
          expected_vars=['ch_0', 'ch_1'],
          corr_values=[0.9, 0.01, 0.0, 0.0, 0.0, 0.8],
          all_vars=['ch_0', 'ch_1', 'control_0', 'control_1'],
          expected_title=(
              'Top 2 variables by pairwise correlations among treatments and'
              f' controls for {constants.NATIONAL_MODEL_DEFAULT_GEO_NAME}'
          ),
      ),
      dict(
          testcase_name='max_vars_ties_national',
          max_vars=3,
          # (ch_1, ch_0) = 0.9. (control_1, rf_ch_0) = 0.8.
          # (control_0, ch_0) = 0.8.
          # (ch_1, ch_0) adds {ch_0, ch_1} (2).
          # Ties at 0.8:
          #   (control_1, rf_ch_0) adds {control_1, rf_ch_0} (4 > 3). SKIP.
          #   (control_0, ch_0) adds {control_0} (3). OK.
          # Stop.
          # Expected: {ch_0, ch_1, control_0}
          expected_vars=['ch_0', 'ch_1', 'control_0'],
          corr_values=[0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
          all_vars=['ch_0', 'ch_1', 'control_0', 'control_1', 'rf_ch_0'],
          expected_title=(
              'Top 3 variables by pairwise correlations among treatments and'
              f' controls for {constants.NATIONAL_MODEL_DEFAULT_GEO_NAME}'
          ),
      ),
      dict(
          testcase_name='max_vars_none_national',
          max_vars=None,
          expected_vars=['ch_0', 'control_0', 'rf_ch_0'],
          corr_values=[0.9, 0.8, 0.7],
          all_vars=['ch_0', 'control_0', 'rf_ch_0'],
          expected_title=(
              'Pairwise correlations among all treatments and controls for'
              f' {constants.NATIONAL_MODEL_DEFAULT_GEO_NAME}'
          ),
      ),
      dict(
          testcase_name='max_vars_equal_total_national',
          max_vars=3,
          expected_vars=['ch_0', 'control_0', 'rf_ch_0'],
          corr_values=[0.9, 0.8, 0.7],
          all_vars=['ch_0', 'control_0', 'rf_ch_0'],
          expected_title=(
              'Pairwise correlations among all treatments and controls for'
              f' {constants.NATIONAL_MODEL_DEFAULT_GEO_NAME}'
          ),
      ),
  )
  def test_plot_pairwise_correlation_max_vars_national(
      self,
      max_vars,
      expected_vars,
      corr_values,
      all_vars,
      expected_title,
  ):
    self._setup_pairwise_correlation_max_vars_mocks(
        corr_values=corr_values, all_vars=all_vars, is_national=True
    )
    self._assert_plot_pairwise_correlation_max_vars(
        max_vars,
        expected_vars,
        expected_title,
        geos=1,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='max_vars_less_than_total_geo',
          max_vars=2,
          expected_vars=['ch_0', 'ch_1'],
          corr_values=[0.9, 0.0, 0.0],
          all_vars=['ch_0', 'ch_1', 'control_0'],
          expected_title=(
              'Top 2 variables by pairwise correlations among treatments and'
              ' controls for geo_0'
          ),
      ),
      dict(
          testcase_name='max_vars_exceeded_by_next_pair_geo',
          max_vars=3,
          # (ch_1, ch_0) = 0.9, (control_1, control_0) = 0.8,
          # (control_0, ch_0) = 0.01
          # (ch_1, ch_0) adds {ch_0, ch_1} (2).
          # (control_1, control_0) adds {control_0, control_1} (4 > 3). STOP.
          # (control_0, ch_0) = 0.01 is NOT evaluated.
          # Expected: {ch_0, ch_1}
          expected_vars=['ch_0', 'ch_1'],
          corr_values=[0.9, 0.01, 0.0, 0.0, 0.0, 0.8],
          all_vars=['ch_0', 'ch_1', 'control_0', 'control_1'],
          expected_title=(
              'Top 2 variables by pairwise correlations among treatments and'
              ' controls for geo_0'
          ),
      ),
      dict(
          testcase_name='max_vars_ties_geo',
          max_vars=3,
          # (ch_1, ch_0) = 0.9. (control_1, rf_ch_0) = 0.8.
          # (control_0, ch_0) = 0.8.
          # (ch_1, ch_0) adds {ch_0, ch_1} (2).
          # Ties at 0.8:
          #   (control_1, rf_ch_0) adds {control_1, rf_ch_0} (4 > 3). SKIP.
          #   (control_0, ch_0) adds {control_0} (3). OK.
          # Stop.
          # Expected: {ch_0, ch_1, control_0}
          expected_vars=['ch_0', 'ch_1', 'control_0'],
          corr_values=[0.9, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
          all_vars=['ch_0', 'ch_1', 'control_0', 'control_1', 'rf_ch_0'],
          expected_title=(
              'Top 3 variables by pairwise correlations among treatments and'
              ' controls for geo_0'
          ),
      ),
      dict(
          testcase_name='max_vars_none_geo',
          max_vars=None,
          expected_vars=['ch_0', 'control_0', 'rf_ch_0'],
          corr_values=[0.9, 0.8, 0.7],
          all_vars=['ch_0', 'control_0', 'rf_ch_0'],
          expected_title=(
              'Pairwise correlations among all treatments and controls for'
              ' geo_0'
          ),
      ),
      dict(
          testcase_name='max_vars_equal_total_geo',
          max_vars=3,
          expected_vars=['ch_0', 'control_0', 'rf_ch_0'],
          corr_values=[0.9, 0.8, 0.7],
          all_vars=['ch_0', 'control_0', 'rf_ch_0'],
          expected_title=(
              'Pairwise correlations among all treatments and controls for'
              ' geo_0'
          ),
      ),
  )
  def test_plot_pairwise_correlation_max_vars_geo(
      self,
      max_vars,
      expected_vars,
      corr_values,
      all_vars,
      expected_title,
  ):
    self._setup_pairwise_correlation_max_vars_mocks(
        corr_values=corr_values, all_vars=all_vars, is_national=False
    )
    self._assert_plot_pairwise_correlation_max_vars(
        max_vars,
        expected_vars,
        expected_title,
        geos=['geo_0'],
    )

  @parameterized.named_parameters(
      dict(testcase_name='max_vars_zero', max_vars=0),
      dict(testcase_name='max_vars_one', max_vars=1),
      dict(testcase_name='max_vars_negative', max_vars=-1),
  )
  def test_plot_pairwise_correlation_invalid_max_vars(self, max_vars):
    with self.assertRaisesRegex(ValueError, 'max_vars must be greater than 1.'):
      self._eda.plot_pairwise_correlation(max_vars=max_vars)

  # ============================================================================
  # Knots Time Series Plot Tests
  # ============================================================================
  def test_plot_national_kpi_with_knots_time_series(self):
    self._meridian.knot_info.knot_locations = [0, 2]
    self._mock_eda_engine.national_kpi_scaled_da = xr.DataArray(
        data=np.array([100.0, 200.0, 300.0]),
        coords={constants.TIME: range(3)},
        dims=[constants.TIME],
        name='value',
    )

    plot = self._eda.plot_national_kpi_with_knots_time_series()

    knot_layer_data = plot.layer[1].data
    actual_kpi = sorted(knot_layer_data['value'].tolist())
    actual_knots = sorted(knot_layer_data['time'].tolist())

    self.assertEqual(actual_kpi, [100.0, 300.0])
    self.assertEqual(actual_knots, [0, 2])

  # ============================================================================
  # Population Correlation Tests
  # ============================================================================
  def test_plot_population_raw_media_correlation_success(self):
    """Tests that the raw media correlation plot uses the correct data."""
    artifact = _create_pop_corr_artifact([0.5, -0.2])
    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.POPULATION_CORRELATION,
        analysis_artifacts=[artifact],
    )

    self.enter_context(
        mock.patch.object(
            self._mock_eda_engine,
            'check_population_corr_raw_media',
            return_value=mock_outcome,
        )
    )

    plot = self._eda.plot_population_raw_media_correlation()

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    np.testing.assert_allclose(actual_values, [-0.2, 0.5])

  def test_plot_population_treatment_correlation_success(self):
    """Tests that the treatment correlation plot uses the correct data."""
    artifact = _create_pop_corr_artifact([0.9, 0.1])
    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.POPULATION_CORRELATION,
        analysis_artifacts=[artifact],
    )

    self.enter_context(
        mock.patch.object(
            self._mock_eda_engine,
            'check_population_corr_scaled_treatment_control',
            return_value=mock_outcome,
        )
    )

    plot = self._eda.plot_population_treatment_correlation()

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    np.testing.assert_allclose(actual_values, [0.1, 0.9])

  # ============================================================================
  # Prior Mean Tests
  # ============================================================================
  def test_plot_prior_mean_success(self):
    artifact = _create_prior_artifact([0.5, -0.2])
    mock_outcome = _create_eda_outcome(
        check_type=eda_outcome.EDACheckType.PRIOR_PROBABILITY,
        analysis_artifacts=[artifact],
    )

    self.enter_context(
        mock.patch.object(
            self._mock_eda_engine,
            'check_prior_probability',
            return_value=mock_outcome,
        )
    )

    plot = self._eda.plot_prior_mean()

    actual_values = sorted(plot.data[eda_constants.VALUE].tolist())
    np.testing.assert_allclose(actual_values, [-0.2, 0.5])

  # ============================================================================
  # Error Scenarios
  # ============================================================================
  def test_plot_error_invalid_geo(self):
    self._mock_eda_engine.kpi_scaled_da = xr.DataArray(
        np.zeros((_N_GEOS, _N_TIMES)),
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: list(_GEO_NAMES),
            constants.TIME: range(_N_TIMES),
        },
    )

    with self.assertRaisesRegex(ValueError, 'Geo fake_geo does not exist'):
      self._eda.plot_kpi_boxplot(geos=['fake_geo'])

  def test_plot_error_duplicate_geos(self):
    self._mock_eda_engine.kpi_scaled_da = xr.DataArray(
        np.zeros((_N_GEOS, _N_TIMES)),
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: list(_GEO_NAMES),
            constants.TIME: range(_N_TIMES),
        },
    )
    with self.assertRaisesRegex(ValueError, 'geos must not contain duplicate'):
      self._eda.plot_kpi_boxplot(geos=['geo_0', 'geo_0'])

  def test_plot_error_integer_out_of_bounds(self):
    self._mock_eda_engine.kpi_scaled_da = xr.DataArray(
        np.zeros((_N_GEOS, _N_TIMES)),
        dims=[constants.GEO, constants.TIME],
        coords={
            constants.GEO: list(_GEO_NAMES),
            constants.TIME: range(_N_TIMES),
        },
    )
    with self.assertRaisesRegex(ValueError, 'positive integer less than'):
      self._eda.plot_kpi_boxplot(geos=_N_GEOS + 1)

  def test_no_data_to_plot(self):
    self._mock_eda_engine.kpi_scaled_da = None
    with self.assertRaisesRegex(
        ValueError,
        'There is no data to plot! Make sure your InputData contains the'
        ' component you are trying to plot.',
    ):
      self._eda.plot_kpi_boxplot()

  @parameterized.named_parameters(
      dict(
          testcase_name='raw_media_correlation',
          plotting_function_name='plot_population_raw_media_correlation',
      ),
      dict(
          testcase_name='treatment_correlation',
          plotting_function_name='plot_population_treatment_correlation',
      ),
  )
  def test_plot_population_correlation_national_raises(
      self, plotting_function_name
  ):
    self._meridian.is_national = True
    with self.assertRaises(eda_engine.GeoLevelCheckOnNationalModelError):
      getattr(self._eda, plotting_function_name)()

  def test_plot_prior_mean_national_raises(self):
    self._meridian.is_national = True
    with self.assertRaises(eda_engine.GeoLevelCheckOnNationalModelError):
      self._eda.plot_prior_mean()

  # ============================================================================
  # Report Generation and HTML Structure Tests
  # ============================================================================
  def _get_output_eda_report_html_dom(self) -> ET.Element:
    """Helper to generate the report to a temp file and parse the HTML DOM."""
    tmpdir = self.create_tempdir()
    outfile_path = os.path.join(tmpdir.full_path, 'eda_summary')
    os.makedirs(outfile_path)
    outfile_name = 'eda.html'
    fpath = os.path.join(outfile_path, outfile_name)

    self._eda.generate_and_save_report(
        filename=outfile_name,
        filepath=outfile_path,
    )

    with open(fpath, 'r') as f:
      written_html_dom = ET.parse(f)

    root = written_html_dom.getroot()
    self.assertEqual(root.tag, 'html')
    return root

  def test_generate_and_save_report_io(self):
    """Tests that the report function actually writes a file to disk."""
    mock_gen = self.enter_context(
        mock.patch.object(self._eda, '_generate_report_html')
    )
    mock_gen.return_value = '<html><body>Test Content</body></html>'

    tmp_dir = self.create_tempdir()
    filename = 'test_report.html'
    self._eda.generate_and_save_report(filename, tmp_dir.full_path)

    expected_path = os.path.join(tmp_dir.full_path, filename)
    self.assertTrue(os.path.exists(expected_path))

    with open(expected_path, 'r') as f:
      content = f.read()
      self.assertEqual(content, '<html><body>Test Content</body></html>')

  def test_generate_report_html_overall_structure(self):
    """Tests that the report contains the correct title and structure."""
    self._stub_plotters()
    self._stub_engine_checks()

    dom = self._get_output_eda_report_html_dom()

    title = dom.find('head/title')
    self.assertIsNotNone(title, 'Title tag not found in head.')
    title_text = title.text
    self.assertIsNotNone(title_text, 'Title text is None.')
    self.assertEqual(title_text.strip(), eda_constants.REPORT_TITLE)

    body_title = dom.find(".//div[@class='title']")
    self.assertIsNotNone(
        body_title, "Div with class 'title' not found in body."
    )
    body_title_text = body_title.text
    self.assertIsNotNone(body_title_text, 'Body title text is None.')
    self.assertEqual(body_title_text.strip(), eda_constants.REPORT_TITLE)

  @parameterized.named_parameters(
      dict(
          testcase_name='summary_card_clean',
          is_national=False,
          card_id=eda_constants.SUMMARY_CARD_ID,
          card_title=eda_constants.SUMMARY_CARD_TITLE,
          expected_ids=[eda_constants.SUMMARY_TABLE_ID],
          missing_ids=[],
          expected_text=[
              'Info',
              eda_constants.SPEND_AND_MEDIA_UNIT_CARD_TITLE,
              eda_constants.RESPONSE_VARIABLES_CARD_TITLE,
          ],
      ),
      dict(
          testcase_name='summary_card_with_error',
          is_national=True,
          card_id=eda_constants.SUMMARY_CARD_ID,
          card_title=eda_constants.SUMMARY_CARD_TITLE,
          vif_findings=[_create_vif_finding(Severity.ERROR)],
          expected_ids=[eda_constants.SUMMARY_TABLE_ID],
          missing_ids=[],
          expected_text=[
              '1 fail(s)',
              eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_TITLE,
              'Info',
          ],
      ),
      dict(
          testcase_name='summary_card_mixed_complex',
          is_national=False,
          card_id=eda_constants.SUMMARY_CARD_ID,
          card_title=eda_constants.SUMMARY_CARD_TITLE,
          cpmu_findings=[
              _create_cpmu_finding(
                  Severity.ATTENTION,
                  Cause.OUTLIER,
              )
          ],
          kpi_findings=[
              _create_generic_finding(
                  Severity.ERROR,
                  Cause.VARIABILITY,
                  mock.create_autospec(
                      eda_outcome.KpiInvariabilityArtifact,
                      instance=True,
                  ),
                  'KPI ERROR Explanation',
              ),
          ],
          expected_ids=[eda_constants.SUMMARY_TABLE_ID],
          missing_ids=[],
          expected_text=[
              '1 fail(s)',
              '1 review(s)',
              eda_constants.SPEND_AND_MEDIA_UNIT_CARD_TITLE,
              eda_constants.RESPONSE_VARIABLES_CARD_TITLE,
          ],
      ),
      dict(
          testcase_name='spend_media_unit_card_data_adequacy',
          is_national=False,
          card_id=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_ID,
          card_title=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_TITLE,
          expected_ids=[
              eda_constants.RELATIVE_SPEND_SHARE_CHART_ID,
              eda_constants.DATA_ADEQUACY_TABLE_ID,
          ],
          missing_ids=[
              eda_constants.SPEND_PER_MEDIA_UNIT_CHART_ID,
              eda_constants.INCONSISTENT_DATA_TABLE_ID,
              eda_constants.COST_PER_MEDIA_UNIT_OUTLIER_TABLE_ID,
          ],
          expected_text=[
              "Please review the channel's share of spend.",
              'As a rough guidance',
          ],
      ),
      dict(
          testcase_name='spend_media_unit_card_geo_clean',
          is_national=False,
          card_id=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_ID,
          card_title=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_TITLE,
          expected_ids=[
              eda_constants.RELATIVE_SPEND_SHARE_CHART_ID,
              eda_constants.DATA_ADEQUACY_TABLE_ID,
          ],
          missing_ids=[
              eda_constants.SPEND_PER_MEDIA_UNIT_CHART_ID,
              eda_constants.INCONSISTENT_DATA_TABLE_ID,
              eda_constants.COST_PER_MEDIA_UNIT_OUTLIER_TABLE_ID,
          ],
          expected_text=[
              "Please review the channel's share of spend.",
          ],
      ),
      dict(
          testcase_name='spend_media_unit_card_national_clean',
          is_national=True,
          card_id=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_ID,
          card_title=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_TITLE,
          expected_ids=[
              eda_constants.RELATIVE_SPEND_SHARE_CHART_ID,
              eda_constants.DATA_ADEQUACY_TABLE_ID,
          ],
          missing_ids=[
              eda_constants.SPEND_PER_MEDIA_UNIT_CHART_ID,
              eda_constants.INCONSISTENT_DATA_TABLE_ID,
              eda_constants.COST_PER_MEDIA_UNIT_OUTLIER_TABLE_ID,
          ],
          expected_text=[
              "Please review the channel's share of spend.",
          ],
      ),
      dict(
          testcase_name='spend_media_unit_card_two_attentions',
          is_national=False,
          card_id=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_ID,
          card_title=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_TITLE,
          cpmu_findings=[
              _create_cpmu_finding(
                  Severity.ATTENTION,
                  Cause.INCONSISTENT_DATA,
              ),
              _create_cpmu_finding(
                  Severity.ATTENTION,
                  Cause.OUTLIER,
              ),
          ],
          expected_ids=[
              eda_constants.RELATIVE_SPEND_SHARE_CHART_ID,
              eda_constants.SPEND_PER_MEDIA_UNIT_CHART_ID,
              eda_constants.INCONSISTENT_DATA_TABLE_ID,
              eda_constants.COST_PER_MEDIA_UNIT_OUTLIER_TABLE_ID,
          ],
          missing_ids=[],
          expected_text=[
              'Please review the patterns for spend',
              'CPMU ATTENTION INCONSISTENT_DATA',
              'CPMU ATTENTION OUTLIER',
          ],
      ),
      dict(
          testcase_name='spend_media_unit_card_inconsistency_attention',
          is_national=False,
          card_id=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_ID,
          card_title=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_TITLE,
          cpmu_findings=[
              _create_cpmu_finding(
                  Severity.ATTENTION,
                  Cause.INCONSISTENT_DATA,
              ),
          ],
          expected_ids=[
              eda_constants.RELATIVE_SPEND_SHARE_CHART_ID,
              eda_constants.SPEND_PER_MEDIA_UNIT_CHART_ID,
              eda_constants.INCONSISTENT_DATA_TABLE_ID,
          ],
          missing_ids=[eda_constants.COST_PER_MEDIA_UNIT_OUTLIER_TABLE_ID],
          expected_text=[
              'Please review the patterns for spend',
              'CPMU ATTENTION INCONSISTENT_DATA',
          ],
      ),
      dict(
          testcase_name='spend_media_unit_card_outlier_attention',
          is_national=False,
          card_id=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_ID,
          card_title=eda_constants.SPEND_AND_MEDIA_UNIT_CARD_TITLE,
          cpmu_findings=[
              _create_cpmu_finding(
                  Severity.ATTENTION,
                  Cause.OUTLIER,
              ),
          ],
          expected_ids=[
              eda_constants.RELATIVE_SPEND_SHARE_CHART_ID,
              eda_constants.SPEND_PER_MEDIA_UNIT_CHART_ID,
              eda_constants.COST_PER_MEDIA_UNIT_OUTLIER_TABLE_ID,
          ],
          missing_ids=[eda_constants.INCONSISTENT_DATA_TABLE_ID],
          expected_text=[
              'Please review the patterns for spend',
              'CPMU ATTENTION OUTLIER',
          ],
      ),
      dict(
          testcase_name='response_variables_card_geo_clean',
          is_national=False,
          card_id=eda_constants.RESPONSE_VARIABLES_CARD_ID,
          card_title=eda_constants.RESPONSE_VARIABLES_CARD_TITLE,
          expected_ids=[
              eda_constants.TREATMENTS_CHART_ID,
              eda_constants.CONTROLS_AND_NON_MEDIA_CHART_ID,
              eda_constants.KPI_CHART_ID,
          ],
          missing_ids=[
              eda_constants.TREATMENT_CONTROL_OUTLIER_TABLE_ID,
              eda_constants.TREATMENT_CONTROL_VARIABILITY_TABLE_ID,
          ],
          expected_text=[
              'Please review the variability of the explanatory and response'
          ],
      ),
      dict(
          testcase_name='response_variables_card_national_clean',
          is_national=True,
          card_id=eda_constants.RESPONSE_VARIABLES_CARD_ID,
          card_title=eda_constants.RESPONSE_VARIABLES_CARD_TITLE,
          expected_ids=[
              eda_constants.TREATMENTS_CHART_ID,
              eda_constants.CONTROLS_AND_NON_MEDIA_CHART_ID,
              eda_constants.KPI_CHART_ID,
          ],
          missing_ids=[
              eda_constants.TREATMENT_CONTROL_OUTLIER_TABLE_ID,
              eda_constants.TREATMENT_CONTROL_VARIABILITY_TABLE_ID,
          ],
          expected_text=[
              'Please review the variability of the explanatory and response'
          ],
      ),
      dict(
          testcase_name='response_variables_card_error_and_two_attentions',
          is_national=False,
          card_id=eda_constants.RESPONSE_VARIABLES_CARD_ID,
          card_title=eda_constants.RESPONSE_VARIABLES_CARD_TITLE,
          kpi_findings=[
              _create_generic_finding(
                  Severity.ERROR,
                  Cause.VARIABILITY,
                  mock.create_autospec(
                      eda_outcome.KpiInvariabilityArtifact,
                      instance=True,
                      spec_set=True,
                  ),
                  'KPI ERROR Explanation',
              ),
          ],
          stdev_findings=[
              _create_stdev_finding(
                  Severity.ATTENTION,
                  Cause.VARIABILITY,
                  constants.TREATMENT_CONTROL_SCALED,
              ),
              _create_stdev_finding(
                  Severity.ATTENTION,
                  Cause.OUTLIER,
                  constants.TREATMENT_CONTROL_SCALED,
              ),
          ],
          expected_ids=[
              eda_constants.TREATMENTS_CHART_ID,
              eda_constants.CONTROLS_AND_NON_MEDIA_CHART_ID,
              eda_constants.KPI_CHART_ID,
              eda_constants.TREATMENT_CONTROL_OUTLIER_TABLE_ID,
              eda_constants.TREATMENT_CONTROL_VARIABILITY_TABLE_ID,
          ],
          missing_ids=[],
          expected_text=[
              'KPI ERROR Explanation',
              'treatment_control_scaled ATTENTION VARIABILITY',
              'treatment_control_scaled ATTENTION OUTLIER',
          ],
      ),
      dict(
          testcase_name='response_variables_card_three_attentions',
          is_national=False,
          card_id=eda_constants.RESPONSE_VARIABLES_CARD_ID,
          card_title=eda_constants.RESPONSE_VARIABLES_CARD_TITLE,
          stdev_findings=[
              _create_stdev_finding(
                  Severity.ATTENTION,
                  Cause.VARIABILITY,
                  constants.TREATMENT_CONTROL_SCALED,
              ),
              _create_stdev_finding(
                  Severity.ATTENTION,
                  Cause.OUTLIER,
                  constants.TREATMENT_CONTROL_SCALED,
              ),
              _create_stdev_finding(
                  Severity.ATTENTION,
                  Cause.VARIABILITY,
                  constants.KPI_SCALED,
              ),
          ],
          expected_ids=[
              eda_constants.TREATMENTS_CHART_ID,
              eda_constants.CONTROLS_AND_NON_MEDIA_CHART_ID,
              eda_constants.KPI_CHART_ID,
              eda_constants.TREATMENT_CONTROL_OUTLIER_TABLE_ID,
              eda_constants.TREATMENT_CONTROL_VARIABILITY_TABLE_ID,
          ],
          missing_ids=[],
          expected_text=[
              'kpi_scaled ATTENTION VARIABILITY',
              'treatment_control_scaled ATTENTION VARIABILITY',
              'treatment_control_scaled ATTENTION OUTLIER',
          ],
      ),
      dict(
          testcase_name=(
              'response_variables_card_no_kpi_finding_chart_still_there'
          ),
          is_national=False,
          card_id=eda_constants.RESPONSE_VARIABLES_CARD_ID,
          card_title=eda_constants.RESPONSE_VARIABLES_CARD_TITLE,
          stdev_findings=[
              _create_stdev_finding(
                  Severity.ATTENTION,
                  Cause.VARIABILITY,
                  constants.TREATMENT_CONTROL_SCALED,
              ),
              _create_stdev_finding(
                  Severity.ATTENTION,
                  Cause.OUTLIER,
                  constants.TREATMENT_CONTROL_SCALED,
              ),
          ],
          expected_ids=[
              eda_constants.TREATMENTS_CHART_ID,
              eda_constants.CONTROLS_AND_NON_MEDIA_CHART_ID,
              eda_constants.KPI_CHART_ID,
              eda_constants.TREATMENT_CONTROL_OUTLIER_TABLE_ID,
              eda_constants.TREATMENT_CONTROL_VARIABILITY_TABLE_ID,
          ],
          missing_ids=[],
          expected_text=[
              'treatment_control_scaled ATTENTION VARIABILITY',
              'treatment_control_scaled ATTENTION OUTLIER',
          ],
      ),
      dict(
          testcase_name='population_scaling_card_geo_clean',
          is_national=False,
          card_id=eda_constants.POPULATION_SCALING_CARD_ID,
          card_title=eda_constants.POPULATION_SCALING_CARD_TITLE,
          expected_ids=[
              eda_constants.POPULATION_RAW_MEDIA_CHART_ID,
              eda_constants.POPULATION_TREATMENT_CHART_ID,
          ],
          missing_ids=[],
          expected_text=[
              'Please review the Spearman correlation between population and'
              ' raw paid'
          ],
          pop_raw_media_artifacts=[_create_pop_corr_artifact([0.5, 0.6])],
          pop_treatment_artifacts=[_create_pop_corr_artifact([0.1, 0.2])],
      ),
      dict(
          testcase_name='relationship_among_variables_card_national_clean',
          is_national=True,
          card_id=eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_ID,
          card_title=eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_TITLE,
          expected_ids=[eda_constants.PAIRWISE_CORRELATION_CHART_ID],
          missing_ids=[
              eda_constants.EXTREME_VIF_ERROR_TABLE_ID,
              eda_constants.EXTREME_VIF_ATTENTION_TABLE_ID,
              eda_constants.R_SQUARED_TIME_TABLE_ID,
              eda_constants.R_SQUARED_GEO_TABLE_ID,
          ],
          expected_text=['Please review the computed pairwise correlations.'],
      ),
      dict(
          testcase_name='relationship_among_variables_card_national_with_error',
          is_national=True,
          card_id=eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_ID,
          card_title=eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_TITLE,
          vif_findings=[_create_vif_finding(Severity.ERROR)],
          expected_ids=[
              eda_constants.PAIRWISE_CORRELATION_CHART_ID,
              eda_constants.EXTREME_VIF_ERROR_TABLE_ID,
          ],
          missing_ids=[eda_constants.EXTREME_VIF_ATTENTION_TABLE_ID],
          expected_text=['Some variables have extreme multicollinearity'],
      ),
      dict(
          testcase_name='relationship_among_variables_card_geo_clean',
          is_national=False,
          card_id=eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_ID,
          card_title=eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_TITLE,
          vif_findings=[
              _create_vif_finding(
                  Severity.ERROR,
                  finding_cause=Cause.NONE,
              )
          ],  # ignores irrelevant findings
          expected_ids=[
              eda_constants.PAIRWISE_CORRELATION_CHART_ID,
              eda_constants.R_SQUARED_TIME_TABLE_ID,
              eda_constants.R_SQUARED_GEO_TABLE_ID,
          ],
          missing_ids=[
              eda_constants.EXTREME_VIF_ERROR_TABLE_ID,
              eda_constants.EXTREME_VIF_ATTENTION_TABLE_ID,
          ],
          expected_text=[
              'Please review the computed pairwise correlations.',
              'This check regresses each variable against time',
              'This check regresses each variable against geo',
          ],
      ),
      dict(
          testcase_name=(
              'relationship_among_variables_card_geo_with_attention_and_error'
          ),
          is_national=False,
          card_id=eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_ID,
          card_title=eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_TITLE,
          vif_findings=[
              _create_vif_finding(Severity.ERROR),
              _create_vif_finding(Severity.ATTENTION),
          ],
          expected_ids=[
              eda_constants.PAIRWISE_CORRELATION_CHART_ID,
              eda_constants.EXTREME_VIF_ERROR_TABLE_ID,
              eda_constants.EXTREME_VIF_ATTENTION_TABLE_ID,
              eda_constants.R_SQUARED_TIME_TABLE_ID,
              eda_constants.R_SQUARED_GEO_TABLE_ID,
          ],
          missing_ids=[],
          expected_text=[
              'Some variables have extreme multicollinearity',
          ],
      ),
      dict(
          testcase_name='prior_specifications_card_geo_clean',
          is_national=False,
          card_id=eda_constants.PRIOR_SPECIFICATIONS_CARD_ID,
          card_title=eda_constants.PRIOR_SPECIFICATIONS_CARD_TITLE,
          expected_ids=[
              eda_constants.PRIOR_CHART_ID,
          ],
          missing_ids=[],
          expected_text=[
              'Negative baseline is equivalent to the treatment effects'
          ],
          prior_artifacts=[_create_prior_artifact([0.5, 0.6])],
      ),
  )
  def test_card_structure_scenarios(
      self,
      is_national: bool,
      card_id: str,
      card_title: str,
      expected_ids: Sequence[str],
      missing_ids: Sequence[str],
      expected_text: Sequence[str],
      cpmu_findings: Sequence[eda_outcome.EDAFinding] | None = None,
      kpi_findings: Sequence[eda_outcome.EDAFinding] | None = None,
      stdev_findings: Sequence[eda_outcome.EDAFinding] | None = None,
      vif_findings: Sequence[eda_outcome.EDAFinding] | None = None,
      pop_raw_media_artifacts: (
          Sequence[eda_outcome.PopulationCorrelationArtifact] | None
      ) = None,
      pop_treatment_artifacts: (
          Sequence[eda_outcome.PopulationCorrelationArtifact] | None
      ) = None,
      prior_artifacts: (
          Sequence[eda_outcome.PriorProbabilityArtifact] | None
      ) = None,
  ) -> None:
    self._stub_plotters()
    self._meridian.is_national = is_national
    self._stub_engine_checks(
        cpmu_findings=cpmu_findings or [],
        kpi_findings=kpi_findings or [],
        stdev_findings=stdev_findings or [],
        vif_findings=vif_findings or [],
        pop_raw_media_artifacts=pop_raw_media_artifacts or [],
        pop_treatment_artifacts=pop_treatment_artifacts or [],
        prior_artifacts=prior_artifacts or [],
    )

    dom = self._get_output_eda_report_html_dom()
    card = dom.find(f".//card[@id='{card_id}']")
    with self.subTest(name='card_exists'):
      self.assertIsNotNone(card, f'Card with id "{card_id}" not found in DOM.')

    found_card_title = card.find('card-title')
    with self.subTest(name='card_title_exists'):
      self.assertIsNotNone(
          found_card_title,
          f'Card title element not found within card with id "{card_id}".',
      )

    title_text_content = found_card_title.text
    with self.subTest(name='card_title_text_correct'):
      self.assertIsNotNone(title_text_content, 'Card title text is None.')
      self.assertEqual(title_text_content.strip(), card_title)

    for chart_id in expected_ids:
      with self.subTest(f'expected_id_{chart_id}'):
        self.assertIsNotNone(
            card.find(f".//*[@id='{chart_id}']"),
            f"Expected element '{chart_id}' missing.",
        )

    for chart_id in missing_ids:
      with self.subTest(f'missing_id_{chart_id}'):
        self.assertIsNone(
            card.find(f".//*[@id='{chart_id}']"),
            f"Found unexpected element '{chart_id}'.",
        )

    full_text = ''.join(card.itertext())
    for snippet in expected_text:
      with self.subTest(f'expected_text_{snippet}'):
        self.assertIn(snippet, full_text)

  @parameterized.named_parameters(
      dict(
          testcase_name='variability',
          finding_cause=Cause.VARIABILITY,
          table_id=eda_constants.TREATMENT_CONTROL_VARIABILITY_TABLE_ID,
          artifact_kwargs={
              'std_ds': xr.Dataset(
                  {
                      eda_constants.STD_WITHOUT_OUTLIERS_VAR_NAME: (
                          [eda_constants.VARIABLE, constants.GEO],
                          np.array([0.0, 0.0, 0.00005, 0.06]).reshape(-1, 1),
                      ),
                      eda_constants.STD_WITH_OUTLIERS_VAR_NAME: (
                          [eda_constants.VARIABLE, constants.GEO],
                          np.array([0.5, 0.0, 0.1, 1.0]).reshape(-1, 1),
                      ),
                  },
                  coords={
                      eda_constants.VARIABLE: ['var1', 'var2', 'var3', 'var4'],
                      constants.GEO: ['geo_0'],
                  },
              ),
              'outlier_df': pd.DataFrame(),
          },
          expected_order=['var2', 'var1', 'var3'],
          unexpected_vars=['var4'],
      ),
      dict(
          testcase_name='outlier',
          finding_cause=Cause.OUTLIER,
          table_id=eda_constants.TREATMENT_CONTROL_OUTLIER_TABLE_ID,
          artifact_kwargs={
              'std_ds': xr.Dataset(),
              'outlier_df': (
                  pd.DataFrame({
                      eda_constants.OUTLIERS_COL_NAME: [-5.0, 10.0, 1.0],
                      eda_constants.ABS_OUTLIERS_COL_NAME: [5.0, 10.0, 1.0],
                      eda_constants.VARIABLE: ['var1', 'var2', 'var3'],
                      constants.GEO: ['geo_0'] * 3,
                      constants.TIME: [0] * 3,
                  }).set_index(
                      [eda_constants.VARIABLE, constants.GEO, constants.TIME]
                  )
              ),
          },
          expected_order=['var2', 'var1', 'var3'],
          unexpected_vars=[],
      ),
  )
  def test_response_variables_stdev_callout_sorting(
      self,
      finding_cause,
      table_id,
      artifact_kwargs,
      expected_order,
      unexpected_vars,
  ) -> None:
    self._stub_plotters()
    self._meridian.is_national = False
    self._meridian.eda_spec = eda_spec.EDASpec(
        std_spec=eda_spec.StandardDeviationSpec(geo_std_threshold=0.05)
    )
    self._mock_eda_engine.spec = self._meridian.eda_spec

    artifact = eda_outcome.StandardDeviationArtifact(
        level=eda_outcome.AnalysisLevel.GEO,
        variable='treatment_control_scaled',
        **artifact_kwargs,
    )
    finding = _create_generic_finding(
        finding_cause=finding_cause,
        severity=Severity.ATTENTION,
        associated_artifact=artifact,
        explanation=f'{finding_cause.name} finding',
    )

    self._stub_engine_checks(stdev_findings=[finding])

    dom = self._get_output_eda_report_html_dom()
    card = dom.find(
        f".//card[@id='{eda_constants.RESPONSE_VARIABLES_CARD_ID}']"
    )
    self.assertIsNotNone(card)

    table = card.find(f".//*[@id='{table_id}']")
    self.assertIsNotNone(table)
    text = ''.join(table.itertext())

    vars_in_text = [var for var in unexpected_vars if var in text]
    self.assertEmpty(vars_in_text)

    # Check order
    self.assertContainsInOrder(expected_order, text)

  def test_relationship_card_values(self) -> None:
    self._stub_plotters()
    self._meridian.is_national = False
    self._stub_engine_checks(
        vif_findings=[
            _create_vif_finding(Severity.ERROR),
            _create_vif_finding(Severity.ATTENTION),
        ],
        pairwise_findings=[_create_pairwise_finding(Severity.ERROR)],
    )

    dom = self._get_output_eda_report_html_dom()
    card = dom.find(
        f".//card[@id='{eda_constants.RELATIONSHIP_BETWEEN_VARIABLES_CARD_ID}']"
    )
    self.assertIsNotNone(card)

    with self.subTest(name='error_table_pairwise_join'):
      table = card.find(
          f".//*[@id='{eda_constants.EXTREME_VIF_ERROR_TABLE_ID}']"
      )
      self.assertIsNotNone(table)
      text = ''.join(table.itertext())
      self.assertIn('ch_0', text)
      self.assertIn('ch_1', text)

    with self.subTest(name='attention_table_filtering'):
      table = card.find(
          f".//*[@id='{eda_constants.EXTREME_VIF_ATTENTION_TABLE_ID}']"
      )
      self.assertIsNotNone(table)
      text = ''.join(table.itertext())
      self.assertNotIn('ch_0', text)

    with self.subTest(name='rsquared_values'):
      table = card.find(f".//*[@id='{eda_constants.R_SQUARED_TIME_TABLE_ID}']")
      self.assertIsNotNone(table)
      text = ''.join(table.itertext())
      self.assertIn('ch_0', text)
      self.assertIn('0.500', text)

  def test_spend_media_unit_table_channels_determines_charts(self):
    self._stub_plotters()
    self._stub_engine_checks(
        cpmu_findings=[
            _create_cpmu_finding(
                Severity.ATTENTION,
                Cause.OUTLIER,
            )
        ]
    )
    self._get_output_eda_report_html_dom()
    self._eda.plot_cost_per_media_unit_time_series.assert_called_with(
        eda_constants.NATIONALIZE, [_MEDIA_CHANNEL_NAMES[0]]
    )  # type: ignore

  def test_display_limit_message_keywords_with_six_rows(self):
    self._stub_plotters()
    self._meridian.is_national = False

    n_rows = 6
    channels = [f'ch_{i}' for i in range(n_rows)]
    geos = [f'geo_{i}' for i in range(n_rows)]
    times = pd.date_range(start='2020-01-01', periods=n_rows)

    outlier_df = pd.DataFrame({
        constants.CHANNEL: channels,
        constants.GEO: geos,
        constants.TIME: times,
        constants.SPEND: [100.0] * n_rows,
        constants.MEDIA_UNITS: [10.0] * n_rows,
        eda_constants.COST_PER_MEDIA_UNIT: [10.0] * n_rows,
        eda_constants.ABS_COST_PER_MEDIA_UNIT: [10.0] * n_rows,
    }).set_index([constants.CHANNEL, constants.GEO, constants.TIME])

    mock_artifact = mock.create_autospec(
        eda_outcome.CostPerMediaUnitArtifact, instance=True
    )
    mock_artifact.outlier_df = outlier_df

    finding = _create_generic_finding(
        finding_cause=Cause.OUTLIER,
        severity=Severity.ATTENTION,
        associated_artifact=mock_artifact,
        explanation='CPMU Outlier Found',
    )

    self._stub_engine_checks(cpmu_findings=[finding])

    dom = self._get_output_eda_report_html_dom()

    card = dom.find(
        f".//card[@id='{eda_constants.SPEND_AND_MEDIA_UNIT_CARD_ID}']"
    )
    self.assertIsNotNone(card)

    table = card.find(
        f".//*[@id='{eda_constants.COST_PER_MEDIA_UNIT_OUTLIER_TABLE_ID}']"
    )
    self.assertIsNotNone(table)

    self.assertIn(
        'to review outliers for 6 channels in 6 times and 6 geos',
        ''.join(table.itertext()),
    )

  def test_population_scaling_card_national_is_none(self):
    self._stub_plotters()
    self._meridian.is_national = True
    self._stub_engine_checks()

    dom = self._get_output_eda_report_html_dom()
    card = dom.find(
        f".//card[@id='{eda_constants.POPULATION_SCALING_CARD_ID}']"
    )

    self.assertIsNone(
        card,
        'Population Scaling card should not be present in the report for'
        ' national models.',
    )

  def test_prior_specifications_card_national_is_none(self):
    self._stub_plotters()
    self._meridian.is_national = True
    self._stub_engine_checks()

    dom = self._get_output_eda_report_html_dom()
    card = dom.find(
        f".//card[@id='{eda_constants.PRIOR_SPECIFICATIONS_CARD_ID}']"
    )

    self.assertIsNone(
        card,
        'Prior Specifications card should not be present in the report for'
        ' national models.',
    )

  def test_spend_media_unit_card_includes_data_param_ratio_message(self):
    self._stub_plotters()
    self._stub_engine_checks()

    # Raw HTML check
    html_content = self._eda._generate_report_html()
    self.assertIn('As a rough guidance', html_content)
    self.assertIn('&#160;&#160;&#8226; ', html_content)

    dom = self._get_output_eda_report_html_dom()
    card = dom.find(
        f".//card[@id='{eda_constants.SPEND_AND_MEDIA_UNIT_CARD_ID}']"
    )
    self.assertIsNotNone(card)

    callout = card.find(f".//*[@id='{eda_constants.DATA_ADEQUACY_TABLE_ID}']")
    self.assertIsNotNone(callout)

    callout_text = ''.join(callout.itertext())
    self.assertIn('Ratio', callout_text)
    self.assertIn('n_geos', callout_text)
    self.assertIn('n_times', callout_text)
    self.assertIn('n_knots', callout_text)
    self.assertIn('n_controls', callout_text)
    self.assertIn('n_treatments', callout_text)
    self.assertIn('n_parameters', callout_text)
    self.assertIn('n_data_points', callout_text)

  def test_spend_media_unit_card_severity_aggregation(self):
    self._stub_plotters()
    finding_cpmu = _create_cpmu_finding(
        severity=eda_outcome.EDASeverity.ATTENTION,
        finding_cause=eda_outcome.FindingCause.OUTLIER,
    )
    self._stub_engine_checks(cpmu_findings=[finding_cpmu])

    dom = self._get_output_eda_report_html_dom()

    summary_card = dom.find(f".//*[@id='{eda_constants.SUMMARY_CARD_ID}']")
    self.assertIsNotNone(summary_card)

    table = summary_card.find(f".//*[@id='{eda_constants.SUMMARY_TABLE_ID}']")
    self.assertIsNotNone(table)

    spend_and_media_unit_row = None
    for tr in table.findall('.//tr'):
      if eda_constants.SPEND_AND_MEDIA_UNIT_CARD_TITLE in ''.join(
          tr.itertext()
      ):
        spend_and_media_unit_row = tr
        break

    self.assertIsNotNone(spend_and_media_unit_row)
    row_text = ''.join(spend_and_media_unit_row.itertext())
    self.assertIn('1 review(s)', row_text)

  def test_cpmu_finding_newline_replacement(self):
    self._stub_plotters()
    finding_cpmu = _create_cpmu_finding(
        severity=eda_outcome.EDASeverity.ATTENTION,
        finding_cause=eda_outcome.FindingCause.OUTLIER,
    )
    finding_cpmu_with_newline = eda_outcome.EDAFinding(
        finding_cause=finding_cpmu.finding_cause,
        severity=finding_cpmu.severity,
        associated_artifact=finding_cpmu.associated_artifact,
        explanation='Line 1\nLine 2',
    )
    self._stub_engine_checks(cpmu_findings=[finding_cpmu_with_newline])

    html_content = self._eda._generate_report_html()

    self.assertNotIn('Line 1\nLine 2', html_content)


class MeridianEdaTest(backend_test_utils.MeridianTestCase):

  # ============================================================================
  # __init__ Tests
  # ============================================================================
  def test_init_eda_engine(self):
    """Tests that the eda_engine is initialized correctly."""
    data = data_test_utils.sample_input_data_revenue(
        n_geos=1,
        n_times=_N_TIMES,
        n_media_channels=3,
        n_controls=3,
        n_non_media_channels=2,
        n_rf_channels=2,
    )
    mmm = model.Meridian(data)
    eda = meridian_eda.MeridianEDA(mmm)
    self.assertIsInstance(eda.eda_engine, sampling_eda_engine.SamplingEDAEngine)
    self.assertIs(eda.eda_engine.spec, mmm.eda_spec)

  def test_call_geo_pairwise_corr_property_on_national_model_raises_error(
      self,
  ):
    data = data_test_utils.sample_input_data_revenue(
        n_geos=1,
        n_times=_N_TIMES,
        n_media_channels=3,
        n_controls=3,
        n_non_media_channels=2,
        n_rf_channels=2,
    )
    eda = meridian_eda.MeridianEDA(model.Meridian(data))
    with self.assertRaises(eda_engine.GeoLevelCheckOnNationalModelError):
      _ = eda.geo_pairwise_correlation_check_outcome

  def test_plot_national_kpi_with_knots_time_series_single_knot(self):
    with self.assertRaisesRegex(
        ValueError, 'This feature requires more than one knot.'
    ):
      data = data_test_utils.sample_input_data_revenue(
          n_geos=1,
          n_times=_N_TIMES,
          n_media_channels=3,
          n_controls=3,
          n_non_media_channels=2,
          n_rf_channels=2,
      )
      eda = meridian_eda.MeridianEDA(model.Meridian(data))
      eda.plot_national_kpi_with_knots_time_series()

  # ============================================================================
  # Chart properties
  # ============================================================================

  @parameterized.named_parameters(
      dict(
          testcase_name='plot_kpi_boxplot',
          plotting_function='plot_kpi_boxplot',
          expected_title='Boxplots of scaled KPI',
          expected_x_title=constants.KPI,
          expected_y_title=constants.KPI_SCALED,
      ),
      dict(
          testcase_name='plot_treatments_without_non_media_boxplot',
          plotting_function='plot_treatments_without_non_media_boxplot',
          expected_title='Boxplots of paid and organic scaled impressions',
          expected_x_title=eda_constants.VARIABLE,
          expected_y_title=eda_constants.MEDIA_IMPRESSIONS_SCALED,
      ),
      dict(
          testcase_name='plot_controls_and_non_media_boxplot',
          plotting_function='plot_controls_and_non_media_boxplot',
          expected_title='Boxplots of scaled controls and non-media treatments',
          expected_x_title=eda_constants.VARIABLE,
          expected_y_title=f'{constants.CONTROLS_SCALED}/{constants.NON_MEDIA_TREATMENTS_SCALED}',
      ),
      dict(
          testcase_name='plot_relative_spend_share_barchart',
          plotting_function='plot_relative_spend_share_barchart',
          expected_title=(
              'Relative spend share of paid media channels (all channels)'
          ),
          expected_x_title=eda_constants.SPEND_SHARE,
          expected_y_title=constants.CHANNEL,
      ),
      dict(
          testcase_name='plot_relative_impression_share_barchart',
          plotting_function='plot_relative_impression_share_barchart',
          expected_title=(
              'Relative scaled impression share of paid and'
              ' organic media channels (all channels)'
          ),
          expected_x_title=eda_constants.IMPRESSION_SHARE_SCALED,
          expected_y_title=constants.CHANNEL,
      ),
  )
  def test_plot_chart_properties_standard(
      self,
      plotting_function,
      expected_title,
      expected_x_title,
      expected_y_title,
  ):
    data = data_test_utils.sample_input_data_revenue(
        n_geos=1,
        n_times=_N_TIMES,
        n_media_channels=3,
        n_controls=3,
        n_non_media_channels=2,
        n_rf_channels=2,
    )
    eda = meridian_eda.MeridianEDA(model.Meridian(data))
    plot = getattr(eda, plotting_function)()

    actual_chart = plot.vconcat[0]

    with self.subTest(name='chart_title'):
      self.assertEqual(
          actual_chart.title,
          f'{expected_title} for {constants.NATIONAL_MODEL_DEFAULT_GEO_NAME}',
      )
    with self.subTest(name='x_axis_title'):
      self.assertEqual(
          actual_chart.encoding.x.to_dict().get('title'),
          expected_x_title,
      )
    with self.subTest(name='y_axis_title'):
      self.assertEqual(
          actual_chart.encoding.y.to_dict().get('title'),
          expected_y_title,
      )

  def test_plot_chart_properties_pairwise_correlation(self):
    data = data_test_utils.sample_input_data_revenue(
        n_geos=1,
        n_times=_N_TIMES,
        n_media_channels=3,
        n_controls=3,
        n_non_media_channels=2,
        n_rf_channels=2,
    )
    eda = meridian_eda.MeridianEDA(model.Meridian(data))
    plot = eda.plot_pairwise_correlation()

    actual_chart = plot.vconcat[0].layer[0]
    expected_title = 'Pairwise correlations among all treatments and controls'

    with self.subTest(name='chart_title'):
      self.assertEqual(
          actual_chart.title.text,
          f'{expected_title} for {constants.NATIONAL_MODEL_DEFAULT_GEO_NAME}',
      )
    with self.subTest(name='x_axis_title'):
      self.assertIsNone(actual_chart.encoding.x.to_dict().get('title'))
    with self.subTest(name='y_axis_title'):
      self.assertIsNone(actual_chart.encoding.y.to_dict().get('title'))

  def test_plot_chart_properties_cost_per_media_unit(self):
    data = data_test_utils.sample_input_data_revenue(
        n_geos=1,
        n_times=_N_TIMES,
        n_media_channels=3,
        n_controls=3,
        n_non_media_channels=2,
        n_rf_channels=2,
    )
    eda = meridian_eda.MeridianEDA(model.Meridian(data))
    plot = eda.plot_cost_per_media_unit_time_series()

    channel_chart_block = plot.vconcat[0].vconcat[0]

    superimposed_chart = channel_chart_block.vconcat[0]
    expected_superimposed_title = (
        f'Time series of {constants.NATIONAL_MODEL_DEFAULT_GEO_NAME} level '
        f'{constants.SPEND} and {constants.MEDIA_UNITS} for ch_0'
    )
    with self.subTest(name='superimposed_chart_title'):
      self.assertEqual(
          superimposed_chart.layer[0].title.text, expected_superimposed_title
      )

    with self.subTest(name='layer_0_y_axis_title'):
      self.assertEqual(
          superimposed_chart.layer[0]
          .encoding.y.to_dict()
          .get('axis')
          .get('title'),
          constants.SPEND,
      )
    with self.subTest(name='layer_0_x_axis_title'):
      self.assertEqual(
          superimposed_chart.layer[0]
          .encoding.x.to_dict()
          .get('axis')
          .get('title'),
          constants.TIME,
      )

    with self.subTest(name='layer_1_y_axis_title'):
      self.assertEqual(
          superimposed_chart.layer[1]
          .encoding.y.to_dict()
          .get('axis')
          .get('title'),
          constants.MEDIA_UNITS,
      )
    with self.subTest(name='layer_1_x_axis_title'):
      self.assertEqual(
          superimposed_chart.layer[1]
          .encoding.x.to_dict()
          .get('axis')
          .get('title'),
          constants.TIME,
      )

    cpmu_chart = channel_chart_block.vconcat[1]
    expected_cpmu_title = (
        f'Time series of {constants.NATIONAL_MODEL_DEFAULT_GEO_NAME} level '
        f'{eda_constants.COST_PER_MEDIA_UNIT} for ch_0'
    )

    with self.subTest(name='cpmu_chart_title'):
      self.assertEqual(cpmu_chart.title.text, expected_cpmu_title)
    with self.subTest(name='cpmu_chart_y_axis_title'):
      self.assertEqual(
          cpmu_chart.encoding.y.to_dict().get('axis').get('title'),
          eda_constants.COST_PER_MEDIA_UNIT,
      )
    with self.subTest(name='cpmu_chart_x_axis_title'):
      self.assertEqual(
          cpmu_chart.encoding.x.to_dict().get('axis').get('title'),
          constants.TIME,
      )

  def test_plot_national_kpi_with_knots_time_series_properties(self):
    data = data_test_utils.sample_input_data_revenue(
        n_geos=2,
        n_times=20,
        n_media_channels=3,
        n_controls=3,
        n_non_media_channels=2,
        n_rf_channels=2,
    )
    plot = meridian_eda.MeridianEDA(
        model.Meridian(data)
    ).plot_national_kpi_with_knots_time_series()

    with self.subTest(name='chart_title'):
      self.assertEqual(
          'Time Series of scaled national KPI with Knots',
          plot.layer[0].title.text,
      )
    with self.subTest(name='x_axis_title'):
      self.assertEqual(
          'time', plot.layer[0].encoding.x.to_dict().get('axis').get('title')
      )
    with self.subTest(name='y_axis_title'):
      self.assertEqual(
          'national kpi_scaled',
          plot.layer[0].encoding.y.to_dict().get('axis').get('title'),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='plot_population_raw_media_correlation',
          plotting_function='plot_population_raw_media_correlation',
          expected_title=(
              'Correlation between population and raw paid and organic media'
              ' variables'
          ),
          expected_x_title='channel',
          expected_y_title='correlation',
      ),
      dict(
          testcase_name='plot_population_treatment_correlation',
          plotting_function='plot_population_treatment_correlation',
          expected_title=(
              'Correlation between population and scaled treatment/scaled'
              ' controls'
          ),
          expected_x_title='channel',
          expected_y_title='correlation',
      ),
      dict(
          testcase_name='prior_mean',
          plotting_function='plot_prior_mean',
          expected_title='Prior mean of contribution by channel',
          expected_x_title='channel',
          expected_y_title='prior_contribution',
      ),
  )
  def test_plot_geo_only_plots_properties(
      self,
      plotting_function,
      expected_title,
      expected_x_title,
      expected_y_title,
  ):
    data = data_test_utils.sample_input_data_revenue(
        n_geos=2,
        n_times=20,
        n_media_channels=3,
        n_controls=3,
        n_non_media_channels=2,
        n_rf_channels=2,
    )
    eda = meridian_eda.MeridianEDA(model.Meridian(data))
    plot = getattr(eda, plotting_function)()
    actual_chart = plot.vconcat[0]

    with self.subTest(name='chart_title'):
      self.assertIn(expected_title, actual_chart.title)
    with self.subTest(name='x_axis_title'):
      self.assertEqual(
          expected_x_title, actual_chart.encoding.x.to_dict().get('title')
      )
    with self.subTest(name='y_axis_title'):
      self.assertEqual(
          expected_y_title, actual_chart.encoding.y.to_dict().get('title')
      )


if __name__ == '__main__':
  absltest.main()
