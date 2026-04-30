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

"""Meridian Sampling EDA Engine."""

from typing import TYPE_CHECKING

from meridian import constants
from meridian.model.eda import constants as eda_constants
from meridian.model.eda import eda_engine
from meridian.model.eda import eda_outcome
from meridian.model.eda import eda_spec as eda_spec_module
import numpy as np
import xarray as xr

if TYPE_CHECKING:
  from meridian.analysis import analyzer as analyzer_module  # pylint: disable=g-bad-import-order,g-import-not-at-top


# TODO: Remove this class once EDAEngine can use Analyzer without
# circular dependencies.
class SamplingEDAEngine(eda_engine.EDAEngine):
  """EDA engine for sampling-based checks."""

  def __init__(
      self,
      analyzer: "analyzer_module.Analyzer",
      spec: eda_spec_module.EDASpec | None = None,
  ):
    """Initializes the instance.

    Args:
      analyzer: The Analyzer instance to use for sampling-based checks. It must
        contain prior samples in its inference_data.
      spec: The EDASpec for configuration.

    Raises:
      ValueError: If the analyzer instance does not have 'prior' in its
        inference_data.
    """

    if spec is None:
      spec = eda_spec_module.EDASpec()

    super().__init__(model_context=analyzer.model_context, spec=spec)
    if constants.PRIOR not in analyzer.inference_data.groups():
      raise ValueError(
          f"Analyzer instance must have '{constants.PRIOR}' in its"
          " inference_data."
      )
    self._analyzer = analyzer

  def check_prior_probability(
      self,
  ) -> eda_outcome.EDAOutcome[eda_outcome.PriorProbabilityArtifact]:
    """Checks prior probabilities of negative baseline and contributions.

    Returns:
      An EDAOutcome object with findings and result values.
    """
    prior_negative_baseline_prob = self._analyzer.negative_baseline_probability(
        use_posterior=False
    )

    outcome = (
        self._input_data.kpi * self._input_data.revenue_per_kpi
        if self._input_data.revenue_per_kpi is not None
        else self._input_data.kpi
    )
    total_outcome = outcome.sum()

    n_channels = len(self._model_context.input_data.get_all_channels())
    # Shape = (n_chains, n_draws, n_channels)
    incremental_outcome = self._analyzer.incremental_outcome(
        use_posterior=False
    )

    if total_outcome == 0:
      # If total_outcome is zero, division would result in inf.
      # We set mean_prior_contribution to np.inf to indicate this.
      mean_prior_contribution = np.full(n_channels, np.inf)
    else:
      prior_contribution_samples = (
          np.array(incremental_outcome) / total_outcome.values
      )
      # Shape = (n_channels,)
      mean_prior_contribution = np.mean(prior_contribution_samples, axis=(0, 1))

    mean_prior_contribution_da = xr.DataArray(
        mean_prior_contribution,
        coords={
            constants.CHANNEL: self._model_context.input_data.get_all_channels()
        },
        dims=[constants.CHANNEL],
    )

    artifact = eda_outcome.PriorProbabilityArtifact(
        level=eda_outcome.AnalysisLevel.OVERALL,
        prior_negative_baseline_prob=float(prior_negative_baseline_prob),
        mean_prior_contribution_da=mean_prior_contribution_da,
    )

    findings = [
        eda_outcome.EDAFinding(
            severity=eda_outcome.EDASeverity.INFO,
            explanation=eda_constants.PRIOR_PROBABILITY_INFO,
            finding_cause=eda_outcome.FindingCause.NONE,
            associated_artifact=artifact,
        )
    ]

    return eda_outcome.EDAOutcome(
        check_type=eda_outcome.EDACheckType.PRIOR_PROBABILITY,
        findings=findings,
        analysis_artifacts=[artifact],
    )
