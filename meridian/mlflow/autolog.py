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

"""MLflow autologging integration for Meridian.

This module enables MLflow tracking for Meridian. When enabled via `autolog()`,
parameters, metrics, and other metadata will be automatically logged to MLflow,
allowing for improved experiment tracking and analysis.

To enable MLflow autologging for your Meridian workflows, simply call
`autolog.autolog()` once before your model run.

Example usage:

```python
import mlflow
from meridian.data import load
from meridian.mlflow import autolog
from meridian.model import model

# Enable autologging (call this once per session)
autolog.autolog(log_metrics=True)

# Start an MLflow run (optionally name it for better grouping)
with mlflow.start_run(run_name="my_run"):
  # Load data
  data = load.CsvDataLoader(...).load()

  # Initialize Meridian model
  mmm = model.Meridian(input_data=data)

  # Run Meridian sampling processes
  mmm.sample_prior(n_draws=100, seed=123)
  mmm.sample_posterior(n_chains=7, n_adapt=500, n_burnin=500, n_keep=1000,
  seed=1)

# After the run completes, you can retrieve run results using the MLflow client.
client = mlflow.tracking.MlflowClient()

# Get the experiment ID for the run you just launched
experiment_id = "0"

# Search for runs matching the run name
runs = client.search_runs(
    experiment_id,
    max_results=1000,
    filter_string=f"attributes.run_name = 'my_run'"
)

# Print details of the run
if runs:
  print(runs[0])
else:
  print("No runs found.")
```
"""

import dataclasses
import inspect
import json
from typing import Any, Callable

import arviz as az
from meridian import backend
from meridian.analysis import visualizer
import mlflow
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from meridian.model import model
from meridian.model import posterior_sampler
from meridian.model import prior_sampler
from meridian.model import spec
from meridian.version import __version__
import numpy as np


FLAVOR_NAME = "meridian"


__all__ = ["autolog"]


def _log_versions() -> None:
  """Logs Meridian and ArviZ versions."""
  mlflow.log_param("meridian_version", __version__)
  mlflow.log_param("arviz_version", az.__version__)


def _log_model_spec(model_spec: spec.ModelSpec) -> None:
  """Logs the `ModelSpec` object."""
  # TODO: Replace with serde api when it's available.
  # PriorDistribution is logged separately.
  excluded_fields = ["prior"]

  for field in dataclasses.fields(model_spec):
    if field.name in excluded_fields:
      continue

    field_value = getattr(model_spec, field.name)

    # Stringify numpy arrays before logging.
    if isinstance(field_value, np.ndarray):
      field_value = json.dumps(field_value.tolist())

    mlflow.log_param(f"spec.{field.name}", field_value)


def _log_priors(model_spec: spec.ModelSpec) -> None:
  """Logs the `PriorDistribution` object."""
  # TODO: Replace with serde api when it's available.
  priors = model_spec.prior
  for field in dataclasses.fields(priors):
    field_value = getattr(priors, field.name)

    # Stringify Distributions and numpy arrays.
    if isinstance(field_value, backend.tfd.Distribution):
      field_value = str(field_value)
    elif isinstance(field_value, np.ndarray):
      field_value = json.dumps(field_value.tolist())

    mlflow.log_param(f"prior.{field.name}", field_value)


@autologging_integration(FLAVOR_NAME)
def autolog(
    disable: bool = False,  # pylint: disable=unused-argument
    silent: bool = False,  # pylint: disable=unused-argument
    log_metrics: bool = False,
) -> None:
  """Enables MLflow tracking for Meridian.

  See https://mlflow.org/docs/latest/tracking/

  Args:
    disable: Whether to disable autologging.
    silent: Whether to suppress all event logs and warnings from MLflow.
    log_metrics: Whether model metrics should be logged. Enabling this option
      involves the creation of post-modeling objects to compute relevant
      performance metrics. Metrics include R-Squared, MAPE, and wMAPE values.
  """

  def patch_meridian_init(
      original: Callable[..., Any], self, *args, **kwargs
  ) -> model.Meridian:
    _log_versions()
    mmm = original(self, *args, **kwargs)
    _log_model_spec(self.model_spec)
    _log_priors(self.model_spec)
    return mmm

  def patch_prior_sampling(original: Callable[..., Any], self, *args, **kwargs):
    mlflow.log_param("sample_prior.n_draws", kwargs.get("n_draws", "default"))
    mlflow.log_param("sample_prior.seed", kwargs.get("seed", "default"))
    return original(self, *args, **kwargs)

  def patch_posterior_sampling(
      original: Callable[..., Any], self, *args, **kwargs
  ):
    excluded_fields = ["current_state", "pins"]
    params = [
        name
        for name, value in inspect.signature(original).parameters.items()
        if name != "self"
        and value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        and name not in excluded_fields
    ]

    for param in params:
      mlflow.log_param(
          f"sample_posterior.{param}", kwargs.get(param, "default")
      )

    original(self, *args, **kwargs)
    if log_metrics:
      model_diagnostics = visualizer.ModelDiagnostics(self.model)
      df_diag = model_diagnostics.predictive_accuracy_table()

      get_metric = lambda n: df_diag[df_diag.metric == n].value.to_list()[0]

      mlflow.log_metric("R_Squared", get_metric("R_Squared"))
      mlflow.log_metric("MAPE", get_metric("MAPE"))
      mlflow.log_metric("wMAPE", get_metric("wMAPE"))

  safe_patch(FLAVOR_NAME, model.Meridian, "__init__", patch_meridian_init)
  safe_patch(
      FLAVOR_NAME,
      prior_sampler.PriorDistributionSampler,
      "__call__",
      patch_prior_sampling,
  )
  safe_patch(
      FLAVOR_NAME,
      posterior_sampler.PosteriorMCMCSampler,
      "__call__",
      patch_posterior_sampling,
  )
