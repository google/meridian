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

"""Progress reporting for prior and posterior sampling.

`Meridian.sample_prior` and `Meridian.sample_posterior` accept an optional
`progress_callback`: a callable that receives a `PriorSamplingProgress` or a
`PosteriorSamplingProgress` snapshot each time sampling advances. Use it to
drive a progress bar, to estimate the remaining time, or to report the state of
a long-running fit to another system. For example:

```python
def print_progress(progress: PosteriorSamplingProgress) -> None:
  print(
      f"{progress.stage.value}: draw {progress.draw}/{progress.n_draws} of"
      f" chain batch {progress.chain_batch + 1}/{progress.n_chain_batches}"
  )


mmm.sample_posterior(
    n_chains=4,
    n_adapt=500,
    n_burnin=500,
    n_keep=1000,
    progress_callback=print_progress,
)
```
"""

from collections.abc import Callable
import dataclasses
import enum


__all__ = [
    "PosteriorProgressCallback",
    "PosteriorSamplingProgress",
    "PosteriorSamplingStage",
    "PriorProgressCallback",
    "PriorSamplingProgress",
    "PriorSamplingStage",
]


@enum.unique
class PosteriorSamplingStage(enum.Enum):
  """Stages of `Meridian.sample_posterior`, in the order they run."""

  # NUTS adaptation (warm-up) draws, `n_adapt` per chain.
  ADAPTATION = "adaptation"
  # Burn-in draws, `n_burnin` per chain.
  BURNIN = "burnin"
  # Kept draws, `n_keep` per chain.
  KEEP = "keep"
  # Computation of the deterministic parameters from the kept draws.
  RECONSTRUCTION = "reconstruction"


@dataclasses.dataclass(frozen=True, kw_only=True)
class PosteriorSamplingProgress:
  """A snapshot of the progress of `Meridian.sample_posterior`.

  MCMC runs once per chain batch (see the `n_chains` argument of
  `sample_posterior`). Within a chain batch, every chain goes through `n_adapt`
  adaptation draws, `n_burnin` burn-in draws and `n_keep` kept draws. Once all
  chain batches are sampled, the deterministic parameters are reconstructed
  from the kept draws in batches.

  Attributes:
    stage: The stage sampling is in: the stage of the most recently completed
      draw or, before any draw of a chain batch has completed, the first stage.
    chain_batch: Zero-based index of the chain batch being sampled.
    n_chain_batches: Total number of chain batches.
    draw: Number of draws (per chain) completed so far in the current chain
      batch, counting adaptation, burn-in and kept draws. Ranges from 0 to
      `n_draws`. On the JAX backend, XLA compilation of the first chain batch
      runs between `draw == 0` and `draw == 1`, so time-remaining estimates
      should measure the draw rate starting from `draw == 1`.
    n_adapt: Number of adaptation draws per chain.
    n_burnin: Number of burn-in draws per chain.
    n_keep: Number of kept draws per chain.
    n_divergent: Number of divergent transitions among the kept draws completed
      so far, summed over all chains and chain batches. Once sampling is
      complete, it equals `inference_data.sample_stats.diverging.sum()`.
    reconstruction_batch: Number of reconstruction batches completed so far.
      Zero until reconstruction starts.
    n_reconstruction_batches: Total number of reconstruction batches. Zero until
      reconstruction starts.
  """

  stage: PosteriorSamplingStage
  chain_batch: int
  n_chain_batches: int
  draw: int
  n_adapt: int
  n_burnin: int
  n_keep: int
  n_divergent: int
  reconstruction_batch: int = 0
  n_reconstruction_batches: int = 0

  @property
  def n_draws(self) -> int:
    """Total number of draws per chain in a chain batch, all stages included."""
    return self.n_adapt + self.n_burnin + self.n_keep


@enum.unique
class PriorSamplingStage(enum.Enum):
  """Stages of `Meridian.sample_prior`, in the order they run.

  Only the stages whose variables are present in the model run.
  """

  # Baseline, controls, knot values and other non-treatment parameters. In
  # models that constrain the aggregate baseline, this stage also samples the
  # non-media treatment parameters.
  BASE = "base"
  MEDIA = "media"
  RF = "rf"
  ORGANIC_MEDIA = "organic_media"
  ORGANIC_RF = "organic_rf"
  NON_MEDIA_TREATMENTS = "non_media_treatments"


@dataclasses.dataclass(frozen=True, kw_only=True)
class PriorSamplingProgress:
  """A snapshot of the progress of `Meridian.sample_prior`.

  A snapshot is reported each time a stage completes.

  Attributes:
    stage: The stage that has just completed.
    stages_completed: Number of stages completed so far, including `stage`.
    n_stages: Total number of stages that run for the model.
  """

  stage: PriorSamplingStage
  stages_completed: int
  n_stages: int


PosteriorProgressCallback = Callable[[PosteriorSamplingProgress], None]
PriorProgressCallback = Callable[[PriorSamplingProgress], None]
