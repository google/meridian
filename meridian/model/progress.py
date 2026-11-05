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

To display progress bars without writing a callback, pass a `TqdmProgressBar`:

```python
progress_bar = TqdmProgressBar()
mmm.sample_prior(500, progress_callback=progress_bar)
mmm.sample_posterior(
    n_chains=4,
    n_adapt=500,
    n_burnin=500,
    n_keep=1000,
    progress_callback=progress_bar,
)
```
"""

from collections.abc import Callable
import dataclasses
import enum
from typing import Generic, TypeVar
import warnings

from tqdm import auto as tqdm_auto

__all__ = [
    "PosteriorProgressCallback",
    "PosteriorSamplingProgress",
    "PosteriorSamplingStage",
    "PriorProgressCallback",
    "PriorSamplingProgress",
    "PriorSamplingStage",
    "TqdmProgressBar",
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
    n_reconstruction_batches: Total number of reconstruction batches. Zero when
      there are no kept draws to reconstruct.
  """

  stage: PosteriorSamplingStage
  chain_batch: int
  n_chain_batches: int
  draw: int
  n_adapt: int
  n_burnin: int
  n_keep: int
  n_divergent: int
  reconstruction_batch: int
  n_reconstruction_batches: int

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


class TqdmProgressBar:
  """A progress callback that displays sampling progress as tqdm progress bars.

  Pass an instance as the `progress_callback` of `Meridian.sample_prior`,
  `Meridian.sample_posterior` or `Meridian.sample_posterior_and_review`. It
  displays a progress bar for each of the following:

  *   Prior sampling, which advances as each prior sampling stage completes.
  *   Each posterior chain batch, which advances after every draw on the JAX
      backend and once per chain batch on the TensorFlow backend. On the JAX
      backend, the first chain batch stays at zero draws while the sampler
      compiles.
  *   The posterior reconstruction, which advances after each reconstruction
      batch.

  Each bar shows the current stage, and the posterior bars show the number of
  divergent transitions among the kept draws so far. The bars are rendered with
  `tqdm.auto`: as widgets in notebooks and as text in a terminal.

  One instance can be passed to several sampling calls, one call at a time.
  """

  def __init__(self):
    self._bar: tqdm_auto.tqdm | None = None
    self._bar_key: tuple[str, int] | None = None

  def __call__(
      self, snapshot: PriorSamplingProgress | PosteriorSamplingProgress
  ) -> None:
    """Updates the progress bars with a progress snapshot."""
    if isinstance(snapshot, PriorSamplingProgress):
      self._update(
          key=("prior", 0),
          desc="Prior sampling",
          unit="stage",
          n=snapshot.stages_completed,
          total=snapshot.n_stages,
          postfix={"stage": snapshot.stage.value},
      )
    elif snapshot.stage is PosteriorSamplingStage.RECONSTRUCTION:
      self._update(
          key=("reconstruction", 0),
          desc="Posterior reconstruction",
          unit="batch",
          n=snapshot.reconstruction_batch,
          total=snapshot.n_reconstruction_batches,
          postfix={"divergent": snapshot.n_divergent},
      )
    else:
      self._update(
          key=("chain_batch", snapshot.chain_batch),
          desc=(
              f"Posterior chain batch {snapshot.chain_batch + 1}/"
              f"{snapshot.n_chain_batches}"
          ),
          unit="draw",
          n=snapshot.draw,
          total=snapshot.n_draws,
          postfix={
              "stage": snapshot.stage.value,
              "divergent": snapshot.n_divergent,
          },
      )

  def _update(
      self,
      *,
      key: tuple[str, int],
      desc: str,
      unit: str,
      n: int,
      total: int,
      postfix: dict[str, str | int],
  ) -> None:
    """Moves the bar identified by `key` to `n` of `total`.

    Starts a new bar when `key` changes, or when progress goes backwards, which
    means that a new sampling call has started after an interrupted one. Closes
    the bar once it is complete.

    Args:
      key: Identifies the sampling phase the bar tracks.
      desc: Description shown before the bar.
      unit: Unit of the counted work.
      n: Units of work completed so far.
      total: Total units of work.
      postfix: Details shown after the bar.
    """
    if self._bar is None or key != self._bar_key or n <= self._bar.n:
      self._close()
      self._bar = tqdm_auto.tqdm(desc=desc, unit=unit, total=total)
      self._bar_key = key
    self._bar.n = n
    self._bar.set_postfix(postfix)  # Also refreshes the bar.
    if n >= total:
      self._close()

  def _close(self) -> None:
    if self._bar is not None:
      self._bar.close()
    self._bar = None
    self._bar_key = None


_ProgressT = TypeVar(
    "_ProgressT", PosteriorSamplingProgress, PriorSamplingProgress
)


class ProgressReporter(Generic[_ProgressT]):
  """Base class of the sampling progress reporters. Not part of the public API.

  It owns the progress callback and reports snapshots to it through `_emit`,
  which is a no-op without a callback, skips a snapshot identical to the last
  one reported, and isolates sampling from failures of the callback.
  """

  def __init__(self, callback: Callable[[_ProgressT], None] | None):
    self._callback = callback
    self._last_snapshot: _ProgressT | None = None
    self._callback_failed = False

  @property
  def enabled(self) -> bool:
    """Whether there is a progress callback to report to."""
    return self._callback is not None

  def _emit(self, snapshot: _ProgressT) -> None:
    """Calls the callback with `snapshot`, unless it equals the last one."""
    callback = self._callback
    if callback is None or snapshot == self._last_snapshot:
      return
    self._last_snapshot = snapshot
    try:
      callback(snapshot)
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Progress reporting must never abort a sampling run that may take hours.
      # On the JAX backend, an exception would moreover escape through a host
      # callback and surface as an opaque XLA error. Report the first failure
      # and keep calling the callback, since the failure may be transient.
      if self._callback_failed:
        return
      self._callback_failed = True
      warnings.warn(
          f"`progress_callback` raised {e!r}; sampling continues and"
          " further failures of the callback are not reported.",
          RuntimeWarning,
      )
