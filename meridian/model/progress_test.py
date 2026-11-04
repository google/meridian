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

from unittest import mock
import warnings

from absl.testing import absltest
from meridian.model import progress
from tqdm import auto as tqdm_auto


def _prior_snapshot(stages_completed: int) -> progress.PriorSamplingProgress:
  return progress.PriorSamplingProgress(
      stage=progress.PriorSamplingStage.BASE,
      stages_completed=stages_completed,
      n_stages=3,
  )


def _no_op_callback(snapshot: progress.PriorSamplingProgress) -> None:
  del snapshot


class PosteriorSamplingProgressTest(absltest.TestCase):

  def test_n_draws_counts_all_stages(self):
    snapshot = progress.PosteriorSamplingProgress(
        stage=progress.PosteriorSamplingStage.BURNIN,
        chain_batch=0,
        n_chain_batches=1,
        draw=7,
        n_adapt=5,
        n_burnin=3,
        n_keep=10,
        n_divergent=0,
        reconstruction_batch=0,
        n_reconstruction_batches=4,
    )
    self.assertEqual(snapshot.n_draws, 18)


class ProgressReporterTest(absltest.TestCase):

  def test_without_callback_is_disabled(self):
    reporter = progress.ProgressReporter(None)

    self.assertFalse(reporter.enabled)
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      reporter._emit(_prior_snapshot(1))

  def test_emit_calls_callback_with_snapshot(self):
    callback = mock.create_autospec(_no_op_callback)
    reporter = progress.ProgressReporter(callback)

    self.assertTrue(reporter.enabled)
    reporter._emit(_prior_snapshot(1))
    reporter._emit(_prior_snapshot(2))

    self.assertEqual(
        callback.call_args_list,
        [mock.call(_prior_snapshot(1)), mock.call(_prior_snapshot(2))],
    )

  def test_emit_skips_snapshot_equal_to_last_one(self):
    callback = mock.create_autospec(_no_op_callback)
    reporter = progress.ProgressReporter(callback)

    reporter._emit(_prior_snapshot(1))
    reporter._emit(_prior_snapshot(1))

    callback.assert_called_once_with(_prior_snapshot(1))

  def test_emit_warns_once_and_keeps_calling_failing_callback(self):
    callback = mock.create_autospec(
        _no_op_callback, side_effect=ValueError("sink unavailable")
    )
    reporter = progress.ProgressReporter(callback)

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      reporter._emit(_prior_snapshot(1))
      reporter._emit(_prior_snapshot(2))

    self.assertLen(caught, 1)
    self.assertIs(caught[0].category, RuntimeWarning)
    self.assertIn("sink unavailable", str(caught[0].message))
    self.assertEqual(callback.call_count, 2)


def _posterior_snapshot(
    *,
    stage: progress.PosteriorSamplingStage = (
        progress.PosteriorSamplingStage.ADAPTATION
    ),
    chain_batch: int = 0,
    draw: int = 0,
    n_divergent: int = 0,
    reconstruction_batch: int = 0,
) -> progress.PosteriorSamplingProgress:
  return progress.PosteriorSamplingProgress(
      stage=stage,
      chain_batch=chain_batch,
      n_chain_batches=2,
      draw=draw,
      n_adapt=2,
      n_burnin=1,
      n_keep=2,
      n_divergent=n_divergent,
      reconstruction_batch=reconstruction_batch,
      n_reconstruction_batches=3,
  )


class TqdmProgressBarTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.bars = []
    tqdm_class = tqdm_auto.tqdm

    def new_bar(**kwargs):
      del kwargs
      bar = mock.create_autospec(tqdm_class, instance=True)
      bar.n = 0
      self.bars.append(bar)
      return bar

    self.mock_tqdm = self.enter_context(
        mock.patch.object(tqdm_auto, "tqdm", autospec=True, side_effect=new_bar)
    )

  def test_prior_shows_one_bar_over_stages(self):
    progress_bar = progress.TqdmProgressBar()

    progress_bar(_prior_snapshot(1))
    progress_bar(_prior_snapshot(2))

    self.mock_tqdm.assert_called_once_with(
        desc="Prior sampling", unit="stage", total=3
    )
    self.assertEqual(self.bars[0].n, 2)
    self.bars[0].set_postfix.assert_called_with({"stage": "base"})
    self.bars[0].close.assert_not_called()

  def test_prior_closes_bar_after_last_stage(self):
    progress_bar = progress.TqdmProgressBar()

    progress_bar(_prior_snapshot(3))

    self.bars[0].close.assert_called_once()

  def test_posterior_shows_one_bar_per_chain_batch(self):
    progress_bar = progress.TqdmProgressBar()

    progress_bar(_posterior_snapshot(chain_batch=0, draw=0))
    progress_bar(_posterior_snapshot(chain_batch=0, draw=1))
    progress_bar(_posterior_snapshot(chain_batch=1, draw=0))

    self.assertEqual(
        self.mock_tqdm.call_args_list,
        [
            mock.call(desc="Posterior chain batch 1/2", unit="draw", total=5),
            mock.call(desc="Posterior chain batch 2/2", unit="draw", total=5),
        ],
    )
    self.assertEqual(self.bars[0].n, 1)
    self.bars[0].close.assert_called_once()
    self.bars[1].close.assert_not_called()

  def test_posterior_shows_stage_and_divergences(self):
    progress_bar = progress.TqdmProgressBar()

    progress_bar(
        _posterior_snapshot(
            stage=progress.PosteriorSamplingStage.KEEP, draw=4, n_divergent=7
        )
    )

    self.bars[0].set_postfix.assert_called_once_with(
        {"stage": "keep", "divergent": 7}
    )

  def test_posterior_closes_chain_batch_bar_after_last_draw(self):
    progress_bar = progress.TqdmProgressBar()

    progress_bar(_posterior_snapshot(draw=0))
    progress_bar(
        _posterior_snapshot(stage=progress.PosteriorSamplingStage.KEEP, draw=5)
    )

    self.mock_tqdm.assert_called_once()
    self.assertEqual(self.bars[0].n, 5)
    self.bars[0].close.assert_called_once()

  def test_posterior_shows_reconstruction_bar(self):
    progress_bar = progress.TqdmProgressBar()
    stage = progress.PosteriorSamplingStage.RECONSTRUCTION

    progress_bar(
        _posterior_snapshot(stage=stage, draw=5, reconstruction_batch=1)
    )
    progress_bar(
        _posterior_snapshot(stage=stage, draw=5, reconstruction_batch=3)
    )

    self.mock_tqdm.assert_called_once_with(
        desc="Posterior reconstruction", unit="batch", total=3
    )
    self.assertEqual(self.bars[0].n, 3)
    self.bars[0].set_postfix.assert_called_with({"divergent": 0})
    self.bars[0].close.assert_called_once()

  def test_restarted_sampling_starts_new_bar(self):
    progress_bar = progress.TqdmProgressBar()

    progress_bar(_posterior_snapshot(draw=3))
    progress_bar(_posterior_snapshot(draw=0))

    self.assertLen(self.bars, 2)
    self.bars[0].close.assert_called_once()
    self.assertEqual(self.bars[1].n, 0)


class TqdmProgressBarRenderingTest(absltest.TestCase):

  def test_renders_prior_and_posterior_progress_without_errors(self):
    progress_bar = progress.TqdmProgressBar()

    with warnings.catch_warnings():
      warnings.simplefilter("error")
      for stages_completed in range(1, 4):
        progress_bar(_prior_snapshot(stages_completed))
      for draw in range(6):
        progress_bar(_posterior_snapshot(draw=draw))
      progress_bar(
          _posterior_snapshot(
              stage=progress.PosteriorSamplingStage.RECONSTRUCTION,
              draw=5,
              reconstruction_batch=3,
          )
      )


if __name__ == "__main__":
  absltest.main()
