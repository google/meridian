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


if __name__ == "__main__":
  absltest.main()
