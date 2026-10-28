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

from absl.testing import absltest
from meridian.model import progress


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
    )
    self.assertEqual(snapshot.n_draws, 18)

  def test_reconstruction_fields_default_to_zero(self):
    snapshot = progress.PosteriorSamplingProgress(
        stage=progress.PosteriorSamplingStage.ADAPTATION,
        chain_batch=0,
        n_chain_batches=1,
        draw=0,
        n_adapt=5,
        n_burnin=3,
        n_keep=10,
        n_divergent=0,
    )
    self.assertEqual(snapshot.reconstruction_batch, 0)
    self.assertEqual(snapshot.n_reconstruction_batches, 0)


if __name__ == "__main__":
  absltest.main()
