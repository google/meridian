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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian.backend import test_utils
from meridian.model import context
from meridian.model import equations


class ModelEquationsTest(test_utils.MeridianTestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_context = mock.create_autospec(
        context.ModelContext,
        instance=True,
    )
    self.equations = equations.ModelEquations(model_context=self.mock_context)

  def test_adstock_hill_media(self):
    pass

  def test_adstock_hill_rf(self):
    pass

  def test_compute_non_media_treatments_baseline(self):
    pass

  def test_linear_predictor_counterfactual_difference_media(self):
    pass

  def test_linear_predictor_counterfactual_difference_rf(self):
    pass

  def test_calculate_beta_x(self):
    pass


if __name__ == "__main__":
  absltest.main()
