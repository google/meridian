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

"""TODO: Implement."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian.model.eda import meridian_eda


class MeridianEdaTest(parameterized.TestCase):

  def test_give_me_a_name(self):
    # TODO: add real test cases
    self.assertIsNotNone(meridian_eda.MeridianEDA(mock.MagicMock()))


if __name__ == "__main__":
  absltest.main()
