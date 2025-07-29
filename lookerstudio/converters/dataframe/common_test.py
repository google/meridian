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

from absl.testing import absltest
from absl.testing import parameterized
from lookerstudio.converters.dataframe import common


class CommonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="name_with_underscore",
          prefix="prefix",
          name="name_with_underscore",
          expected="prefix_name_with_underscore",
      ),
      dict(
          testcase_name="name_with_multiple_spaces",
          prefix="prefix",
          name="name    with    multiple    spaces",
          expected="prefix_name_with_multiple_spaces",
      ),
      dict(
          testcase_name="prefix_with_underscore",
          prefix="prefix_with_underscore",
          name="name",
          expected="prefix_with_underscore_name",
      ),
      dict(
          testcase_name="prefix_with_multiple_spaces",
          prefix="prefix   with   multiple   spaces",
          name="name",
          expected="prefix_with_multiple_spaces_name",
      ),
  )
  def test_create_grid_sheet_name(self, prefix, name, expected):
    got = common.create_grid_sheet_name(prefix, name)
    self.assertEqual(expected, got)


if __name__ == "__main__":
  absltest.main()
