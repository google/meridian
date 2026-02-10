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
from absl.testing import parameterized
from meridian.data import arg_builder


class ArgBuilderTest(parameterized.TestCase):

  def test_ordered_list_argument_builder(self):
    builder = arg_builder.OrderedListArgumentBuilder(
        ["channel_1", "channel_2", "channel_3"]
    )

    self.assertEqual(
        builder(channel_1=1, channel_2=2, channel_3=3),
        [1, 2, 3],
    )

    self.assertEqual(
        builder(channel_2=2, channel_3=3, channel_1=1),
        [1, 2, 3],
    )

    self.assertEqual(
        builder(**{
            "channel_3": 3,
            "channel_1": 1,
            "channel_2": 2,
        }),
        [1, 2, 3],
    )

  def test_ordered_list_argument_builder_missing_argument_without_default(self):
    builder = arg_builder.OrderedListArgumentBuilder(
        ["channel_1", "channel_2", "channel_3"]
    )

    with self.assertRaisesRegex(
        ValueError,
        "All coordinates must be present in the given keyword arguments",
    ):
      builder(channel_1=1, channel_2=2)

  def test_ordered_list_argument_builder_missing_argument_with_default(self):
    builder = arg_builder.OrderedListArgumentBuilder(
        ["channel_1", "channel_2", "channel_3"]
    ).with_default_value(0.3)

    self.assertEqual(
        builder(channel_1=1, channel_2=2),
        [1, 2, 0.3],
    )


if __name__ == "__main__":
  absltest.main()
