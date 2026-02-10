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
import bidict
from meridian import constants as c
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from schema.utils import proto_enum_converter


_MediaEffectsDist = meridian_pb.MediaEffectsDistribution


class ProtoEnumConverterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.converter = proto_enum_converter.ProtoEnumConverter(
        enum_display_name="Media effects distribution",
        enum_message=_MediaEffectsDist,
        mapping=bidict.bidict({
            c.MEDIA_EFFECTS_LOG_NORMAL: "LOG_NORMAL",
            "fake": "FAKE",
        }),
        enum_unspecified=_MediaEffectsDist.MEDIA_EFFECTS_DISTRIBUTION_UNSPECIFIED,
        default_when_unspecified=c.MEDIA_EFFECTS_LOG_NORMAL,
    )

  def test_to_proto_returns_correct_proto_enum(self):
    string_value = self.converter.to_proto(c.MEDIA_EFFECTS_LOG_NORMAL)
    self.assertEqual(string_value, _MediaEffectsDist.LOG_NORMAL)

  def test_to_proto_raises_error_for_unmatched_value(self):
    unmatched_string_value = c.MEDIA_EFFECTS_NORMAL
    with self.assertRaisesRegex(
        ValueError,
        f"Unmatched {self.converter.enum_message.DESCRIPTOR.name} value:"
        f" {unmatched_string_value}.",
    ):
      self.converter.to_proto(unmatched_string_value)

  def test_to_proto_missing_enum_warns(self):
    with self.assertWarnsRegex(
        Warning,
        f"Invalid {self.converter.enum_message.DESCRIPTOR.name} value: fake."
        " Resolving to"
        f" {self.converter.enum_message.Name(self.converter.enum_unspecified)}",
    ):
      self.assertEqual(
          self.converter.to_proto("fake"), self.converter.enum_unspecified
      )

  def test_from_proto_returns_correct_string_value(self):
    proto_name = self.converter.from_proto(_MediaEffectsDist.LOG_NORMAL)
    self.assertEqual(proto_name, c.MEDIA_EFFECTS_LOG_NORMAL)

  def test_from_proto_returns_default_when_unspecified_and_warns(self):
    with self.assertWarnsRegex(
        Warning,
        f"{self.converter.enum_display_name} is unspecified. Resolving to"
        " default:"
        f" {self.converter.enum_message.Name(self.converter.enum_unspecified)}.",
    ):
      proto_name = self.converter.from_proto(
          _MediaEffectsDist.MEDIA_EFFECTS_DISTRIBUTION_UNSPECIFIED
      )
    self.assertEqual(proto_name, self.converter.default_when_unspecified)

  def test_from_proto_raises_error_for_invalid_proto(self):
    with self.assertRaisesRegex(
        ValueError,
        f"Invalid {self.converter.enum_message.DESCRIPTOR.name} proto enum"
        " value: -1.",
    ):
      self.converter.from_proto(-1)

  def test_from_proto_raises_error_for_nonexistent_proto_name(self):
    with self.assertRaisesRegex(
        KeyError,
        "Protobuf enum name 'NORMAL' is not configured in the mapping.",
    ):
      self.converter.from_proto(_MediaEffectsDist.NORMAL)


if __name__ == "__main__":
  absltest.main()
