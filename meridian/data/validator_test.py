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
from meridian import constants
from meridian.data import validator
import numpy as np
import xarray as xr


class ValidatorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="valid_time",
          coords={
              constants.TIME: ["2023-01-01", "2023-01-02"],
              "geo": ["g1", "g2", "g3"],
          },
          dims=[constants.TIME, "geo"],
      ),
      dict(
          testcase_name="valid_media_time",
          coords={
              constants.MEDIA_TIME: ["2023-01-01", "2023-01-02"],
              "channel": ["c1", "c2", "c3"],
          },
          dims=[constants.MEDIA_TIME, "channel"],
      ),
  )
  def test_validate_time_coord_format_valid(self, coords, dims):
    dim1, dim2 = dims
    data = xr.DataArray(
        np.zeros((len(coords[dim1]), len(coords[dim2]))),
        coords=coords,
        dims=dims,
    )
    validator.validate_time_coord_format(data)

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_time",
          coords={
              constants.TIME: ["2023/01/01", "2023-01-02"],
              "geo": ["g1", "g2", "g3"],
          },
          dims=[constants.TIME, "geo"],
          expected_coord_name=constants.TIME,
          expected_invalid_label="2023/01/01",
      ),
      dict(
          testcase_name="invalid_media_time",
          coords={
              constants.MEDIA_TIME: ["2023-01-01", "01-02-2023"],
              "channel": ["c1", "c2", "c3"],
          },
          dims=[constants.MEDIA_TIME, "channel"],
          expected_coord_name=constants.MEDIA_TIME,
          expected_invalid_label="01-02-2023",
      ),
  )
  def test_validate_time_coord_format_invalid(
      self, *, coords, dims, expected_coord_name, expected_invalid_label
  ):
    dim1, dim2 = dims
    data = xr.DataArray(
        np.zeros((len(coords[dim1]), len(coords[dim2]))),
        coords=coords,
        dims=dims,
    )
    with self.assertRaisesRegex(
        ValueError,
        f"Invalid {expected_coord_name} label: '{expected_invalid_label}'. "
        f"Expected format: '{constants.DATE_FORMAT}'",
    ):
      validator.validate_time_coord_format(data)

  def test_validate_time_coord_format_none(self):
    validator.validate_time_coord_format(None)

  def test_validate_time_coord_format_no_time_coords(self):
    data_no_time = xr.DataArray(
        np.zeros((2, 3)),
        coords={"geo": ["g1", "g2"], "channel": ["c1", "c2", "c3"]},
        dims=["geo", "channel"],
    )
    validator.validate_time_coord_format(data_no_time)


if __name__ == "__main__":
  absltest.main()
