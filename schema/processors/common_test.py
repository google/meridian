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
from meridian import constants
from mmm.v1.common import estimate_pb2 as estimate_pb
from schema.processors import common
import numpy as np
import xarray as xr

from tensorflow.python.util.protobuf import compare


class CommonTest(absltest.TestCase):

  def test_to_estimate_returns_correct_estimate_proto(self):
    data = xr.DataArray(
        data=np.array([100.0, 90.0, 110.0]),
        dims=[constants.METRIC],
        coords={
            constants.METRIC: [
                constants.MEAN,
                constants.CI_LO,
                constants.CI_HI,
            ]
        },
    )

    estimate_proto = common.to_estimate(
        dataarray=data, confidence_level=constants.DEFAULT_CONFIDENCE_LEVEL
    )
    expected_estimate_proto = estimate_pb.Estimate(
        value=100.0,
        uncertainties=[
            estimate_pb.Estimate.Uncertainty(
                probability=constants.DEFAULT_CONFIDENCE_LEVEL,
                lowerbound=90.0,
                upperbound=110.0,
            )
        ],
    )
    compare.assertProtoEqual(self, estimate_proto, expected_estimate_proto)


if __name__ == "__main__":
  absltest.main()
