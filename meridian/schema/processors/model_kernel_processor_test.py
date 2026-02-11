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

from absl.testing import absltest
from absl.testing import parameterized
import meridian
from meridian.model import model
from mmm.v1 import mmm_pb2 as pb
from mmm.v1.model import mmm_kernel_pb2 as kernel_pb
from meridian.schema.processors import model_kernel_processor
from meridian.schema.serde import meridian_serde
import semver

from tensorflow.python.util.protobuf import compare


class ModelKernelProcessorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._model_id = 'test_model'
    self._meridian_version = semver.VersionInfo.parse(meridian.__version__)
    self._mock_meridian = mock.MagicMock(spec=model.Meridian)
    self._processor = model_kernel_processor.ModelKernelProcessor(
        meridian_model=self._mock_meridian,
        model_id=self._model_id,
    )

  @mock.patch.object(meridian_serde.MeridianSerde, 'serialize')
  def test_call(self, mock_serialize):
    mock_serialize.return_value = kernel_pb.MmmKernel()
    output = pb.Mmm()
    self._processor(output)
    self.assertTrue(output.HasField('mmm_kernel'))
    mock_serialize.assert_called_once_with(
        self._mock_meridian,
        self._model_id,
        self._meridian_version,
        include_convergence_info=True,
    )
    compare.assertProtoEqual(self, output.mmm_kernel, kernel_pb.MmmKernel())


if __name__ == '__main__':
  absltest.main()
