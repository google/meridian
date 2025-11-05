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
import arviz as az
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from schema.serde import inference_data as infdata
import numpy as np
import xarray as xr
from tensorflow.python.util.protobuf import compare


mock = absltest.mock


def _mock_infdata_with_datasets(
    include_prior=False,
    include_posterior=False,  # Also include debug sample_stats and trace mocks.
    to_netcdf_returns_memoryview=False,
) -> mock.MagicMock:
  idata = mock.MagicMock(spec=az.InferenceData)
  mock_groups = {}

  attrs = {"created_at": "2024-08-20T12:00:00.000000000Z"}
  get_return_value = (
      lambda bytes: memoryview(bytes) if to_netcdf_returns_memoryview else bytes
  )

  if include_prior:
    prior = mock.MagicMock(spec=xr.Dataset)
    prior.name = "prior"
    prior.attrs = attrs.copy()
    prior.to_netcdf.return_value = get_return_value(b"test-prior-bytes")
    prior.copy.return_value = prior
    idata.prior = prior
    mock_groups["prior"] = prior

  if include_posterior:
    posterior = mock.MagicMock(spec=xr.Dataset)
    posterior.name = "posterior"
    posterior.attrs = attrs.copy()
    posterior.to_netcdf.return_value = get_return_value(b"test-posterior-bytes")
    posterior.copy.return_value = posterior
    idata.posterior = posterior
    mock_groups["posterior"] = posterior

    sample_stats = mock.MagicMock(spec=xr.Dataset)
    sample_stats.name = "sample_stats"
    sample_stats.attrs = attrs.copy()
    sample_stats.to_netcdf.return_value = get_return_value(b"test-stats-bytes")
    sample_stats.copy.return_value = sample_stats
    mock_groups["sample_stats"] = sample_stats

    trace = mock.MagicMock(spec=xr.Dataset)
    trace.name = "trace"
    trace.attrs = attrs.copy()
    trace.to_netcdf.return_value = get_return_value(b"test-trace-bytes")
    trace.copy.return_value = trace
    mock_groups["trace"] = trace

  idata.groups.return_value = mock_groups.keys()

  def _get(group: str) -> xr.Dataset:
    return mock_groups[group]

  idata.get.side_effect = _get

  return idata


def _create_random_infdata(group: str) -> az.InferenceData:
  shape = (1, 2, 3, 4, 5)
  dataset = az.convert_to_inference_data(np.random.randn(*shape), group=group)

  idata = az.InferenceData()
  idata.extend(dataset, join="right")
  return idata


def _create_prior_infdata() -> az.InferenceData:
  return _create_random_infdata("prior")


def _create_posterior_infdata() -> az.InferenceData:
  return az.concat(
      _create_random_infdata("posterior"),
      _create_random_infdata("trace"),
      _create_random_infdata("sample_stats"),
  )


def _create_fully_fitted_infdata() -> az.InferenceData:
  prior = _create_prior_infdata()
  posterior = _create_posterior_infdata()
  prior.extend(posterior, join="right")
  return prior


class InferenceDataTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.serde = infdata.InferenceDataSerde()

  def test_serialize_no_sampled_data(self):
    infdata_proto = self.serde.serialize(az.InferenceData())
    compare.assertProtoEqual(
        self,
        infdata_proto,
        meridian_pb.InferenceData(),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="prior_only",
          idata=_mock_infdata_with_datasets(
              include_prior=True,
              include_posterior=False,
          ),
          expected_infdata_proto=meridian_pb.InferenceData(
              prior=b"test-prior-bytes",
          ),
      ),
      dict(
          testcase_name="prior_only_memoryview",
          idata=_mock_infdata_with_datasets(
              include_prior=True,
              include_posterior=False,
              to_netcdf_returns_memoryview=True,
          ),
          expected_infdata_proto=meridian_pb.InferenceData(
              prior=b"test-prior-bytes",
          ),
      ),
      dict(
          testcase_name="posterior_only",
          idata=_mock_infdata_with_datasets(
              include_prior=False,
              include_posterior=True,
          ),
          expected_infdata_proto=meridian_pb.InferenceData(
              posterior=b"test-posterior-bytes",
              auxiliary_data={
                  "sample_stats": b"test-stats-bytes",
                  "trace": b"test-trace-bytes",
              },
          ),
      ),
      dict(
          testcase_name="posterior_only_memoryview",
          idata=_mock_infdata_with_datasets(
              include_prior=False,
              include_posterior=True,
              to_netcdf_returns_memoryview=True,
          ),
          expected_infdata_proto=meridian_pb.InferenceData(
              posterior=b"test-posterior-bytes",
              auxiliary_data={
                  "sample_stats": b"test-stats-bytes",
                  "trace": b"test-trace-bytes",
              },
          ),
      ),
      dict(
          testcase_name="fully_fitted",
          idata=_mock_infdata_with_datasets(
              include_prior=True,
              include_posterior=True,
          ),
          expected_infdata_proto=meridian_pb.InferenceData(
              prior=b"test-prior-bytes",
              posterior=b"test-posterior-bytes",
              auxiliary_data={
                  "sample_stats": b"test-stats-bytes",
                  "trace": b"test-trace-bytes",
              },
          ),
      ),
      dict(
          testcase_name="fully_fitted_memoryview",
          idata=_mock_infdata_with_datasets(
              include_prior=True,
              include_posterior=True,
              to_netcdf_returns_memoryview=True,
          ),
          expected_infdata_proto=meridian_pb.InferenceData(
              prior=b"test-prior-bytes",
              posterior=b"test-posterior-bytes",
              auxiliary_data={
                  "sample_stats": b"test-stats-bytes",
                  "trace": b"test-trace-bytes",
              },
          ),
      ),
  )
  def test_serialize(
      self,
      idata: mock.MagicMock,
      expected_infdata_proto: meridian_pb.InferenceData,
  ):
    infdata_proto = self.serde.serialize(idata)
    compare.assertProtoEqual(self, infdata_proto, expected_infdata_proto)
    self.assertNotIn("created_at", idata.attrs)

  def test_deserialize_empty(self):
    infdata_proto = meridian_pb.InferenceData()
    idata = self.serde.deserialize(infdata_proto)
    self.assertEqual(idata, az.InferenceData())

  @parameterized.named_parameters(
      dict(
          testcase_name="prior_only",
          idata=_create_prior_infdata(),
      ),
      dict(
          testcase_name="posterior_only",
          idata=_create_posterior_infdata(),
      ),
      dict(
          testcase_name="fully_fitted",
          idata=_create_fully_fitted_infdata(),
      ),
  )
  def test_serialize_deserialize(self, idata: az.InferenceData):
    ser = self.serde.serialize(idata)
    de = self.serde.deserialize(ser)
    self.assertEqual(de, idata)


if __name__ == "__main__":
  absltest.main()
