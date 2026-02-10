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

"""Serialization and deserialization of `InferenceData` container for sampled priors and posteriors."""

import io

import arviz as az
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from schema.serde import serde
import xarray as xr


_NETCDF_FORMAT = "NETCDF3_64BIT"  # scipy only supports up to v3
_PRIOR_FIELD = "prior"
_POSTERIOR_FIELD = "posterior"
_CREATED_AT_ATTRIBUTE = "created_at"


def _remove_created_at_attribute(dataset: xr.Dataset) -> xr.Dataset:
  dataset_copy = dataset.copy()
  if _CREATED_AT_ATTRIBUTE in dataset_copy.attrs:
    del dataset_copy.attrs[_CREATED_AT_ATTRIBUTE]
  return dataset_copy


class InferenceDataSerde(
    serde.Serde[meridian_pb.InferenceData, az.InferenceData]
):
  """Serializes and deserializes an `InferenceData` container in Meridian.

  Meridian uses `InferenceData` as a container to store sampled prior and
  posterior containers.
  """

  def serialize(self, obj: az.InferenceData) -> meridian_pb.InferenceData:
    """Serializes the given Meridian inference data container into an `InferenceData` proto."""
    if hasattr(obj, _PRIOR_FIELD):
      prior_dataset_copy = _remove_created_at_attribute(obj.prior)  # pytype: disable=attribute-error
      prior_bytes = bytes(prior_dataset_copy.to_netcdf(format=_NETCDF_FORMAT))
    else:
      prior_bytes = None

    if hasattr(obj, _POSTERIOR_FIELD):
      posterior_dataset_copy = _remove_created_at_attribute(obj.posterior)  # pytype: disable=attribute-error
      posterior_bytes = bytes(
          posterior_dataset_copy.to_netcdf(format=_NETCDF_FORMAT)
      )
    else:
      posterior_bytes = None

    aux = {}
    for group in obj.groups():
      if group in (_PRIOR_FIELD, _POSTERIOR_FIELD):
        continue
      aux_dataset_copy = _remove_created_at_attribute(obj.get(group))
      aux[group] = bytes(aux_dataset_copy.to_netcdf(format=_NETCDF_FORMAT))

    return meridian_pb.InferenceData(
        prior=prior_bytes,
        posterior=posterior_bytes,
        auxiliary_data=aux,
    )

  def deserialize(
      self, serialized: meridian_pb.InferenceData, serialized_version: str = ""
  ) -> az.InferenceData:
    """Deserializes the given `InferenceData` proto.

    Args:
      serialized: The serialized `InferenceData` proto.
      serialized_version: The version of the serialized model. This is used to
        handle changes in deserialization logic across different versions.

    Returns:
      A Meridian inference data container.
    """
    groups = {}

    if serialized.HasField(_PRIOR_FIELD):
      prior_dataset = xr.open_dataset(io.BytesIO(serialized.prior))
      groups[_PRIOR_FIELD] = prior_dataset

    if serialized.HasField(_POSTERIOR_FIELD):
      posterior_dataset = xr.open_dataset(io.BytesIO(serialized.posterior))
      groups[_POSTERIOR_FIELD] = posterior_dataset

    for name, data in serialized.auxiliary_data.items():
      groups[name] = xr.open_dataset(io.BytesIO(data))

    idata = az.InferenceData()
    if groups:
      idata.add_groups(groups)
    return idata
