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

import copy

from absl.testing import absltest
from meridian.model.eda import eda_spec
from meridian.schema.serde import eda_spec as eda_spec_serde
from meridian.schema.serde import function_registry as function_registry_utils
import numpy as np
import xarray as xr


def _custom_agg_fn(x: xr.DataArray) -> np.ndarray:
  return np.mean(x).values


def _custom_agg_fn_other(x: xr.DataArray) -> np.ndarray:
  return np.sum(x).values


class EDASpecSerdeTest(absltest.TestCase):

  def test_serialize_deserialize_default_eda_spec(self):
    serde = eda_spec_serde.EDASpecSerde(
        function_registry_utils.FunctionRegistry()
    )
    original_spec = eda_spec.EDASpec()
    serialized = serde.serialize(original_spec)
    deserialized = serde.deserialize(serialized)
    self.assertEqual(original_spec, deserialized)

  def test_serialize_deserialize_custom_vif_spec(self):
    serde = eda_spec_serde.EDASpecSerde(
        function_registry_utils.FunctionRegistry()
    )
    custom_vif = eda_spec.VIFSpec(
        geo_threshold=100.0, overall_threshold=200.0, national_threshold=300.0
    )
    original_spec = eda_spec.EDASpec(vif_spec=custom_vif)
    serialized = serde.serialize(original_spec)
    deserialized = serde.deserialize(serialized)
    self.assertEqual(original_spec, deserialized)

  def test_serialize_deserialize_custom_aggregation_config(self):
    function_registry = function_registry_utils.FunctionRegistry(
        {"custom_agg": _custom_agg_fn, "numpy_sum": np.sum}
    )
    serde = eda_spec_serde.EDASpecSerde(function_registry=function_registry)
    custom_agg = eda_spec.AggregationConfig(
        control_variables={"var1": _custom_agg_fn},
        non_media_treatments={"var2": np.sum},
    )
    original_spec = eda_spec.EDASpec(aggregation_config=custom_agg)
    serialized = serde.serialize(original_spec)
    deserialized = serde.deserialize(serialized)
    self.assertEqual(original_spec, deserialized)

  def test_deserialize_with_changed_function_registry_raises_error(self):
    function_registry = function_registry_utils.FunctionRegistry(
        {"custom_agg": _custom_agg_fn}
    )
    serde = eda_spec_serde.EDASpecSerde(function_registry=function_registry)
    custom_agg = eda_spec.AggregationConfig(
        control_variables={"var1": _custom_agg_fn},
    )
    original_spec = eda_spec.EDASpec(aggregation_config=custom_agg)
    serialized = serde.serialize(original_spec)

    # When we try to deserialize with a modified function registry, it fails
    changed_function_registry = function_registry_utils.FunctionRegistry(
        {"custom_agg": _custom_agg_fn_other}
    )
    serde_with_changed_registry = eda_spec_serde.EDASpecSerde(
        function_registry=changed_function_registry
    )
    with self.assertRaisesRegex(
        ValueError, "An issue found during deserializing EDASpec"
    ):
      serde_with_changed_registry.deserialize(serialized)

  def test_deserialize_with_force_deserialization_succeeds(self):
    function_registry = function_registry_utils.FunctionRegistry(
        {"custom_agg": _custom_agg_fn}
    )
    serde = eda_spec_serde.EDASpecSerde(function_registry=function_registry)
    custom_agg = eda_spec.AggregationConfig(
        control_variables={"var1": _custom_agg_fn},
    )
    original_spec = eda_spec.EDASpec(aggregation_config=custom_agg)
    serialized = serde.serialize(original_spec)

    # If force_deserialization=True, deserialization succeeds even with
    # a mismatched function registry.
    changed_function_registry = function_registry_utils.FunctionRegistry(
        {"custom_agg": _custom_agg_fn_other}
    )
    serde_with_changed_registry = eda_spec_serde.EDASpecSerde(
        function_registry=changed_function_registry
    )
    with self.assertWarnsRegex(
        Warning,
        "You're attempting to deserialize an EDASpec while ignoring changes",
    ):
      deserialized = serde_with_changed_registry.deserialize(
          serialized, force_deserialization=True
      )
    # The functions will be different, but the rest of the spec is identical.
    self.assertEqual(deserialized.vif_spec, original_spec.vif_spec)
    self.assertEqual(
        deserialized.aggregation_config.control_variables["var1"],
        _custom_agg_fn_other,
    )

  def test_serialize_raises_error_for_function_not_in_registry(self):
    # Serde is initialized without np.sum in its registry
    function_registry = function_registry_utils.FunctionRegistry(
        {"custom_agg": _custom_agg_fn}
    )
    serde = eda_spec_serde.EDASpecSerde(function_registry=function_registry)
    # Spec uses np.sum, which is not in registry
    custom_agg = eda_spec.AggregationConfig(
        non_media_treatments={"var2": np.sum},
    )
    spec_with_unregistered_fn = eda_spec.EDASpec(aggregation_config=custom_agg)

    with self.assertRaisesRegex(
        ValueError,
        "Custom aggregation function `var2` in `non_media_treatments` detected,"
        " but not found in registry.",
    ):
      serde.serialize(spec_with_unregistered_fn)

  def test_deserialize_raises_error_for_missing_function_key(self):
    function_registry = function_registry_utils.FunctionRegistry(
        {"custom_agg": _custom_agg_fn}
    )
    serde = eda_spec_serde.EDASpecSerde(function_registry=function_registry)
    custom_agg = eda_spec.AggregationConfig(
        control_variables={"var1": _custom_agg_fn},
    )
    original_spec = eda_spec.EDASpec(aggregation_config=custom_agg)
    serialized = serde.serialize(original_spec)

    # Manually remove function_key from proto
    serialized_missing_key = copy.deepcopy(serialized)
    serialized_missing_key.aggregation_config.control_variables[
        "var1"
    ].ClearField("function_key")

    with self.assertRaisesRegex(
        ValueError,
        "Function key is required in `AggregationFunction` proto message. The"
        " function key for var1 is empty.",
    ):
      serde.deserialize(serialized_missing_key)

  def test_deserialize_raises_error_for_function_key_not_in_registry(self):
    function_registry = function_registry_utils.FunctionRegistry(
        {"custom_agg": _custom_agg_fn}
    )
    serde = eda_spec_serde.EDASpecSerde(function_registry=function_registry)
    custom_agg = eda_spec.AggregationConfig(
        control_variables={"var1": _custom_agg_fn},
    )
    original_spec = eda_spec.EDASpec(aggregation_config=custom_agg)
    serialized = serde.serialize(original_spec)

    # When we try to deserialize with a registry missing 'custom_agg', it fails
    # in _from_aggregation_function_proto, if we bypass main validation.
    serde_missing_key = eda_spec_serde.EDASpecSerde(
        function_registry=function_registry_utils.FunctionRegistry()
    )
    with self.assertRaisesRegex(
        ValueError, "Function key `custom_agg` not found in registry."
    ):
      serde_missing_key.deserialize(serialized, force_deserialization=True)


if __name__ == "__main__":
  absltest.main()
