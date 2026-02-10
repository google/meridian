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

"""Unit tests for model_test_data."""

import collections
import dataclasses
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from meridian import backend
from meridian.backend import test_utils as backend_test_utils
from meridian.data import input_data
from meridian.model import model_test_data
import xarray as xr

_PRIVATE_PREFIX = "_"


def _get_private_attr_name(property_name: str) -> str:
  """Returns the private attribute name for a given property name."""
  return _PRIVATE_PREFIX + property_name


def _assert_tensors_equal(
    test_case: absltest.TestCase,
    tensor1: backend.Tensor,
    tensor2: backend.Tensor,
):
  """Asserts that two backend tensors are equal."""
  test_case.assertEqual(tensor1.shape, tensor2.shape)
  test_case.assertEqual(tensor1.dtype, tensor2.dtype)
  backend_test_utils.assert_allclose(tensor1, tensor2)


def _assert_data_array_equal(test_case, arr1, arr2):
  if arr1 is None and arr2 is None:
    return
  if arr1 is None or arr2 is None:
    test_case.fail(
        f"One DataArray is None while the other is not: {arr1}, {arr2}"
    )
  # xr.DataArray.equals compares data, coords, dims, and attrs.
  test_case.assertTrue(
      arr1.equals(arr2), f"DataArrays are not equal: {arr1.name}"
  )


def _assert_input_data_equal(
    test_case: absltest.TestCase,
    data1: input_data.InputData,
    data2: input_data.InputData,
):
  """Asserts that two InputData objects are equal."""
  for field in dataclasses.fields(data1):
    name = field.name
    val1 = getattr(data1, name)
    val2 = getattr(data2, name)
    if isinstance(val1, xr.DataArray):
      _assert_data_array_equal(test_case, val1, val2)
    else:
      test_case.assertEqual(val1, val2, f"Field '{name}' differs.")


def _assert_ordered_dict_tensors_equal(
    test_case: absltest.TestCase,
    dict1: collections.OrderedDict[str, backend.Tensor],
    dict2: collections.OrderedDict[str, backend.Tensor],
):
  """Asserts that two OrderedDicts of tensors are equal."""
  test_case.assertEqual(dict1.keys(), dict2.keys())
  for key in dict1:
    _assert_tensors_equal(test_case, dict1[key], dict2[key])


def _assert_dict_tensors_equal(
    test_case: absltest.TestCase,
    dict1: dict[str, backend.Tensor],
    dict2: dict[str, backend.Tensor],
):
  """Asserts that two dicts of tensors are equal."""
  test_case.assertEqual(dict1.keys(), dict2.keys())
  for key in dict1:
    _assert_tensors_equal(test_case, dict1[key], dict2[key])


class WithInputDataSamplesTest(
    parameterized.TestCase, model_test_data.WithInputDataSamples
):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.setup()

  def _get_property(self, property_name: str) -> Any:
    """Returns the value of a public property."""
    return getattr(self, property_name)

  def _get_private_attr(self, property_name: str) -> Any:
    """Returns the value of the private attribute for a given property name."""
    return getattr(self, _get_private_attr_name(property_name))

  @parameterized.named_parameters(
      (
          "input_data_non_revenue_no_revenue_per_kpi",
          "input_data_non_revenue_no_revenue_per_kpi",
      ),
      (
          "input_data_media_and_rf_non_revenue_no_revenue_per_kpi",
          "input_data_media_and_rf_non_revenue_no_revenue_per_kpi",
      ),
      ("input_data_with_media_only", "input_data_with_media_only"),
      ("input_data_with_rf_only", "input_data_with_rf_only"),
      ("input_data_with_media_and_rf", "input_data_with_media_and_rf"),
      (
          "input_data_with_media_and_rf_no_controls",
          "input_data_with_media_and_rf_no_controls",
      ),
      ("short_input_data_with_media_only", "short_input_data_with_media_only"),
      (
          "short_input_data_with_media_only_no_controls",
          "short_input_data_with_media_only_no_controls",
      ),
      ("short_input_data_with_rf_only", "short_input_data_with_rf_only"),
      (
          "short_input_data_with_media_and_rf",
          "short_input_data_with_media_and_rf",
      ),
      ("national_input_data_media_only", "national_input_data_media_only"),
      ("national_input_data_media_and_rf", "national_input_data_media_and_rf"),
      (
          "national_input_data_non_media_and_organic",
          "national_input_data_non_media_and_organic",
      ),
      ("input_data_non_media_and_organic", "input_data_non_media_and_organic"),
      (
          "short_input_data_non_media_and_organic",
          "short_input_data_non_media_and_organic",
      ),
      ("short_input_data_non_media", "short_input_data_non_media"),
      (
          "input_data_non_media_and_organic_same_time_dims",
          "input_data_non_media_and_organic_same_time_dims",
      ),
      ("input_data_organic_only", "input_data_organic_only"),
      (
          "national_input_data_organic_only",
          "national_input_data_organic_only",
      ),
  )
  def test_input_data_properties_value_equality(self, property_name):
    """Tests that InputData properties return values equal to private members."""
    property_value = self._get_property(property_name)
    private_value = self._get_private_attr(property_name)
    _assert_input_data_equal(self, property_value, private_value)

  @parameterized.named_parameters(
      (
          "input_data_non_revenue_no_revenue_per_kpi",
          "input_data_non_revenue_no_revenue_per_kpi",
      ),
      (
          "input_data_media_and_rf_non_revenue_no_revenue_per_kpi",
          "input_data_media_and_rf_non_revenue_no_revenue_per_kpi",
      ),
      ("input_data_with_media_only", "input_data_with_media_only"),
      ("input_data_with_rf_only", "input_data_with_rf_only"),
      ("input_data_with_media_and_rf", "input_data_with_media_and_rf"),
      (
          "input_data_with_media_and_rf_no_controls",
          "input_data_with_media_and_rf_no_controls",
      ),
      ("short_input_data_with_media_only", "short_input_data_with_media_only"),
      (
          "short_input_data_with_media_only_no_controls",
          "short_input_data_with_media_only_no_controls",
      ),
      ("short_input_data_with_rf_only", "short_input_data_with_rf_only"),
      (
          "short_input_data_with_media_and_rf",
          "short_input_data_with_media_and_rf",
      ),
      ("national_input_data_media_only", "national_input_data_media_only"),
      ("national_input_data_media_and_rf", "national_input_data_media_and_rf"),
      (
          "national_input_data_non_media_and_organic",
          "national_input_data_non_media_and_organic",
      ),
      ("input_data_non_media_and_organic", "input_data_non_media_and_organic"),
      (
          "short_input_data_non_media_and_organic",
          "short_input_data_non_media_and_organic",
      ),
      ("short_input_data_non_media", "short_input_data_non_media"),
      (
          "input_data_non_media_and_organic_same_time_dims",
          "input_data_non_media_and_organic_same_time_dims",
      ),
      ("input_data_organic_only", "input_data_organic_only"),
      (
          "national_input_data_organic_only",
          "national_input_data_organic_only",
      ),
  )
  def test_input_data_properties_deep_copy(self, property_name):
    """Tests that InputData properties return deep copies."""
    val1 = self._get_property(property_name)
    val2 = self._get_property(property_name)
    self.assertIsNot(val1, val2)

    # Check that individual DataArrays are also different instances and deep
    # copies.
    data_array_fields = [
        f.name
        for f in dataclasses.fields(val1)
        if isinstance(getattr(val1, f.name), xr.DataArray)
    ]
    for field_name in data_array_fields:
      arr1 = getattr(val1, field_name)
      arr2 = getattr(val2, field_name)
      self.assertIsNot(
          arr1, arr2, f"DataArray '{field_name}' is the same instance."
      )
      if arr1.size > 0:
        original_val = arr1.values.flat[0]
        # Modify a value in arr1
        arr1.values.flat[0] = original_val + 1
        # Check if arr2 is unchanged
        self.assertEqual(
            arr2.values.flat[0],
            original_val,
            f"DataArray '{field_name}' was not deeply copied.",
        )

  @parameterized.named_parameters(
      ("test_dist_media_and_rf", "test_dist_media_and_rf"),
      ("test_dist_media_only", "test_dist_media_only"),
      ("test_dist_media_only_no_controls", "test_dist_media_only_no_controls"),
      ("test_dist_rf_only", "test_dist_rf_only"),
  )
  def test_ordered_dict_properties_value_equality(self, property_name):
    """Tests that OrderedDict properties return values equal to private members."""
    property_value = self._get_property(property_name)
    private_value = self._get_private_attr(property_name)
    _assert_ordered_dict_tensors_equal(self, property_value, private_value)

  @parameterized.named_parameters(
      ("test_dist_media_and_rf", "test_dist_media_and_rf"),
      ("test_dist_media_only", "test_dist_media_only"),
      ("test_dist_media_only_no_controls", "test_dist_media_only_no_controls"),
      ("test_dist_rf_only", "test_dist_rf_only"),
  )
  def test_ordered_dict_properties_deep_copy(self, property_name):
    """Tests that OrderedDict properties return deep copies."""
    val1 = self._get_property(property_name)
    val2 = self._get_property(property_name)
    self.assertIsNot(val1, val2)

    # Modify a tensor within val1 and check val2
    key_to_modify = next(iter(val1))
    original_tensor = val1[key_to_modify]
    modified_tensor = backend.zeros_like(original_tensor)
    val1[key_to_modify] = modified_tensor
    _assert_tensors_equal(self, original_tensor, val2[key_to_modify])

  @parameterized.named_parameters(
      (
          "test_posterior_states_media_and_rf",
          "test_posterior_states_media_and_rf",
      ),
      ("test_posterior_states_media_only", "test_posterior_states_media_only"),
      (
          "test_posterior_states_media_only_no_controls",
          "test_posterior_states_media_only_no_controls",
      ),
      ("test_posterior_states_rf_only", "test_posterior_states_rf_only"),
  )
  def test_named_tuple_properties_immutability(self, property_name):
    """Tests immutability of NamedTuple properties and their tensor attributes."""
    nt = self._get_property(property_name)

    for field_name in nt._fields:
      original_value = getattr(nt, field_name)

      # Test 1: NamedTuple attribute immutability. Cannot modify fields directly
      with self.assertRaisesRegex(AttributeError, "can't set attribute"):
        setattr(nt, field_name, "some_other_value")

      # Test 2: Backend.Tensor immutability
      self.assertIsInstance(original_value, backend.Tensor)
      # Check if the tensor has any elements.
      if all(d > 0 for d in original_value.shape):
        # Attempt to modify an element within the tensor.
        # backend.Tensor objects are immutable, so this should raise TypeError
        idx = (0,) * original_value.ndim
        with self.assertRaises(TypeError):
          # This line attempts to perform an in-place modification
          original_value[idx] = original_value[idx] + 1

  def test_trace_value_equality(self):
    """Tests that test_trace property returns value equal to private member."""
    property_value = self._get_property("test_trace")
    private_value = self._get_private_attr("test_trace")
    _assert_dict_tensors_equal(self, property_value, private_value)

  def test_trace_deep_copy(self):
    """Tests that test_trace property returns a deep copy."""
    dict1 = self.test_trace
    dict2 = self.test_trace
    self.assertIsNot(dict1, dict2)

    # Modify a tensor within dict1 and check dict2
    key_to_modify = next(iter(dict1))
    original_tensor = dict1[key_to_modify]
    modified_tensor = backend.zeros_like(original_tensor)
    dict1[key_to_modify] = modified_tensor
    _assert_tensors_equal(self, original_tensor, dict2[key_to_modify])


if __name__ == "__main__":
  absltest.main()
