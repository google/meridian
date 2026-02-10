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

import inspect
import warnings

from absl.testing import absltest
from absl.testing import parameterized
from schema.serde import function_registry


def _func1(x):
  return x + 1


def _func2(x):
  return x + 2


def _func1_modified(x):
  return x + 10


class FunctionRegistryUtilsTest(parameterized.TestCase):

  def test_hashed_registry(self):
    registry = function_registry.FunctionRegistry(
        {"func1": _func1, "func2": _func2}
    )
    hashed_registry = registry.hashed_registry
    self.assertEqual(
        hashed_registry,
        {
            "func1": function_registry._get_hash(inspect.getsource(_func1)),
            "func2": function_registry._get_hash(inspect.getsource(_func2)),
        },
    )

  def test_init_with_kwargs(self):
    registry = function_registry.FunctionRegistry(func1=_func1, func2=_func2)
    self.assertEqual(registry, {"func1": _func1, "func2": _func2})

  def test_validate_matching_registries_succeeds(self):
    registry = function_registry.FunctionRegistry({"func1": _func1})
    hashed_registry = registry.hashed_registry
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      registry.validate(
          stored_hashed_function_registry=hashed_registry,
      )
      self.assertEmpty(w)

  def test_validate_mismatched_hash_raises_error(self):
    registry = function_registry.FunctionRegistry({"func1": _func1})
    hashed_registry = registry.hashed_registry

    registry_modified = function_registry.FunctionRegistry(
        {"func1": _func1_modified}
    )
    with self.assertRaisesRegex(
        ValueError, "Function registry hash mismatch for func1"
    ):
      registry_modified.validate(
          stored_hashed_function_registry=hashed_registry,
      )

  def test_validate_no_stored_registry_warns(self):
    registry = function_registry.FunctionRegistry({"func1": _func1})
    with self.assertWarnsRegex(UserWarning, "A function registry was provided"):
      registry.validate(
          stored_hashed_function_registry={},
      )

  def test_validate_empty_registry_and_stored_registry_raises_error(self):
    registry = function_registry.FunctionRegistry()
    with self.assertRaisesRegex(
        ValueError,
        "Function 'func1' is required by the serialized object but is missing",
    ):
      registry.validate(
          stored_hashed_function_registry={"func1": "hash1"},
      )

  def test_validate_both_empty_succeeds(self):
    registry = function_registry.FunctionRegistry()
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      registry.validate(
          stored_hashed_function_registry={},
      )
      self.assertEmpty(w)

  def test_hashed_registry_with_lambda_warns(self):
    registry = function_registry.FunctionRegistry(
        {"lambda_func": lambda x: x + 1}
    )
    with self.assertWarns(function_registry.LambdaSourceCodeWarning):
      _ = registry.hashed_registry

  def test_validate_with_lambda_warns(self):
    registry = function_registry.FunctionRegistry(
        {"lambda_func": lambda x: x + 1}
    )
    with self.assertWarns(function_registry.LambdaSourceCodeWarning):
      hashed_registry = registry.hashed_registry

    with self.assertWarns(function_registry.LambdaSourceCodeWarning):
      registry.validate(
          stored_hashed_function_registry=hashed_registry,
      )

  def test_hashed_registry_with_builtin_raises_error(self):
    registry = function_registry.FunctionRegistry({"builtin_func": sum})
    with self.assertRaisesRegex(
        function_registry.SourceCodeRetrievalError,
        "Source code of function.*is not retrievable",
    ):
      _ = registry.hashed_registry

  def test_validate_with_builtin_raises_error(self):
    registry = function_registry.FunctionRegistry({"builtin_func": sum})
    dummy_hashed_registry = {"builtin_func": "some_hash"}
    with self.assertRaisesRegex(
        ValueError,
        "Failed to retrieve source code of function builtin_func.",
    ):
      registry.validate(stored_hashed_function_registry=dummy_hashed_registry)

  @parameterized.named_parameters(
      dict(
          testcase_name="found",
          func=_func1,
          registry={"func1": _func1, "func2": _func2},
          expected="func1",
      ),
      dict(
          testcase_name="not_found",
          func=_func1_modified,
          registry={"func1": _func1, "func2": _func2},
          expected=None,
      ),
  )
  def test_get_function_key(self, func, registry, expected):
    registry = function_registry.FunctionRegistry(registry)
    self.assertEqual(
        registry.get_function_key(
            func=func,
        ),
        expected,
    )


if __name__ == "__main__":
  absltest.main()
