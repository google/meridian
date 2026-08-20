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

"""Tests for the currency module."""

from typing import Any
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from meridian.common import currency


class CurrencyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="none", currency_code=None, expected="$"),
      dict(testcase_name="empty", currency_code="", expected="$"),
      dict(testcase_name="whitespace", currency_code="   ", expected="$"),
      dict(testcase_name="mock", currency_code=mock.MagicMock(), expected="$"),
      dict(testcase_name="int", currency_code=123, expected="$"),
  )
  def test_get_currency_symbol_none_or_empty_returns_default(
      self, currency_code: Any, expected: str
  ):
    self.assertEqual(currency.get_currency_symbol(currency_code), expected)

  def test_get_currency_symbol_custom_default(self):
    self.assertEqual(
        currency.get_currency_symbol(None, default="custom"), "custom"
    )
    self.assertEqual(
        currency.get_currency_symbol("", default="custom"), "custom"
    )

  @parameterized.named_parameters(
      dict(testcase_name="usd", currency_code="USD", expected="$"),
      dict(testcase_name="eur", currency_code="EUR", expected="€"),
      dict(testcase_name="gbp", currency_code="GBP", expected="£"),
      dict(testcase_name="jpy", currency_code="JPY", expected="¥"),
      dict(testcase_name="inr", currency_code="INR", expected="₹"),
      dict(testcase_name="brl", currency_code="BRL", expected="R$"),
      dict(testcase_name="krw", currency_code="KRW", expected="₩"),
      dict(testcase_name="cad", currency_code="CAD", expected="CA$"),
      dict(testcase_name="aud", currency_code="AUD", expected="A$"),
      dict(testcase_name="chf", currency_code="CHF", expected="CHF"),
  )
  def test_get_currency_symbol_known_currencies(
      self, currency_code: str, expected: str
  ):
    self.assertEqual(currency.get_currency_symbol(currency_code), expected)

  @parameterized.named_parameters(
      dict(testcase_name="lowercase_usd", currency_code="usd", expected="$"),
      dict(testcase_name="mixedcase_eur", currency_code="Eur", expected="€"),
      dict(testcase_name="spaced_gbp", currency_code=" gbp ", expected="£"),
  )
  def test_get_currency_symbol_case_and_whitespace_insensitivity(
      self, currency_code: str, expected: str
  ):
    self.assertEqual(currency.get_currency_symbol(currency_code), expected)

  def test_get_currency_symbol_unknown_code_returns_code(self):
    self.assertEqual(currency.get_currency_symbol("XYZ"), "XYZ")
    self.assertEqual(currency.get_currency_symbol("unknown"), "UNKNOWN")

  def test_is_valid_currency_code(self):
    self.assertTrue(currency.is_valid_currency_code("USD"))
    self.assertTrue(currency.is_valid_currency_code("eur"))
    self.assertTrue(currency.is_valid_currency_code(" GBP "))
    self.assertTrue(currency.is_valid_currency_code("JPY"))
    self.assertFalse(currency.is_valid_currency_code("INVALID"))
    self.assertFalse(currency.is_valid_currency_code(""))
    self.assertFalse(currency.is_valid_currency_code(None))

  def test_normalize_currency_code(self):
    self.assertEqual(currency.normalize_currency_code("usd"), "USD")
    self.assertEqual(currency.normalize_currency_code(" EUR "), "EUR")
    self.assertEqual(currency.normalize_currency_code("GBP"), "GBP")
    self.assertIsNone(currency.normalize_currency_code("INVALID"))
    self.assertIsNone(currency.normalize_currency_code(""))
    self.assertIsNone(currency.normalize_currency_code(None))


if __name__ == "__main__":
  absltest.main()
