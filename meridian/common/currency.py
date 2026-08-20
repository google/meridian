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

"""Currency utilities for Meridian."""

from babel import numbers as babel_numbers
from meridian import constants

__all__ = [
    'get_currency_symbol',
    'is_valid_currency_code',
    'normalize_currency_code',
]


def get_currency_symbol(
    currency_code: str | None = None,
    default: str = constants.DEFAULT_CURRENCY_SYMBOL,
) -> str:
  """Resolves an ISO 4217 currency code into its display currency symbol.

  Uses `babel.numbers.get_currency_symbol()` to resolve the display symbol for
  the currency. If the code is None or empty, returns `default`. If the symbol
  cannot be resolved, returns the uppercase currency code itself.

  Args:
    currency_code: An optional ISO 4217 currency code (e.g., 'USD', 'EUR',
      'JPY').
    default: The fallback symbol if `currency_code` is None or empty. Defaults
      to `constants.DEFAULT_CURRENCY_SYMBOL` ('$').

  Returns:
    The currency symbol string (e.g., '$', '€', '£', '¥').
  """
  if not isinstance(currency_code, str) or not currency_code.strip():
    return default

  code_upper = currency_code.strip().upper()
  symbol = babel_numbers.get_currency_symbol(code_upper, locale='en_US')
  if symbol:
    return symbol

  return code_upper


def is_valid_currency_code(currency_code: str | None) -> bool:
  """Returns True if the currency code is valid.

  Args:
    currency_code: An optional ISO 4217 currency code.

  Returns:
    True if the currency code is valid, False otherwise.
  """
  if not isinstance(currency_code, str) or not currency_code.strip():
    return False
  return babel_numbers.is_currency(currency_code.strip().upper())


def normalize_currency_code(currency_code: str | None) -> str | None:
  """Normalizes an ISO 4217 currency code to uppercase string, or None if invalid.

  Args:
    currency_code: An optional ISO 4217 currency code.

  Returns:
    The uppercase normalized ISO 4217 currency code if valid, or None.
  """
  if not isinstance(currency_code, str) or not currency_code.strip():
    return None
  return babel_numbers.normalize_currency(currency_code.strip().upper())
