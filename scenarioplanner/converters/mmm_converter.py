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

"""Converts a fully specified trained model and analysis output.

A fully specified trained model and its analyses are in its canonical proto
form. This module provides the API for its conversion to secondary forms
(e.g. flat CSV tables collated in a Sheets file) for immediate consumption
(e.g. as data sources for a Looker Studio dashboard).
"""

import abc
from collections.abc import Mapping
from typing import Generic, TypeVar

from mmm.v1 import mmm_pb2 as pb
from scenarioplanner.converters import mmm


__all__ = ['ModelConverter']


# The output type of a converter.
O = TypeVar('O')


class ModelConverter(abc.ABC, Generic[O]):
  """Converts a fully specified trained model to secondary form(s) `O`.

  Attributes:
    mmm: An `Mmm` proto containing a trained model and its optional analyses.
  """

  def __init__(
      self,
      mmm_proto: pb.Mmm,
  ):
    self._mmm = mmm.Mmm(mmm_proto)

  @property
  def mmm(self) -> mmm.Mmm:
    return self._mmm

  @abc.abstractmethod
  def __call__(self, **kwargs) -> Mapping[str, O]:
    """Converts bound `MmmOutput` proto to named secondary form(s) `O`."""
    raise NotImplementedError()
