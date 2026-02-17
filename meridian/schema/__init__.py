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

"""Module containing MMM schema library."""

try:
  # A quick check for schema dependencies.
  # If this fails, it's likely because meridian was installed without
  # `pip install google-meridian[schema]`.
  from mmm.v1.model.meridian import meridian_model_pb2
except ModuleNotFoundError as exc:
  raise ImportError(
      'Schema dependencies not found. Please install meridian with '
      '`pip install google-meridian[schema]`.'
  ) from exc

# pylint: disable=g-import-not-at-top
from meridian.schema import mmm_proto_generator
from meridian.schema import model_consumer
from meridian.schema import processors
from meridian.schema import serde
from meridian.schema import utils
