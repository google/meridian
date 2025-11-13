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
from meridian.model.eda import eda_outcome
import numpy as np
import pandas as pd
import xarray as xr


class EdaOutcomeTest(parameterized.TestCase):
  _GEO_ARTIFACT = eda_outcome.PairwiseCorrArtifact(
      level=eda_outcome.AnalysisLevel.GEO,
      corr_matrix=xr.DataArray(
          data=np.array([[1, 2], [3, 4]]),
          dims=['var1', 'var2'],
          coords={
              'var1': ['var1_1', 'var1_2'],
              'var2': ['var2_1', 'var2_2'],
          },
      ),
      extreme_corr_var_pairs=pd.DataFrame({
          'var1': ['var1_1', 'var1_2'],
          'var2': ['var2_1', 'var2_2'],
          'correlation': [0.5, 0.6],
      }),
      extreme_corr_threshold=0.5,
  )
  _NATIONAL_ARTIFACT = eda_outcome.PairwiseCorrArtifact(
      level=eda_outcome.AnalysisLevel.NATIONAL,
      corr_matrix=xr.DataArray(
          data=np.array([[1, 2], [3, 4]]),
          dims=['var1', 'var2'],
          coords={
              'var1': ['var1_1', 'var1_2'],
              'var2': ['var2_1', 'var2_2'],
          },
      ),
      extreme_corr_var_pairs=pd.DataFrame({
          'var1': ['var1_1', 'var1_2'],
          'var2': ['var2_1', 'var2_2'],
          'correlation': [0.5, 0.6],
      }),
      extreme_corr_threshold=0.5,
  )
  _OVERALL_ARTIFACT = eda_outcome.PairwiseCorrArtifact(
      level=eda_outcome.AnalysisLevel.OVERALL,
      corr_matrix=xr.DataArray(
          data=np.array([[1, 2], [3, 4]]),
          dims=['var1', 'var2'],
          coords={
              'var1': ['var1_1', 'var1_2'],
              'var2': ['var2_1', 'var2_2'],
          },
      ),
      extreme_corr_var_pairs=pd.DataFrame({
          'var1': ['var1_1', 'var1_2'],
          'var2': ['var2_1', 'var2_2'],
          'correlation': [0.5, 0.6],
      }),
      extreme_corr_threshold=0.5,
  )
  _GEO_OUTCOME = eda_outcome.EDAOutcome(
      check_type=eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
      findings=[],
      analysis_artifacts=[_OVERALL_ARTIFACT, _GEO_ARTIFACT],
  )
  _NATIONAL_OUTCOME = eda_outcome.EDAOutcome(
      check_type=eda_outcome.EDACheckType.PAIRWISE_CORRELATION,
      findings=[],
      analysis_artifacts=[_NATIONAL_ARTIFACT],
  )

  def test_get_national_artifact(self):
    self.assertEqual(
        self._NATIONAL_OUTCOME.get_national_artifact, self._NATIONAL_ARTIFACT
    )
    self.assertIsNone(self._GEO_OUTCOME.get_national_artifact)

  def test_get_geo_artifact(self):
    self.assertEqual(self._GEO_OUTCOME.get_geo_artifact, self._GEO_ARTIFACT)
    self.assertIsNone(self._NATIONAL_OUTCOME.get_geo_artifact)


if __name__ == '__main__':
  absltest.main()
