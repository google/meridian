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

from collections.abc import Collection
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from meridian import constants
from meridian.data import input_data
from meridian.data import test_utils
from meridian.model import knots
import numpy as np


class L1DistanceWeightsTest(parameterized.TestCase):
  """Tests for knots.l1_distance_weights()."""

  @parameterized.named_parameters(
      (
          "knots at 0 and 2",
          np.array([0, 2]),
          np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]),
      ),
      (
          "knots at 0 and 1",
          np.array([0, 1]),
          np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
      ),
      (
          "knots at 1 and 2",
          np.array([1, 2]),
          np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
      ),
      ("full knots", np.array([0, 1, 2]), np.eye(3)),
  )
  def test_correctness(self, knot_locations, expected):
    """Check correctness."""
    knot_locations = knot_locations.astype(np.int32)
    result = knots.l1_distance_weights(3, knot_locations)
    self.assertTrue(np.allclose(result, expected))

  @parameterized.named_parameters(
      ("knots not 1D", np.array([[1, 5]]), "must be one-dimensional"),
      ("knots not sorted", np.array([5, 1]), "must be sorted"),
      ("at least two knots", np.array([5]), "must be greater than 1"),
      ("knots not unique", np.array([1, 5, 5]), "must be unique"),
      ("negative knot", np.array([-1, 5]), "must be positive"),
      ("all negative knots", np.array([-5, -1]), "must be positive"),
      ("knot too big", np.array([5, 10]), "must be less than `n_times`"),
      ("all knots too big", np.array([10, 11]), "must be less than `n_times`"),
  )
  def test_error_checking(self, knot_locations, error_str):
    """Tests that proper errors are raised with impermissible arguments."""
    knot_locations = knot_locations.astype(np.int32)
    with self.assertRaisesRegex(ValueError, error_str):
      knots.l1_distance_weights(10, knot_locations)


class GetKnotInfoTest(parameterized.TestCase):
  """Tests for knots.get_knot_info()."""

  @parameterized.named_parameters(
      dict(
          testcase_name="list_no_right_endpoint",
          n_times=5,
          knots_param=[0, 2, 3],
          is_national=False,
          expected_n_knots=3,
          expected_knot_locations=[0, 2, 3],
          expected_weights=np.array([
              [1, 0.5, 0, 0, 0],
              [0, 0.5, 1, 0, 0],
              [0, 0, 0, 1, 1],
          ]),
      ),
      dict(
          testcase_name="list_no_left_endpoint",
          n_times=5,
          knots_param=[2, 4],
          is_national=False,
          expected_n_knots=2,
          expected_knot_locations=[2, 4],
          expected_weights=np.array([
              [1, 1, 1, 0.5, 0],
              [0, 0, 0, 0.5, 1],
          ]),
      ),
      dict(
          testcase_name="list_no_endpoints",
          n_times=5,
          knots_param=[2, 3],
          is_national=False,
          expected_n_knots=2,
          expected_knot_locations=[2, 3],
          expected_weights=np.array([
              [1, 1, 1, 0, 0],
              [0, 0, 0, 1, 1],
          ]),
      ),
      dict(
          testcase_name="list_both_endpoints",
          n_times=5,
          knots_param=[0, 2, 4],
          is_national=False,
          expected_n_knots=3,
          expected_knot_locations=[0, 2, 4],
          expected_weights=np.array([
              [1, 0.5, 0, 0, 0],
              [0, 0.5, 1, 0.5, 0],
              [0, 0, 0, 0.5, 1],
          ]),
      ),
      dict(
          testcase_name="list_with_gap",
          n_times=7,
          knots_param=[0, 1, 6],
          is_national=False,
          expected_n_knots=3,
          expected_knot_locations=[0, 1, 6],
          expected_weights=np.array([
              [1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0.8, 0.6, 0.4, 0.2, 0],
              [0, 0, 0.2, 0.4, 0.6, 0.8, 1],
          ]),
      ),
      dict(
          testcase_name="int",
          n_times=6,
          knots_param=3,
          is_national=False,
          expected_n_knots=3,
          expected_knot_locations=[0, 2, 5],
          expected_weights=np.array([
              [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.5, 1.0, 0.6666667, 0.33333334, 0.0],
              [0.0, 0.0, 0.0, 0.33333334, 0.6666667, 1.0],
          ]),
      ),
      dict(
          testcase_name="int_same_as_n_times",
          n_times=6,
          knots_param=6,
          is_national=False,
          expected_n_knots=6,
          expected_knot_locations=[0, 1, 2, 3, 4, 5],
          expected_weights=np.array([
              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
          ]),
      ),
      dict(
          testcase_name="none",
          n_times=4,
          knots_param=None,
          is_national=False,
          expected_n_knots=4,
          expected_knot_locations=[0, 1, 2, 3],
          expected_weights=np.array([
              [1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0],
          ]),
      ),
      dict(
          testcase_name="none_and_national",
          n_times=6,
          knots_param=None,
          is_national=True,
          expected_n_knots=1,
          expected_knot_locations=[0],
          expected_weights=np.array([
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          ]),
      ),
  )
  def test_sample_list_of_knots(
      self,
      n_times,
      knots_param,
      is_national,
      expected_n_knots,
      expected_knot_locations,
      expected_weights,
  ):
    match knots.get_knot_info(
        n_times=n_times, knots=knots_param, is_national=is_national
    ):
      case knots.KnotInfo(n_knots, knot_locations, weights):
        self.assertEqual(n_knots, expected_n_knots)
        self.assertTrue((knot_locations == expected_knot_locations).all())
        self.assertTrue(np.allclose(weights, expected_weights))
      case _:
        self.fail("Unexpected return type.")

  @parameterized.named_parameters(
      dict(
          testcase_name="too_many",
          knots_arg=201,
          is_national=False,
          msg=(
              "The number of knots (201) cannot be greater than the number of"
              " time periods in the kpi (200)."
          ),
      ),
      dict(
          testcase_name="less_than_one",
          knots_arg=0,
          is_national=False,
          msg="If knots is an integer, it must be at least 1.",
      ),
      dict(
          testcase_name="same_as_n_times_national",
          knots_arg=200,
          is_national=True,
          msg=(
              "Number of knots (200) must be less than number of time periods"
              " (200) in a nationally aggregated model."
          ),
      ),
      dict(
          testcase_name="negative",
          knots_arg=[-2, 17],
          is_national=False,
          msg="Knots must be all non-negative.",
      ),
      dict(
          testcase_name="too_large",
          knots_arg=[3, 202],
          is_national=False,
          msg="Knots must all be less than the number of time periods.",
      ),
  )
  def test_wrong_knots_fails(
      self,
      knots_arg: int | Collection[int] | None,
      is_national: bool,
      msg: str,
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        msg,
    ):
      knots.get_knot_info(n_times=200, knots=knots_arg, is_national=is_national)

  @parameterized.named_parameters(
      dict(
          testcase_name="geo",
          is_national=False,
          n_times=5,
          expected_n_knots=5,
          expected_knot_locations=[0, 1, 2, 3, 4],
      ),
      dict(
          testcase_name="national",
          is_national=True,
          n_times=5,
          expected_n_knots=1,
          expected_knot_locations=[0],
      ),
  )
  def test_get_knot_info_aks_returns_empty_falls_back_to_defaults(
      self,
      is_national,
      n_times,
      expected_n_knots,
      expected_knot_locations,
  ):
    """Tests that if AKS returns empty knots, we fall back to default logic."""
    mock_result = mock.create_autospec(knots.AKSResult, instance=True)
    mock_result.knots = np.array([], dtype=int)
    mock_aks = self.enter_context(
        mock.patch.object(
            knots.AKS, "automatic_knot_selection", autospec=True, spec_set=True
        )
    )
    mock_aks.return_value = mock_result

    info = knots.get_knot_info(
        n_times=n_times,
        knots=None,
        enable_aks=True,
        data=mock.create_autospec(
            input_data.InputData, instance=True, spec_set=True
        ),
        is_national=is_national,
    )

    self.assertEqual(info.n_knots, expected_n_knots)
    np.testing.assert_array_equal(info.knot_locations, expected_knot_locations)

  def test_get_knot_info_aks_returns_knots_uses_them(self):
    mock_result = mock.create_autospec(
        knots.AKSResult,
        instance=True,
    )
    mock_result.knots = np.array([2, 4], dtype=int)
    mock_aks = self.enter_context(
        mock.patch.object(
            knots.AKS,
            "automatic_knot_selection",
            autospec=True,
            spec_set=True,
        )
    )
    mock_aks.return_value = mock_result

    info = knots.get_knot_info(
        n_times=10,
        knots=None,
        enable_aks=True,
        data=mock.create_autospec(
            input_data.InputData, instance=True, spec_set=True
        ),
    )

    self.assertEqual(info.n_knots, 2)
    np.testing.assert_array_equal(info.knot_locations, [2, 4])


class AKSTest(parameterized.TestCase):
  """Tests for knots.AKS class."""

  @parameterized.named_parameters(
      dict(
          testcase_name="when_min_gt_initial_knots",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=1,
                  n_times=2,
                  n_media_times=10,
                  n_controls=2,
                  n_media_channels=5,
              ),
              "non_revenue",
          ),
          expected_error=(
              "The minimum number of internal knots cannot be greater than the"
              " total number of initial knots."
          ),
      ),
      dict(
          testcase_name="when_max_lt_min_media",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=2,
                  n_times=10,
                  n_media_times=10,
                  n_controls=2,
                  n_media_channels=5,
              ),
              "non_revenue",
          ),
          expected_error=(
              "The maximum number of internal knots cannot be less than the"
              " minimum number of internal knots."
          ),
      ),
      dict(
          testcase_name="when_max_lt_min_rf",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=2,
                  n_times=10,
                  n_media_times=10,
                  n_controls=2,
                  n_rf_channels=4,
                  n_media_channels=1,
              ),
              "non_revenue",
          ),
          expected_error=(
              "The maximum number of internal knots cannot be less than the"
              " minimum number of internal knots."
          ),
      ),
      dict(
          testcase_name="when_max_lt_min_organic_media",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=2,
                  n_times=10,
                  n_media_times=10,
                  n_controls=2,
                  n_organic_media_channels=4,
                  n_media_channels=1,
              ),
              "non_revenue",
          ),
          expected_error=(
              "The maximum number of internal knots cannot be less than the"
              " minimum number of internal knots."
          ),
      ),
      dict(
          testcase_name="when_max_lt_min_organic_rf",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=2,
                  n_times=10,
                  n_media_times=10,
                  n_controls=2,
                  n_organic_rf_channels=4,
                  n_media_channels=1,
              ),
              "non_revenue",
          ),
          expected_error=(
              "The maximum number of internal knots cannot be less than the"
              " minimum number of internal knots."
          ),
      ),
      dict(
          testcase_name="when_max_lt_min_non_media",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=2,
                  n_times=10,
                  n_media_times=10,
                  n_controls=2,
                  n_non_media_channels=4,
                  n_media_channels=1,
              ),
              "non_revenue",
          ),
          expected_error=(
              "The maximum number of internal knots cannot be less than the"
              " minimum number of internal knots."
          ),
      ),
  )
  def test_aks_internal_knots_guardrail_raises(self, data, expected_error):
    aks_obj = knots.AKS(data)

    with self.assertRaisesWithLiteralMatch(ValueError, expected_error):
      aks_obj.automatic_knot_selection()

  @parameterized.named_parameters(
      dict(
          testcase_name="20_geos",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=20,
                  n_times=117,
                  n_media_times=117,
                  n_controls=2,
                  n_media_channels=5,
              ),
              "non_revenue",
          ),
          expected_knots=[
              0,
              11,
              14,
              38,
              39,
              41,
              43,
              45,
              48,
              50,
              55,
              87,
              89,
              90,
              116,
          ],
      ),
      dict(
          testcase_name="national_geos",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=1,
                  n_times=117,
                  n_media_times=117,
                  n_controls=2,
                  n_media_channels=5,
              ),
              "non_revenue",
          ),
          expected_knots=[
              0,
              1,
              2,
              3,
              5,
              6,
              15,
              16,
              19,
              20,
              21,
              24,
              25,
              27,
              28,
              30,
              33,
              35,
              38,
              39,
              42,
              43,
              44,
              47,
              53,
              54,
              59,
              60,
              64,
              65,
              66,
              67,
              68,
              72,
              73,
              74,
              78,
              79,
              81,
              83,
              85,
              87,
              89,
              90,
              91,
              93,
              95,
              96,
              98,
              99,
              102,
              103,
              104,
              114,
              116,
          ],
      ),
      dict(
          testcase_name="5_geos",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=5,
                  n_times=117,
                  n_media_times=117,
                  n_controls=2,
                  n_media_channels=5,
              ),
              "non_revenue",
          ),
          expected_knots=[
              0,
              4,
              17,
              20,
              27,
              28,
              31,
              32,
              33,
              41,
              42,
              50,
              54,
              59,
              61,
              76,
              77,
              78,
              81,
              116,
          ],
      ),
      dict(
          testcase_name="50_geos",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=50,
                  n_times=117,
                  n_media_times=117,
                  n_controls=2,
                  n_media_channels=5,
              ),
              "non_revenue",
          ),
          expected_knots=[0, 2, 7, 24, 25, 38, 39, 49, 114, 116],
      ),
      dict(
          testcase_name="50_times",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=20,
                  n_times=50,
                  n_media_times=50,
                  n_controls=2,
                  n_media_channels=5,
              ),
              "non_revenue",
          ),
          expected_knots=[0, 1, 5, 13, 15, 16, 23, 27, 31, 32, 38, 42, 49],
      ),
      dict(
          testcase_name="200_times",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=20,
                  n_times=200,
                  n_media_times=200,
                  n_controls=2,
                  n_media_channels=5,
              ),
              "non_revenue",
          ),
          expected_knots=[
              0,
              4,
              10,
              12,
              19,
              20,
              22,
              29,
              30,
              33,
              34,
              35,
              42,
              43,
              48,
              49,
              57,
              58,
              61,
              63,
              65,
              66,
              73,
              77,
              78,
              83,
              84,
              105,
              106,
              107,
              111,
              121,
              134,
              135,
              138,
              143,
              151,
              157,
              159,
              160,
              166,
              173,
              175,
              179,
              195,
              196,
              197,
              199,
          ],
      ),
      dict(
          testcase_name="flat_kpi",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=1,
                  n_times=50,
                  n_media_times=50,
                  n_controls=2,
                  n_media_channels=5,
                  kpi_data_pattern="flat",
              ),
              "non_revenue",
          ),
          expected_knots=[0, 17, 25, 49],
      ),
      dict(
          testcase_name="seasonal",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=1,
                  n_times=50,
                  n_media_times=50,
                  n_controls=2,
                  n_media_channels=5,
                  kpi_data_pattern="seasonal",
              ),
              "non_revenue",
          ),
          expected_knots=[
              0,
              1,
              4,
              5,
              9,
              10,
              14,
              16,
              17,
              21,
              22,
              29,
              30,
              33,
              35,
              36,
              40,
              41,
              45,
              47,
              49,
          ],
      ),
      dict(
          testcase_name="peak",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=1,
                  n_times=50,
                  n_media_times=50,
                  n_controls=2,
                  n_media_channels=5,
                  kpi_data_pattern="peak",
              ),
              "non_revenue",
          ),
          expected_knots=[0, 24, 25, 26, 49],
      ),
      dict(
          testcase_name="minimal_initial_knots",
          data=test_utils.sample_input_data_from_dataset(
              test_utils.random_dataset(
                  n_geos=20,
                  n_times=15,
                  n_media_times=15,
                  n_controls=4,
                  n_media_channels=7,
              ),
              "non_revenue",
          ),
          expected_knots=[0, 3, 14],
      ),
  )
  def test_aks(self, data: input_data.InputData, expected_knots: list[int]):
    aks_obj = knots.AKS(data)
    aks_result = aks_obj.automatic_knot_selection()
    actual_knots, actual_model = aks_result.knots, aks_result.model
    self.assertListEqual(actual_knots.tolist(), expected_knots)
    self.assertIsNotNone(actual_model)

  def test_aks_no_input_data_raises(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "If enable_aks is true then input data must be provided.",
    ):
      knots.get_knot_info(n_times=200, knots=None, enable_aks=True)

  def test_aks_returns_correct_knot_info(self):
    data, expected_knot_info = (
        test_utils.sample_input_data_for_aks_with_expected_knot_info()
    )
    actual_knot_info = knots.get_knot_info(
        n_times=117, knots=None, enable_aks=True, data=data
    )
    self.assertEqual(actual_knot_info.n_knots, expected_knot_info.n_knots)
    np.testing.assert_equal(
        actual_knot_info.knot_locations, expected_knot_info.knot_locations
    )
    np.testing.assert_equal(
        actual_knot_info.weights, expected_knot_info.weights
    )

  def test_aspline(self):
    x = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    y = np.array([10, 2, 3, 4, 50, 6, 30, 4, 5, 6, 3, 4, 5, 6, 6])
    input_knots = np.array([1.0, 3, 15])

    result = knots.AKS(mock.MagicMock()).aspline(
        x, y, input_knots, np.array([-10, 50, 0])
    )
    np.testing.assert_allclose(
        result[constants.SELECTION_COEFS],
        [
            np.array([5.17673881e-11, 1.00000000e00, 1.26217745e-19]),
            np.array([1.41647864e-12, 1.00000000e00, 3.15544362e-20]),
            np.array([1.0, 1.0, 1.0]),
        ],
    )
    self.assertLen(result[constants.KNOTS_SELECTED], 3)
    np.testing.assert_allclose(
        result[constants.KNOTS_SELECTED][0], np.array([3.0])
    )
    np.testing.assert_allclose(
        result[constants.KNOTS_SELECTED][1], np.array([3.0])
    )
    np.testing.assert_allclose(
        result[constants.KNOTS_SELECTED][2], np.array([1.0, 3.0, 15.0])
    )
    np.testing.assert_allclose(
        result[constants.REGRESSION_COEFS],
        [
            np.array([0.0, 2.68338214, 0.0, 18.35816252, 2.3545022]),
            np.array([0.0, 2.68338214, 0.0, 18.35816252, 2.3545022]),
            np.array([10.0, -1.35407537, 20.77037686, 0.04119194, 0.0]),
        ],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        result[constants.SELECTED_MATRIX],
        np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    np.testing.assert_allclose(
        result[constants.AIC], np.array([6.07028377, 6.07028377, 10.06504776])
    )
    np.testing.assert_allclose(
        result[constants.BIC], np.array([8.194435, 8.194435, 13.60529853])
    )
    np.testing.assert_allclose(
        result[constants.EBIC],
        np.array([12.79960525, 12.79960525, 13.60529853]),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="x_2d",
          x=np.array([[1], [2]]),
          y=np.array([10, 2, 3, 4, 50, 6, 30, 4, 5, 6, 3, 4, 5, 6, 6]),
          expected_error=(
              "Provided x and y args for aspline must both be 1 dimensional!"
          ),
      ),
      dict(
          testcase_name="y_2d",
          x=np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
          y=np.array([[10], [11]]),
          expected_error=(
              "Provided x and y args for aspline must both be 1 dimensional!"
          ),
      ),
  )
  def test_aspline_invalid_args(self, x, y, expected_error):
    with self.assertRaisesRegex(ValueError, expected_error):
      knots.AKS(mock.MagicMock()).aspline(
          x, y, np.array([1.0, 3, 15]), np.array([-10, 50, 0])
      )

  def test_user_provided_base_penalty(self):
    data = test_utils.sample_input_data_from_dataset(
        test_utils.random_dataset(
            n_geos=50,
            n_times=117,
            n_media_times=117,
            n_controls=2,
            n_media_channels=5,
        ),
        "non_revenue",
    )
    aks_obj = knots.AKS(data)
    base_penalty = np.array([0.5] * 100)
    aks_result = aks_obj.automatic_knot_selection(base_penalty=base_penalty)
    actual_knots, _ = aks_result.knots, aks_result.model
    self.assertListEqual(
        actual_knots.tolist(),
        [
            0,
            2,
            7,
            8,
            10,
            11,
            14,
            15,
            16,
            17,
            21,
            22,
            24,
            25,
            26,
            30,
            31,
            34,
            35,
            36,
            38,
            39,
            40,
            42,
            43,
            45,
            49,
            52,
            54,
            57,
            60,
            61,
            64,
            67,
            68,
            69,
            72,
            73,
            74,
            79,
            81,
            83,
            84,
            85,
            86,
            89,
            93,
            94,
            95,
            98,
            101,
            102,
            103,
            104,
            105,
            107,
            109,
            110,
            113,
            114,
            116,
        ],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="min_equals_max",
          min_internal_knots=8,
          max_internal_knots=8,
          expected_knots=[0, 2, 7, 24, 25, 38, 39, 49, 114, 116],
      ),
      dict(
          testcase_name="min_lt_max_",
          min_internal_knots=2,
          max_internal_knots=15,
          expected_knots=[0, 2, 7, 24, 25, 38, 39, 49, 114, 116],
      ),
  )
  def test_aks_user_provided_min_max_internal_knots(
      self, min_internal_knots, max_internal_knots, expected_knots
  ):
    data = test_utils.sample_input_data_from_dataset(
        test_utils.random_dataset(
            n_geos=50,
            n_times=117,
            n_media_times=117,
            n_controls=2,
            n_media_channels=5,
        ),
        "non_revenue",
    )
    aks_obj = knots.AKS(data)
    aks_result = aks_obj.automatic_knot_selection(
        min_internal_knots=min_internal_knots,
        max_internal_knots=max_internal_knots,
    )
    actual_knots, _ = aks_result.knots, aks_result.model
    self.assertListEqual(actual_knots.tolist(), expected_knots)

  def test_aks_user_provided_min_max_internal_knots_raises(self):
    data = test_utils.sample_input_data_from_dataset(
        test_utils.random_dataset(
            n_geos=50,
            n_times=117,
            n_media_times=117,
            n_controls=2,
            n_media_channels=5,
        ),
        "non_revenue",
    )
    aks_obj = knots.AKS(data)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The range [10, 10] does not contain any of the available knot lengths:"
        " [0, 1, 4, 6, 8, 22, 39, 44, 48, 51, 55, 60, 62, 69, 70, 79, 80, 84,"
        " 87, 89]",
    ):
      aks_obj.automatic_knot_selection(
          min_internal_knots=10,
          max_internal_knots=10,
      )


if __name__ == "__main__":
  absltest.main()
