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
from meridian.model import aks
import numpy as np
from patsy import highlevel


class AKS(parameterized.TestCase):

  def test_mat2rot(self):
    band_mat = np.array([
        [999, 1, 2, 0, 0, 0, 0, 0],
        [10, 11, 12, 13, 0, 0, 0, 0],
        [20, 21, 22, 23, 24, 0, 0, 0],
        [0, 31, 32, 33, 34, 35, 0, 0],
        [0, 0, 42, 43, 44, 45, 46, 0],
        [0, 0, 0, 53, 54, 55, 56, 57],
        [0, 0, 0, 0, 64, 65, 66, 67],
        [0, 0, 0, 0, 0, 75, 76, 77],
    ])
    expected_rot_mat = np.array([
        [999, 1, 2],
        [11, 12, 13],
        [22, 23, 24],
        [33, 34, 35],
        [44, 45, 46],
        [55, 56, 57],
        [66, 67, 0],
        [77, 0, 0],
    ])
    np.testing.assert_array_equal(aks._mat2rot(band_mat), expected_rot_mat)

  def test_ldl(self):
    rot_mat = np.array([[2, 1, 0], [3, 1, 1], [2, 0, 0]])
    expected_ldl = np.array([[2, 0, 0], [3, 0, 1], [2, 0, 0]])
    np.testing.assert_array_almost_equal(aks._ldl(rot_mat), expected_ldl)

  @parameterized.named_parameters(
      dict(
          testcase_name='1d_rhs_mat',
          rhs_mat=np.array([1, 2, 3]),
          expected_result=np.array([[0], [0], [1]]),
      ),
      dict(
          testcase_name='2d_rhs_mat',
          rhs_mat=np.array([[1.0], [2.0], [3.0]]),
          expected_result=np.array([[[0.5]], [[-1.11022302e-16]], [[1.5]]]),
      ),
      dict(
          testcase_name='rhs_mat_none',
          rhs_mat=None,
          expected_result=np.array([
              [0.625, -0.25, 0.125],
              [-0.25, 0.5, -0.25],
              [0.125, -0.25, 0.625],
          ]),
      ),
  )
  def test_bandsolve(self, rhs_mat, expected_result):
    rot_mat = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]])
    result = aks._bandsolve(aks._mat2rot(rot_mat), rhs_mat)
    np.testing.assert_array_almost_equal(result, expected_result, decimal=3)

  @parameterized.named_parameters(
      dict(
          testcase_name='rot_mat_last_col_not_zero',
          rot_mat=np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]]),
          rhs_mat=np.array([1, 2, 3]),
          expected_error='rot_mat should be a rotated matrix!',
      ),
      dict(
          testcase_name='rot_mat_first_index_not_zero',
          rot_mat=np.array([[2, 1, 0], [1, 3, 1], [0, 1, 0]]),
          rhs_mat=np.array([1, 2, 3]),
          expected_error='rot_mat should be a rotated matrix!',
      ),
      dict(
          testcase_name='1d_rhs_mat_inconsistent_dimension',
          rot_mat=np.array([[2, 1, 0], [1, 3, 0], [0, 0, 0]]),
          rhs_mat=np.array([1, 2]),
          expected_error='Dimension problem!',
      ),
      dict(
          testcase_name='2d_rhs_mat_inconsistent_dimension',
          rot_mat=np.array([[2, 1, 0], [1, 3, 0], [0, 0, 0]]),
          rhs_mat=np.array([[1, 2], [1, 2]]),
          expected_error='Dimension problem!',
      ),
      dict(
          testcase_name='3d_rhs_mat',
          rot_mat=np.array([[2, 1, 0], [1, 3, 0], [0, 0, 0]]),
          rhs_mat=np.array([[[1, 2, 3]], [[1, 2, 3]], [[1, 2, 3]]]),
          expected_error='rhs_mat must either be a vector or a matrix',
      ),
  )
  def test_bandsolve_error(self, rot_mat, rhs_mat, expected_error):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        expected_error,
    ):
      aks._bandsolve(rot_mat, rhs_mat)

  def test_wridge_solver(self):
    x = np.linspace(0, 1, 10)
    y = (
        2 * x + 3 + np.random.normal(0, 0.01, 10)
    )  # Linear relationship plus noise
    degree = 1
    knots = np.array([])  # No internal knots, only the boundaries
    bs_cmd = (
        f"bs(x,knots=[{','.join(map(str, knots))}],degree={degree},include_intercept=True)-1"
    )

    xmat = highlevel.dmatrix(bs_cmd, {'x': x})
    ncol = xmat.shape[1]
    xx = xmat.T.dot(xmat)
    pen = 0.01  # Small pen for linear case
    w = (
        np.ones(xmat.shape[1] - degree - 1)
        if xmat.shape[1] - degree - 1 > 0
        else np.array([])
    )  # Example weights
    xx_rot = np.concat(
        [
            aks._mat2rot(xx + (1e-20 * np.identity(ncol))),
            np.zeros(ncol)[:, np.newaxis],
        ],
        axis=1,
    )
    xy = xmat.T.dot(y)
    old_par = np.ones(xmat.shape[1])
    par = aks._wridge_solver(
        xx_rot, xy, degree, pen, w, old_par, max_iter=100, tol=1e-6
    )

    # Expected parameters are close to [1, 2]
    self.assertIsNotNone(par)
    np.testing.assert_allclose(par[:2], [3, 5], rtol=0.1)

  @parameterized.named_parameters(
      dict(
          testcase_name='diff_1',
          w=np.array([1, 2, 3]),
          diff=1,
          expected_result=np.array([
              [1, -1],
              [3, -2],
              [5, -3],
              [3, 0],
          ]),
      ),
      dict(
          testcase_name='diff_2',
          w=np.array([1, 2, 3]),
          diff=2,
          expected_result=np.array([
              [1, -2, 1],
              [6, -6, 2],
              [12, -10, 3],
              [14, -6, 0],
              [3, 0, 0],
          ]),
      ),
  )
  def test_band_weight(self, w, diff, expected_result):
    result = aks._band_weight(w, diff)
    np.testing.assert_array_almost_equal(result, expected_result)

  def test_aspline(self):
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5, 6])
    knots = np.array([1, 3])
    result = aks.aspline(x, y, knots, pen=np.array([0, 5]))
    np.testing.assert_allclose(
        result['sel'],
        [np.array([1.0, 0.0]), np.array([9.99999978e-01, 2.17678349e-13])],
    )
    np.testing.assert_allclose(
        result['knots_sel'], [np.array([1]), np.array([1])]
    )
    np.testing.assert_allclose(
        result['par'],
        [np.array([1.0, 0.0, 2.0, 6.0]), np.array([1.0, 0.0, 2.0, 6.0])],
    )
    np.testing.assert_allclose(
        result['sel_mat'], np.array([[1.0, 1.0], [0.0, 0.0]])
    )
    np.testing.assert_allclose(
        result['aic'], np.array([1.6348564e28, 1.6348564e28])
    )
    np.testing.assert_allclose(
        result['bic'], np.array([1.6348564e28, 1.6348564e28])
    )
    np.testing.assert_allclose(
        result['ebic'], np.array([1.6348564e28, 1.6348564e28])
    )


if __name__ == '__main__':
  absltest.main()
