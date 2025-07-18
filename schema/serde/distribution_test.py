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

from collections import abc
import pickle
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
from meridian import constants
from meridian.model import prior_distribution as pd
from mmm.v1.model.meridian import meridian_model_pb2 as meridian_pb
from schema.serde import distribution
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.util.protobuf import compare
from tensorflow.core.framework import tensor_shape_pb2  # pylint: disable=g-direct-tensorflow-import


_DEFAULT_PRIORS_PROTO = text_format.Parse(
    """
    knot_values {
      name: "knot_values"
      normal {
        locs: 0
        scales: 5
      }
    }
    tau_g_excl_baseline {
      name: "tau_g_excl_baseline"
      normal {
        locs: 0
        scales: 5
      }
    }
    beta_m {
      name: "beta_m"
      half_normal {
        scales: 5
      }
    }
    beta_rf {
      name: "beta_rf"
      half_normal {
        scales: 5
      }
    }
    beta_om {
      name: "beta_om"
      half_normal {
        scales: 5
      }
    }
    beta_orf {
      name: "beta_orf"
      half_normal {
        scales: 5
      }
    }
    eta_m {
      name: "eta_m"
      half_normal {
        scales: 1
      }
    }
    eta_rf {
      name: "eta_rf"
      half_normal {
        scales: 1
      }
    }
    eta_om {
      name: "eta_om"
      half_normal {
        scales: 1
      }
    }
    eta_orf {
      name: "eta_orf"
      half_normal {
        scales: 1
      }
    }
    gamma_c {
      name: "gamma_c"
      normal {
        locs: 0
        scales: 5
      }
    }
    gamma_n {
      name: "gamma_n"
      normal {
        locs: 0
        scales: 5
      }
    }
    xi_c {
      name: "xi_c"
      half_normal {
        scales: 5
      }
    }
    xi_n {
      name: "xi_n"
      half_normal {
        scales: 5
      }
    }
    alpha_m {
      name: "alpha_m"
      uniform {
        low: 0
        high: 1
      }
    }
    alpha_rf {
      name: "alpha_rf"
      uniform {
        low: 0
        high: 1
      }
    }
    alpha_om {
      name: "alpha_om"
      uniform {
        low: 0
        high: 1
      }
    }
    alpha_orf {
      name: "alpha_orf"
      uniform {
        low: 0
        high: 1
      }
    }
    ec_m {
      name: "ec_m"
      truncated_normal {
        locs: 0.80
        scales: 0.80
        low: 0.10
        high: 10
      }
    }
    ec_rf {
      name: "ec_rf"
      transformed {
        distribution {
          name: "LogNormal"
          log_normal {
            locs: 0.70
            scales: 0.40
          }
        }
        bijector {
          name: "shift"
          shift {
            shifts: 0.10
          }
        }
      }
    }
    ec_om {
      name: "ec_om"
      truncated_normal {
        locs: 0.80
        scales: 0.80
        low: 0.10
        high: 10
      }
    }
    ec_orf {
      name: "ec_orf"
      transformed {
        distribution {
          name: "LogNormal"
          log_normal {
            locs: 0.70
            scales: 0.40
          }
        }
        bijector {
          name: "shift"
          shift {
            shifts: 0.10
          }
        }
      }
    }
    slope_m {
      name: "slope_m"
      deterministic {
        locs: 1
      }
    }
    slope_rf {
      name: "slope_rf"
      log_normal {
        locs: 0.70
        scales: 0.40
      }
    }
    slope_om {
      name: "slope_om"
      deterministic {
        locs: 1
      }
    }
    slope_orf {
      name: "slope_orf"
      log_normal {
        locs: 0.70
        scales: 0.40
      }
    }
    sigma {
      name: "sigma"
      half_normal {
        scales: 5
      }
    }
    roi_m {
      name: "roi_m"
      log_normal {
        locs: 0.20
        scales: 0.90
      }
    }
    roi_rf {
      name: "roi_rf"
      log_normal {
        locs: 0.20
        scales: 0.90
      }
    }
    mroi_m {
      name: "mroi_m"
      log_normal {
        locs: 0.0
        scales: 0.5
      }
    }
    mroi_rf {
      name: "mroi_rf"
      log_normal {
        locs: 0.0
        scales: 0.5
      }
    }
    contribution_m {
      name: "contribution_m"
      beta {
        alpha: 1.00
        beta: 99.00
      }
    }
    contribution_rf {
      name: "contribution_rf"
      beta {
        alpha: 1.00
        beta: 99.00
      }
    }
    contribution_om {
      name: "contribution_om"
      beta {
        alpha: 1.00
        beta: 99.00
      }
    }
    contribution_orf {
      name: "contribution_orf"
      beta {
        alpha: 1.00
        beta: 99.00
      }
    }
    contribution_n {
      name: "contribution_n"
      truncated_normal {
        locs: 0.00
        scales: 0.10
        low: -1.00
        high: 1.00
      }
    }
    """,
    meridian_pb.PriorDistributions(),
)


def _make_tensor_shape_proto(
    dims: Sequence[int],
) -> tensor_shape_pb2.TensorShapeProto:
  tensor_shape = tensor_shape_pb2.TensorShapeProto()
  for dim in dims:
    tensor_shape.dim.append(tensor_shape_pb2.TensorShapeProto.Dim(size=dim))
  return tensor_shape


class DistributionSerdeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.serde = distribution.DistributionSerde()

  @parameterized.named_parameters(
      dict(
          testcase_name='scalar_deterministic',
          dist=tfp.distributions.Deterministic(
              0.7, name='scalar_deterministic'
          ),
          expected_dist_proto=meridian_pb.Distribution(
              name='scalar_deterministic',
              deterministic=meridian_pb.Distribution.Deterministic(locs=[0.7]),
          ),
      ),
      dict(
          testcase_name='list_deterministic',
          dist=tfp.distributions.Deterministic(
              [1.0, 1.1, 1.2, 1.3, 1.4, 1.5], name='list_deterministic'
          ),
          expected_dist_proto=meridian_pb.Distribution(
              name='list_deterministic',
              deterministic=meridian_pb.Distribution.Deterministic(
                  locs=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
              ),
          ),
      ),
      dict(
          testcase_name='scalar_log_normal',
          dist=tfp.distributions.LogNormal(1.0, 0.4, name='scalar_log_normal'),
          expected_dist_proto=meridian_pb.Distribution(
              name='scalar_log_normal',
              log_normal=meridian_pb.Distribution.LogNormal(
                  locs=[1.0], scales=[0.4]
              ),
          ),
      ),
      dict(
          testcase_name='list_log_normal',
          dist=tfp.distributions.LogNormal(
              [0.1, 0.2, 0.3], [1.1, 1.2, 1.3], name='list_log_normal'
          ),
          expected_dist_proto=meridian_pb.Distribution(
              name='list_log_normal',
              log_normal=meridian_pb.Distribution.LogNormal(
                  locs=[0.1, 0.2, 0.3], scales=[1.1, 1.2, 1.3]
              ),
          ),
      ),
      dict(
          testcase_name='batch_broadcast',
          dist=tfp.distributions.BatchBroadcast(
              tfp.distributions.HalfNormal(5.0), 3, name='batch_broadcast'
          ),
          expected_dist_proto=meridian_pb.Distribution(
              name='batch_broadcast',
              batch_broadcast=meridian_pb.Distribution.BatchBroadcast(
                  distribution=meridian_pb.Distribution(
                      name='HalfNormal',
                      half_normal=meridian_pb.Distribution.HalfNormal(
                          scales=[5.0]
                      ),
                  ),
                  batch_shape=_make_tensor_shape_proto(dims=[3]),
              ),
          ),
      ),
      dict(
          testcase_name='transformed_distribution',
          dist=tfp.distributions.TransformedDistribution(
              tfp.distributions.LogNormal(0.7, 0.4),
              tfp.bijectors.Shift(0.1),
              name='transformed_distribution',
          ),
          expected_dist_proto=meridian_pb.Distribution(
              name='transformed_distribution',
              transformed=meridian_pb.Distribution.Transformed(
                  distribution=meridian_pb.Distribution(
                      name='LogNormal',
                      log_normal=meridian_pb.Distribution.LogNormal(
                          locs=[0.7], scales=[0.4]
                      ),
                  ),
                  bijector=meridian_pb.Distribution.Bijector(
                      name='shift',
                      shift=meridian_pb.Distribution.Bijector.Shift(
                          shifts=[0.1]
                      ),
                  ),
              ),
          ),
      ),
  )
  def test_to_distribution_proto(
      self,
      dist: tfp.distributions.Distribution,
      expected_dist_proto: meridian_pb.Distribution,
  ):
    compare.assertProto2Equal(
        self,
        distribution._to_distribution_proto(dist),
        expected_dist_proto,
        precision=2,
    )

  def test_serialize_default_priors(self):
    prior_dist_proto = self.serde.serialize(pd.PriorDistribution())

    compare.assertProto2Equal(
        self,
        prior_dist_proto,
        _DEFAULT_PRIORS_PROTO,
        precision=2,
    )

  def test_serialize_broadcast_priors(self):
    broadcast_priors = pd.PriorDistribution().broadcast(
        n_geos=10,
        n_media_channels=6,
        n_rf_channels=4,
        n_organic_media_channels=2,
        n_organic_rf_channels=1,
        n_non_media_channels=5,
        n_controls=3,
        unique_sigma_for_each_geo=True,
        n_knots=5,
        is_national=False,
        set_total_media_contribution_prior=False,
        kpi=1.0,
        total_spend=np.array([]),
    )

    broadcast_dist_proto = self.serde.serialize(broadcast_priors)

    expected_prior_dist_proto = text_format.Parse(
        """
        knot_values {
          name: "knot_values"
          batch_broadcast {
            distribution {
              name: "knot_values"
              normal {
                locs: 0
                scales: 5
              }
            }
            batch_shape {
              dim {
                size: 5
              }
            }
          }
        }
        tau_g_excl_baseline {
          name: "tau_g_excl_baseline"
          batch_broadcast {
            distribution {
              name: "tau_g_excl_baseline"
              normal {
                locs: 0
                scales: 5
              }
            }
            batch_shape {
              dim {
                size: 9
              }
            }
          }
        }
        beta_m {
          name: "beta_m"
          batch_broadcast {
            distribution {
              name: "beta_m"
              half_normal {
                scales: 5
              }
            }
            batch_shape {
              dim {
                size: 6
              }
            }
          }
        }
        beta_rf {
          name: "beta_rf"
          batch_broadcast {
            distribution {
              name: "beta_rf"
              half_normal {
                scales: 5
              }
            }
            batch_shape {
              dim {
                size: 4
              }
            }
          }
        }
        beta_om {
          name: "beta_om"
          batch_broadcast {
            distribution {
              name: "beta_om"
              half_normal {
                scales: 5
              }
            }
            batch_shape {
              dim {
                size: 2
              }
            }
          }
        }
        beta_orf {
          name: "beta_orf"
          batch_broadcast {
            distribution {
              name: "beta_orf"
              half_normal {
                scales: 5
              }
            }
            batch_shape {
              dim {
                size: 1
              }
            }
          }
        }
        eta_m {
          name: "eta_m"
          batch_broadcast {
            distribution {
              name: "eta_m"
              half_normal {
                scales: 1
              }
            }
            batch_shape {
              dim {
                size: 6
              }
            }
          }
        }
        eta_rf {
          name: "eta_rf"
          batch_broadcast {
            distribution {
              name: "eta_rf"
              half_normal {
                scales: 1
              }
            }
            batch_shape {
              dim {
                size: 4
              }
            }
          }
        }
        eta_om {
          name: "eta_om"
          batch_broadcast {
            distribution {
              name: "eta_om"
              half_normal {
                scales: 1
              }
            }
            batch_shape {
              dim {
                size: 2
              }
            }
          }
        }
        eta_orf {
          name: "eta_orf"
          batch_broadcast {
            distribution {
              name: "eta_orf"
              half_normal {
                scales: 1
              }
            }
            batch_shape {
              dim {
                size: 1
              }
            }
          }
        }
        gamma_c {
          name: "gamma_c"
          batch_broadcast {
            distribution {
              name: "gamma_c"
              normal {
                locs: 0
                scales: 5
              }
            }
            batch_shape {
              dim {
                size: 3
              }
            }
          }
        }
        gamma_n {
          name: "gamma_n"
          batch_broadcast {
            distribution {
              name: "gamma_n"
              normal {
                locs: 0
                scales: 5
              }
            }
            batch_shape {
              dim {
                size: 5
              }
            }
          }
        }
        xi_c {
          name: "xi_c"
          batch_broadcast {
            distribution {
              name: "xi_c"
              half_normal {
                scales: 5
              }
            }
            batch_shape {
              dim {
                size: 3
              }
            }
          }
        }
        xi_n {
          name: "xi_n"
          batch_broadcast {
            distribution {
              name: "xi_n"
              half_normal {
                scales: 5
              }
            }
            batch_shape {
              dim {
                size: 5
              }
            }
          }
        }
        alpha_m {
          name: "alpha_m"
          batch_broadcast {
            distribution {
              name: "alpha_m"
              uniform {
                low: 0
                high: 1
              }
            }
            batch_shape {
              dim {
                size: 6
              }
            }
          }
        }
        alpha_rf {
          name: "alpha_rf"
          batch_broadcast {
            distribution {
              name: "alpha_rf"
              uniform {
                low: 0
                high: 1
              }
            }
            batch_shape {
              dim {
                size: 4
              }
            }
          }
        }
        alpha_om {
          name: "alpha_om"
          batch_broadcast {
            distribution {
              name: "alpha_om"
              uniform {
                low: 0
                high: 1
              }
            }
            batch_shape {
              dim {
                size: 2
              }
            }
          }
        }
        alpha_orf {
          name: "alpha_orf"
          batch_broadcast {
            distribution {
              name: "alpha_orf"
              uniform {
                low: 0
                high: 1
              }
            }
            batch_shape {
              dim {
                size: 1
              }
            }
          }
        }
        ec_m {
          name: "ec_m"
          batch_broadcast {
            distribution {
              name: "ec_m"
              truncated_normal {
                locs: 0.80
                scales: 0.80
                low: 0.10
                high: 10
              }
            }
            batch_shape {
              dim {
                size: 6
              }
            }
          }
        }
        ec_rf {
          name: "ec_rf"
          batch_broadcast {
            distribution {
              name: "ec_rf"
              transformed {
                distribution {
                  name: "LogNormal"
                  log_normal {
                    locs: 0.70
                    scales: 0.40
                  }
                }
                bijector {
                  name: "shift"
                  shift {
                    shifts: 0.10
                  }
                }
              }
            }
            batch_shape {
              dim {
                size: 4
              }
            }
          }
        }
        ec_om {
          name: "ec_om"
          batch_broadcast {
            distribution {
              name: "ec_om"
              truncated_normal {
                locs: 0.80
                scales: 0.80
                low: 0.10
                high: 10
              }
            }
            batch_shape {
              dim {
                size: 2
              }
            }
          }
        }
        ec_orf {
          name: "ec_orf"
          batch_broadcast {
            distribution {
              name: "ec_orf"
              transformed {
                distribution {
                  name: "LogNormal"
                  log_normal {
                    locs: 0.70
                    scales: 0.40
                  }
                }
                bijector {
                  name: "shift"
                  shift {
                    shifts: 0.10
                  }
                }
              }
            }
            batch_shape {
              dim {
                size: 1
              }
            }
          }
        }
        slope_m {
          name: "slope_m"
          batch_broadcast {
            distribution {
              name: "slope_m"
              deterministic {
                locs: 1
              }
            }
            batch_shape {
              dim {
                size: 6
              }
            }
          }
        }
        slope_rf {
          name: "slope_rf"
          batch_broadcast {
            distribution {
              name: "slope_rf"
              log_normal {
                locs: 0.70
                scales: 0.40
              }
            }
            batch_shape {
              dim {
                size: 4
              }
            }
          }
        }
        slope_om {
          name: "slope_om"
          batch_broadcast {
            distribution {
              name: "slope_om"
              deterministic {
                locs: 1
              }
            }
            batch_shape {
              dim {
                size: 2
              }
            }
          }
        }
        slope_orf {
          name: "slope_orf"
          batch_broadcast {
            distribution {
              name: "slope_orf"
              log_normal {
                locs: 0.70
                scales: 0.40
              }
            }
            batch_shape {
              dim {
                size: 1
              }
            }
          }
        }
        sigma {
          name: "sigma"
          batch_broadcast {
            distribution {
              name: "sigma"
              half_normal {
                scales: 5
              }
            }
            batch_shape {
              dim {
                size: 10
              }
            }
          }
        }
        roi_m {
          name: "roi_m"
          batch_broadcast {
            distribution {
              name: "roi_m"
              log_normal {
                locs: 0.20
                scales: 0.90
              }
            }
            batch_shape {
              dim {
                size: 6
              }
            }
          }
        }
        roi_rf {
          name: "roi_rf"
          batch_broadcast {
            distribution {
              name: "roi_rf"
              log_normal {
                locs: 0.20
                scales: 0.90
              }
            }
            batch_shape {
              dim {
                size: 4
              }
            }
          }
        }
        mroi_m {
          name: "mroi_m"
          batch_broadcast {
            distribution {
              name: "mroi_m"
              log_normal {
                locs: 0.00
                scales: 0.50
              }
            }
            batch_shape {
              dim {
                size: 6
              }
            }
          }
        }
        mroi_rf {
          name: "mroi_rf"
          batch_broadcast {
            distribution {
              name: "mroi_rf"
              log_normal {
                locs: 0.00
                scales: 0.50
              }
            }
            batch_shape {
              dim {
                size: 4
              }
            }
          }
        }
        contribution_m {
          name: "contribution_m"
          batch_broadcast {
            distribution {
              name: "contribution_m"
              beta {
                alpha: 1.00
                beta: 99.00
              }
            }
            batch_shape {
              dim {
                size: 6
              }
            }
          }
        }
        contribution_rf {
          name: "contribution_rf"
          batch_broadcast {
            distribution {
              name: "contribution_rf"
              beta {
                alpha: 1.00
                beta: 99.00
              }
            }
            batch_shape {
              dim {
                size: 4
              }
            }
          }
        }
        contribution_om {
          name: "contribution_om"
          batch_broadcast {
            distribution {
              name: "contribution_om"
              beta {
                alpha: 1.00
                beta: 99.00
              }
            }
            batch_shape {
              dim {
                size: 2
              }
            }
          }
        }
        contribution_orf {
          name: "contribution_orf"
          batch_broadcast {
            distribution {
              name: "contribution_orf"
              beta {
                alpha: 1.00
                beta: 99.00
              }
            }
            batch_shape {
              dim {
                size: 1
              }
            }
          }
        }
        contribution_n {
          name: "contribution_n"
          batch_broadcast {
            distribution {
              name: "contribution_n"
              truncated_normal {
                locs: 0.00
                scales: 0.10
                low: -1.00
                high: 1.00
              }
            }
            batch_shape {
              dim {
                size: 5
              }
            }
          }
        }
        """,
        meridian_pb.PriorDistributions(),
    )

    compare.assertProto2Equal(
        self,
        broadcast_dist_proto,
        expected_prior_dist_proto,
        precision=2,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='scalar_halfnormal',
          dist_proto=text_format.Parse(
              """
              name: "scalar_halfnormal"
              half_normal {
                scales: 1.0
              }
              """,
              meridian_pb.Distribution(),
          ),
          expected_tfp_dist=tfp.distributions.HalfNormal(
              1.0, name='scalar_halfnormal'
          ),
      ),
      dict(
          testcase_name='list_halfnormal',
          dist_proto=text_format.Parse(
              """
              name: "list_halfnormal"
              half_normal {
                scales: [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
              }
              """,
              meridian_pb.Distribution(),
          ),
          expected_tfp_dist=tfp.distributions.HalfNormal(
              [1.0, 1.1, 1.2, 1.3, 1.4, 1.5], name='list_halfnormal'
          ),
      ),
      dict(
          testcase_name='truncated_normal',
          dist_proto=text_format.Parse(
              """
              name: "truncated_normal"
              truncated_normal {
                locs: 0.7
                scales: 0.4
                low: 0.1
                high: 10
              }
              """,
              meridian_pb.Distribution(),
          ),
          expected_tfp_dist=tfp.distributions.TruncatedNormal(
              0.7, 0.4, 0.1, 10, name='truncated_normal'
          ),
      ),
      dict(
          testcase_name='batch_broadcast',
          dist_proto=text_format.Parse(
              """
              name: "batch_broadcast"
              batch_broadcast {
                distribution {
                  name: "HalfNormal"
                  half_normal {
                    scales: [5.0]
                  }
                }
                batch_shape {
                  dim {
                    size: 3
                  }
                }
              }
              """,
              meridian_pb.Distribution(),
          ),
          expected_tfp_dist=tfp.distributions.BatchBroadcast(
              tfp.distributions.HalfNormal(5.0), 3, name='batch_broadcast'
          ),
      ),
      dict(
          testcase_name='transformed_distribution',
          dist_proto=text_format.Parse(
              """
              name: "transformed_distribution"
              transformed {
                distribution {
                  name: "LogNormal"
                  log_normal {
                    locs: [0.7]
                    scales: [0.4]
                  }
                }
                bijector {
                  name: "shift"
                  shift {
                    shifts: 0.1
                  }
                }
              }
              """,
              meridian_pb.Distribution(),
          ),
          expected_tfp_dist=tfp.distributions.TransformedDistribution(
              tfp.distributions.LogNormal(0.7, 0.4),
              tfp.bijectors.Shift(0.1),
              name='transformed_distribution',
          ),
      ),
  )
  def test_from_distribution_proto(self, dist_proto, expected_tfp_dist):
    tfp_dist = distribution._from_distribution_proto(dist_proto)
    self._check_equal_distributions(tfp_dist, expected_tfp_dist)

  def _check_equal_distributions(
      self,
      dist1: tfp.distributions.Distribution,
      dist2: tfp.distributions.Distribution,
  ):
    self.assertIsInstance(
        dist1, type(dist2), msg=f'{type(dist1)} is not a {type(dist2)}'
    )
    self.assertSetEqual(
        set(dist1.parameters),
        set(dist2.parameters),
        msg=f'Non equal parameters: {dist1.parameters} != {dist2.parameters}',
    )
    for param in dist1.parameters:
      if param in ('kwargs_split_fn', 'parameters'):
        continue
      dist1_param_value = getattr(dist1, param)
      dist2_param_value = getattr(dist2, param)
      if isinstance(dist1_param_value, abc.Sequence):
        self.assertSequenceEqual(
            dist1_param_value,
            dist2_param_value,
            msg=(
                f'Non equal parameter {param}: {dist1_param_value} != '
                f'{dist2_param_value}'
            ),
        )
      elif tf.is_tensor(dist1_param_value):
        tf.debugging.assert_equal(
            dist1_param_value,
            dist2_param_value,
            message=(
                f'Non equal tensor parameter {param}: {dist1_param_value} != '
                f'{dist2_param_value}'
            ),
        )
      elif isinstance(dist1_param_value, tfp.distributions.Distribution):
        self._check_equal_distributions(dist1_param_value, dist2_param_value)
      else:
        self.assertEqual(
            dist1_param_value,
            dist2_param_value,
            msg=(
                f'Non equal parameter {param}: {dist1_param_value} != '
                f'{dist2_param_value}'
            ),
        )

  def test_deserialize_non_default_priors(self):
    priors_proto = text_format.Parse(
        """
        knot_values {
          name: "knot_values"
          normal {
            locs: 2
            locs: 3
            locs: 4
            scales: 7
            scales: 8
            scales: 9
          }
        }
        tau_g_excl_baseline {
          name: "tau_g_excl_baseline"
          normal {
            locs: 3
            scales: 10
          }
        }
        beta_m {
          name: "beta_m"
          half_normal {
            scales: 4
          }
        }
        beta_rf {
          name: "beta_rf"
          half_normal {
            scales: 2
            scales: 3
          }
        }
        alpha_om {
          name: "alpha_om"
          uniform {
            low: 1
            high: 2
          }
        }
        ec_rf {
        name: "ec_rf"
        transformed {
          distribution {
            name: "LogNormal"
            log_normal {
              locs: 0.70
              scales: 0.40
            }
          }
          bijector {
            name: "scale"
            scale {
              scales: 11
            }
          }
        }
      }
        """,
        # The rest are defaults.
        meridian_pb.PriorDistributions(),
    )
    priors = self.serde.deserialize(priors_proto)

    expected_priors = pd.PriorDistribution(
        knot_values=tfp.distributions.Normal(
            [2, 3, 4], [7, 8, 9], name='knot_values'
        ),
        tau_g_excl_baseline=tfp.distributions.Normal(
            3, 10, name='tau_g_excl_baseline'
        ),
        beta_m=tfp.distributions.HalfNormal(4, name='beta_m'),
        beta_rf=tfp.distributions.HalfNormal([2, 3], name='beta_rf'),
        alpha_om=tfp.distributions.Uniform(1, 2, name='alpha_om'),
        ec_rf=tfp.distributions.TransformedDistribution(
            tfp.distributions.LogNormal(0.7, 0.4),
            tfp.bijectors.Scale(11),
            name='ec_rf',
        ),
        # The rest are defaults.
    )

    for param in constants.ALL_PRIOR_DISTRIBUTION_PARAMETERS:
      if hasattr(priors, param):
        self._check_equal_distributions(
            getattr(priors, param), getattr(expected_priors, param)
        )

  @parameterized.named_parameters(
      dict(
          testcase_name='default_priors',
          priors=pd.PriorDistribution(),
      ),
      dict(
          testcase_name='non_default_priors',
          priors=pd.PriorDistribution(
              alpha_m=tfp.distributions.Uniform(1.1, 2.0, name='alpha_m'),
              ec_m=tfp.distributions.TruncatedNormal(
                  [0.8, 0.9], [0.85, 0.95], 0.1, 10, name='ec_m'
              ),
              ec_rf=tfp.distributions.TransformedDistribution(
                  tfp.distributions.LogNormal(0.8, 0.5),
                  tfp.bijectors.Shift(0.2),
                  name='ec_rf',
              ),
              slope_m=tfp.distributions.Deterministic(2, name='slope_m'),
              xi_n=tfp.distributions.HalfNormal(
                  [1.0, 1.1, 1.2, 1.3, 1.4, 1.5], name='xi_n'
              ),
          ),
      ),
  )
  def test_serialize_deserialize_default_priors(
      self, priors: pd.PriorDistribution
  ):
    serialized = self.serde.serialize(priors)
    deserialized = self.serde.deserialize(serialized)
    serialized_again = self.serde.serialize(deserialized)

    compare.assertProto2Equal(
        self,
        serialized,
        serialized_again,
        precision=2,
    )

  def test_b_414895509_deserialize_completely_deconstructs_protobuf(self):
    """Regression test for b/414895509.

    When deserializing a `PriorDistributions` proto, all protobuf structures
    must be fully deconstructed, even simple sequence types.
    """
    priors_proto = text_format.Parse(
        """
        roi_m {
          name: "roi_m"
          log_normal {
            locs: 0.12
            locs: 0.23
            locs: 0.34
            scales: 0.90
          }
        }
        """,
        meridian_pb.PriorDistributions(),
    )
    deserialized = self.serde.deserialize(priors_proto)
    pickle.dumps(deserialized)


if __name__ == '__main__':
  absltest.main()
