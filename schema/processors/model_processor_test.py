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

from collections.abc import Sequence
import datetime
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from meridian.data import time_coordinates
from mmm.v1 import mmm_pb2 as pb
from mmm.v1.common import date_interval_pb2 as date_interval_pb
from mmm.v1.fit import model_fit_pb2 as fit_pb
from schema.processors import model_processor

from google.type import date_pb2 as date_pb
from tensorflow.python.util.protobuf import compare


class MySpec(model_processor.Spec):

  def validate(self):
    pass


class MyModelFitProcessor(
    model_processor.ModelProcessor[MySpec, fit_pb.ModelFit]
):

  @classmethod
  def spec_type(cls):
    return MySpec

  @classmethod
  def output_type(cls):
    return fit_pb.ModelFit

  def execute(self, specs: Sequence[MySpec]) -> fit_pb.ModelFit:
    return fit_pb.ModelFit()

  def _set_output(self, output: pb.Mmm, result: fit_pb.ModelFit):
    output.model_fit.CopyFrom(result)


class DatedSpecTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._start_date = datetime.date(2024, 1, 1)
    self._end_date = datetime.date(2024, 12, 31)

  def test_start_date_is_after_end_date(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Start date must be before end date."
    ):
      model_processor.DatedSpec(
          start_date=self._end_date,
          end_date=self._start_date,
      )

  def test_validates_successfully(self):
    tag = "tag"
    spec = model_processor.DatedSpec(
        start_date=self._start_date,
        end_date=self._end_date,
        date_interval_tag=tag,
    )

    with self.subTest(name="StartDate"):
      self.assertEqual(spec.start_date, self._start_date)
    with self.subTest(name="EndDate"):
      self.assertEqual(spec.end_date, self._end_date)
    with self.subTest(name="DateIntervalTag"):
      self.assertEqual(spec.date_interval_tag, tag)


class OptimizationSpecTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._start_date = datetime.date(2024, 1, 1)
    self._end_date = datetime.date(2024, 12, 31)
    self._date_interval_tag = "tag"
    self._optimization_name = "Opt Name"
    self._grid_name = "grid_name"
    self._confidence_level = 0.95

  def test_validates_successfully(self):
    spec = model_processor.OptimizationSpec(
        start_date=self._start_date,
        end_date=self._end_date,
        date_interval_tag=self._date_interval_tag,
        optimization_name=self._optimization_name,
        grid_name=self._grid_name,
        confidence_level=self._confidence_level,
    )

    with self.subTest(name="OptimizationName"):
      self.assertEqual(spec.optimization_name, self._optimization_name)
    with self.subTest(name="GridName"):
      self.assertEqual(spec.grid_name, self._grid_name)
    with self.subTest(name="ConfidenceLevel"):
      self.assertEqual(spec.confidence_level, self._confidence_level)

  def test_empty_optimization_name(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Optimization name must not be empty or blank."
    ):
      model_processor.OptimizationSpec(
          start_date=self._start_date,
          end_date=self._end_date,
          date_interval_tag=self._date_interval_tag,
          optimization_name="",
          grid_name=self._grid_name,
          confidence_level=self._confidence_level,
      )

  def test_blank_optimization_name(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Optimization name must not be empty or blank."
    ):
      model_processor.OptimizationSpec(
          start_date=self._start_date,
          end_date=self._end_date,
          date_interval_tag=self._date_interval_tag,
          optimization_name=" ",
          grid_name=self._grid_name,
          confidence_level=self._confidence_level,
      )

  def test_empty_grid_name(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Grid name must not be empty or blank."
    ):
      model_processor.OptimizationSpec(
          start_date=self._start_date,
          end_date=self._end_date,
          date_interval_tag=self._date_interval_tag,
          optimization_name=self._optimization_name,
          grid_name="",
          confidence_level=self._confidence_level,
      )

  def test_blank_grid_name(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Grid name must not be empty or blank."
    ):
      model_processor.OptimizationSpec(
          start_date=self._start_date,
          end_date=self._end_date,
          date_interval_tag=self._date_interval_tag,
          optimization_name=self._optimization_name,
          grid_name=" ",
          confidence_level=self._confidence_level,
      )

  def test_confidence_level_out_of_range(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Confidence level must be between 0 and 1."
    ):
      model_processor.OptimizationSpec(
          start_date=self._start_date,
          end_date=self._end_date,
          date_interval_tag=self._date_interval_tag,
          optimization_name=self._optimization_name,
          grid_name=self._grid_name,
          confidence_level=2.0,
      )


class ModelProcessorTest(absltest.TestCase):

  def test_call(self):
    output = pb.Mmm()
    MyModelFitProcessor()([MySpec()], output)
    self.assertTrue(output.HasField("model_fit"))

  def test_call_wrong_spec_type(self):
    class WrongSpec(model_processor.Spec):

      def validate(self):
        pass

    output = pb.Mmm()
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Not all specs are of type <class '__main__.MySpec'>"
    ):
      MyModelFitProcessor()([MySpec(), WrongSpec()], output)


class TrainedModelTest(absltest.TestCase):

  def test_init_unfitted_model(self):
    mmm = mock.MagicMock()
    mmm.inference_data.posterior = None
    with self.assertRaisesWithLiteralMatch(
        ValueError, "MMM model has not been fitted."
    ):
      model_processor.TrainedModel(mmm=mmm)

  def test_init_fitted_model(self):
    mmm = mock.MagicMock()
    tmodel = model_processor.TrainedModel(mmm=mmm)
    self.assertIs(tmodel.mmm, mmm)

  def test_get_time_coordinates(self):
    mmm = mock.MagicMock()
    tmodel = model_processor.TrainedModel(mmm=mmm)
    self.assertIs(tmodel.time_coordinates, mmm.input_data.time_coordinates)

  def test_get_internal_analyzer(self):
    mmm = mock.MagicMock()
    tmodel = model_processor.TrainedModel(mmm=mmm)
    analyzer = tmodel.internal_analyzer
    self.assertIsNotNone(analyzer)
    self.assertIs(analyzer._meridian, mmm)

  def test_get_internal_optimizer(self):
    mmm = mock.MagicMock()
    tmodel = model_processor.TrainedModel(mmm=mmm)
    optimizer = tmodel.internal_optimizer
    self.assertIsNotNone(optimizer)
    self.assertIs(optimizer._meridian, mmm)

  def test_get_internal_model_diagnostics(self):
    mmm = mock.MagicMock()
    tmodel = model_processor.TrainedModel(mmm=mmm)
    model_diagnostics = tmodel.internal_model_diagnostics
    self.assertIsNotNone(model_diagnostics)
    self.assertIs(model_diagnostics._meridian, mmm)


class DatedSpecResolverTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._time_coordinates = time_coordinates.TimeCoordinates.from_dates([
        "2024-01-01",
        "2024-01-08",
        "2024-01-15",
        "2024-01-22",
    ])

  def test_resolve_to_enumerated_selected_times_default_start_and_end_dates(
      self,
  ):
    spec = model_processor.DatedSpec(start_date=None, end_date=None)
    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )
    selected_times = resolver.resolve_to_enumerated_selected_times()

    self.assertIsNone(selected_times)

  def test_resolve_to_bool_selected_times_default_start_and_end_dates(
      self,
  ):
    spec = model_processor.DatedSpec(start_date=None, end_date=None)
    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )
    selected_times = resolver.resolve_to_bool_selected_times()

    self.assertIsNone(selected_times)

  def test_resolve_to_enumerated_selected_times_start_date_not_in_coordinates(
      self,
  ):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "start_date (2021-01-08) must be in the time coordinates!"
    ):
      spec = model_processor.DatedSpec(start_date=datetime.date(2021, 1, 8))
      model_processor.DatedSpecResolver(
          spec=spec, time_coordinates=self._time_coordinates
      ).resolve_to_enumerated_selected_times()

  def test_resolve_to_enumerated_selected_times_default_end_date(self):
    spec = model_processor.DatedSpec(start_date=datetime.date(2024, 1, 8))

    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )
    selected_times = resolver.resolve_to_enumerated_selected_times()

    self.assertEqual(selected_times, ["2024-01-08", "2024-01-15", "2024-01-22"])

  def test_resolve_to_bool_selected_times_default_end_date(self):
    spec = model_processor.DatedSpec(start_date=datetime.date(2024, 1, 8))

    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )
    selected_times = resolver.resolve_to_bool_selected_times()

    self.assertEqual(selected_times, [False, True, True, True])

  def test_resolve_to_enumerated_selected_times_default_start_date(self):
    spec = model_processor.DatedSpec(end_date=datetime.date(2024, 1, 15))

    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )
    selected_times = resolver.resolve_to_enumerated_selected_times()
    # 7 days before 1/15
    self.assertEqual(selected_times, ["2024-01-01", "2024-01-08"])

  def test_resolve_to_bool_selected_times_default_start_date(self):
    spec = model_processor.DatedSpec(end_date=datetime.date(2024, 1, 15))

    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )
    selected_times = resolver.resolve_to_bool_selected_times()
    self.assertEqual(selected_times, [True, True, False, False])

  def test_to_closed_date_interval_tuple_default(self):
    spec = model_processor.DatedSpec(start_date=None, end_date=None)
    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )
    date_interval_tup = resolver.to_closed_date_interval_tuple()

    self.assertEqual(date_interval_tup, (None, None))

  def test_to_closed_date_interval_tuple_default_end_date(self):
    spec = model_processor.DatedSpec(start_date=datetime.date(2024, 1, 1))
    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )
    date_interval_tup = resolver.to_closed_date_interval_tuple()

    self.assertEqual(
        date_interval_tup,
        ("2024-01-01", None),
    )

  def test_to_closed_date_interval_tuple_default_start_date(self):
    spec = model_processor.DatedSpec(end_date=datetime.date(2024, 12, 29))

    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )
    date_interval_tup = resolver.to_closed_date_interval_tuple()

    self.assertEqual(
        date_interval_tup,
        (None, "2024-12-22"),  # 7 days before 12/29 to close it
    )

  def test_to_closed_date_interval_tuple(self):
    spec = model_processor.DatedSpec(
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 12, 29),
    )

    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )
    date_interval_tup = resolver.to_closed_date_interval_tuple()

    self.assertEqual(
        date_interval_tup,
        ("2024-01-01", "2024-12-22"),  # 7 days before 12/29 to close it
    )

  def test_collapse_to_date_interval_proto_default_selected_times(self):
    spec = model_processor.DatedSpec(date_interval_tag="tag")
    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )
    date_interval_proto = resolver.collapse_to_date_interval_proto()

    compare.assertProto2Equal(
        self,
        date_interval_proto,
        date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=1),
            # 7 days after 1/22
            end_date=date_pb.Date(year=2024, month=1, day=29),
            tag="tag",
        ),
    )

  def test_collapse_to_date_interval_proto_with_selected_times(self):
    spec = model_processor.DatedSpec(
        start_date=datetime.date(2024, 1, 8),
        end_date=datetime.date(2024, 1, 29),
        date_interval_tag="tag",
    )

    date_interval_proto = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    ).collapse_to_date_interval_proto()

    compare.assertProto2Equal(
        self,
        date_interval_proto,
        date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=8),
            # 7 days after 1/22
            end_date=date_pb.Date(year=2024, month=1, day=29),
            tag="tag",
        ),
    )

  def test_transform_to_date_interval_protos_default_selected_times(self):
    spec = model_processor.DatedSpec(date_interval_tag="tag")
    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )

    actual_date_interval_protos = resolver.transform_to_date_interval_protos()

    expected_date_interval_protos = [
        date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=1),
            end_date=date_pb.Date(year=2024, month=1, day=8),
            tag="tag",
        ),
        date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=8),
            end_date=date_pb.Date(year=2024, month=1, day=15),
            tag="tag",
        ),
        date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=15),
            end_date=date_pb.Date(year=2024, month=1, day=22),
            tag="tag",
        ),
        date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=22),
            end_date=date_pb.Date(year=2024, month=1, day=29),
            tag="tag",
        ),
    ]

    compare.assertProto2CountEqual(
        self,
        actual_date_interval_protos,
        expected_date_interval_protos,
    )

  def test_transform_to_date_interval_protos_with_selected_times(self):
    spec = model_processor.DatedSpec(
        start_date=datetime.date(2024, 1, 8),
        end_date=datetime.date(2024, 1, 29),
        date_interval_tag="tag",
    )
    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )

    actual_date_interval_protos = resolver.transform_to_date_interval_protos()

    expected_date_interval_protos = [
        date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=8),
            end_date=date_pb.Date(year=2024, month=1, day=15),
            tag="tag",
        ),
        date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=15),
            end_date=date_pb.Date(year=2024, month=1, day=22),
            tag="tag",
        ),
        date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=22),
            # 7 days after 1/22
            end_date=date_pb.Date(year=2024, month=1, day=29),
            tag="tag",
        ),
    ]

    compare.assertProto2CountEqual(
        self,
        actual_date_interval_protos,
        expected_date_interval_protos,
    )

  def test_resolve_to_date_interval_open_end(self):
    spec = model_processor.DatedSpec(
        start_date=datetime.date(2024, 1, 1),
    )
    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )

    start, end = resolver.resolve_to_date_interval_open_end()

    self.assertEqual(start, datetime.date(2024, 1, 1))
    self.assertEqual(end, datetime.date(2024, 1, 29))  # 7 days after 1/22

  def test_resolve_to_date_interval_proto_default(self):
    spec = model_processor.DatedSpec(date_interval_tag="tag")
    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )

    date_interval_proto = resolver.resolve_to_date_interval_proto()

    compare.assertProto2Equal(
        self,
        date_interval_proto,
        date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=1),
            # 7 days after 1/22
            end_date=date_pb.Date(year=2024, month=1, day=29),
            tag="tag",
        ),
    )

  def test_resolve_to_date_interval_proto_end_date_specified(self):
    spec = model_processor.DatedSpec(
        end_date=datetime.date(2024, 1, 15),
        date_interval_tag="tag",
    )
    resolver = model_processor.DatedSpecResolver(
        spec=spec, time_coordinates=self._time_coordinates
    )

    date_interval_proto = resolver.resolve_to_date_interval_proto()

    compare.assertProto2Equal(
        self,
        date_interval_proto,
        date_interval_pb.DateInterval(
            start_date=date_pb.Date(year=2024, month=1, day=1),
            end_date=date_pb.Date(year=2024, month=1, day=15),  # not adjusted
            tag="tag",
        ),
    )


if __name__ == "__main__":
  absltest.main()
