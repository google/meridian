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

from collections.abc import Sequence
import functools
from unittest import mock

from absl.testing import absltest
from mmm.v1 import mmm_pb2 as mmm_pb
from mmm.v1.fit import model_fit_pb2 as fit_pb
from mmm.v1.marketing.optimization import marketing_optimization_pb2 as opt_pb
from mmm.v1.model import mmm_kernel_pb2 as kernel_pb
from meridian.schema import model_consumer
from meridian.schema.processors import model_processor
from meridian.schema.serde import meridian_serde


class FooSpec(model_processor.Spec):

  def validate(self):
    pass


class FooProcessor(model_processor.ModelProcessor[FooSpec, fit_pb.ModelFit]):

  def __init__(self, trained_model: model_processor.TrainedModel):
    self._trained_model = trained_model

  @classmethod
  def spec_type(cls):
    return FooSpec

  @classmethod
  def output_type(cls):
    return fit_pb.ModelFit

  def execute(self, specs: Sequence[FooSpec]) -> fit_pb.ModelFit:
    return fit_pb.ModelFit()

  def _set_output(self, output: mmm_pb.Mmm, result: fit_pb.ModelFit):
    output.model_fit.CopyFrom(result)


class BarSpec(model_processor.Spec):

  def validate(self):
    pass


class BarProcessor(
    model_processor.ModelProcessor[BarSpec, opt_pb.MarketingOptimization]
):

  def __init__(
      self,
      trained_model: model_processor.TrainedModel,
  ):
    self._trained_model = trained_model

  @classmethod
  def spec_type(cls):
    return BarSpec

  @classmethod
  def output_type(cls):
    return opt_pb.MarketingOptimization

  def execute(self, specs: Sequence[BarSpec]) -> opt_pb.MarketingOptimization:
    return opt_pb.MarketingOptimization()

  def _set_output(
      self, output: mmm_pb.Mmm, result: opt_pb.MarketingOptimization
  ):
    output.marketing_optimization.CopyFrom(result)


class ModelConsumerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_mmm = mock.MagicMock()

    # Patch the trained model class.
    self.mock_trained_model = self.enter_context(
        mock.patch.object(model_processor, 'TrainedModel', autospec=True)
    )(mmm=self.mock_mmm)
    self.consumer = model_consumer.ModelConsumer([FooProcessor, BarProcessor])
    self.enter_context(
        mock.patch.object(
            meridian_serde.MeridianSerde,
            'serialize',
            autospec=True,
            return_value=kernel_pb.MmmKernel(),
        )
    )

  def test_specs_to_processors(self):
    self.assertEqual(
        self.consumer.specs_to_processors_classes,
        {FooSpec: FooProcessor, BarSpec: BarProcessor},
    )

  def test_specs_to_processors_error_on_duplicate_spec_types(self):
    class DuplicateProcessor(
        model_processor.ModelProcessor[FooSpec, fit_pb.ModelFit]
    ):

      @classmethod
      def spec_type(cls):
        return FooSpec

      @classmethod
      def output_type(cls):
        return fit_pb.ModelFit

      def execute(self, specs: Sequence[FooSpec]) -> fit_pb.ModelFit:
        return fit_pb.ModelFit()

      def _set_output(self, output: mmm_pb.Mmm, result: fit_pb.ModelFit):
        output.model_fit.CopyFrom(result)

    with self.assertRaises(ValueError):
      _ = model_consumer.ModelConsumer(
          [FooProcessor, BarProcessor, DuplicateProcessor]
      ).specs_to_processors_classes

  def test_consumer_call_dispatches_to_processors(self):
    # This context dict will be used to verify that the correct processors are
    # called.
    context = {
        'foo': False,  # if FooProcessor.execute is called.
        'bar': False,  # if BarProcessor.execute is called.
    }

    def _patch_foo_execute(
        slf, specs: Sequence[FooSpec], context
    ) -> fit_pb.ModelFit:
      self.assertLen(specs, 2)
      self.assertTrue(all([isinstance(spec, FooSpec) for spec in specs]))
      self.assertIs(slf._trained_model, self.mock_trained_model)
      context['foo'] = True
      return fit_pb.ModelFit()

    FooProcessor.execute = functools.partialmethod(
        _patch_foo_execute, context=context
    )

    def _patch_bar_execute(
        slf, specs: Sequence[BarSpec], context
    ) -> opt_pb.MarketingOptimization:
      self.assertLen(specs, 1)
      self.assertTrue(all([isinstance(spec, BarSpec) for spec in specs]))
      self.assertIs(slf._trained_model, self.mock_trained_model)
      context['bar'] = True
      return opt_pb.MarketingOptimization()

    BarProcessor.execute = functools.partialmethod(
        _patch_bar_execute, context=context
    )

    self.consumer = model_consumer.ModelConsumer([FooProcessor, BarProcessor])

    # Calling the model consumer should execute both FooProcessor and
    # BarProcessor: FooProcessor should be given two FooSpecs and BarProcessor
    # should be given one BarSpec.
    output = self.consumer(self.mock_mmm, [FooSpec(), FooSpec(), BarSpec()])

    self.assertTrue(output.HasField('mmm_kernel'))
    self.assertTrue(output.HasField('model_fit'))
    self.assertTrue(output.HasField('marketing_optimization'))
    self.assertTrue(context['foo'], 'FooProcessor.execute was not called.')
    self.assertTrue(context['bar'], 'BarProcessor.execute was not called.')


if __name__ == '__main__':
  absltest.main()
