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

"""Meridian Model Processor Library.

This package provides a collection of processors designed to operate on trained
Meridian models. These processors facilitate various post-training tasks,
including model analysis, insight generation, and budget optimization.

The processors are built upon a common framework defined in the
`model_processor` module, which establishes the base classes and interfaces for
builtin processors in this package, as well as for creating custom processors.
Each processor typically takes a trained Meridian model object and additional
specifications as input, producing structured output, in protobuf format.

These structured outputs can then be used to generate insights, visualizations,
and other artifacts that help users understand and optimize their marketing
strategy. For instance, the `schema.converters` package provides tools to
flatten these outputs into tabular Google Sheets tables suitable for a Meridian
Looker Studio dashboard's data sources.

Available Processor Modules:

-   `model_processor`: Defines the abstract base classes `ModelProcessor` and
    `ModelProcessorSpec`, which serve as the foundation for all processors
    in this package.
-   `model_kernel_processor`: A processor to extract and serialize the core
    components and parameters of the trained Meridian model.
-   `model_fit_processor`: Generates various goodness-of-fit statistics and
    diagnostic metrics for the trained model.
-   `marketing_processor`: Performs marketing mix analysis, including
    contribution analysis, response curves, and ROI calculations.
-   `budget_optimization_processor`: Provides tools for optimizing marketing
    budgets based on the model's predictions to achieve specific goals.
-   `reach_frequency_processor`: Analyzes and optimizes based on reach and
    frequency metrics, if applicable to the model structure.

Each processor defines its own spec language. For instance, the budget
optimization processor would take a `BudgetOptimizationSpec` object as input,
which defines the constraints and parameters of the optimization problem a
user wants to explore.

A trained Meridian model is generally a requisite input for all processors.
Generally, a `model_processor.TrainedModel` wrapper object is passed to each
processor, along with its processor-specific spec. For example:

```python
# Assuming 'trained_model' is a loaded Meridian model object
processor = model_fit_processor.ModelFitProcessor(trained_model)
result = processor([model_fit_processor.ModelFitSpec()])

# `result` is a structured `ModelFit` proto that describes the model's goodness
# of fit analysis.
```

For more details on these processors' sub-API, please refer to the documentation
of the individual modules.
"""

from schema.processors import budget_optimization_processor
from schema.processors import common
from schema.processors import marketing_processor
from schema.processors import model_fit_processor
from schema.processors import model_kernel_processor
from schema.processors import model_processor
from schema.processors import reach_frequency_optimization_processor
