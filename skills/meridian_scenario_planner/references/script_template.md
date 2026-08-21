# Python Script Template for Scenario Planner

This reference provides a complete Python script template for loading a fitted
Meridian model, generating Scenario Planner data, serializing it to disk, and
handing it off to the Meridian Colab notebook for Looker Studio visualization.

```python
import os
from meridian.schema.serde import meridian_serde
from meridian.schema.processors import budget_optimization_processor
from meridian.schema.processors import marketing_processor
from meridian.schema.processors import model_fit_processor
from meridian.schema.utils import date_range_bucketing
from meridian.scenarioplanner import mmm_ui_proto_generator as mmm_ui_gen

def main():
    # Model and output file paths
    model_path = "meridian_model.binpb"
    output_proto_path = "model_build/scenario_planner_data.binpb"

    # 1. Load Model
    print(f"Loading model from {model_path}...")
    model = meridian_serde.load_meridian(model_path)

    # 2. Define Spec Configurations (User can modify these)
    optimization_name = "Scenario Planner"
    include_non_paid_channels = True  # Whether to include non-paid channels in the analysis.
    start_date = None  # String format: 'YYYY-MM-DD' (e.g., '2021-01-01')
    end_date = None  # String format: 'YYYY-MM-DD' (e.g., '2021-12-31')

    # Time breakdown setup
    yearly = False  # Whether to optimize budget for every year
    quarterly = True  # Whether to optimize budget for every quarter
    monthly = False  # Whether to optimize budget for every month

    # Interactive budget optimization grid setup
    min_spend_shift_ratio = 0.3
    max_spend_shift_ratio = 0.3

    # Optimal frequency settings for Reach & Frequency channels
    use_optimal_frequency = True
    max_frequency = 10.0

    # 3. Build Specs
    time_breakdown_generators = []
    if yearly:
        time_breakdown_generators.append(
            date_range_bucketing.YearlyDateRangeGenerator
        )
    if quarterly:
        time_breakdown_generators.append(
            date_range_bucketing.QuarterlyDateRangeGenerator
        )
    if monthly:
        time_breakdown_generators.append(
            date_range_bucketing.MonthlyDateRangeGenerator
        )

    channel_constraints = []
    if min_spend_shift_ratio is not None and max_spend_shift_ratio is not None:
        for channel in model.input_data.get_all_paid_channels():
            channel_constraints.append(
                budget_optimization_processor.ChannelConstraintRel(
                    channel_name=channel,
                    spend_constraint_lower=min_spend_shift_ratio,
                    spend_constraint_upper=max_spend_shift_ratio,
                )
            )

    grid_name_prefix = "-".join(optimization_name.lower().split(" "))

    budget_opt_spec = budget_optimization_processor.BudgetOptimizationSpec(
        start_date=start_date,
        end_date=end_date,
        optimization_name=optimization_name,
        grid_name=grid_name_prefix,
        constraints=channel_constraints,
        use_optimal_frequency=use_optimal_frequency,
        max_frequency=max_frequency,
    )

    model_fit_spec = model_fit_processor.ModelFitSpec()
    summary_spec = marketing_processor.MediaSummarySpec(include_non_paid_channels=include_non_paid_channels)
    marketing_spec = marketing_processor.MarketingAnalysisSpec(media_summary_spec=summary_spec)

    # 4. Generate Scenario Planner Proto
    print("Generating Scenario Planner data...")
    mmm_proto = mmm_ui_gen.create_mmm_ui_data_proto(
        mmm=model,
        specs=[model_fit_spec, marketing_spec, budget_opt_spec],
        time_breakdown_generators=time_breakdown_generators,
    )

    # 5. Save Proto for Colab Handoff
    os.makedirs(os.path.dirname(output_proto_path) or ".", exist_ok=True)
    with open(output_proto_path, "wb") as f:
        f.write(mmm_proto.SerializeToString())
    print(f"Scenario planner data saved to {output_proto_path}.")
    print(
        "Upload this file to the Meridian Scenario Planner Colab notebook: "
        "https://colab.research.google.com/github/google/meridian/blob/main/demo/Meridian_Scenario_Planner_Beta.ipynb"
    )

if __name__ == '__main__':
    main()
```
