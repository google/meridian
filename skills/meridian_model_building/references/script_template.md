# Python Script Template

When generating the final Python script for the user, use this clean structure:

```python
import os
import pandas as pd
from meridian.data.data_frame_input_data_builder import DataFrameInputDataBuilder
from meridian.model.spec import ModelSpec
from meridian.model.prior_distribution import PriorDistribution
from meridian.model.model import Meridian
from meridian.model.eda import meridian_eda
from meridian.schema.serde import meridian_serde
from meridian import backend

def main():
    # 1. Handle Output Paths
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")
    output_dir = os.path.join(workspace_dir, "model_build")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Data & Build InputData
    # For full details, see references/data_builder_template.md
    df = pd.read_csv("path/to/data.csv")
    builder = DataFrameInputDataBuilder(kpi_type="non_revenue")
    builder.with_kpi(df, kpi_col="kpi_column", time_col="date_column", geo_col="region_column")
    builder.with_revenue_per_kpi(df, revenue_per_kpi_col="revenue_per_kpi_column")

    media_channels = ["tv", "radio", "social"]
    builder.with_media(
        df,
        media_cols=[f"{ch}_impressions" for ch in media_channels],
        media_spend_cols=[f"{ch}_spend" for ch in media_channels],
        media_channels=media_channels,
    )
    input_data = builder.build()

    # 3. Configure ModelSpec
    # For full details, see references/model_spec_template.md
    tfd = backend.tfd
    prior_distribution = PriorDistribution()
    model_spec = ModelSpec(prior=prior_distribution, knots=10)

    mmm = Meridian(input_data=input_data, model_spec=model_spec)

    # 4. Sample Prior
    mmm.sample_prior(n_draws=100)

    # 5. Run EDA
    eda = meridian_eda.MeridianEDA(mmm)
    eda.generate_and_save_report(filename='eda.html', filepath=output_dir)

    # 6. Fit Model
    mmm.sample_posterior(n_chains=1, n_adapt=100, n_burnin=100, n_keep=100)

    # 7. Save Model
    save_path = os.path.join(output_dir, "meridian_model.binpb")
    meridian_serde.save_meridian(mmm, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
```
