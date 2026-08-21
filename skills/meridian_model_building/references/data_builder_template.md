# Data Builder Template

When building the data loading section, use the `DataFrameInputDataBuilder` from
`meridian.data.data_frame_input_data_builder`.

This template shows a comprehensive example of how to construct the input data,
handling various types of regressors (controls, media, organic media, etc.) and
accounting for both National and Geo-level configurations.

```python
import pandas as pd
from meridian.data.data_frame_input_data_builder import DataFrameInputDataBuilder

def main():

    # 1. Load the data
    df = pd.read_csv("path/to/data.csv")

    # 2. Initialize the builder
    # kpi_type must be either 'revenue' or 'non_revenue'
    builder = DataFrameInputDataBuilder(kpi_type="non_revenue")

    # 3. Configure KPI, Time, and Geo
    # Map the dataframe and its columns. Omit geo_col if building a National-level model.
    builder.with_kpi(df, kpi_col="kpi_column", time_col="date_column", geo_col="region_column")

    # 4. Configure Revenue Per KPI (Required if kpi_type is 'non_revenue' and you want ROI)
    builder.with_revenue_per_kpi(df, revenue_per_kpi_col="revenue_per_kpi_column")

    # 4. Configure Media (Spend and Impressions)
    media_channels = ["tv", "radio", "social"]
    builder.with_media(
        df,
        media_cols=[f"{ch}_impressions" for ch in media_channels],
        media_spend_cols=[f"{ch}_spend" for ch in media_channels],
        media_channels=media_channels,
        # These columns are optional here if they match the default or what was used in with_kpi
        time_col="date_column",
        geo_col="region_column",
    )

    # 5. Configure Controls (Optional)
    # Controls are variables outside your control that impact the KPI (e.g., seasonality, competitors).
    control_variables = ["competitor_spend", "holiday_flag", "seasonality_index"]
    builder.with_controls(df, control_cols=control_variables, time_col="date_column", geo_col="region_column")

    # 6. Configure Population (REQUIRED for Geo models)
    # Important for normalizing data across geos of different sizes.
    builder.with_population(df, population_col="population", geo_col="region_column")

    # 7. Configure Organic/Non-Media Variables (Optional)
    # Factors you control but don't strictly "spend" on (e.g., newsletter signups, SEO).
    organic_media = ["email_sends", "organic_search_volume"]
    builder.with_organic_media(df, organic_media_cols=organic_media)

    # 8. Configure Revenue per KPI (Optional)
    # This is only relevant if `kpi_type` is 'non_revenue' and you want to calculate ROI based on a varying revenue value per KPI unit.
    # builder.with_revenue_per_kpi(df, revenue_per_kpi_col="avg_order_value", time_col="date_column", geo_col="region_column")

    # 9. Configure Reach and Frequency (Optional - Requires R&F data instead of just impressions)
    # If using Reach and Frequency, DO NOT use `with_media`.
    # builder.with_rf(
    #     df,
    #     reach_cols=[f"{ch}_reach" for ch in media_channels],
    #     frequency_cols=[f"{ch}_frequency" for ch in media_channels],
    #     media_spend_cols=[f"{ch}_spend" for ch in media_channels],
    #     media_channels=media_channels,
    # )

    # 10. Build the InputData object
    input_data = builder.build()
```
