# CSV Data Format Reference

This document describes the expected tabular structure and guardrails for CSV
data files used with the `meridian-model-building` skill.

## General Structure

*   **Headers**: The first row of the CSV file MUST contain the column headers
    (variable names). Do NOT place headers in the first column.
*   **One Variable Per Column**: Each column must represent a single variable
    (e.g., KPI, a specific media channel's impressions, or a control variable).
*   **One Observation Per Row**: Each row represents a specific combination of
    `time` and `geo` (for geo-level models).

## Data Quality Guardrails

The agent MUST check for and attempt to fix the following issues:

1.  **Date Format**: The `time` column should contain dates in a consistent,
    recognizable format (ideally `YYYY-MM-DD`). If strings are used, the agent
    should attempt to parse them to datetime objects.
2.  **Numeric Values**: Columns mapped to KPI, Media, Spend, and Population MUST
    contain numeric data.
    *   Remove currency symbols ($, €, etc.) and commas used as thousands
        separators.
    *   Convert string representations of numbers to float or int.
3.  **Non-Negative Constraints**: Values for Media exposures, Media spend, and
    Population MUST be non-negative. If negative values are found, the agent
    should investigate or ask the user (unless it's a control variable where
    negative values might be valid).
4.  **Missing Values (NaN)**:
    *   For Media and Spend, missing values often imply zero activity. The agent
        should consider filling them with 0 after confirming with data context
        or user.
    *   For KPI and controls, missing values might require interpolation or
        dropping rows.
5.  **Summable Metrics**: Verify that the KPI and Media exposure metrics are
    summable (e.g., use total conversions, not conversion rate).
6.  **Duplicate Records**: Ensure there are no duplicate rows for the same `geo`
    and `time` combination.

## Example Format (Geo-Level Data)

Here is an example of a valid CSV structure:

```csv
geo,time,Channel0_impression,Channel0_spend,conversions,revenue_per_conversion,population
Geo0,2021-01-25,280668,2058.06,1954576,0.02,136670
Geo0,2021-02-01,366206,3667.39,2000000,0.02,136670
Geo1,2021-01-25,150000,1000.00,500000,0.05,50000
```
