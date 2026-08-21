# Results Summary

Use `summarizer.Summarizer` to generate the HTML results summary.

```python
from meridian.analysis import summarizer

mmm_summarizer = summarizer.Summarizer(model, use_kpi=True)
mmm_summarizer.output_model_results_summary(
    filename="results_summary.html",
    filepath=output_dir,
    start_date="2021-01-01",  # Optional
    end_date="2021-12-31",  # Optional
)
```
