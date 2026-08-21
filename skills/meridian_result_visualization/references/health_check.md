# Health Checks

Use `reviewer.ModelReviewer` to run health checks, and
`output_model_health_card` to save the result to HTML.

```python
from meridian.analysis.review import reviewer

model_reviewer = reviewer.ModelReviewer(
    model_context=model.model_context, inference_data=model.inference_data
)
review_summary = model_reviewer.run()

# Save health check report to HTML
review_summary.output_model_health_card(
    filename="health_check.html",
    filepath=output_dir
)
```
