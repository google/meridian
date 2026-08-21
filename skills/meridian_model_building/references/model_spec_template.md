# Model Spec Template

When configuring the model, use `ModelSpec` and `PriorDistribution` from
`meridian.model.spec`.

This template provides a comprehensive example of setting up priors and the
model specification, illustrating how to customize knot configurations for time
effects and how to tailor specific prior distributions.

```python
from meridian.model.spec import ModelSpec
from meridian.model.prior_distribution import PriorDistribution
from meridian.model.model import Meridian
from meridian import backend

def main():

    # Access distributions via the backend abstraction layer
    tfd = backend.tfd

    # 1. Define Prior Distributions
    # You can customize the priors for ROI, media effects, Adstock, and Hill functions.
    # The defaults are generally robust, but overriding them is common when incorporating past experiment data.
    prior_distribution = PriorDistribution(
        # Example: Setting a tighter prior on ROI if we have strong beliefs from past MMMs.
        # By default, Meridian uses a log-normal distribution for ROI priors.
        roi_m=tfd.LogNormal(loc=0.2, scale=0.5),

        # Example: Customizing Adstock decay limits.
        # Alpha represents the decay rate (0 to 1).
        alpha_m=tfd.Uniform(low=0.1, high=0.8),

        # Example: Customizing the Hill function's half-saturation point (ec).
        ec_m=tfd.TruncatedNormal(loc=0.5, scale=0.2, low=0.1, high=10.0),

        # You can also customize slope (hill_m), control variables (gamma_c), etc.
    )

    # 2. Define the Model Specification
    model_spec = ModelSpec(
        prior=prior_distribution,

        # Knots control the flexibility of the baseline time trend.
        # Rule of thumb: Total time periods divided by knots determines periods per knot.
        # Too many knots = overfitting. Too few = underfitting the baseline.
        knots=10,

        # Maximum lag for the Adstock decay function (in time periods).
        max_lag=8,

        # Whether the Hill function is applied before or after Adstock.
        # Default is False (Adstock -> Hill).
        hill_before_adstock=False,

        # How to parameterize the media effects. Options are 'roi' or 'mroi'.
        # 'roi' is generally recommended and easier to interpret as a prior.
        media_effects_dist='roi',
    )

    # 3. Initialize the Meridian model
    mmm = Meridian(input_data=input_data, model_spec=model_spec)

    # 4. Sample the Prior (Crucial step to validate the model setup before fitting)
    # This checks that the specified priors are valid and the model compiles.
    mmm.sample_prior(n_draws=100)
```
