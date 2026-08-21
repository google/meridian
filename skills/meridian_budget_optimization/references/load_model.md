# Load Model

Use `meridian_serde.load_meridian()` to load a serialized Meridian model.

```python
from meridian.schema.serde import meridian_serde

model = meridian_serde.load_meridian(model_path)
```

> [!IMPORTANT] `meridian_model.binpb` is a **binary file**. Do NOT try to read
> its content directly using file viewing tools or grep, as this will produce
> invalid UTF-8 errors. Always use `meridian_serde.load_meridian()` in a Python
> script to load it.
