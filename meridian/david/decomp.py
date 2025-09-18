"""Helpers to inspect baseline decomposition contributions by control."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import xarray as xr


__all__ = ["control_contribution"]


_METRICS = ("mean", "median", "ci_low", "ci_high")


def _ensure_sequence(values: Any) -> Sequence[Any] | None:
  if values is None:
    return None
  if isinstance(values, (str, bytes)):
    return [values]
  if isinstance(values, Sequence):
    return values
  try:
    return list(values)
  except TypeError:  # pragma: no cover - defensive fallback
    return [values]


def _resolve_control_index(mmm: Any, control: str | int) -> tuple[int, str]:
  n_controls = int(getattr(mmm, "n_controls", 0))
  if isinstance(control, int):
    c_idx = int(control)
    if c_idx < 0 or c_idx >= n_controls:
      raise IndexError(f"Control index {c_idx} is out of range (n_controls={n_controls}).")
    names = _ensure_sequence(getattr(mmm.input_data, "control_variable", None))
    if names is None:
      control_name = f"control_{c_idx}"
    else:
      control_name = names[c_idx]
    return c_idx, str(control_name)

  names = _ensure_sequence(getattr(mmm.input_data, "control_variable", None))
  if not names:
    raise ValueError("Control variable names are not available on input_data.")
  try:
    c_idx = list(names).index(control)
  except ValueError as exc:
    raise KeyError(
        f"Control '{control}' not found in control_variable names: {list(names)}"
    ) from exc
  return c_idx, str(control)


def _normalise_selection(
    labels: Sequence[Any],
    selection: Sequence[Any] | Iterable[bool] | None,
    *,
    allow_bool: bool,
) -> np.ndarray:
  all_idx = np.arange(len(labels), dtype=int)
  if selection is None:
    return all_idx

  if isinstance(selection, (str, bytes)):
    selection = [selection]

  if allow_bool:
    seq = _ensure_sequence(selection)
    if seq is not None and len(seq) == len(labels) and all(
        isinstance(val, (bool, np.bool_)) for val in seq
    ):
      mask = np.asarray(seq, dtype=bool)
      return all_idx[mask]

  seq = np.asarray(list(selection))
  if seq.dtype.kind in {"i", "u"}:
    return seq.astype(int)

  wanted = {str(val) for val in selection}
  return np.array(
      [idx for idx, label in enumerate(labels) if str(label) in wanted],
      dtype=int,
  )


def control_contribution(
    analyzer: Any,
    control: str | int,
    *,
    use_posterior: bool = True,
    use_kpi: bool = False,
    selected_geos: Sequence[Any] | Iterable[bool] | None = None,
    selected_times: Sequence[Any] | Iterable[bool] | None = None,
    aggregate_geos: bool = True,
    aggregate_times: bool = True,
    confidence_level: float = 0.95,
) -> xr.DataArray:
  """Compute contribution statistics for a single control variable."""
  if not 0.0 < confidence_level < 1.0:
    raise ValueError("confidence_level must be between 0 and 1.")

  mmm = analyzer._meridian
  controls_scaled = getattr(mmm, "controls_scaled", None)
  if controls_scaled is None or getattr(mmm, "n_controls", 0) == 0:
    raise ValueError("This Meridian model has no control variables.")

  c_idx, control_name = _resolve_control_index(mmm, control)

  geos = list(_ensure_sequence(getattr(mmm.input_data, "geo", range(mmm.n_geos))) or [])
  if not geos:
    geos = list(range(mmm.n_geos))
  times = list(_ensure_sequence(getattr(mmm.input_data, "time", range(mmm.n_times))) or [])
  if not times:
    times = list(range(mmm.n_times))

  if selected_geos is not None and getattr(mmm, "is_national", False):
    geo_idx = np.arange(mmm.n_geos, dtype=int)
  else:
    geo_idx = _normalise_selection(geos, selected_geos, allow_bool=False)
  time_idx = _normalise_selection(times, selected_times, allow_bool=True)

  controls_scaled = np.asarray(controls_scaled, dtype=np.float32)
  controls_scaled = controls_scaled[geo_idx][:, time_idx, c_idx]

  group = "posterior" if use_posterior else "prior"
  inference_data = getattr(mmm, "inference_data", None)
  if inference_data is None or not hasattr(inference_data, group):
    raise ValueError(f"Inference data does not contain '{group}' group.")

  ds = getattr(inference_data, group)
  if not hasattr(ds, "__getitem__") or "gamma_gc" not in ds:
    raise KeyError("Parameter 'gamma_gc' not found in inference data.")
  gamma = np.asarray(ds["gamma_gc"].values, dtype=np.float32)
  if gamma.ndim != 4 or gamma.shape[-2] < len(geo_idx):
    raise ValueError(f"Unexpected shape for gamma_gc: {gamma.shape}")
  gamma = gamma[:, :, geo_idx, c_idx]

  contrib = gamma[..., :, np.newaxis] * controls_scaled[np.newaxis, np.newaxis, :, :]

  if not use_kpi:
    revenue_per_kpi = getattr(mmm, "revenue_per_kpi", None)
    if revenue_per_kpi is None:
      if getattr(getattr(mmm, "input_data", None), "kpi_type", "").upper() == "REVENUE":
        revenue_per_kpi = np.ones((mmm.n_geos, mmm.n_times), dtype=np.float32)
      else:
        raise ValueError(
            "Revenue requested (use_kpi=False) but revenue_per_kpi is missing."
        )
    revenue_per_kpi = np.asarray(revenue_per_kpi, dtype=np.float32)
    revenue_per_kpi = revenue_per_kpi[geo_idx][:, time_idx]
    contrib = contrib * revenue_per_kpi[np.newaxis, np.newaxis, :, :]

  if aggregate_geos:
    contrib = contrib.sum(axis=2)
  if aggregate_times:
    contrib = contrib.sum(axis=-1)

  draws = contrib.reshape((contrib.shape[0] * contrib.shape[1],) + contrib.shape[2:])
  mean = draws.mean(axis=0)
  median = np.median(draws, axis=0)
  half_alpha = (1.0 - confidence_level) / 2.0
  lo = np.quantile(draws, half_alpha, axis=0)
  hi = np.quantile(draws, 1.0 - half_alpha, axis=0)

  stats = np.stack([mean, median, lo, hi], axis=-1)

  coords: dict[str, Any] = {}
  dims: list[str] = []
  if not aggregate_geos:
    coords["geo"] = [geos[i] for i in geo_idx]
    dims.append("geo")
  if not aggregate_times:
    coords["time"] = [times[i] for i in time_idx]
    dims.append("time")
  coords["metric"] = list(_METRICS)
  dims.append("metric")

  stats = stats.astype(np.float32)
  return xr.DataArray(
      stats,
      dims=dims,
      coords=coords,
      name=f"{control_name}_contribution",
      attrs={
          "distribution": group,
          "scale": "kpi" if use_kpi else "revenue",
          "control_variable": control_name,
          "confidence_level": confidence_level,
      },
  )
