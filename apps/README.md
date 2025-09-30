# Advantage Medical Professionals MMM App

This directory hosts a Streamlit experience that wraps Meridian's modeling and
analysis capabilities in a branded workflow for the Advantage Medical
Professionals team.

## Features

* Upload or explore sample Advantage Medical Professionals marketing data.
* Configure KPI, revenue, control, and media channel mappings without editing
  code.
* Launch a lightweight Meridian run and explore ROI, incremental contribution,
  and optimization levers through interactive summaries.
* Generate scenario simulations by adjusting channel investments and comparing
  the projected impact on high-value staffing KPIs.

## Getting started

1. Install the Google Meridian package. Pick the command that matches your
   environment:
   * **CPU-only / macOS**
     ```bash
     pip install --upgrade 'google-meridian'
     ```
   * **Linux with CUDA GPU**
     ```bash
     pip install --upgrade 'google-meridian[and-cuda]'
     ```
   * **Latest development build**
     ```bash
     pip install --upgrade git+https://github.com/google/meridian.git
     ```
2. Install the UI extras from this repository:
   ```bash
   pip install -e .[ui]
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run apps/advantage_streamlit_app.py
   ```
4. The app opens in your browser and loads the sample Advantage dataset by
   default. Use the sidebar to map your own columns or upload fresh CSV exports
   from Advantage Medical Professionals' systems.

The app is designed to run locally for rapid experimentation. When you are
ready to productionize, pair it with a proper data pipeline and model
monitoring strategy.
