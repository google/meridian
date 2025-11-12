# About MMM Proto Schema

The MMM Proto Schema is a language-agnostic data standard for representing
trained Marketing Mix Models (MMMs) and their analysis artifacts in a
consistent, serializable format. It establishes a common language for MMM
outputs, allowing results from different tools and methodologies to be uniformly
represented, stored, shared, and compared. This standardized representation
enhances interoperability and facilitates downstream applications like scenario
planning, optimization, and reporting.

## Install Meridian with MMM Proto Schema

Using PIP:

```sh
pip install --upgrade mmm-proto-schema
```

Alternatively, to install this package from source code:

```sh
git clone https://github.com/google/meridian.git
cd meridian
pip install .[schema]
```
