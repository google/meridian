# About MMM Proto Schema

The MMM Proto Schema is a framework-agnostic data standard that provides a
consistent and serializable way to represent a trained Marketing Mix Model
(MMM). Its core purpose is to establish a common language for the outputs of an
MMM, primarily within its current Python-centric implementation. This allows the
results from models built using various tools or methodologies to be uniformly
represented, stored, shared, and compared by Python-based applications and
workflows. By offering this standardized representation, the schema aims to
enhance interoperability and facilitate downstream applications, such as scenario
planning, optimization, and consistent reporting, independent of how the original
model was constructed.

## Install Meridian with MMM Proto Schema
Currently, this package can only be installed from source code:
```sh
git clone https://github.com/google/meridian.git;
cd meridian;
pip install .[schema];
```
