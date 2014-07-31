## Physutils: Yet another Python library for electrophysiology

### Introduction:

The purpose of physutils is to collect together a useful set of classes and functions for analyzing extracellular neurophysiology data, namely, spikes and local field potentials.

Currently, `import physutils` imports all functions in `core` and the `LFPset` class in `/classes/lfpclasses` into the `physutils` namespace.

### Organization:

* `core` contains all core functions in the library, including those for decimation, filtering, histogramming, time-frequency analysis, and some plotting.
* `classes` includes helpful classes for analysis.
  * `lfpclasses` contains the `LFPset` class, which is a thin wrapper around a Pandas dataframe (dataframe + metadata) with methods for decimation, filtering, time-frequency analysis, and significance testing.
  * `unionfind` is an implementation of the union-find data structure used by the cluster-finding algorithm in `bootstrap`.
* `bootstrap` contains functions to implement a permutation significance test for differences between time-frequency plots for pairs of conditions.
* `examples` contains iPython notebooks illustrating typical use cases of the classes and functions in the package.

### Dependencies:
* Numpy
* SciPy
* Pandas
* Matplotlib
