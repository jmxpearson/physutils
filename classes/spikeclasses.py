"""
Spikeset is a subclass of the Pandas dataframe with methods for analysis of spike trains.
"""

from .. import core
import numpy as np
import pandas as pd
import warnings

class Spikeset(pd.DataFrame):
    # because we're inheriting directly from DataFrame, we don't need __init__
    
    @classmethod
    def from_csv(cls, fname):
        return Spikeset(super(Spikeset, cls).from_csv(fname))
    
    def bin(self, dt, timecolumn='time'):
        # if self is a frame of spike times, return a histogrammed set of spike
        # counts in each time bin

        id_keys = list(set(self.columns).difference({timecolumn}))

        grp = self.groupby(id_keys)
        binned = grp.apply(lambda x: core.binspikes(x, dt))
        binned = binned.unstack(level=id_keys)

        return Spikeset(binned)

