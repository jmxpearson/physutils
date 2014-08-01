"""
Spikeset is a subclass of the Pandas dataframe with methods for analysis 
of spike trains.
"""

from .. import core
import numpy as np
import pandas as pd
import warnings

class Spikeset(pd.DataFrame):
    # because we're inheriting directly from DataFrame, we don't need __init__
    
    def bin(self, dt, timecolumn='time'):
        # if self is a frame of spike times, return a histogrammed set of spike
        # counts in each time bin

        id_keys = list(set(self.columns).difference({timecolumn}))

        grp = self.groupby(id_keys)
        binned = grp.apply(lambda x: core.binspikes(x, dt))
        binned = binned.unstack(level=id_keys)

        # drop counts label that clutters MultiIndex
        binned.columns = binned.columns.droplevel(0)

        return Spikeset(binned)

    def evtsplit(self, events, Tpre, Tpost, t0=0.0, dt=0.05, 
        timecolumn='time'):
        # split frame into chunks (Tpre, Tpost) around each event in events
        # Tpre should be < 0 for times before event
        # if multiple series are passed, return a list of dataframes

        # first, check if self is binned; if not, do so
        if not self.index.name == 'time':
            binned = self.bin(dt, timecolumn)
        else:
            binned = self

        chunklist = []
        for col in binned.columns.values:
            chunklist.append(Spikeset(core.evtsplit(binned[col], events, 
                Tpre, Tpost, t0)))

        return chunklist