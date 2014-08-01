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

    def evtsplit(self, events, Tpre, Tpost, t0=0.0, dt=0.005, 
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
        idx = binned.columns

        return chunklist, idx

    def psth(self, events, Tpre, Tpost, t0=0.0, rate=True, dt=0.005, timecolumn='time'):
        """
        Construct a peri-stimulus time histogram of the data in self in
        an interval (Tpre, Tpost) relative to each event in events. Tpre
        should be negative for times preceding events. Accepts either
        a dataframe of timestamps or a dataframe of binned counts. Returns
        a Spikeset, one column per unique combination of columns in self
        (excluding timestamps) in the case of timestamp input or one 
        column per column in the case of binned input. If rate=True, 
        returns the mean spike rate across events. If rate is false, returns
        raw counts in each time bin.
        """

        # if already binned, override dt
        if self.index.name == 'time':
            dt = self.index[1] - self.index[0]

        chunks, colnames = self.evtsplit(events, Tpre, Tpost, t0, dt, timecolumn)

        if rate:
            means = pd.concat([ck.mean(axis=1) for ck in chunks], axis=1)
            outframe = means / dt
        else:
            outframe = pd.concat([ck.sum(axis=1) for ck in chunks], axis=1)

        outframe.columns = colnames
        return Spikeset(outframe)

    def smooth(self, window):
        """
        *Causal* boxcar smooth of the data in dataframe. window is the length
        of the smoothing window (in seconds).
        """

        dt = self.index[1] - self.index[0]
        winlen = np.floor(window / dt)

        return Spikeset(pd.stats.moments.rolling_mean(self, winlen, 
            min_periods=0))



