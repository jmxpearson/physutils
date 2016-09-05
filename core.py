import numpy as np
import pandas as pd
import scipy.signal as ssig
import warnings
from functools import reduce

def _arrdecimate(x, decfrac, axis=-1):
    """
    just like in the scipy source code, except I use filtfilt
    and opt for hardcode defaults
    q is the fraction to decimate by; length of returned data is len(x)/q
    """
    if decfrac > 15:
        wrnstr = """You are attempting to decimate by a factor > 15. You are risking numerical instability. Consider performing multiple successive decimations instead."""
        warnings.warn(wrnstr)

    n = 8
    b, a = ssig.filter_design.cheby1(n, 0.05, 0.8 / decfrac)

    y = ssig.filtfilt(b, a, x, axis=axis)

    sl = [slice(None)] * y.ndim
    sl[axis] = slice(None, None, decfrac)
    return y[sl]

def decimate(df, decfrac):
    """
    Decimate a dataframe, handling indices and columns appropriately.
    decfrac can be an iterable of successive decimations
    """
    newdf = pd.DataFrame(df)  # upcast from Series, if needed

    # if we passed an int, make it a tuple
    if isinstance(decfrac, int):
        decfrac = (decfrac,)

    for frac in decfrac:
        tindex = newdf.index[::frac]
        parts = [pd.DataFrame(_arrdecimate(aa[1], frac), columns=[aa[0]]) for aa in newdf.items()]
        outdf = pd.concat(parts, axis=1)
        outdf.index = tindex
        outdf.index.name = newdf.index.name
    return outdf

def _binseries(df, dt, col='time'):
    """
    Convert df, a Pandas dataframe of spike timestamps, to a binned
    histogram with bin width dt.
    """
    maxT = np.max(df[col])
    maxbin = np.ceil(maxT / dt)
    binT = np.arange(0, maxbin * dt, dt)
    binned = np.histogram(df[col], binT)[0]
    dframe = pd.DataFrame(binned, index=pd.Index(binT[:-1], name='time'),
        columns=['count'])
    return dframe

def binspikes(df, dt, timecolumn='time'):
    # if df is a frame of spike times, return a histogrammed set of spike
    # counts in each time bin

    id_keys = list(set(df.columns).difference({timecolumn}))

    grp = df.groupby(id_keys)
    binned = grp.apply(lambda x: _binseries(x, dt, timecolumn))
    binned = binned.unstack(level=id_keys)

    # drop counts label that clutters MultiIndex
    binned.columns = binned.columns.droplevel(0)

    return binned

def smooth(df, wid):
    """
    *Causal* boxcar smooth of the data in dataframe. window is the length
    of the smoothing window (in seconds).
    """

    ts = df.index[1] - df.index[0]
    wlen = np.round(wid/ts)
    return pd.rolling_mean(df, wlen, min_periods=1)

def norm_by_trial(timetuple, method='division'):
    """
    Given a list (one per trial) of dataframes, return a function that
    returns a list of the same type in which each trial is normalized by 
    the mean of the dataframe values in the range given by timetuple.
    """
    def norm_by_range(df):
        baseline = df[slice(*timetuple)].mean()
        if method == 'division':
            return df.div(baseline)
        elif method == 'subtraction':
            return df - baseline

    def normalize(framelist):
        return list(map(norm_by_range, framelist))

    return normalize

def norm_by_mean(timetuple, method='division'):
    """
    Given a list (one per trial) of dataframes, return a normalizer 
    function that returns a list of the same type in which all trials 
    are normalized by the mean (across time) of the mean (across frames)
    of the dataframe values in the range given by timetuple.
    """
    def get_baseline(df):
        return df[slice(*timetuple)].mean()

    def normalize(framelist):
        all_baselines = list(map(get_baseline, framelist))
        mean_baseline = reduce(lambda x, y: x.add(y, fill_value=0), all_baselines) / len(framelist)
        if method == 'division':
            return [x.div(mean_baseline) for x in framelist]
        elif method == 'subtraction':
            return [x - mean_baseline for x in framelist]

    return normalize

def _splitseries(df, ts, Tpre, Tpost, t0=0.0):
    """
    Split time series data into peri-event chunks. Data are in df.
    Times of events around which to split are in ts. 
    Code grabs Tpre:Tpost bins relative to event, so times before 
    the event have Tpre < 0. The first time bin is assumed to have 
    timestamp t0.
    """
    dt = df.index[1] - df.index[0]
    xx = df.values.squeeze()

    nevt = ts.size
    nstart = np.ceil(Tpre / dt)
    nend = np.ceil(Tpost / dt)
    binT = np.arange(nstart, nend) * dt

    evtrel = ts - t0

    elist = []
    for time in evtrel:
        offset = np.around(time / dt)
        ss = slice(nstart + offset, nend + offset)
        elist.append(pd.DataFrame(xx[ss], columns=[time]))
    alltrials = pd.concat(elist, axis=1)
    alltrials = alltrials.set_index(binT)
    alltrials.index.name = 'time'
    alltrials.columns = pd.Index(np.arange(nevt), name='trial')
    return alltrials

def evtsplit(df, events, Tpre, Tpost, t0=0.0, dt=0.001, timecolumn='time', 
    return_by_event=False):
    """
    split frame into chunks (Tpre, Tpost) around each event in events
    Tpre should be < 0 for times before event
    if multiple series are passed, return a list of dataframes, one per series
    if return_by_event is True, return a list of dataframes, one per event
    """

    # first, check if df is binned; if not, do so
    if not df.index.name == 'time':
        binned = binspikes(df, dt, timecolumn)
    else:
        binned = df

    chunklist = []
    for col in binned.columns.values:
        chunklist.append(_splitseries(binned[col], events, Tpre, Tpost, t0))
    idx = binned.columns

    if return_by_event:
        # make a panel (3D array) of all this data
        panel = pd.Panel(dict(list(zip(idx, chunklist))))
        # transpose so that panel is events x time x series
        panel = panel.transpose(2, 1, 0)
        idx = panel.items
        chunklist = [p for event, p in panel.items()]

    return chunklist, idx

def _bandlimit_series(df, band=(0.01, 120)):
    """
    Computes bandpass-filtered version of time series in df.
    Band is either a two-element indexed sequence or a conventionally
    defined electrophysiological frequency band.
    """
    dt = df.index[1] - df.index[0]
    band_dict = {'delta': (0.1, 4), 'theta': (4, 8), 'alpha': (8, 13), 
    'beta': (13, 30), 'gamma': (30, 80)}

    # if band isn't a two-element sequence, it should be a string
    if isinstance(band, str):
        fband = band_dict[band]
    else:
        fband = band

    b, a = ssig.iirfilter(2, [2 * dt * f for f in fband], rp=0.1, rs=40,
        ftype='ellip')
    return df.apply(lambda x: ssig.filtfilt(b, a, x), raw=True)

def bandlimit(df, filters=None):
    """
    Convenience function for bandlimiting data frames. Handles
    indices and columns appropriately.
    """
    if filters is None:
        return df

    df = pd.DataFrame(df)
    nchan = df.shape[1]
    bands = [_bandlimit_series(df, f) for f in filters]
    allbands = pd.concat(bands, axis=1)
    
    # attend to labeling
    fstr = list(map(str, filters))
    bandpairs = list(zip(np.repeat(fstr, nchan), allbands.columns))
    bandnames = [b[0] + '.' + str(b[1]) for b in bandpairs]
    allbands.columns = bandnames

    return allbands

def psth(df, events, Tpre, Tpost, t0=0.0, rate=True, dt=0.001, timecolumn='time'):
    """
    Construct a peri-stimulus time histogram of the data in df in
    an interval (Tpre, Tpost) relative to each event in events. Tpre
    should be negative for times preceding events. Accepts either
    a dataframe of timestamps or a dataframe of binned counts. Returns
    a dataframe, one column per unique combination of columns in df
    (excluding timestamps) in the case of timestamp input or one 
    column per column in the case of binned input. If rate=True, 
    returns the mean spike rate across events. If rate is false, returns
    raw counts in each time bin.
    """

    # if already binned, override dt
    if df.index.name == 'time':
        dt = df.index[1] - df.index[0]

    chunks, colnames = evtsplit(df, events, Tpre, Tpost, t0, dt, timecolumn)

    if rate:
        means = pd.concat([ck.mean(axis=1) for ck in chunks], axis=1)
        outframe = means / dt
    else:
        outframe = pd.concat([ck.sum(axis=1) for ck in chunks], axis=1)

    outframe.columns = colnames
    return outframe


if __name__ == '__main__':
    pass
