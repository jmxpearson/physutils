import numpy as np
import pandas as pd
import scipy.signal as ssig
from matplotlib.mlab import specgram
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import warnings

def decimate(x, decfrac, axis=-1):
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

def dfdecimate(df, decfrac):
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
        parts = [pd.DataFrame(decimate(aa[1], frac), columns=[aa[0]]) 
        for aa in newdf.iteritems()]
        newdf = pd.concat(parts, axis=1)
        newdf.index = tindex
        newdf.index.name = df.index.name
    return newdf

def binspikes(df, dt):
    """
    Convert df, a Pandas dataframe of spike timestamps, to a binned
    histogram with bin width dt.
    """
    maxT = np.max(df['time'])
    maxbin = np.ceil(maxT / dt)
    binT = np.arange(0, maxbin * dt, dt)
    binned = np.histogram(df['time'], binT)[0]
    dframe = pd.DataFrame(binned, index=pd.Index(binT[:-1], name='time'),
        columns=['count'])
    return dframe

def dfsmooth(df, wid):
    """
    performs smoothing by a window of width wid (in s); data in df
    reflect data at both ends to minimize edge effect
    smooths using a centered window, which is non-causal
    """
    ts = df.index[1] - df.index[0]
    x = df.values.squeeze()
    wlen = np.round(wid/ts)
    return pd.DataFrame(smooth(x, wlen), index=df.index, columns=[''])

def smooth(x, wlen):
    """
    Performs smoothing on x via a hanning window of wlen samples
    centered on x.
    """
    ww = np.hanning(wlen)
    # grab first wlen samples, reverse them, append to front,
    # grab last wlen samples, reverse, append to end
    xx = np.r_[x[wlen-1:0:-1], x, x[-1:-wlen:-1]]
    y = np.convolve(ww/ww.sum(),xx, mode='valid')
    return y[(wlen/2 - 1):-wlen/2]


def spectrogram(series, winlen, frac_overlap):
    """
    Given a Pandas series, a window length (in s), and a percent 
    overlap between successive windows, calculate the spectrogram
    based on the moving window stft.
    Returns a DataFrame with frequencies in columns and time in rows
    Uses matplotlib.mlab.specgram
    """
    dt = series.index[1] - series.index[0]
    sr = 1. / dt
    Nwin = int(np.ceil(winlen / dt))
    Npad = int(2 ** np.ceil(np.log2(Nwin)))  # next closest power of 2 to pad to
    Noverlap = min(int(np.ceil(frac_overlap * Nwin)), Nwin - 1)

    spec = specgram(series.values, NFFT=Nwin, Fs=sr, noverlap=Noverlap, 
        pad_to=Npad)
    return pd.DataFrame(spec[0].T, columns=spec[1], index=spec[2] + series.index[0])

def continuous_wavelet(series, freqs=None, *args, **kwargs):
    """
    Construct a continuous wavelet transform for the data series.
    Extra pars are parameters for the Morlet wavelet.
    Returns a tuple (time-frequency matrix, frequencies, times)
    """
    if not freqs:
        # define some default LFP frequencies of interest
        freqlist = [np.arange(1, 13), np.arange(15, 30, 3), np.arange(35, 100, 5)]
        freqs = np.concatenate(freqlist)

    dt = series.index[1] - series.index[0]
    scales = 1. / (freqs * dt)  # widths need to be in samples, not seconds
    if 'w' in kwargs:
        wav = make_morlet(kwargs['w'])
    else:
        wav = make_morlet(*args)
    rwavelet = lambda N, b: np.real(wav(N, b))
    iwavelet = lambda N, b: np.imag(wav(N, b))
    tfr = ssig.cwt(series.values, rwavelet, scales)
    tfi = ssig.cwt(series.values, iwavelet, scales)
    tf = tfr ** 2 + tfi ** 2

    return pd.DataFrame(tf.T, columns=freqs, index=series.index)

def make_morlet(w=np.sqrt(2) * np.pi):
    """
    Coded to conform to the requirements of scipy.signal.cwt.
    WARNING: scipy.signal.morlet does not follow the recommended convention!
    Default value of w corresponds (up to a rescaling) to the 1-1 
    complex Morlet wavelet in Matlab.
    """
    def wav(N, b):
        x = (np.arange(0, N) - (N - 1.0) / 2) / b
        output = np.exp(1j * w * x) - np.exp(-0.5 * (w ** 2))
        output *= np.exp(-0.5 * (x ** 2)) * (np.pi ** (-0.25))
        output /= np.sqrt(b)
        return output

    return wav

def plot_time_frequency(spectrum, **kwargs):
    """
    Time-frequency plot. Modeled after image_nonuniform.py example 
    spectrum is a dataframe with frequencies in columns and time in rows
    """
    times = spectrum.index
    freqs = spectrum.columns
    z = 10 * np.log10(spectrum.T)
    ax = plt.figure().add_subplot(111)
    extent = (times[0], times[-1], freqs[0], freqs[-1])
    if 'interpolation' in kwargs:
        interp = kwargs['interpolation']
    else:
        interp = 'bilinear'
    im = NonUniformImage(ax, interpolation=interp, extent=extent)
    im.set_data(times, freqs, z)
    if 'clim' in kwargs:
        im.set_clim(kwargs['clim'])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.images.append(im)
    plt.colorbar(im, label='Power (dB/Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    return plt.gcf() 

def avg_time_frequency(series, tffun, events, Tpre, Tpost, *args, **kwargs):
    """
    Given a Pandas series, split it into chunks of (Tpre, Tpost) around
    events, do the time-frequency on each using the function tffun,
    and return a DataFrame with time as the index and frequency as the 
    column label.
    Note that, as per evtsplit, events before the events have Tpre < 0.
    *args and **kwargs are passed on to tffun
    """
    specmats, times, freqs = per_event_time_frequency(series, tffun, events, Tpre, Tpost, *args, **kwargs)
    
    return mean_from_events(specmats, times, freqs)

def mean_from_events(specmats, times, freqs):
    """
    Given a list containing time-frequency spectra, average them and
    return a dataframe (time x frequency).
    """
    allspecs = np.dstack(specmats)
    meanspec = np.nanmean(allspecs, axis=2)

    return pd.DataFrame(meanspec, index=times, columns=freqs)

def norm_by_trial(timetuple):
    """
    Given a list (one per trial) of dataframes, return a function that
    returns a list of the same type in which each trial is normalized by 
    the mean of the dataframe values in the range given by timetuple.
    """
    def norm_by_range(df):
        baseline = df[slice(*timetuple)].mean()
        return df.div(baseline)

    def normalize(framelist):
        return map(norm_by_range, framelist)

    return normalize

def norm_by_mean(timetuple):
    """
    Given a list (one per trial) of dataframes, return a function that
    returns a list of the same type in which all trials are normalized
    by the mean (across time) of the mean (across frames) 
    of the dataframe values in the range given by timetuple.
    """
    def get_baseline(df):
        return df[slice(*timetuple)].mean()

    def normalize(framelist):
        all_baselines = map(get_baseline, framelist)
        mean_baseline = reduce(lambda x, y: x.add(y, fill_value=0), all_baselines) / len(framelist)
        return map(lambda x: x.div(mean_baseline), framelist)

    return normalize

def per_event_time_frequency(series, tffun, events, Tpre, Tpost, *args, **kwargs):
    """
    Given a Pandas series, split it into chunks of (Tpre, Tpost) around
    events, do the time-frequency on each using the function tffun,
    and return a tuple containing the list of time-frequency matrices
    (time x frequency), an array of times, and an array of frequencies.
    """
    df = evtsplit(series, events, Tpre, Tpost)
    spectra = [tffun(ser, *args, **kwargs) for (name, ser) in df.iteritems()]
    if 'normfun' in kwargs:
        spectra = kwargs['normfun'](spectra)
    specmats = map(lambda x: x.values, spectra)
    times = spectra[0].index
    freqs = spectra[0].columns
    return (specmats, times, freqs)

def evtsplit(df, ts, Tpre, Tpost, t0=0):
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

def bandlimit(df, band=(0.01, 120)):
    """
    Computes bandpass-filtered version of time series in df.
    Band is either a two-element indexed sequence or a conventionally
    defined electrophysiological frequency band.
    """
    dt = df.index[1] - df.index[0]
    band_dict = {'delta': (0.1, 4), 'theta': (4, 8), 'alpha': (8, 13), 
    'beta': (13, 30), 'gamma': (30, 100)}

    # if band isn't a two-element sequence, it should be a string
    if isinstance(band, str):
        fband = band_dict[band]
    else:
        fband = band

    b, a = ssig.iirfilter(2, [2 * dt * f for f in fband], rp=0.1, rs=40,
        ftype='ellip')
    return df.apply(lambda x: ssig.filtfilt(b, a, x), raw=True)

def dfbandlimit(df, filters=None):
    """
    Convenience function for bandlimiting data frames. Handles
    indices and columns appropriately.
    """
    if filters is None:
        return df

    df = pd.DataFrame(df)
    nchan = df.shape[1]
    bands = [bandlimit(df, f) for f in filters]
    allbands = pd.concat(bands, axis=1)
    
    # attend to labeling
    fstr = map(str, filters)
    bandpairs = zip(np.repeat(fstr, nchan), allbands.columns)
    bandnames = [b[0] + '.' + str(b[1]) for b in bandpairs]
    allbands.columns = bandnames

    return allbands

def rle(x):
    """
    Perform run length encoding on the numpy array x. Returns a tuple
    of start indices, run lengths, and values for each run.
    """
    # add infinity to beginning to x[0] always starts a run
    dx = np.diff(np.insert(x.flatten(), 0, np.inf))
    starts = np.flatnonzero(dx)
    # make sure stops always include last element
    stops = np.append(starts[1:], np.size(x))
    runlens = stops - starts
    values = x[starts]
    return starts, runlens, values

def remove_short_runs(x, minlen, replace_val):
    """
    Given an array x, replace all runs shorter than minlen with the 
    value in replace_val. Returns a copy.
    """
    starts, runlens, values = rle(x)
    to_replace = runlens < minlen
    stops = starts + runlens
    rngs = zip(starts[to_replace], stops[to_replace])
    newx = x.copy()
    for rng in rngs:
        newx[slice(*rng)] = replace_val
    return newx

def censor_railing(x, thresh=4, toler=1e-2, minlen=10, smooth_wid=300):
    """
    Given a numpy array x, censor the dataset by detecting and removing 
    artifacts due to railing signal, returning a boolean 
    array with the same shape as a flattened x, suitable for 
    masking (i.e., True for censored data).

    Censoring happens as follows:
    * Mark all data exceeding a threshold of thresh * sig, where sig
    is an estimate of the standard deviation of the data based on the median.
    * Mark all points at which the derivative of the data is less than 
    a tolerance toler * sig.
    * Take the intersection of these sets. Remove any sets of consecutive 
    points smaller than minlen as putative false positives.
    * Expand the censoring region by smearing with a kernel of size 
    smooth_wid.
    """

    # first off, flatten and de-mean
    xx = x.ravel() - np.mean(x)

    sig = np.median(abs(xx)) / 0.67449  # median(abs(xx)) / Phi^{-1}(0.75)

    dx = np.diff(xx)
    dx = np.insert(dx, 0, np.inf)  # make same length as xx

    is_artifact = np.logical_and(np.abs(dx) < toler * sig, 
        np.abs(xx) > thresh * sig)

    min_removed = remove_short_runs(is_artifact, minlen, replace_val=False)
    return smooth(min_removed, smooth_wid).astype('bool')




if __name__ == '__main__':
    pass