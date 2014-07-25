import h5py
import numpy as np
import pandas as pd
import pandas.io.pytables as pdtbl
import scipy.signal as ssig
from matplotlib.mlab import specgram
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
from unionfind import UnionFind
import warnings

def make_path(*tup):
    abbr = ['p', 'd', 'c', 'u'][:len(tup)]
    nstrs = map(str, tup)
    pieces = [a + b for a,b in zip(abbr, nstrs)]
    return '/'.join(pieces)

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

def smooth(df, wid):
    """
    performs smoothing by a window of width wid (in s); data in df
    reflect data at both ends to minimize edge effect
    smooths using a centered window, which is non-causal
    """
    ts = df.index[1] - df.index[0]
    x = df.values.squeeze()
    wlen = np.round(wid/ts)
    ww = np.hanning(wlen)
    # grab first wlen samples, reverse them, append to front,
    # grab last wlen samples, reverse, append to end
    xx = np.r_[x[wlen-1:0:-1], x, x[-1:-wlen:-1]]
    y = np.convolve(ww/ww.sum(),xx, mode='valid')
    y = y[(wlen/2 - 1):-wlen/2]
    return pd.DataFrame(y, index=df.index, columns=[''])

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
    im = NonUniformImage(ax, interpolation='bilinear', extent=extent)
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

def diff_mean_pow(arraylist, labels):
    """
    Given a list of arrays and an array of labels designating conditions,
    calculate the mean for each label on a decibel scale. Return 
    difference in dB between the two conditions.
    """
    multarray = np.array(arraylist) # convert to array along dim 0
    lls = np.array(labels)  # make sure this is an array
    arr0 = multarray[lls == 0]
    arr1 = multarray[lls == 1]
    m0 = np.nanmean(arr0, 0)
    m1 = np.nanmean(arr1, 0)
    z0 = 10 * np.log10(m0)
    z1 = 10 * np.log10(m1)
    return z0 - z1

def diff_t_stat(arraylist, labels):
    multarray = np.array(arraylist) # convert to array along dim 0
    lls = np.array(labels)  # make sure this is an array
    arr0 = multarray[lls == 0]
    arr1 = multarray[lls == 1]
    tmap = tstats(arr0, arr1, equal_var=False)[0]
    return tmap

def tstats(a, b, axis=0, equal_var=True):
    """
    Version of scipy.stats.ttest_ind rewritten to handle NaNs
    in input. Only returns tstatistic and df, not p-value.
    """
    if a.size == 0 or b.size == 0:
        return (np.nan, np.nan)

    v1 = np.nanvar(a, axis, ddof=1)
    v2 = np.nanvar(b, axis, ddof=1)
    n1 = a.shape[axis]
    n2 = b.shape[axis]

    if (equal_var):
        df = n1 + n2 - 2
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(df)
        denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        df = ((vn1 + vn2)**2) / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

        # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
        # Hence it doesn't matter what df is as long as it's not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = np.sqrt(vn1 + vn2)

    d = np.nanmean(a, axis) - np.nanmean(b, axis)
    t = np.divide(d, denom)

    return t, df

def make_thresholded_diff(arraylist, labels, lo=-np.inf, hi=np.inf, diff_fun=diff_t_stat):
    """
    Given a list of arrays and an array of labels designating conditions, 
    calculate a difference map based on diff_fun. Return a masked array
    censored outside the interval (lo, hi).
    """
    
    diffarray = diff_fun(arraylist, labels) 
    return np.ma.masked_inside(diffarray, lo, hi)

def select_clusters(arr, cluster_inds):
    """
    Given an array with entries corresponding to cluster labels
    and an iterable of cluster indices, return a Boolean array 
    corresponding to which entries in arr are labeled with any of
    the indices in cluster_inds.
    """
    boolarr = np.empty( (len(cluster_inds),) + arr.shape)
    for ind, cnum in enumerate(cluster_inds):
        boolarr[ind] = arr == cnum
    return np.any(boolarr, 0)

def threshold_clusters(arraylist, labels, lo=-np.inf, hi=np.inf, keeplo=None, keephi=None, diff_fun=diff_t_stat):
    """
    Given a list of arrays corresponding to single trials, a list of labels
    corresponding to classes, lo and hi thresholds, and keeplo and keephi
    significance thresholds for clusters, return an array
    the size of arraylist[0] corresponding to True whenver the entry 
    corresponds to a cluster larger than the significance threshold.
    """
    # get entries above hi threshold and below low threshold
    pos = make_thresholded_diff(arraylist, labels, hi=hi, diff_fun=diff_fun)
    neg = make_thresholded_diff(arraylist, labels, lo=lo, diff_fun=diff_fun)

    # label clusters 
    posclus = label_clusters(pos)
    negclus = label_clusters(neg)
   
    # get cluster masses
    szhi = get_cluster_masses(pos, posclus)
    szlo = get_cluster_masses(neg, negclus)
   
    # which cluster numbers exceed size thresholds? 
    hi_inds = np.flatnonzero(szhi >= keephi) 
    lo_inds = np.flatnonzero(szlo <= keeplo)
    
    # get rid of cluster labeled 0, which is background
    hi_inds = np.setdiff1d(hi_inds, np.array(0))
    lo_inds = np.setdiff1d(lo_inds, np.array(0))
    
    # get Boolean mask for positive and negative clusters
    hi_img = select_clusters(posclus, hi_inds)
    lo_img = select_clusters(negclus, lo_inds)
    
    newmask = ~np.logical_or(hi_img, lo_img)
    final_clusters = pos.copy()
    final_clusters.mask = newmask
    return final_clusters

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

def label_clusters(img):
    """
    Given a masked array, label all masked elements with 0 and each unmasked
    elements with an integer index of the connected component to which
    it belongs. 
    Assumes a 4-neighborhood for connectivity. Returns labeled array
    of same shape as input.
    """
    clust_map = np.zeros(img.shape, dtype='int64')
    uf = UnionFind()

    # extract mask from passed image
    # if no masking, img.mask will only be a scalar False,
    # in which case expand to array
    if img.mask.size == 1:
        im_mask = img.mask * np.zeros_like(img)
    else:
        im_mask = img.mask

    # first loop: traverse image by pixels, constructing union-find
    # for connected components
    it = np.nditer(img, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index

        # if present cell is not masked
        if not im_mask[idx]:
            uf.add(idx)  # add to union-find
            if idx[0] > 0:
                left = (idx[0] - 1, idx[1])
                if uf.find(left):
                    uf.union(left, idx)  # attach to left neighbor
            if idx[1] > 0:
                above = (idx[0], idx[1] - 1)
                if uf.find(above):
                    uf.union(above, idx)  # attach to upper neighbor

        it.iternext()

    # get roots of union-find, construct a code dict to relabel them
    # as integers
    roots = set(map(lambda x: x[0], uf.nodes.values()))
    code_dict = dict(zip(roots, np.arange(1, len(roots) + 1)))

    # second pass: label by root
    it = np.nditer(img, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index

        if not im_mask[idx]:
            clust_map[idx] = code_dict[uf.find(idx)[0]]

        it.iternext()

    # get unique cluster labels and their mappings into the image
    # note that since uniques are sorted, the value 0 in the image
    # always maps to the zeroth position in uniques, so labeling of
    # non-cluster background is preserved
    uniques, indices = np.unique(clust_map, return_inverse=True)

    return np.reshape(indices, img.shape)

def get_cluster_masses(arr, indices):
    """
    Given a masked array of activations and an array of the same size
    containing labels, return an array of cluster masses, caculated
    as the sum of all weights in arr for each nonzero index in inds.
    """
    # get counts of each index, weighted by arr
    counts = np.bincount(indices.ravel(), arr.ravel())
    counts = counts[1:]  # get rid of "cluster" labeled 0
    return counts
    
def fetch(dbname, node, *args):
    """
    Given a node ('lfp', 'spikes', 'events', 'censor'), and
    a tuple of (patient, dataset, channel, unit), retrieves data.
    """
    target = node + '/' + make_path(*args)
    return pd.read_hdf(dbname, target)

def fetch_metadata(dbname, node, *args):
    # have to get metadata this way because we didn't store it via pandas
    # (which currently has no support for dataframe metadata)
    target = node + '/' + make_path(*args)
    fobj = h5py.File(dbname, 'r')
    attdict = dict(fobj[target].attrs)
    fobj.close()
    return attdict

def fetch_all_such(dbname, node, *args, **kwargs):
    """
    Given an incomplete specification in args, get all datasets consistent
    with it. Return a single dataframe.
    If database keys are precomputed to save time, they can be specified.
    """
    if 'keys' in kwargs:
        keys = kwargs['keys']
    else:
        keys = pdtbl.HDFStore(dbname).keys()

    glob = node + '/' + make_path(*args)
    # do really simple regex matching
    matches = [k for k in keys if glob in k]

    if matches:
        parts = []
        for m in matches:
            parts.append(fetch(dbname, m))
        excl = pd.concat(parts)
    else:
        excl = pd.DataFrame([])

    return excl

def get_censor(dbname, taxis, *args):
    """
    Convenience function for retrieving censoring intervals from the db
    and converting to logical arrays, one entry for each time point.
    args = patient, dataset, channel
    Assumes timestamp range equal to that of lfp.
    """
    
    # make sure tindex is of type float64
    ##### should be able to remove when pandas is upgraded to 0.13, which 
    ##### allows for float64 indices
    taxis = pd.Series(taxis.values.astype('float64'))

    # get censoring intervals, group by channel
    censors = fetch_all_such(dbname, 'censor', *args)
    if not censors.empty:
        censors = censors.groupby('channel')
    else:
        return censors
    
    # arrange start and stop times into linear sequence
    # (the extra pair of braces around the return value is to prevent
    # pandas from converting the time array to a series when apply gets only
    # a single return value (i.e., when censors has only a single group))
    if censors.ngroups > 1:
        flatfun = lambda x: x[['start', 'stop']].values.ravel()
    else:
        flatfun = lambda x: [x[['start', 'stop']].values.ravel()]
    censbins = censors.apply(flatfun)
    
    # append 0 and inf to bins
    censbins = censbins.apply(lambda x: np.append([0], x))
    censbins = censbins.apply(lambda x: np.append(x, np.inf))
    # bin times in taxis; censored bins will have even indices
    binnum = censbins.apply(lambda x: np.digitize(taxis, x))
    binnum = binnum.apply(lambda x: x % 2 == 0)

    excludes = pd.concat([pd.Series(b) for b in binnum], axis=1)
    excludes.columns = binnum.index  # channel names
    excludes.index = taxis
    
    return excludes
   
def censor_spikes(df, dbname, dtup):
    excludes = get_censor(dbname, df.index, *dtup)
    if not excludes.empty:
        excludes = excludes[excludes.columns.intersection(
            df.columns)]
        excl_vec = np.any(excludes.values, axis=1)
        newdf = df.copy()
        newdf[excl_vec] = np.nan
        return newdf
    else:
        return df

def load_spikes(dbname, dtup):
    spks = fetch(dbname, 'spikes', *dtup)

    spkbin = binspikes(spks, 0.05)  # use 50 ms bins

    return censor_spikes(spkbin, dbname, dtup)

if __name__ == '__main__':

    # get all spikes for a given unit 
    # df = getSpikes(18, 1, 1, 1)
    dbname = '/home/jmp33/data/bartc/plexdata/bartc.hdf5'
    df = fetch(dbname, 'spikes', 18, 1, 1, 1)

    binsize = 0.050  # 50 ms bin
    binned = binspikes(df, binsize)

    evt = fetch(dbname, 'events', 18, 1)['banked'].dropna()

    psth = evtsplit(binned, evt, -1, 1).mean(axis=1)

    smpsth = smooth(psth, 0.4)

    df = fetch_all_such(dbname, 'spikes', 17, 2)

    df = fetch(dbname, 'lfp', 18, 1, 17)
    df = df.set_index('time')