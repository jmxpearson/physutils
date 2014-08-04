import numpy as np

def _rle(x):
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

def _remove_short_runs(x, minlen, replace_val):
    """
    Given an array x, replace all runs shorter than minlen with the 
    value in replace_val. Returns a copy.
    """
    starts, runlens, values = _rle(x)
    to_replace = runlens < minlen
    stops = starts + runlens
    rngs = zip(starts[to_replace], stops[to_replace])
    newx = x.copy()
    for rng in rngs:
        newx[slice(*rng)] = replace_val
    return newx

def _hansmooth(x, wlen):
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

def censor_railing(x, thresh=3, toler=1e-2, minlen=10, smooth_wid=300):
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

    min_removed = _remove_short_runs(is_artifact, minlen, replace_val=False)
    return _hansmooth(min_removed, smooth_wid).astype('bool')
