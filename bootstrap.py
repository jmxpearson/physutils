"""
Functions needed to allow for bootstrap significance testing of 
time-frequency data.
"""

import numpy as np
from .classes import unionfind
from . import tf
from scipy.signal import convolve2d

def diff_t_stat(arraylist, labels):
    multarray = np.array(arraylist) # convert to array along dim 0
    lls = np.array(labels)  # make sure this is an array
    arr0 = multarray[lls == 0]
    arr1 = multarray[lls == 1]
    tmap = tstats(arr0, arr1, equal_var=False)[0]
    return tmap

def t_of_log(arraylist, labels):
    multarray = np.array(arraylist) # convert to array along dim 0
    lls = np.array(labels)  # make sure this is an array
    arr0 = 10 * np.log10(multarray[lls == 0])
    arr1 = 10 * np.log10(multarray[lls == 1])
    tmap = tstats(arr0, arr1, equal_var=False)[0]
    return tmap

def F_stat(arraylist, labels):
    multarray = np.array(arraylist) # convert to array along dim 0
    lls = np.array(labels)  # make sure this is an array
    arr0 = multarray[lls == 0]
    arr1 = multarray[lls == 1]

    # if each element of arr0 is chi2(1), then the mean of d such 
    # arrays is chi2(d), and a ratio of such chi2 variables is F(d1, d2)
    chi2n = np.nanmean(arr0, axis=0) / np.nanstd(arr0, axis=0)
    chi2d = np.nanmean(arr1, axis=0) / np.nanstd(arr1, axis=0)
    Fmap = chi2n / chi2d
    return Fmap

def normalized_diff_mean_power(arraylist, labels):
    multarray = np.array(arraylist) # convert to array along dim 0
    lls = np.array(labels)  # make sure this is an array
    # convert to log scale 
    arr0 = np.log(multarray[lls == 0])
    arr1 = np.log(multarray[lls == 1])

    m0 = np.nanmean(arr0, axis=0)
    m1 = np.nanmean(arr1, axis=0)

    s0 = np.nanstd(arr0, axis=0)
    s1 = np.nanstd(arr1, axis=0)

    n0 = np.sum(lls == 0)
    n1 = np.sum(lls == 1)

    numer = m0 - m1 + 0.5 * (s0 ** 2 - s1 ** 2)
    denom = np.sqrt((s0 ** 2 / n0) + (s1 **2 / n1) + (s0 ** 4 / (n0 - 1)) +
        (s1 ** 4 / (n1 - 1)))

    return numer / denom


def log_F_stat(arraylist, labels):
    return 10 * np.log10(F_stat(arraylist, labels))

def tstats(a, b, axis=0, equal_var=True):
    """
    Version of scipy.stats.ttest_ind rewritten to handle NaNs
    in input. Only returns tstatistic and df, not p-value.
    """
    if a.size == 0 or b.size == 0:
        return (np.nan, np.nan)

    # we will use this kernel to smooth our variance estimate
    smoother = np.array([[0, 1., 0], [1., 2., 1.], [0, 1., 0]])
    # smoother = np.array([[1, 1., 1], [1., 1., 1.], [1, 1., 1]])
    # smoother = np.array([[0.5, 1., 0.5], [1., 2., 1.], [0.5, 1., 0.5]])
    smoother = smoother / np.sum(smoother)

    v1 = np.nanvar(a, axis, ddof=1)
    v2 = np.nanvar(b, axis, ddof=1)

    v1 = convolve2d(np.nan_to_num(v1), smoother, mode='same')
    v2 = convolve2d(np.nan_to_num(v2), smoother, mode='same')

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

def label_clusters(img):
    """
    Given a masked array, label all masked elements with 0 and each unmasked
    elements with an integer index of the connected component to which
    it belongs. 
    Assumes a 4-neighborhood for connectivity. Returns labeled array
    of same shape as input.
    """
    clust_map = np.zeros(img.shape, dtype='int64')
    uf = unionfind.UnionFind()

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
    counts = counts
    return counts

def significant_time_frequency(series, times, Tpre, Tpost, thresh, expand=1.0, niter=1000, pval=0.05, method='wav', doplot=True, normfun=None, diff_fun=t_of_log, **kwargs): 
    """
    Given a data series determined by channel, a two-element iterable, 
    times, containing times for a pair of events, pre and post-event 
    windows to grab, a scalar threshold (for symmetric 
    thresholding) or a pair of thresholds (lo, hi) at which to 
    cut clusters, a number of bootstrap iterations, and a p-value for
    statistical significance,
    return a time-frequency dataframe (suitable for plotting) containing
    the statistically significant clusters in the contrast between the 
    two conditions (times[0] - times[1]).
    """
    if method == 'wav':
        callback = tf.continuous_wavelet
    else:
        callback = tf.spectrogram

    dT = Tpost - Tpre
    Tpre_x = Tpre - expand * dT 
    Tpost_x = Tpost + expand * dT

    # make a dataframe containing all times, labeled by event type
    labels0 = np.zeros_like(times[0])
    labels1 = np.ones_like(times[1])
    alllabels = np.concatenate((labels0, labels1))

    # get time-frequency matrix for each event
    spec0, taxis, faxis = tf._per_event_time_frequency(series,
        callback, times[0], Tpre_x, Tpost_x, **kwargs)
    spec1, taxis, faxis = tf._per_event_time_frequency(series,
        callback, times[1], Tpre_x, Tpost_x, **kwargs)

    # normalize
    if normfun:
        spec0 = normfun(spec0)
        spec1 = normfun(spec1)

    # combine
    spectra = spec0 + spec1

    # convert from dataframes to ndarrays
    spectra = [s.values for s in spectra]

    try: 
        thlo = thresh[0]
        thhi = thresh[1]
    except:
        thlo = -thresh
        thhi = thresh

    # now loop
    cluster_masses = []
    for ind in np.arange(niter):
        labels = np.random.permutation(alllabels)
        pos = make_thresholded_diff(spectra, labels, hi=thhi, diff_fun=diff_fun)
        neg = make_thresholded_diff(spectra, labels, lo=thlo, diff_fun=diff_fun)

        posclus = label_clusters(pos)
        negclus = label_clusters(neg)

        # get all masses for clusters other than cluster 0 (= background)
        cluster_masses = np.concatenate([
            cluster_masses,
            get_cluster_masses(pos, posclus)[1:],
            get_cluster_masses(neg, negclus)[1:]
            ])


    # extract cluster size thresholds based on null distribution
    cluster_masses = np.sort(cluster_masses)
    plo = pval / 2.0
    phi = 1 - plo
    Nlo = np.floor(cluster_masses.size * plo)
    Nhi = np.ceil(cluster_masses.size * phi)
    Clo = cluster_masses[Nlo]
    Chi = cluster_masses[Nhi]

    # get significance-masked array for statistic image
    truelabels = alllabels
    signif = threshold_clusters(spectra, truelabels, lo=thlo,
        hi=thhi, keeplo=Clo, keephi=Chi, diff_fun=diff_fun)

    # make contrast image
    img0 = tf._mean_from_events(np.array(spectra)[truelabels == 0], taxis, faxis)
    img1 = tf._mean_from_events(np.array(spectra)[truelabels == 1], taxis, faxis)
    contrast = (img0 / img1)

    # use mask from statistic map to mask original data
    # mcontrast = contrast.copy().mask(signif.mask)
    mcontrast = np.ma.masked_array(data=contrast.values, mask=signif.mask)
    to_return = np.logical_and(taxis >= Tpre, taxis < Tpost)

    return mcontrast[to_return], taxis[to_return], faxis