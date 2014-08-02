"""
LFPset.py
LFPset is a wrapper class for a Pandas dataframe object containing LFP data.
Implements methods from py.

As per Pandas convention, these should return new dataframes.
"""
from .. import core
from .. import bootstrap as boot
from .. import tf
from .. import cleaning
import numpy as np
from scipy.signal import hilbert
import pandas as pd

class LFPset(object):
    def __init__(self, data, meta={}):
        # wrap passed data in constructor in case it's a series
        self.dataframe = pd.DataFrame(data)
        self.meta = meta  # dict of metadata 

    def __getattr__(self, name):
        return getattr(self.dataframe, name)

    def __str__(self):
        return self.dataframe.__str__()

    def __repr__(self):
        return 'LFP dataset object containing a\n' + self.dataframe.__repr__()

    def decimate(self, decfrac):
        newdf = core.decimate(self.dataframe, decfrac)
        newmeta = self.meta.copy()
        newmeta['sr'] = newmeta.get('sr', None) / np.product(decfrac)
        return LFPset(newdf, newmeta)

    def bandlimit(self, *args):
        newdf = core.bandlimit(self.dataframe, *args)
        newmeta = self.meta.copy()
        return LFPset(newdf, newmeta)

    def demean(self):
        dmn = lambda x: (x - x.mean())
        newdf = self.dataframe.apply(dmn)
        newmeta = self.meta.copy()
        return LFPset(newdf, newmeta)

    def interpolate(self):
        newdf = self.dataframe.interpolate()
        newmeta = self.meta.copy()
        return LFPset(newdf, newmeta)

    def zscore(self):
        zsc = lambda x: (x - x.mean()) / x.std()
        newdf = self.dataframe.apply(zsc)
        newmeta = self.meta.copy()
        return LFPset(newdf, newmeta)

    def instpwr(self):
        # by padding up to next highest power of 2, we get a huge 
        # performance boost; truncate afterward
        Nstart = self.dataframe.shape[0]
        Nfft = 2 ** np.ceil(np.log2(Nstart))
        hilbert_pad = lambda x: hilbert(x, N=Nfft)[:Nstart]
        newdf = self.dataframe.apply(hilbert_pad, raw=True)
        newdf = newdf.apply(np.absolute) ** 2
        newmeta = self.meta.copy()
        return LFPset(newdf, newmeta)

    def smooth(self, winlen):
        wid = np.around(winlen * self.meta['sr'])
        newdf = pd.rolling_mean(self.dataframe, wid, 
            min_periods=1)
        newdf = newdf.apply(pd.Series.interpolate)
        newmeta = self.meta.copy()
        return LFPset(newdf, newmeta)

    def censor(self, maskarr=None, get_censor=None):
        """
        Censor lfp, replacing with NaNs. Censoring has a default 
        behavior of removing railing artifacts separately for each
        column, but this behavior can be overridden by specifying either:
        1) maskarr, a Boolean numpy array suitable to serve as a mask
        for the data (i.e., True for censored values)
        2) get_censor, a function that is passed self and should 
        return a mask the same shape as self.values
        Option 1 takes precedence over option 2
        """

        if maskarr:
            m = maskarr
        elif get_censor:
            m = get_censor(self)
        else:
            m = np.apply_along_axis(cleaning.censor_railing, axis=0, 
                arr=self.dataframe.values)
        
        return LFPset(self.mask(m), self.meta.copy())

    def evtsplit(self, times, Tpre, Tpost, t0=0):
        # note: Tpre < 0 for times before the events in times
        split_to_series = lambda x: core._splitseries(x, times, Tpre, Tpost, 
            t0).unstack()
        return self.dataframe.apply(split_to_series)

    def avg_time_frequency(self, channel, times, Tpre, Tpost, method='wav', doplot=True, **kwargs):
        """
        Do a time frequency decomposition, averaging across chunks split at 
        times. **kwargs are passed on as parameters to the method call:
        method 'wav': w parameter
        method 'spec': window length (in s) and fraction overlap
        """
        series = self.dataframe[channel]
        if method == 'wav':
            callback = tf.continuous_wavelet
        else:
            callback = tf.spectrogram

        avgtf = tf.avg_time_frequency(series, callback, times, Tpre, 
            Tpost, **kwargs)

        if doplot:
            fig = tf.plot_time_frequency(avgtf)
        else:
            fig = None

        return avgtf, fig

    def contrast_time_frequency(self, channel, times, Tpre, Tpost, method='wav', doplot=True, **kwargs):
        """
        Do a contrast analysis for two sets of events. times is an iterable
        containing times for each. Returned value is a ratio of time-frequency
        power in the first set vs. the second. **kwargs are passed on as 
        parameters to the method call:
        method 'wav': w parameter
        method 'spec': window length (in s) and fraction overlap
        """
        series = self.dataframe[channel]
        if method == 'wav':
            callback = tf.continuous_wavelet
        else:
            callback = tf.spectrogram

        tf0 = tf.avg_time_frequency(series, callback, times[0], Tpre, 
            Tpost, **kwargs)
        tf1 = tf.avg_time_frequency(series, callback, times[1], Tpre, 
            Tpost, **kwargs)

        if doplot:
            fig = tf.plot_time_frequency(tf0 / tf1) 
        else:
            fig = None

        return tf0 / tf1, fig

    def significant_time_frequency(self, channel, times, Tpre, Tpost, thresh, niter=1000, pval=0.05, method='wav', doplot=True, **kwargs):
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
        series = self.dataframe[channel]
        if method == 'wav':
            callback = tf.continuous_wavelet
        else:
            callback = tf.spectrogram

        # make a dataframe containing all times, labeled by event type
        t0 = pd.DataFrame({'time': times[0], 'label': 0})
        t1 = pd.DataFrame({'time': times[1], 'label': 1})
        alltimes = pd.concat([t0, t1])

        # get time-frequency matrix for each event
        spectra, taxis, faxis = tf._per_event_time_frequency(series,
            callback, alltimes['time'], Tpre, Tpost, **kwargs)

        try: 
            thlo = thresh[0]
            thhi = thresh[1]
        except:
            thlo = -thresh
            thhi = thresh

        # now loop
        cluster_masses = []
        for ind in np.arange(niter):
            labels = np.random.permutation(alltimes['label'])
            pos = boot.make_thresholded_diff(spectra, labels, hi=thhi)
            neg = boot.make_thresholded_diff(spectra, labels, lo=thlo)

            posclus = boot.label_clusters(pos)
            negclus = boot.label_clusters(neg)

            cluster_masses = np.concatenate([
                cluster_masses,
                boot.get_cluster_masses(pos, posclus), 
                boot.get_cluster_masses(neg, negclus)
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
        truelabels = alltimes['label'].values
        signif = boot.threshold_clusters(spectra, truelabels, lo=thlo,
            hi=thhi, keeplo=Clo, keephi=Chi)

        # make contrast image
        img0 = tf._mean_from_events(np.array(spectra)[truelabels == 0], taxis, faxis)
        img1 = tf._mean_from_events(np.array(spectra)[truelabels == 1], taxis, faxis)
        contrast = img0 / img1

        # use mask from statistic map to mask original data
        mcontrast = contrast.copy().mask(signif.mask)

        if doplot:
            color_lims = (np.amin(10 * np.log10(contrast.values)), np.amax(10 * np.log10(contrast.values)))
            fig = tf.plot_time_frequency(mcontrast, clim=color_lims)
        else:
            fig = None

        return mcontrast, fig 
