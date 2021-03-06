import numpy as np
import pandas as pd
from matplotlib.mlab import specgram
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import scipy.signal as ssig
from . import core

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

def continuous_wavelet(series, freqs=None, bandwidth=4.5, phase=False, **kwargs):
    """
    Construct a continuous wavelet transform for the data series.
    Extra pars are parameters for the Morlet wavelet.
    Returns a tuple (time-frequency matrix, frequencies, times)
    If phase=True, returns the phase, else returns the amplitude
    """
    if freqs is None:
        # define some default LFP frequencies of interest
        freqlist = [np.arange(1, 13), np.arange(15, 30, 3), np.arange(35, 100, 5)]
        freqs = np.concatenate(freqlist)

    dt = series.index[1] - series.index[0]
    wav = _make_morlet(bandwidth)
    scales = bandwidth / (2 * np.pi * freqs * dt)
    rwavelet = lambda N, b: np.real(wav(N, b))
    iwavelet = lambda N, b: np.imag(wav(N, b))
    tfr = ssig.cwt(series.values, rwavelet, scales)
    tfi = ssig.cwt(series.values, iwavelet, scales)

    tf = tfr ** 2 + tfi ** 2
    if phase:
        # return tf rescaled to unit circle
        tf = (tfr + 1j * tfi) / tf

    return pd.DataFrame(tf.T, columns=freqs, index=series.index)

def _make_morlet(w):
    """
    Coded to conform to the requirements of scipy.signal.cwt.
    WARNING: scipy.signal.morlet does not follow the recommended convention!
    w = np.sqrt(2) * np.pi corresponds to the 1-1 complex Morlet wavelet 
    in Matlab.
    """
    def wav(N, b):
        x = (np.arange(0, N) - (N - 1.0) / 2) / b
        output = np.exp(1j * w * x) - np.exp(-0.5 * (w ** 2))
        output *= np.exp(-0.5 * (x ** 2)) * (np.pi ** (-0.25))
        output /= np.sqrt(b)
        return output

    return wav

def plot_time_frequency(spectrum, interpolation='bilinear', 
    background_color=None, clim=None, dbscale=True, **kwargs):
    """
    Time-frequency plot. Modeled after image_nonuniform.py example 
    spectrum is a dataframe with frequencies in columns and time in rows
    """
    if spectrum is None:
        return None
    
    times = spectrum.index
    freqs = spectrum.columns
    if dbscale:
        z = 10 * np.log10(spectrum.T)
    else:
        z = spectrum.T
    ax = plt.figure().add_subplot(111)
    extent = (times[0], times[-1], freqs[0], freqs[-1])
    
    im = NonUniformImage(ax, interpolation=interpolation, extent=extent)

    if background_color:
        im.get_cmap().set_bad(kwargs['background_color'])
    else:
        z[np.isnan(z)] = 0.0  # replace missing values with 0 color

    if clim:
        im.set_clim(clim)

    if 'cmap' in kwargs:
        im.set_cmap(kwargs['cmap'])

    im.set_data(times, freqs, z)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.images.append(im)
    if 'colorbar_label' in kwargs:
        plt.colorbar(im, label=kwargs['colorbar_label'])
    else:
        plt.colorbar(im, label='Power (dB/Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    return plt.gcf() 

def avg_time_frequency(series, tffun, events, Tpre, Tpost, expand=1.0, 
    normfun=None, splitfirst=True, **kwargs):
    """
    Given a Pandas series, split it into chunks of (Tpre, Tpost) around
    events, do the time-frequency on each using the function tffun,
    and return a DataFrame with time as the index and frequency as the 
    column label.
    To reduce edge artifacts, the interval is expanded by a fraction expand 
    on either side of the requested interval and truncated before returning.
    Note that, as per evtsplit, events before the events have Tpre < 0.
    **kwargs is passed on to tffun
    """
    orig_slice = slice(Tpre, Tpost)
    dT = Tpost - Tpre
    Tpre_x = Tpre - expand * dT 
    Tpost_x = Tpost + expand * dT

    # if we want phase information, we cannot split the series into chunks 
    # before doing the time-frequency transform
    if 'phase' in kwargs and kwargs['phase']:
        splitfirst = False

    specmats, times, freqs = _per_event_time_frequency(series, tffun, events, Tpre_x, Tpost_x, splitfirst, **kwargs)

    if len(specmats) == 0:
        return None

    if normfun: 
        specmats = normfun(specmats) 
    
    mfe = _mean_from_events(specmats, times, freqs)[orig_slice]

    if 'phase' in kwargs and kwargs['phase']:
        mfe = np.abs(mfe)

    return mfe

def _mean_from_events(specmats, times, freqs):
    """
    Given a list containing time-frequency spectra, average them and
    return a dataframe (time x frequency).
    """
    allspecs = np.dstack(specmats)
    meanspec = np.nanmean(allspecs, axis=2)

    return pd.DataFrame(meanspec, index=times, columns=freqs)

def _per_event_time_frequency(series, tffun, events, Tpre, Tpost, complete_only=True, splitfirst=True, **kwargs):
    """
    Given a Pandas series, split it into chunks of (Tpre, Tpost) around
    events, do the time-frequency on each using the function tffun,
    and return a tuple containing the list of time-frequency matrices
    (time x frequency), an array of times, and an array of frequencies.
    """
    if splitfirst:
        df = core._splitseries(series, events, Tpre, Tpost)
        if complete_only:
            spectra = [tffun(ser, **kwargs) for (name, ser) in df.items() if not np.any(np.isnan(ser))]
        else:
            spectra = [tffun(ser, **kwargs) for (name, ser) in df.items()]
    else:
        tf_all = tffun(series, **kwargs)
        spectra, freqs = core.evtsplit(tf_all, events, Tpre, Tpost, return_by_event=True)
        if complete_only:
            spectra = [s for s in spectra if not np.any(np.isnan(s))]

    if len(spectra) > 0:
        times = spectra[0].index
        freqs = spectra[0].columns
    else:
        times = None
        freqs = None

    return (spectra, times, freqs)

def phase_amplitude_coupling(fser, gser, lag=0):
    """
    Compute the product of two time series for calculation of phase-amplitude
    coupling. That is, if the Hilbert transform of fser is A_f exp(\phi_f),
    then the phase-amplitude product of fser and gser is 
    f * g = A_f exp(\phi_g). Lag is the offset of f's amplitude from g's
    phase: A(t) exp(\phi(t - lag)). This is useful for examining asymptotic
    statistics in large-lag regime, where biases in the joint distribution
    of A_f and \phi_g are not due to phase-amplitude coupling, but to the 
    individual signals (cf. Canolty et al., Sciece, 2006, SI).
    Function returns a series of the same length, but first and last lag 
    elements are NaN.
    """
    fh = ssig.hilbert(fser)
    gh = ssig.hilbert(gser)

    Af = np.abs(fh)
    ephig = gh / np.abs(gh)

    if lag == 0:
        pac = Af * ephig
    else:
        pac = Af * np.roll(ephig, lag)
        pac[:lag] = np.nan
        pac[-lag:] = np.nan

    return pac






