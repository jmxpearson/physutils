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
        wav = _make_morlet(kwargs['w'])
    else:
        wav = _make_morlet(*args)
    rwavelet = lambda N, b: np.real(wav(N, b))
    iwavelet = lambda N, b: np.imag(wav(N, b))
    tfr = ssig.cwt(series.values, rwavelet, scales)
    tfi = ssig.cwt(series.values, iwavelet, scales)
    tf = tfr ** 2 + tfi ** 2

    return pd.DataFrame(tf.T, columns=freqs, index=series.index)

def _make_morlet(w=np.sqrt(2) * np.pi):
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
    z = spectrum.T
    ax = plt.figure().add_subplot(111)
    extent = (times[0], times[-1], freqs[0], freqs[-1])
    if 'interpolation' in kwargs:
        interp = kwargs['interpolation']
    else:
        interp = 'bilinear'
    im = NonUniformImage(ax, interpolation=interp, extent=extent)
    if 'background_color' in kwargs:
        im.get_cmap().set_bad(kwargs['background_color'])
    else:
        z[np.isnan(z)] = 0.0  # replace missing values with 0 color
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
    specmats, times, freqs = _per_event_time_frequency(series, tffun, events, Tpre, Tpost, *args, **kwargs)
    
    return _mean_from_events(specmats, times, freqs)

def _mean_from_events(specmats, times, freqs):
    """
    Given a list containing time-frequency spectra, average them and
    return a dataframe (time x frequency).
    """
    allspecs = np.dstack(specmats)
    meanspec = np.nanmean(allspecs, axis=2)

    return pd.DataFrame(meanspec, index=times, columns=freqs)

def _per_event_time_frequency(series, tffun, events, Tpre, Tpost, *args, **kwargs):
    """
    Given a Pandas series, split it into chunks of (Tpre, Tpost) around
    events, do the time-frequency on each using the function tffun,
    and return a tuple containing the list of time-frequency matrices
    (time x frequency), an array of times, and an array of frequencies.
    NOTE: Returns power on a decibel scale!
    """
    df = core._splitseries(series, events, Tpre, Tpost)
    spectra = [tffun(ser, *args, **kwargs) for (name, ser) in df.iteritems()]
    spectra = map(lambda x: 10 * np.log10(x), spectra)
    if 'normfun' in kwargs:
        spectra = kwargs['normfun'](spectra)
    specmats = map(lambda x: x.values, spectra)
    times = spectra[0].index
    freqs = spectra[0].columns
    return (specmats, times, freqs)

