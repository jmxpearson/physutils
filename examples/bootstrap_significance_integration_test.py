import numpy as np
import pandas as pd
import physutils as phys

lfp = phys.LFPset(pd.DataFrame.from_csv("sample_lfp.csv"))
evt = pd.DataFrame.from_csv('sample_events.csv')

stops = evt['stop inflating'].dropna()
pops = evt['popped'].dropna()

Tpre = -1.5
Tpost = 0.5
baseline_interval = (-1.5, -1.35)
lothresh = -2.56
hithresh = 2.56
thresh = (lothresh, hithresh)

freqs = np.exp(np.linspace(np.log(2.5), np.log(50)))

series = lfp.dataframe['17']

nf = phys.norm_by_mean(baseline_interval)

lfp.significant_time_frequency('17', [stops, pops], Tpre, Tpost, thresh, niter=100, method='wav', doplot=False, normfun=nf, freqs=freqs)