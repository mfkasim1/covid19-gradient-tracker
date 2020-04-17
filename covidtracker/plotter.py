import numpy as np
import matplotlib.pyplot as plt

def plot_interval(t, ysamples, color="C0"): # numpy operation
    ymedian = np.median(ysamples, axis=0)
    yl1 = np.percentile(ysamples, 25., axis=0)
    yu1 = np.percentile(ysamples, 75., axis=0)
    yl2 = np.percentile(ysamples, 2.5, axis=0)
    yu2 = np.percentile(ysamples, 97.5, axis=0)
    yl3 = np.percentile(ysamples, 0.5, axis=0)
    yu3 = np.percentile(ysamples, 99.5, axis=0)
    plt.plot(t, ymedian, color=color, label="Median")
    plt.fill_between(t, yl1, yu1, color=color, alpha=0.6, label="50% CI")
    plt.fill_between(t, yl2, yu2, color=color, alpha=0.3, label="95% CI")
    plt.fill_between(t, yl3, yu3, color=color, alpha=0.15, label="99% CI")
