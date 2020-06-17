import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import loggamma
from covidtracker.data.clusters.posteriorfit import log_prob, loaddata

plot_lprob = 1
plot_hist = 1

def print_stats(x, name):
    print("%s:" % name)
    print("* Mean: %.3e" % np.mean(x))
    print("* Median: %.3e" % np.median(x))
    print("* 95pct CI: (%.3e - %.3e)" % (np.percentile(x, 2.5), np.percentile(x, 97.5)))

with open("batam-res.pkl", "rb") as fb:
    obj = pickle.load(fb)

n, ks = loaddata("batam-compiled.txt")

res = np.reshape(obj.get_chain(), (-1,2))
res = res[res.shape[0]//2:,:]

print_stats(res[:,0], "Mean")
print_stats(res[:,1], "K")

if plot_lprob:
    lprobs = np.exp(np.array([log_prob(r, n, with_prior=False, sumall=False) for r in res]))
    plt.plot(n, lprobs[::100].T, color='C1', alpha=0.02)
    plt.bar(n, ks / np.sum(ks))
    plt.ylabel("Frequency")
    plt.xlabel("# secondary cases")
    plt.show()

# show the histogram
if plot_hist:
    nbins = 30
    plt.subplot(1,2,1)
    plt.hist(res[:,0], bins=nbins)
    plt.title("Mean")
    plt.subplot(1,2,2)
    plt.hist(res[:,1], bins=nbins)
    plt.title("K")
    plt.show()
