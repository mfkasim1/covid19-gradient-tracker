import numpy as np
import emcee
import pickle
from scipy.stats import nbinom
from scipy.special import loggamma

def log_prob(mean_alpha, ks, with_prior=True, sumall=True):
    mu, r = mean_alpha
    # r = 1./alpha
    if r <= 0 or mu <= 0:
        return -9e99
    loglhood = loggamma(r + ks) - loggamma(r) - loggamma(ks + 1) + r * np.log(r/(r+mu)) + ks * np.log(mu/(r+mu))
    if sumall:
        loglhood = np.sum(loglhood)
    if with_prior:
        logprior = -np.log(r)
    else:
        logprior = 0
    lprob = loglhood + logprior
    return lprob

def loaddata(fname="batam-compiled.txt"):
    data = np.loadtxt(fname, skiprows=1, delimiter=",")
    n = data[:,0]
    ks = data[:,1]
    return n, ks

if __name__ == "__main__":
    ndim, nwalkers = 2, 100
    mean_alpha = np.random.rand(nwalkers, ndim)
    n, ks = loaddata()
    ks_series = []
    for ni, ki in zip(n, ks):
        ks_series += [int(ni) for _ in range(int(ki))]
    print("R:", np.sum(n * ks) / np.sum(ks))
    print(ks_series)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[np.array(ks_series).astype(np.float64)])
    res = sampler.run_mcmc(mean_alpha, 10000, progress=True)
    with open("batam-res.pkl", "wb") as fb:
        pickle.dump(sampler, fb)
