import os
from abc import abstractmethod, abstractproperty
import pickle
import torch
import numpy as np
import pyro
from pyro.distributions import Normal, Uniform, Laplace, MultivariateNormal
from pyro.infer import MCMC, NUTS
import pyro.poutine as poutine
import matplotlib.pyplot as plt
from covidtracker.dataloader import DataLoader

class BaseModel(object):
    @abstractproperty
    def output_type():
        pass # "yt", "logyt"

    @abstractmethod
    def forward(self, t):
        pass

    @abstractmethod
    def simulate_samples(self, samples):
        pass

class Model1(BaseModel):
    def __init__(self, a_bounds=(-2,3),
        b_bounds=(-1,0.5), logs_bounds=(-8,1),
        log_lscale_bounds=(0,3.0)):
        """
        This model assumes:
            log(y(t)) ~ Normal(mu(t), sigma)
            mu(t) ~ a + integral(b(t)*dt)
            log(sigma) ~ Uniform(*logs_bounds)
            b ~ GP(mean=0, kernel=RBF(sigma=bsigma, length_scale=lscale))
            bsigma ~ Uniform(*b_bounds)
            log(lscale) ~ Uniform(*log_lscale_bounds)
            a ~ Uniform(*a_bounds)
        """
        self.a_bounds = a_bounds
        self.b_bounds = b_bounds
        self.logs_bounds = logs_bounds
        self.log_lscale_bounds = log_lscale_bounds

    @property
    def output_type(self):
        return "logyt"

    def forward(self, t):
        # t, yt: (n,)
        n = len(t)
        dt = t[1] - t[0]
        nzero = torch.zeros(n)

        # sample the prior
        a = pyro.sample("a", Uniform(*self.a_bounds))
        log_lscale = pyro.sample("log_lscale", Uniform(*self.log_lscale_bounds))
        lscale = torch.exp(log_lscale)
        log_sigma = pyro.sample("log_sigma", Uniform(*self.logs_bounds))
        tdist = t.unsqueeze(-1) - t # (n,n)
        b_sigma = pyro.sample("b_sigma", Uniform(*self.b_bounds))
        b_cov = b_sigma*b_sigma*torch.exp(-tdist*tdist / (2*lscale*lscale)) + torch.eye(n) * 1e-5
        b = pyro.sample("b", MultivariateNormal(nzero, b_cov))

        # calculate the rate
        int_bdt = torch.cumsum(b, dim=0) * dt
        mu = a + int_bdt # (n,)

        # simulate the observation
        logysim = pyro.sample("logyt", Normal(mu, torch.exp(log_sigma)))

        return logysim

    def simulate_samples(self, samples):
        a = samples["a"] # (nsamples,)
        b = samples["b"] # (nsamples, n)
        int_bdt = torch.cumsum(b, dim=-1)# * self.dt # (nsamples, n)
        mu = a.unsqueeze(-1) + int_bdt
        yt = torch.exp(mu)
        return yt # (nsamples, n)

def conditioned_model(model, t, yt):
    # model must be a BaseModel
    assert isinstance(model, BaseModel)
    if model.output_type == "yt":
        obs = yt
    elif model.output_type == "logyt":
        obs = torch.log(torch.clamp(yt, min=1.0))

    return poutine.condition(model.forward, data={model.output_type: obs})(t)

def infer(args, model, t, yt):
    nuts_kernel = NUTS(conditioned_model, jit_compile=args.jit)
    mcmc = MCMC(nuts_kernel,
            num_samples=args.nsamples,
            warmup_steps=args.nwarmups,
            num_chains=args.nchains)
    mcmc.run(model, t, yt)
    mcmc.summary(prob=0.95)
    return mcmc

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

def main(dtype=torch.float):
    import argparse
    parser = argparse.ArgumentParser(description='MCMC')
    parser.add_argument('data', type=str)
    parser.add_argument('--nsamples', type=int, default=1000,
                        help='number of MCMC samples (default: 1000)')
    parser.add_argument('--nchains', type=int, default=1,
                        help='number of parallel MCMC chains (default: 1)')
    parser.add_argument('--nwarmups', type=int, default=1000,
                        help='number of MCMC samples for warmup (default: 1000)')
    parser.add_argument('--jit', action='store_true', default=False)
    parser.add_argument('--restart', action='store_true', default=False)
    args = parser.parse_args()

    # load the data
    dl = DataLoader(args.data)
    yt = torch.tensor(dl.ytime, dtype=dtype)
    t = torch.arange(yt.shape[0], dtype=dtype) * 1.0
    model = Model1()

    samples_fname = dl.get_fname()
    print("Samples file: %s" % samples_fname)
    # load the samples
    if os.path.exists(samples_fname) and not args.restart:
        with open(samples_fname, "rb") as fb:
            samples = pickle.load(fb)
    # create the samples and save it
    else:
        mcmc = infer(args, model, t, yt)
        samples = mcmc.get_samples()
        with open(samples_fname, "wb") as fb:
            pickle.dump(samples, fb)

    b = samples["b"].detach().numpy() # (nsamples, n)
    mu = model.simulate_samples(samples).detach().numpy() # (nsamples, n)
    tnp = t.numpy()

    dltest = DataLoader("id_new_tests")
    ytest = dltest.ytime
    ttest = np.arange(ytest.shape[0])

    plt.figure(figsize=(12,4))
    ncols = 3
    plt.subplot(1,ncols,1)
    plot_interval(tnp, b, color="C2")
    plt.plot(tnp, tnp*0, "k--")
    plt.ylabel("Faktor eksponensial")
    plt.xticks(tnp[::7], dl.tdate[::7], rotation=90)
    plt.title(dl.ylabel)
    plt.legend()
    plt.subplot(1,ncols,2)
    plt.bar(tnp, yt, color="C1", alpha=0.6)
    plot_interval(tnp, mu, color="C1")
    plt.xticks(tnp[::7], dl.tdate[::7], rotation=90)
    plt.title(dl.ylabel)

    if ncols == 3:
        plt.subplot(1,ncols,3)
        a = 7
        yy = ytest[:((yt.shape[0]//a)*a)].reshape(-1,a).sum(axis=-1)
        ty = ttest[:((yt.shape[0]//a)*a)].reshape(-1,a)[:,0]#.sum(axis=-1)
        plt.bar(ty, yy, width=a-0.5)
        # plt.bar(ttest, ytest, color="C0")
        plt.title("Pemeriksaan per minggu")
        # plot_interval(tnp, mu, color="C1")
        plt.xticks(ttest[::7], dltest.tdate[::7], rotation=90)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
