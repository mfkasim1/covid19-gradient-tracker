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
from covidtracker.plotter import plot_interval

dtype=torch.float

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
    fcn = model.forward
    if model.output_type == "yt":
        obs = yt
    elif model.output_type == "logyt":
        obs = torch.log(torch.clamp(yt, min=1.0))
        # fcn = poutine.mask(fcn, mask=(yt > 0))

    return poutine.condition(fcn, data={model.output_type: obs})(t)

def infer(args, model, t, yt):
    nuts_kernel = NUTS(conditioned_model, jit_compile=args.jit)
    mcmc = MCMC(nuts_kernel,
            num_samples=args.nsamples,
            warmup_steps=args.nwarmups,
            num_chains=args.nchains)
    mcmc.run(model, t, yt)
    mcmc.summary(prob=0.95)
    return mcmc

class EmptyClass(object):
    def __init__(self):
        pass

def update_samples(data, nsamples=1000, nchains=1, nwarmups=1000, jit=True,
        restart=False, **kwargs):

    args = EmptyClass()
    args.__dict__["data"] = data
    args.__dict__["nsamples"] = nsamples
    args.__dict__["nchains"] = nchains
    args.__dict__["nwarmups"] = nwarmups
    args.__dict__["jit"] = jit
    args.__dict__["restart"] = restart

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

    res = EmptyClass()
    res.__dict__["samples"] = samples
    res.__dict__["model"] = model
    res.__dict__["dataloader"] = dl
    res.__dict__["ysim"] = model.simulate_samples(samples).detach().numpy()
    res.__dict__["yobs"] = dl.ytime
    return res

def main(args):
    res = update_samples(**args.__dict__)
    dl = res.dataloader
    model = res.model
    samples = res.samples
    ysim = res.ysim
    yobs = res.yobs

    # simulating the samples
    b = samples["b"].detach().numpy() # (nsamples, n)
    tnp = np.arange(len(yobs))

    ncols = 3
    plt.figure(figsize=(4*ncols,4))

    plt.subplot(1,ncols,1)
    plot_interval(tnp, b, color="C2")
    plt.plot(tnp, tnp*0, "k--")
    plt.ylabel("Faktor eksponensial")
    plt.xticks(tnp[::7], dl.tdate[::7], rotation=90)
    plt.title(dl.ylabel)
    plt.legend(loc="upper right")
    plt.subplot(1,ncols,2)
    plt.bar(tnp, yobs, color="C1", alpha=0.6)
    plot_interval(tnp, ysim, color="C1")
    plt.xticks(tnp[::7], dl.tdate[::7], rotation=90)
    plt.title(dl.ylabel)
    plt.legend(loc="upper left")

    if args.data == "id_new_cases":
        # show the tests
        dltest = DataLoader("id_new_tests")
        ytest = dltest.ytime
        ttest = np.arange(ytest.shape[0])

        plt.subplot(1,ncols,3)
        a = 7
        yy = ytest[:((yobs.shape[0]//a)*a)].reshape(-1,a).sum(axis=-1)
        ty = ttest[:((yobs.shape[0]//a)*a)].reshape(-1,a)[:,0]
        plt.bar(ty, yy, width=a-0.5)
        plt.title("Pemeriksaan per minggu")
        plt.xticks(ttest[::7], dltest.tdate[::7], rotation=90)
    plt.tight_layout()
    if args.savefig is None:
        plt.show()
    else:
        plt.savefig(args.savefig)
        plt.close()

if __name__ == "__main__":
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
    parser.add_argument('--savefig', type=str, default=None)
    args = parser.parse_args()

    main(args)
