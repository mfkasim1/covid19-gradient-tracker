import torch
import pyro
from pyro.distributions import Normal, Uniform, Laplace, MultivariateNormal
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

class Model1(object):
    def __init__(self, a_bounds=(-2,3),
        b_bounds=(-1,0.5), logs_bounds=(-2,1),
        log_lscale_bounds=(0,3.0)):
        """
        This model assumes:

            y(t) ~ Poisson(mu(t))
            mu(t) = exp(eps + a + integral(b(t)*dt))
            eps ~ Laplace(0, sigma)
            log(sigma) ~ Uniform(-2, 10)
            b ~ GP(mean=0, kernel=RBF(sigma=bsigma, length_scale=lscale))
            bsigma ~ Uniform(-1, 0.5)
            log(lscale) ~ Uniform(0, 3.0)
            a ~ Uniform(-2, 10)
        """

        self.a_bounds = a_bounds
        self.b_bounds = b_bounds
        self.logs_bounds = logs_bounds
        self.log_lscale_bounds = log_lscale_bounds

    def forward(self, t):
        # t, yt: (n,)
        n = len(t)
        dt = t[1] - t[0]
        nzero = torch.zeros(n)

        # sample the prior
        a = pyro.sample("a", Uniform(*self.a_bounds))
        log_lscale = pyro.sample("log_lscale", Uniform(*self.log_lscale_bounds))
        lscale = torch.exp(log_lscale)
        log_sigma_lb = nzero + self.logs_bounds[0]
        log_sigma_ub = nzero + self.logs_bounds[1]
        log_sigma = pyro.sample("log_sigma", Uniform(log_sigma_lb, log_sigma_ub))
        tdist = t.unsqueeze(-1) - t # (n,n)
        b_sigma = pyro.sample("b_sigma", Uniform(*self.b_bounds))
        b_cov = b_sigma*b_sigma*torch.exp(-tdist*tdist / (2*lscale*lscale)) + torch.eye(n) * 1e-5
        b = pyro.sample("b", MultivariateNormal(nzero, b_cov))
        eps = pyro.sample("eps", Laplace(nzero, torch.exp(log_sigma)))

        # calculate the rate
        int_bdt = torch.cumsum(b, dim=0) * dt
        mu = torch.exp(a + int_bdt) # (n,)
        plt.plot(b)
        plt.plot(int_bdt)
        plt.show()
        plt.plot(eps)
        plt.show()

        # simulate the observation
        # NOTE: I'm using Normal here to approximate Poisson
        mu_std = torch.clamp(torch.sqrt(mu), min=1.0)
        ysim = pyro.sample("yt", Normal(mu, mu_std+torch.exp(log_sigma)))

        return ysim

def conditioned_model(model, t, yt):
    return poutine.condition(model.forward, data={"yt": yt})(t)

def infer(args, model, t, yt):
    nuts_kernel = NUTS(conditioned_model, jit_compile=args.jit)
    mcmc = MCMC(nuts_kernel,
            num_samples=args.num_samples,
            warmup_steps=args.warmup_steps,
            num_chains=args.num_chains)
    mcmc.run(model, t, yt)
    mcmc.summary(prob=0.5)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='MCMC')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='number of MCMC samples (default: 1000)')
    parser.add_argument('--num-chains', type=int, default=1,
                        help='number of parallel MCMC chains (default: 1)')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='number of MCMC samples for warmup (default: 1000)')
    parser.add_argument('--jit', action='store_true', default=False)
    args = parser.parse_args()

    t = torch.linspace(0, 30, 100)
    yt = torch.exp(0.14 * t)
    model = Model1()
    infer(args, model, t, yt)

if __name__ == "__main__":
    t = torch.linspace(0, 30, 100)
    yt = torch.exp(0.14 * t)
    model = Model1()

    alpha = 1.0
    for i in range(1):
        y = model.forward(t)
        plt.plot(t, y, 'C0-', alpha=alpha)
    plt.plot(t, yt, 'C1-')
    # plt.gca().set_yscale("log")
    plt.show()
