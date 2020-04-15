from abc import abstractmethod, abstractproperty
import torch
import pyro
from pyro.distributions import Normal, Uniform, Laplace, MultivariateNormal
import matplotlib.pyplot as plt

class BaseModel(object):
    @abstractproperty
    def output_type():
        pass # "yt", "logyt"

    @abstractmethod
    def forward(self, t):
        pass

class Model1(BaseModel):
    def __init__(self, a_bounds=(-2,3),
        b_bounds=(-1,0.5), logs_bounds=(-2,1),
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

if __name__ == "__main__":
    # main()
    t = torch.linspace(0, 30, 100)
    yt = torch.exp(0.14 * t)
    logyt = torch.log(yt)
    model = Model1()

    alpha = 1.0 * 0.1
    for i in range(100):
        logy = model.forward(t)
        plt.plot(t, logy, 'C0-', alpha=alpha)
    plt.plot(t, logyt, 'C1-')
    # plt.gca().set_yscale("log")
    plt.show()
