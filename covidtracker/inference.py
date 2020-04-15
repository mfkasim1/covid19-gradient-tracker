import torch
import pyro
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS
from covidtracker.models import BaseModel, Model1

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
            num_samples=args.num_samples,
            warmup_steps=args.warmup_steps,
            num_chains=args.num_chains)
    mcmc.run(model, t, yt)
    mcmc.summary(prob=0.95)

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

    t = torch.linspace(0, 10, 11)
    yt = torch.exp(0.14 * t)
    model = Model1()
    infer(args, model, t, yt)

if __name__ == "__main__":
    main()
