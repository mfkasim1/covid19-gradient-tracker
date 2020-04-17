import os
import numpy as np
from jinja2 import Template
import matplotlib.pyplot as plt
import covidtracker as ct
from covidtracker.dataloader import DataLoader
from covidtracker.models import update_samples
from covidtracker.plotter import plot_interval

def plot_gradient(res):
    dl = res.dataloader
    b = res.samples["b"].detach().numpy()
    tnp = np.arange(b.shape[1])

    plot_interval(tnp, b, color="C2")
    plt.plot(tnp, tnp*0, "k--")
    plt.ylabel("Faktor eksponensial")
    plt.xticks(tnp[::7], dl.tdate[::7], rotation=90)
    plt.title(dl.ylabel)
    plt.legend(loc="upper right")

def plot_data_and_sim(res):
    yobs = res.yobs
    ysim = res.ysim
    dl = res.dataloader
    tnp = np.arange(len(yobs))

    plt.bar(tnp, yobs, color="C1", alpha=0.6)
    plot_interval(tnp, ysim, color="C1")
    plt.xticks(tnp[::7], dl.tdate[::7], rotation=90)
    plt.title(dl.ylabel)
    plt.legend(loc="upper left")

def plot_weekly_tests(res):
    yobs = res.yobs

    # show the tests
    dltest = DataLoader("id_new_tests")
    ytest = dltest.ytime
    ttest = np.arange(ytest.shape[0])

    a = 7
    yy = ytest[:((yobs.shape[0]//a)*a)].reshape(-1,a).sum(axis=-1)
    ty = ttest[:((yobs.shape[0]//a)*a)].reshape(-1,a)[:,0]
    plt.bar(ty, yy, width=a-0.5)
    plt.title("Pemeriksaan per minggu")
    plt.xticks(ttest[::7], dltest.tdate[::7], rotation=90)

def main(img_path, file_path):
    fields = ["id_new_cases", "idprov_jakarta_new_cases", \
        "idprov_jabar_new_cases", "idprov_jatim_new_cases",
        "idprov_sulsel_new_cases"]
    names = ["Indonesia", "Jakarta", "Jabar", "Jatim", "Sulsel"]

    ftemplate = os.path.join(os.path.split(ct.__file__)[0], "templates", "template-idcovid19.md")

    nsamples = 1000
    nwarmups = 1000
    nchains = 1

    places = []
    for i,df in enumerate(fields):
        print("Field: %s" % df)

        # get the samples or resample
        res = update_samples(df, nsamples=nsamples, nchains=nchains, nwarmups=nwarmups,
            jit=True, restart=False)

        dl = res.dataloader
        model = res.model
        samples = res.samples
        ysim = res.ysim
        yobs = res.yobs

        ################## creating the figures ##################
        # simulating the samples
        b = samples["b"].detach().numpy() # (nsamples, n)
        tnp = np.arange(len(yobs))

        ncols = 3
        plt.figure(figsize=(4*ncols,4))

        plt.subplot(1,ncols,1)
        plot_gradient(res)
        plt.subplot(1,ncols,2)
        plot_data_and_sim(res)

        if df == "id_new_cases":
            # show the tests
            plt.subplot(1,ncols,3)
            plot_weekly_tests(res)

        plt.tight_layout()
        plt.savefig(os.path.join(img_path, "%s.png"%df))
        plt.close()

        ## save the information for the templating
        places.append({
            "dataid": df,
            "name": names[i],
            "flatcurve_result": "belum dapat disimpulkan"
        })

    with open(ftemplate, "r") as f:
        template = Template(f.read())
    content = template.render(places=places)
    with open(file_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    fpath = "/mnt/c/Users/firma/Documents/Projects/Git/mfkasim91.github.io"
    img_path = os.path.join(fpath, "assets", "idcovid19-daily")
    file_path = os.path.join(fpath, "idcovid19.md")
    main(img_path, file_path)
