import os
import datetime
import numpy as np
from jinja2 import Template
from scipy.stats import hypergeom
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

def get_weekly_sum(y, a=7):
    yy = y[:((y.shape[-1]//a)*a)].reshape(-1,a).sum(axis=-1)
    return yy

def get_in_week(t, a=7, i=0):
    ty = t[:((t.shape[-1]//a)*a)].reshape(-1,a)[:,i]
    return ty

def plot_weekly_tests(res):
    yobs = res.yobs

    # show the tests
    dltest = DataLoader("id_new_tests")
    ytest = dltest.ytime
    ttest = np.arange(ytest.shape[0])

    a = 7
    yy = get_weekly_sum(ytest)
    ty = get_in_week(ttest, i=0)
    # yy = ytest[:((yobs.shape[0]//a)*a)].reshape(-1,a).sum(axis=-1)
    # ty = ttest[:((yobs.shape[0]//a)*a)].reshape(-1,a)[:,0]
    plt.bar(ty, yy, width=a-0.5)
    plt.title("Pemeriksaan per minggu")
    plt.xticks(ttest[::7], dltest.tdate[::7], rotation=90)

def plot_weekly_tests_prov(res):
    yobs = res.yobs # new positives / day

    # get all the tests and the positive cases from all over the country
    dltest = DataLoader("id_new_tests")
    dlcase = DataLoader("id_new_cases")
    ttest = dltest.tdate
    ytest = dltest.ytime
    ycase = dlcase.ytime
    assert len(ytest) == len(ycase)
    ntest_days = len(ytest)
    nobs_days = len(yobs)
    missing_days = ntest_days - nobs_days
    offset_test = int(np.ceil(missing_days / 7.0)) * 7
    offset_obs = offset_test - missing_days

    # offset the positive tests to match the weekly test
    yobs = yobs[offset_obs:]
    ytest = ytest[offset_test:]
    ycase = ycase[offset_test:]
    ttest = ttest[offset_test:]
    assert len(yobs) == len(ytest)
    assert len(yobs) == len(ycase)
    # yobs and ytest should have the same lengths by now

    # get the weekly data
    ycase = get_weekly_sum(ycase)
    yobs = get_weekly_sum(yobs)
    ytest = get_weekly_sum(ytest)
    ttest = get_in_week(ttest, i=0)
    ndata = len(yobs)

    # calculate the posterior distribution of the number of tests
    yall_positives = ycase.astype(np.int)
    yall_tests = ytest.astype(np.int)
    y_positives = yobs.astype(np.int)
    max_tests = yall_tests.max()
    posteriors = np.zeros((ndata, max_tests+1))
    for i in range(ndata):
        yall_positive = yall_positives[i]
        yall_test = yall_tests[i]
        y_positive = y_positives[i]
        y_test = np.arange(yall_test+1)

        lhood = hypergeom.pmf(y_positive, yall_test, yall_positive, y_test)
        if np.sum(lhood) == 0:
            print(y_positive, yall_test, yall_positive, res.dataloader.dataidentifier)
        posteriors[i,:len(lhood)] = lhood / np.sum(lhood) # (max_tests+1)

    cdf = np.cumsum(posteriors, axis=-1)
    def h(cdf, q):
        return np.sum(cdf < q, axis=-1)

    x = np.arange(ndata)
    ymed = h(cdf, 0.5)
    yl1 = h(cdf, 0.025)
    yu1 = h(cdf, 0.975)
    plt.bar(x, height=ymed, alpha=0.5, label="Median")
    plt.errorbar(x, ymed, [ymed-yl1, yu1-ymed], fmt="o", label="95% CI")
    plt.xticks(x, ttest, rotation=90)
    plt.legend()
    plt.title("Perkiraan jumlah pemeriksaan mingguan")

def main(img_path, file_path):
    provinces = ["Jakarta", "Jabar", "Jatim", "Sulsel"]
    fields = ["id_new_cases"] + ["idprov_%s_new_cases" % p.lower() for p in provinces]
    names = ["Indonesia"] + provinces

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
        elif df.startswith("idprov_") and df.endswith("_new_cases"):
            plt.subplot(1,ncols,3)
            plot_weekly_tests_prov(res)

        plt.tight_layout()
        plt.savefig(os.path.join(img_path, "%s.png"%df))
        plt.close()

        ################## deciding the results ##################
        b_last = b[:,-1]
        decline_portion = np.sum(b_last < 0) * 1.0 / b.shape[0]
        if decline_portion > 0.99:
            flatcurve_res = "**turun**"
        elif decline_portion > 0.95:
            flatcurve_res = "**kemungkinan** turun"
        elif decline_portion > 0.75:
            flatcurve_res = "ada indikasi penurunan, tapi belum pasti"
        else:
            flatcurve_res = "belum dapat disimpulkan"

        ## save the information for the templating
        places.append({
            "dataid": df,
            "name": names[i],
            "flatcurve_result": flatcurve_res
        })

    with open(ftemplate, "r") as f:
        template = Template(f.read())
    today = datetime.date.today()
    content = template.render(places=places, date=today.strftime("%d/%m/%Y"))
    with open(file_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    fpath = "/mnt/c/Users/firma/Documents/Projects/Git/mfkasim91.github.io"
    img_path = os.path.join(fpath, "assets", "idcovid19-daily")
    file_path = os.path.join(fpath, "idcovid19.md")
    main(img_path, file_path)
