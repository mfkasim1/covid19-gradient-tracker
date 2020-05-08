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

    plot_interval(tnp, (np.exp(b)-1)*100, color="C2")
    plt.plot(tnp, tnp*0, "k--")
    plt.ylabel("Persentase pertumbuhan")
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
    # yy = y[:((y.shape[-1]//a)*a)].reshape(-1,a).sum(axis=-1)
    yy = y[y.shape[-1]%7:].reshape(-1,a).sum(axis=-1)
    return yy

def get_in_week(t, a=7, i=0):
    # ty = t[:((t.shape[-1]//a)*a)].reshape(-1,a)[:,i]
    ty = t[t.shape[-1]%7:].reshape(-1,a)[:,i]
    return ty

def get_total_cases(res, total_deaths_data):
    model = res.model
    samples = res.samples
    ysim = res.ysim

    unreported_ratio_death = 2200 / 785. # from reuter's report https://uk.reuters.com/article/us-health-coronavirus-indonesia-casualti/exclusive-more-than-2200-indonesians-have-died-with-coronavirus-symptoms-data-shows-idUKKCN22A04N
    total_deaths_from_cases_fullifr = model.predict_total_deaths(samples, ifr=1.0) * unreported_ratio_death # (nsamples,)
    unreported_ratio_fullifr = total_deaths_data / total_deaths_from_cases_fullifr
    total_cases_fullifr = unreported_ratio_fullifr * np.sum(ysim, axis=-1)

    total_cases = None
    ntrial = 100
    ifrs = np.random.randn(ntrial) * 0.0047 + 0.0086 # (0.39 - 1.33)% from https://www.medrxiv.org/content/10.1101/2020.03.09.20033357v1.full.pdf
    ifrs[ifrs < 0.0039] = 0.0039
    for i,ifr in enumerate(ifrs):
        total_cases1 = total_cases_fullifr / ifr # (nsamples,)
        if i == 0:
            total_cases = total_cases1
        else:
            total_cases = np.concatenate((total_cases, total_cases1))

    # get the statistics of the total cases
    total_cases_median = int(np.round(np.median(total_cases)))
    total_cases_025    = int(np.round(np.percentile(total_cases, 2.5)))
    total_cases_975    = int(np.round(np.percentile(total_cases, 97.5)))

    def formatstr(a):
        b = int(float("%.2g"%a)) # round to some significant figures
        c = f"{b:,}"
        d = c.replace(",", ".")
        return d

    return formatstr(total_cases_median),\
           formatstr(total_cases_025),\
           formatstr(total_cases_975)

def plot_weekly_tests(res):
    yobs = res.yobs

    # show the tests
    dltest = DataLoader("id_new_tests")
    ytest = dltest.ytime
    ttest = np.arange(ytest.shape[0])
    tticks = dltest.tdate

    a = 7
    ytest = get_weekly_sum(ytest)
    ttest = get_in_week(ttest, i=0)
    tticks = get_in_week(tticks, i=0)
    plt.bar(ttest, ytest, width=a-0.5)
    # plt.bar(ttest, ytest)
    plt.title("Pemeriksaan per minggu")
    plt.xticks(ttest, tticks, rotation=90)
    return ytest

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
    return ymed, (h(cdf, 0.975)-h(cdf, 0.025))/2.0

def main(img_path, file_path, idx=None):
    provinces = ["Jakarta", "Jabar", "Jatim", "Jateng", "Sulsel"]
    fields = ["id_new_cases"] + ["idprov_%s_new_cases" % p.lower() for p in provinces]
    names = ["Indonesia"] + provinces

    if idx is not None:
        provinces = provinces[idx:idx+1]
        fields = fields[idx:idx+1]
        names = names[idx:idx+1]

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

        if idx is not None:
            return

        dl = res.dataloader
        model = res.model
        samples = res.samples
        ysim = res.ysim
        res.yobs = res.yobs[:ysim.shape[1]]
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

        total_cases_median = ""
        total_cases_025 = ""
        total_cases_975 = ""
        if df == "id_new_cases":
            # show the tests
            plt.subplot(1,ncols,3)
            test_weekly = plot_weekly_tests(res)
            # calculate the ratio for the last week
            test_ratio = test_weekly[-1] / test_weekly[-2]
            test_ratio_2std = 1e-8 # very small std

            # calculate the estimated infection cases
            total_deaths_data = DataLoader("id_cum_deaths").ytime[-1]
            total_cases_median, total_cases_025, total_cases_975 = get_total_cases(res, total_deaths_data)

        elif df.startswith("idprov_") and df.endswith("_new_cases"):
            plt.subplot(1,ncols,3)
            test_weekly, test_weekly_2std = plot_weekly_tests_prov(res)
            # NOTE: comment the code below to use the ratio from the national figures
            # test_ratio = test_weekly[-1] / test_weekly[-2]
            # test_ratio_2std = ((test_weekly_2std[-1] / test_weekly[-1])**2 +\
            #                    (test_weekly_2std[-2] / test_weekly[-2])**2)**.5 *\
            #                     test_ratio

            # calculate the estimated infection cases
            total_deaths_data = DataLoader(df.replace("_new_cases", "_cum_deaths")).ytime[-1]
            total_cases_median, total_cases_025, total_cases_975 = get_total_cases(res, total_deaths_data)

        plt.tight_layout()
        plt.savefig(os.path.join(img_path, "%s.png"%df))
        plt.close()

        ################## deciding the results ##################
        b_last = b[:,-1]
        # calculate the exponential factor from the weekly ratio
        b_test = np.log(test_ratio) / 7.0
        b_test_std = test_ratio_2std / test_ratio
        lower_grad_portion = np.sum(b_last < b_test) * 1.0 / b.shape[0]

        # calculate the probability of the curve going down
        decline_portion = np.sum(b_last < 0) * 1.0 / b.shape[0]

        # calculate the total probability of it's really going down
        if b_test < 0:
            decline_prob = lower_grad_portion
        else:
            decline_prob = decline_portion

        if decline_prob > 0.95:
            flatcurve_res = "**turun**"
        elif decline_prob > 0.75:
            flatcurve_res = "**kemungkinan** turun"
        elif decline_prob > 0.5:
            flatcurve_res = "ada indikasi penurunan, tapi belum dapat dipastikan"
        elif decline_prob < 0.5 and decline_portion > 0.5:
            flatcurve_res = "kurva terlihat turun, tapi jumlah tes juga menurun"
        else:
            flatcurve_res = "belum dapat disimpulkan"

        # calculate the probability of the curve going down not because of the test
        # i.e. compare the test gradient and the curve gradient


        ## save the information for the templating
        places.append({
            "dataid": df,
            "name": names[i],
            "flatcurve_result": flatcurve_res,
            "decline_prob": int(np.round(decline_prob * 100)),

            # predicted cases
            "total_cases_median": total_cases_median,
            "total_cases_025": total_cases_025,
            "total_cases_975": total_cases_975,
        })

    with open(ftemplate, "r") as f:
        template = Template(f.read())
    today = datetime.date.today()
    content = template.render(places=places, date=today.strftime("%d/%m/%Y"))
    with open(file_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=None)
    args = parser.parse_args()

    fpath = "../../mfkasim91.github.io"
    img_path = os.path.join(fpath, "assets", "idcovid19-daily")
    file_path = os.path.join(fpath, "idcovid19.md")
    main(img_path, file_path, idx=args.idx)
