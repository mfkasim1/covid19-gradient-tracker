import os
import datetime
import pickle
import geopandas as gp
import pandas as pd
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
import matplotlib.pyplot as plt

class ProvinceFinder(object):
    def __init__(self):
        self.to_province_map = {}

    def find_province(self, map_geom, point, point_id):
        if point_id in self.to_province_map:
            return self.to_province_map[point_id]

        res = None
        for i in range(len(map_geom)):
            geom = map_geom.geometry[i]
            if not geom.is_valid:
                geom = geom.convex_hull
            if geom.contains(point):
                res = map_geom.iloc[i].state
                break

        self.to_province_map[point_id] = res
        return res

def add_key(dct, value, *keys):
    present_dct = dct
    for key in keys[:-1]:
        if key not in present_dct:
            present_dct[key] = {}
        present_dct = present_dct[key]

    key = keys[-1]
    if key not in present_dct:
        present_dct[key] = value
    else:
        present_dct[key] += value

class Information(object):
    def __init__(self):
        self.fmap = "data/geojson/indonesia.geojson"
        self.fmovements_format = "data/fb/movements/movement_%Y-%m-%d_%H%M.csv"
        self.fsave_format = "data/fb/movements/movement_%Y-%m-%d_%H%M.pkl"
        self.fsave_format1 = "data/fb/movements/movement_inter_%Y-%m-%d_%H%M.pkl"
        self.tstart = datetime.datetime.strptime("2020-03-31 00:00", "%Y-%m-%d %H:%M")
        self.tend = datetime.datetime.today()
        self.tdelta = datetime.timedelta(hours=8)
        self.country = "ID"

    def get_fsave(self, tnow, skip_intra=False):
        fsave_format = self.fsave_format if not skip_intra else self.fsave_format1
        fsave = datetime.datetime.strftime(tnow, fsave_format)
        return fsave

    def get_fmovement(self, tnow):
        fname = datetime.datetime.strftime(tnow, self.fmovements_format)
        return fname

    def datetime_iter(self):
        tnow = self.tstart
        while tnow < self.tend:
            yield tnow
            tnow = tnow + self.tdelta

class ProvinceMovement(object):
    def __init__(self, dct=None):
        if dct is None:
            self.dct = {}
        else:
            self.dct = dct

    def add_nbaseline(self, start_province, end_province, value):
        add_key(self.dct, value, start_province, end_province, "n_baseline")

    def add_ncrisis(self, start_province, end_province, value):
        add_key(self.dct, value, start_province, end_province, "n_crisis")

    def get_dct(self):
        return self.dct

    def load_dct(self, dct):
        self.dct = dct

    def get_outgoing_province_changes(self, outprovince=True, inprovince=True):
        res_baseline = {}
        res_crisis = {}
        for start_province in self.dct:
            start_dct = self.dct[start_province]
            total_baseline = 0
            total_crisis = 0
            # total_changes = 0.0
            for end_province in start_dct:
                if not outprovince and start_province != end_province: continue
                if not inprovince and start_province == end_province: continue
                pair_dct = start_dct[end_province]
                total_baseline += pair_dct["n_baseline"]
                total_crisis += pair_dct["n_crisis"]
                # total_changes += pair_dct["n_crisis"] / pair_dct["n_baseline"] - 1.0
            # change = total_crisis / total_baseline - 1.0
            res_baseline[start_province] = total_baseline
            res_crisis[start_province] = total_crisis
            # res[start_province] = total_changes / len(start_dct)
        return res_baseline, res_crisis

def convert(skip_intra=False):
    # the general helper class
    info = Information()
    province_finder = ProvinceFinder()

    # load the geometry of the map
    map_geom = gp.read_file(info.fmap)

    n_missing_points = 0
    for tnow in info.datetime_iter():
        fname = info.get_fmovement(tnow)
        fsave = info.get_fsave(tnow, skip_intra=skip_intra)
        if not os.path.exists(fname) or os.path.exists(fsave):
            continue

        movement_geom = pd.read_csv(fname)
        provinces_maps = {}
        print("Reading %s" % fname)
        for i in tqdm(range(movement_geom.shape[0])):
            rowdata = movement_geom.iloc[i]
            if rowdata.country != info.country:
                continue
            if skip_intra and rowdata.start_polygon_id == rowdata.end_polygon_id:
                continue

            start_point = Point(rowdata.start_lon, rowdata.start_lat)
            end_point = Point(rowdata.end_lon, rowdata.end_lat)

            # get the starting and ending provinces
            start_province = province_finder.find_province(map_geom, start_point, rowdata.start_polygon_id)
            if start_province is None:
                n_missing_points += 1
                continue
            end_province = province_finder.find_province(map_geom, end_point, rowdata.end_polygon_id)
            if end_province is None:
                n_missing_points += 1
                continue

            add_key(provinces_maps, rowdata.n_baseline, start_province, end_province, "n_baseline")
            add_key(provinces_maps, rowdata.n_crisis, start_province, end_province, "n_crisis")

        # save the provinces maps
        with open(fsave, "wb") as fb:
            pickle.dump(provinces_maps, fb)

    print("Missing %d points" % n_missing_points)

def get_changes(outprovince=True, inprovince=True):
    info = Information()

    dates = []
    all_baselines = {}
    all_crises = {}
    for i,tnow in enumerate(info.datetime_iter()):
        fprovince = info.get_fsave(tnow)
        if not os.path.exists(fprovince):
            continue

        # load the dictionary from the pickled files
        with open(fprovince, "rb") as fb:
            province_movements_obj = ProvinceMovement(pickle.load(fb))

        # get the outgoing changes
        # {"start_province": changes}
        outgoing_baseline, outgoing_crisis = \
            province_movements_obj.get_outgoing_province_changes(outprovince, inprovince)
        for province in outgoing_crisis:
            if i == 0:
                all_baselines[province] = [outgoing_baseline[province]]
                all_crises[province] = [outgoing_crisis[province]]
            else:
                all_baselines[province].append(outgoing_baseline[province])
                all_crises[province].append(outgoing_crisis[province])
        dates.append(tnow)

    return dates, all_baselines, all_crises

def main(skip_intra=False, fdirsave=None, provinces=None):
    outprovince = True
    inprovince = False

    # set the default images directory
    if fdirsave is None:
        fdirsave = "images/"

    # convert the unconverted files
    convert(skip_intra=skip_intra)
    dates, all_baselines, all_crises = get_changes(outprovince, inprovince)
    ntime_day = 3

    if provinces is None:
        provinces = all_baselines.keys()
    for key in provinces:
    # key = "Jawa Timur"
        name = key if key != "Jakarta Raya" else "DKI Jakarta"
        print(name)

        # set the title and the filename to save
        if outprovince and not inprovince:
            travel_type = "antar-provinsi dari"
        elif not outprovince and inprovince:
            travel_type = "antar-kecamatan dalam"
        title = "Perubahan jumlah perjalanan %s %s" % (travel_type, name)
        fname = "%s_%s.png" % (travel_type, name.lower())
        fname = fname.replace(" ", "_")

        n_baseline = np.asarray(all_baselines[key])
        n_crisis = np.asarray(all_crises[key])

        n_baseline = n_baseline.reshape(-1, ntime_day).sum(axis=-1)
        n_crisis = n_crisis.reshape(-1, ntime_day).sum(axis=-1)
        changes = n_crisis / n_baseline - 1.0
        dates_day = dates[::ntime_day]

        # plot the data
        x = range(len(changes))
        plt.plot(x, changes*100, '.-')
        plt.plot(x, changes*0, 'C0--')
        plt.fill_between(x, changes*0, changes*100, color="C0", alpha=0.3)
        day_interval = 5
        xticklabel = [datetime.datetime.strftime(date, "%d/%m/%y") for date in dates_day[::day_interval]]
        plt.ylim([-100, 50])

        # set the ticklabels
        yticks, _ = plt.yticks()
        plt.yticks(yticks, [("%s%s" % (str(y), "%")) for y in yticks])
        plt.xticks(x[::day_interval], xticklabel)
        plt.xlabel("Tanggal")
        plt.title(title)
        plt.savefig(os.path.join(fdirsave, fname))
        plt.close()
        # plt.show()

if __name__ == "__main__":
    main(skip_intra=True,
         fdirsave="/mnt/c/Users/firma/Documents/Projects/Git/mfkasim91.github.io/assets/idcovid19-mobility/",
         provinces=["Jakarta Raya", "Jawa Barat", "Jawa Timur", "Sulawesi Selatan"])
