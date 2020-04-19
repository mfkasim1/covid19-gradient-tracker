import os
import datetime
import pickle
import geopandas as gp
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

def find_province(map_geom, point):
    for i in range(len(map_geom)):
        geom = map_geom.geometry[i]
        if not geom.is_valid:
            geom = geom.convex_hull
        if geom.contains(point):
            return map_geom.iloc[i].state
    return None

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
        self.tstart = datetime.datetime.strptime("2020-03-31 00:00", "%Y-%m-%d %H:%M")
        self.tend = datetime.datetime.today()
        self.tdelta = datetime.timedelta(hours=8)
        self.country = "ID"

    def get_fsave(self, tnow):
        fsave = datetime.datetime.strftime(tnow, self.fsave_format)
        return fsave

    def get_fmovement(self, tnow):
        fname = datetime.datetime.strftime(tnow, self.fmovements_format)
        return fname

    def datetime_iter(self):
        tnow = self.tstart
        while tnow < self.tend:
            yield tnow
            tnow = tnow + self.tdelta

def convert():
    # the general helper class
    info = Information()

    # load the geometry of the map
    map_geom = gp.read_file(info.fmap)

    n_missing_points = 0
    for tnow in info.datetime_iter():
        fname = info.get_fmovement(tnow)
        fsave = info.get_fsave(tnow)
        if not os.path.exists(fname) or os.path.exists(fsave):
            continue

        movement_geom = pd.read_csv(fname)
        provinces_maps = {}
        print("Reading %s" % fname)
        for i in tqdm(range(movement_geom.shape[0])):
            rowdata = movement_geom.iloc[i]
            if rowdata.country != info.country:
                continue
            start_point = Point(rowdata.start_lon, rowdata.start_lat)
            end_point = Point(rowdata.end_lon, rowdata.end_lat)

            # get the starting and ending provinces
            start_province = find_province(map_geom, start_point)
            if start_province is None:
                n_missing_points += 1
                continue
            end_province = find_province(map_geom, end_point)
            if end_province is None:
                n_missing_points += 1
                continue

            key = "%s-%s" % (start_province, end_province)
            add_key(provinces_maps, rowdata.n_baseline, start_province, end_province, "n_baseline")
            add_key(provinces_maps, rowdata.n_crisis, start_province, end_province, "n_crisis")

        # save the provinces maps
        with open(fsave, "wb") as fb:
            pickle.dump(provinces_maps, fb)

    print("Missing %d points" % n_missing_points)

def main():
    # convert the unconverted files
    convert()


if __name__ == "__main__":
    convert()
