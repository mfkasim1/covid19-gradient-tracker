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

def convert():
    # specify the files
    fmap = "data/geojson/indonesia.geojson"
    fmovements_format = "data/fb/movements/movement_%Y-%m-%d_%H%M.csv"
    fsave_format = "data/fb/movements/movement_%Y-%m-%d_%H%M.pkl"
    tstart = datetime.datetime.strptime("2020-03-31 00:00", "%Y-%m-%d %H:%M")
    tend = datetime.datetime.strptime("2020-05-01 00:00", "%Y-%m-%d %H:%M")
    tdelta = datetime.timedelta(hours=8)
    country = "ID"

    # load the geometry of the map
    map_geom = gp.read_file(fmap)

    tnow = tstart
    n_missing_points = 0
    while tnow < tend:
        # load the file
        fname = datetime.datetime.strftime(tnow, fmovements_format)
        fsave = datetime.datetime.strftime(tnow, fsave_format)
        if not os.path.exists(fname) or os.path.exists(fsave):
            tnow = tnow + tdelta
            continue

        movement_geom = pd.read_csv(fname)
        provinces_maps = {}
        print("Reading %s" % fname)
        for i in tqdm(range(movement_geom.shape[0])):
            row = movement_geom.iloc[i]
            if row.country != country:
                continue
            start_point = Point(row.start_lon, row.start_lat)
            end_point = Point(row.end_lon, row.end_lat)

            # get the starting and ending provinces
            start_province = find_province(map_geom, start_point)
            end_province = find_province(map_geom, end_point)
            if start_province is None or end_province is None:
                n_missing_points += 1
                continue

            key = "%s-%s" % (start_province, end_province)
            add_key(provinces_maps, row.n_baseline, start_province, end_province, "n_baseline")
            add_key(provinces_maps, row.n_crisis, start_province, end_province, "n_crisis")

        # save the provinces maps
        with open(fsave, "wb") as fb:
            pickle.dump(provinces_maps, fb)

        # advancing the time
        tnow = tnow + tdelta
        # break

    print("Missing %d points" % n_missing_points)

if __name__ == "__main__":
    convert()
