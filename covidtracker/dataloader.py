import os
import datetime
import pandas as pd
import numpy as np
import covidtracker as ct

class DataLoader(object):
    def __init__(self, dataidentifier):
        self.address = {
            "id_new_cases": {
                "file": "data/indonesia.csv",
                "xticks": lambda pddata: pddata.tanggal,
                "retrieve_fcn": lambda pddata: pddata.kasus_baru,
            },
            "id_new_deaths": {
                "file": "data/indonesia.csv",
                "xticks": lambda pddata: pddata.tanggal,
                "retrieve_fcn": lambda pddata: pddata.meninggal_baru,
            }
        }[dataidentifier]
        self.dataidentifier = dataidentifier
        self.ct_path = os.path.split(ct.__file__)[0]

        pddata = pd.read_csv(self.get_full_address(self.address["file"]))
        self._data = np.asarray(self.address["retrieve_fcn"](pddata))
        self._xticks = np.asarray(self.address["xticks"](pddata))

    @property
    def ytime(self):
        return self._data

    @property
    def tdate(self):
        return self._xticks

    def get_full_address(self, reladdr):
        return os.path.join(self.ct_path, reladdr)

    def get_fname(self):
        today = datetime.date.today()
        fname = "%s-%s.pkl" % (today.strftime("%y%m%d"), self.dataidentifier)
        fpath = self.get_full_address(os.path.join("samples", fname))
        return fpath
