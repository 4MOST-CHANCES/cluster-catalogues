from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from icecream import ic
from matplotlib import pyplot as plt, ticker
import numpy as np
import os

from plottery.plotutils import savefig, update_rcParams

from astro.clusters import ClusterCatalog

update_rcParams()


def main():
    lowz = ClusterCatalog("chances-lowz")
    evol = ClusterCatalog("chances-evol")
    chances = vstack([lowz.catalog, evol.catalog])
    chances["coord"] = SkyCoord(ra=chances["ra"], dec=chances["dec"], unit="deg")
    for cl in chances:
        wget_decals(cl)
    return


def wget_decals(cl, size=2048):
    name = cl["name"].replace(" ", "")
    ra = cl["coord"].ra.deg
    dec = cl["coord"].dec.deg
    url = "https://www.legacysurvey.org/viewer/fits-cutout?" \
        f"ra={ra}&dec={dec}&layer=ls-dr10&pixscale=0.262&size={size}"
    for band in "grz":
        cmd = "wget -nc -nd -O imgs/decals/{name}_{band}.fits {url}&bands={band}"
        os.system(cmd)
    return

main()
