import argparse
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table, join, vstack
from astroquery.sdss import SDSS
from icecream import ic
from matplotlib import pyplot as plt, ticker
import numpy as np
import os

from astro.clusters import Catalog
from plottery.plotutils import savefig, update_rcParams
update_rcParams()


def main():
    args = parse_args()
    cat = chances_catalog(args)
    #catalogs = [load_catalog(args, cat, name) for name in ('psz2','advact')]
    cat, psz = load_catalog(args, cat, 'psz2')
    ic(np.sort(cat.colnames))
    ic(cat)
    #sdss = query_sdss(args, cat)
    return


def chances_catalog(args):
    if args.sample == 'lowz':
        file = 'CHANCES low-z clusters.csv'
    else:
        file = 'CHANCES Evolution clusters.csv'
    cat = ascii.read(file, format='csv')
    # happens in low-z
    if 'col12' in cat.colnames:
        ic(cat['col12'].value)
        cat.remove_column('col12')
    cat['coords'] = SkyCoord(
        ra=cat['RA_J2000'], dec=cat['Dec_J2000'], unit='deg')
    return cat


#### Ancillary data ####


def load_catalog(args, chances, name):
    cat = Catalog(name)
    ic(cat)
    match_catalog(chances, cat)
    return chances, cat


def match_catalog(chances, cat, radius=5*u.arcmin):
    dist = chances['coords'].separation(cat.coords[:,None])
    closest = np.min(dist, axis=0)
    ic(closest.shape)
    matches = (closest < radius)
    ic(matches.sum())
    idx = -99 * np.ones(matches.size, dtype=int)
    idx[matches] = np.argmin(dist, axis=0)[matches]
    chances[f'{cat.name}_idx'] = idx
    chances[cat.name] \
        = [cat.clusters[i] if i > -99 else '' for i in idx]
    return chances, cat


def meerkat(args):
    mk = ascii.read('aux/meerkat/meerkat_legacy.csv', format='csv')
    return



def query_sdss(args, cat):
    output = f'aux/sdss/sdss_spec_{args.sample}.tbl'
    if os.path.isfile(output):
        return ascii.read(output, format='fixed_width')
    sdss = SDSS.query_region(cat['coords'], spectro=True, radius=1*u.deg)
    ic(sdss)
    if len(sdss.colnames) == 1 and 'htmlhead' in sdss.colnames:
        return
    sdss.write(output, format='ascii.fixed_width', overwrite=True)
    return sdss
    

def parse_args():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('sample', choices=('evolution', 'lowz'), default='lowz')
    add('--debug', action='store_true')
    args = parser.parse_args()
    if not args.debug:
        ic.disable()
    return args


main()
