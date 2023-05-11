"""Identify unique CHANCES fields

For each entry in the provided file, search through that and the
two other lists from the same run (e.g., lowz_complete, lowz_lowmass,
lowz_cherrypick) for clusters that would be included in the same
observation, i.e., all systems within 5r200. Then, calculate what
fraction of 5r200 of those clusters would be included in that
observation
"""

import argparse
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import time


def main():
    args = parse_args()
    tbl = ascii.read(args.filename, format='fixed_width')
    d200 = tbl['d200'] * u.arcmin
    coords = SkyCoord(ra=tbl['ra'], dec=tbl['dec'], unit='deg')
    tbl_matches = Table(
        {'name': [], 'Cluster Name': [], 'z': [], 'd200 (arcmin)': [],
         'N(<5r200)': [],
         'Separations (arcmin)': [], 'Fraction of r200 included': [],
         'redshifts': [], 'Names': []},
         dtype=[int, str, float, float, int, str, str, str, str])
    for i, cl in enumerate(tbl):
        c = SkyCoord(ra=cl['ra'], dec=cl['dec'], unit='deg')
        maxdist = 5 * d200[i]
        sep = c.separation(coords).to(u.arcmin)
        j = np.argsort(sep)
        sep = sep[j]
        within = (sep < maxdist)
        sep_d200 = ((maxdist-sep[within]) / d200[j][within]).to(1).value
        tbl_matches.add_row(
            [cl['name'], cl['Cluster Name'], cl['z'], cl['d200'],
             within.sum()-1,
             ','.join([f'{s:.1f}' for s in sep[within].value][1:]),
             ','.join([f'{d:.1f}' for d in sep_d200][1:]),
             ','.join([f'{n:.3  f}' for n in tbl['z'][j][within][1:]]),
             ','.join([f'{n:.0f}' for n in tbl['name'][j][within]][1:])])
    sample = 'lowz' if 'lowz' in args.filename else 'evolution'
    output = f'output/withinfields_{sample}.txt'
    tbl_matches.write(
        output, format='ascii.fixed_width', overwrite=True)
    print(f'Saved to {output}')
    return


def parse_args():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('filename', type=str,
        help='Cluster sample table filename, typically from output/')
    args = parser.parse_args()
    assert 'masscounts' in args.filename
    return args


main()
