"""
This code works with the output of ``selection.py``
"""
import argparse
from astropy.io import ascii, fits
from astropy.table import Table, vstack
from matplotlib import pyplot
import numpy as np
import os
import sys
import time


def main():
    args = parse_args()
    cluster_sample = ClusterSample(
        args.filename, args.sample, args.zbins)
    for subsample in ('complete', 'lowmass', 'cherrypick'):
        print(f'{subsample.capitalize()} sample:')
        tbl = []
        for zbin in range(4):
            s = cluster_sample.zbin(subsample, zbin)
            #print(s)
            mk = ~s['MeerKAT'].mask
            lv = ~s['LoVoCCS'].mask
            hf = ~s['HIFLUGCS'].mask
            ch = ~s['CHANCES'].mask
            mask = (mk | lv | hf | ch)
            if args.sample == 'lowz':
                wi = ~s['WINGS'].mask
                sp = ~s['S-PLUS'].mask
                mask = mask | wi | sp
            if np.sum(mask) == 0:
                continue
            print(f'*** zbin={zbin} ***')
            if subsample == 'complete' or True:
                ti = s
            else:
                ti = s[mask]
            ti.sort('ra')
            print(ti)
            sm = (np.isin(ti['name'], ['93192518']).sum())
            if sm  > 0:
                print('Found it!')
                return
            ti.add_column(
                (1+zbin)*np.ones(ti['ra'].size, dtype=int), index=0, name='zbin')
            tbl.append(ti)
            # write csv for easy visualization
        tbl = vstack(tbl)
        tbl.write(f'cluster-samples/{args.sample}_{subsample}.csv', format='ascii.csv',
                  overwrite=True)
        print()
    return


class ClusterSample:

    def __init__(self, filename, sample='evolution', zbins='default'):
        self.filename = filename
        self.sample = sample
        self.catalog = ascii.read(self.filename, format='fixed_width')
        if zbins == 'default':
            zbins = [0.07, 0.15, 0.25, 0.35, 0.452] \
                if self.sample == 'evolution' \
                else [0, 0.01, 0.03, 0.05, 0.07]#[0, 0.02, 0.04, 0.055, 0.07]
        self.zbins = np.array(zbins)

    @property
    def complete(self):
        return np.array([v == 'True' for v in self.catalog['complete'].value])

    @property
    def lowmass(self):
        return np.array([v == 'True' for v in self.catalog['lowmass'].value])

    @property
    def cherrypick(self):
        return np.array(
                [v == 'False' for v in self.catalog['complete'].value]) \
            & np.array(
                [v == 'False' for v in self.catalog['lowmass'].value])

    @property
    def in_erosita(self):
        return self.catalog['gal_l'] > 180

    @property
    def in_splus(self):
        return self.catalog['in_splus'] != ''

    @property
    def clusters(self):
        return self.catalog['name']

    @property
    def ra(self):
        return self.catalog['ra']

    @property
    def dec(self):
        return self.catalog['dec']

    @property
    def l(self):
        return self.catalog['gal_l']

    @property
    def z(self):
        return self.catalog['z']

    @property
    def zbinned(self):
        dig = np.digitize(self.z, self.zbins)
        return [self.catalog[dig == i+1] for i in range(self.zbins.size)]

    def zbin(self, subsample, zbin_idx):
        assert subsample in ('complete', 'lowmass', 'cherrypick')
        zbin = (np.digitize(self.z, self.zbins) == zbin_idx + 1)
        if subsample == 'complete': mask = self.complete
        elif subsample == 'lowmass': mask = self.lowmass
        elif subsample == 'cherrypick': mask = self.cherrypick
        return self.catalog[zbin & mask]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--sample', type=str, default='evolution')
    parser.add_argument('--zbins', type=str, default='default')
    args = parser.parse_args()
    return args


main()