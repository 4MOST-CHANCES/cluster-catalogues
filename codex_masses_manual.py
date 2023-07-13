from astropy import units as u
from astropy.cosmology import Planck18
from astropy.io import ascii, fits
from astropy.table import Table
from colossus.cosmology import cosmology
from colossus.halo import concentration
import numpy as np

from profiley.nfw import NFW


def main():
    cosmology.fromAstropy(Planck18, sigma8=0.81, ns=0.966)
    lowz = ascii.read('catalogues/clusters_chances_lowz.csv', format='csv')
    lowz.sort('name')
    print(np.sort(lowz.colnames))
    in_codex = ~(lowz['CODEXID'].mask)
    print(in_codex.sum(), in_codex.size)
    codex = fits.open('aux/xray/codex3_lsdr10_info.fits')
    codex = Table(codex[1].data)
    print('name  z  M200c  r200(Mpc)  d200(arcmin)')
    for cluster in lowz:
        j = (codex['CODEX3'] == cluster['CODEXID'])
        if j.sum() == 0 or not np.any(j):
            continue
        name = cluster['name']
        z = cluster['z']
        codex_id = cluster['CODEXID']
        m200c = codex['M200c'][j][0]
        c = concentration.concentration(m200c, '200c', z, model='ishiyama21')
        nfw = NFW(m200c, c, z, overdensity=200, frame='physical', cosmo=Planck18)
        r200 = nfw.rdelta(200)[0]
        kpc2arcmin = Planck18.arcsec_per_kpc_proper(z)
        d200 = (r200*u.Mpc * kpc2arcmin).to('arcmin').value
        print(f'{name:8s} {z:5.3f} {m200c/1e14:5.2f} {r200:.2f} {d200:6.2f}')
    return


if __name__ == "__main__":
    main()