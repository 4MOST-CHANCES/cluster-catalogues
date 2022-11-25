import argparse
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import ascii, fits
from astropy.table import Table, join, vstack
from astroquery.ned import Ned
from astroquery.sdss import SDSS
from colossus.cosmology import cosmology
from colossus.halo import concentration
from icecream import ic
from matplotlib import pyplot as plt, ticker
import numpy as np
import os
from profiley.nfw import NFW
from time import time
from tqdm import tqdm
import urllib3

from astro.clusters import Catalog
from plottery.plotutils import savefig, update_rcParams
update_rcParams()


def main():
    args = parse_args()
    cat = chances_catalog(args)
    n = cat['RA_J2000'].size
    #catalogs = [load_catalog(args, cat, name) for name in ('psz2','advact')]
    cat, psz = load_catalog(args, cat, 'psz2')
    cat, act = load_catalog(args, cat, 'act-dr5')
    cat, mcxc = load_catalog(args, cat, 'mcxc')
    ic(np.sort(psz.colnames))
    ic(np.sort(act.colnames))
    ic(np.sort(mcxc.colnames))
    # other catalogs
    cat, mk = meerkat(args, cat)
    cat, mkd = meerkat_diffuse(args, cat)
    #cat, rass = load_rass(args, cat)
    cat, xmm = load_xmm(args, cat)
    ic(cat)

    # add masses
    cat = add_masses(cat, psz, 'MSZ', 10**0.13)
    cat = add_masses(cat, act, 'M500cCal', None)
    cat = add_masses(cat, mcxc, 'M500', 10**0.18)
    cat['m500'] = cat['m500_ACT']
    cat['m500'][cat['m500'] == -1] = cat['m500_PSZ2_corr'][cat['m500'] == -1]
    cat['m500'][cat['m500'] == -1] = cat['m500_MCXC_corr'][cat['m500'] == -1]
    cat = calculate_m200(cat)
    ic(cat)

    # match galaxies to spectra
    cat, spec = load_spec(args, cat)

    cat.write(f'catalogues/chances_clusters_{args.sample}.txt',
              format='ascii.fixed_width', overwrite=True)
    cat.write(f'catalogues/chances_clusters_{args.sample}.csv',
              format='ascii.csv', overwrite=True)
    
    summarize_ancillary(args, cat)
    return

    # is there any cluster that has ancillary data but not a mass?
    # ic(np.sort(cat.colnames))
    # nomass = cat['m500'] == -1
    # if args.sample == 'lowz':
    #     has_spec = cat['AAOzs'] > 0
    #     cols = 'Cluster Name','RA_J2000','Dec_J2000','z','AAOzs','SPLUS','SPLUS comments'
    # else:
    #     # just so that it doesn't fail with the evolution sample
    #     has_spec = np.ones(n, dtype=bool)
    #     cols = 'Cluster Name','RA_J2000','Dec_J2000','z','SPLUS'
    # has_aux = (cat['XMM'] != '') | (cat['MeerKAT'] != '')
    # ic(has_spec.sum(), nomass.sum(), has_aux.sum(), (nomass & has_aux).sum(),
    #    (has_spec & nomass).sum(), (has_spec & has_aux).sum())
    # ic(cat[cols][nomass & has_spec])

    # are we missing massive clusters?
    # these are the minimal constraints: redshift and declination
    psz_z = (psz['REDSHIFT'] > args.zrng[0]) & (psz['REDSHIFT'] < args.zrng[1]) \
        & (psz['DEC'] > -80) & (psz['DEC'] < 5)
    act_z = (act['redshift'] > args.zrng[0]) & (act['redshift'] < args.zrng[1]) \
        & (act['decDeg'] > -80) & (act['decDeg'] < 5)
    ic(psz_z.sum(), act_z.sum())

    massive = {'psz2': np.argsort(psz['MSZ'][psz_z])[-20:], 
               'act-dr5': np.argsort(act['M500cCal'][act_z])[-20:]}
    ic(np.sort(psz['MSZ'][psz_z][massive['psz2']].value))
    ic(np.sort(act['M500cCal'][act_z][massive['act-dr5']].value))
    ic(psz['NAME'][psz_z][massive['psz2']].value)
    #ic(cat['PSZ2')
    massive_in_chances = {
        'psz2': np.in1d(psz['NAME'][psz_z][massive['psz2']], cat['PSZ2']),
        'act-dr5': np.in1d(act['name'][act_z][massive['act-dr5']], cat['ACT-DR5'])
    }
    ic(massive_in_chances['psz2'].size, massive_in_chances['act-dr5'].size)
    ic(massive_in_chances['psz2'].sum(), massive_in_chances['act-dr5'].sum())
    ic(psz['NAME','REDSHIFT','MSZ'][psz_z][massive['psz2']][massive_in_chances['psz2']])
    ic(psz['NAME','REDSHIFT','MSZ'][psz_z][massive['psz2']][~massive_in_chances['psz2']])
    ic(act['name','redshift','M500cCal'][act_z][massive['act-dr5']][massive_in_chances['act-dr5']])
    ic(act['name','redshift','M500cCal'][act_z][massive['act-dr5']][~massive_in_chances['act-dr5']])


    #query_muse(args, cat)
    #query_tgss(args, cat)
    #query_ned(args, cat)
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
    # cat = Catalog('CHANCES', catalog=cat,
    #               base_cols=('Cluster Name','RA_J2000','Dec_J2000','Z'))
    cat.rename_column('Z', 'z')
    cat['coords'] = SkyCoord(
        ra=cat['RA_J2000'], dec=cat['Dec_J2000'], unit='deg')
    return cat


def calculate_m200(chances, cosmo=Planck18):
    m200, c200, r200, d200 = calculate_m200_from_m500(
        1e14*chances['m500'], chances['z'], cosmo=cosmo)
    chances['m200'] = m200 / 1e14
    chances['c200'] = c200
    chances['r200'] = r200
    chances['d200'] = d200
    for col in ('m200', 'c200', 'r200', 'd200'):
        chances[col][chances['m500'] == -1] = -1
        chances[col].format = '%.2f'
        chances[col][np.isnan(chances[col])] = -1
    return chances


def calculate_m200_from_m500(m500, z, cosmo=Planck18, model='ishiyama21'):
    cosmology.fromAstropy(cosmo, sigma8=0.81, ns=0.966)
    c = concentration.concentration(m500, '500c', z, model=model)
    nfw = NFW(m500, c, z, overdensity=500, frame='physical', cosmo=cosmo)
    m200, c200 = nfw.mdelta(200)
    r200 = nfw.rdelta(200)
    kpc2arcmin = cosmo.arcsec_per_kpc_proper(z)
    d200 = (r200*u.Mpc * kpc2arcmin).to('arcmin').value
    return m200, c200, r200, d200


#### MUSE query ####


def run_query(args, catalog):
    names = ['index', 'name', 'ra', 'dec', 'z', 'm200',
             'object', 'texp(s)', 'dist(Mpc)', 'proposal', 'dp_id']
    write_formats = {
        'index': '%5d', 'ra': '%10.6f', 'dec': '%9.5f',
        'z': '%.3f', 'lambda': '%5.1f',
        **{name: '%s'
           for name in ('name','proposal','object','dp_id','texp(s)','dist(Mpc)')}
        }
    ncl = catalog['name'].size
    nblocks = ncl // args.block_size + 1
    query = None
    eso = Eso()
    Eso.login('cjsifon', store_password=True)
    for iblock in range(nblocks):
        print(f'Block # {iblock+1:2d} / {nblocks}')
        start = iblock * args.block_size
        end = min((iblock+1)*args.block_size, ncl)
        ic(catalog['ra','dec'][start:end])
        if args.threads == 1:
            q = [query_cluster(args, cosmo, eso, i, cluster)
                 for i, cluster in tqdm(enumerate(
                    catalog[start:end], iblock*args.block_size),
                    total=end-start)]
        else:
            ti = time()
            pool = ThreadPool(args.threads)
            q = [pool.apply_async(query_cluster,
                                  args=(args,cosmo,eso,i,cluster))
                 for i, cluster in enumerate(catalog[start:end],
                                             iblock*args.block_size)]
            pool.close()
            pool.join()
            q = [i.get() for i in q]
            print(f'Done in {(time()-ti)/60:.2f} min')
        q = [i for i in q if i is not None]
        if len(q) == 0:
            continue
        q = [[obs[j] for obs in q] for j in range(len(q[0]))]
        ic(len(q), len(q[0]))
        ic(names, len(names))
        if query is None:
            query = Table(q, names=names)
        else:
            query = vstack([query, Table(q, names=names)])
        print(f'Have found data for {len(query["name"])} clusters')
        query.write(
            'archive_info/programs.fits', format='fits', overwrite=True)
        query.write('archive_info/programs.txt', format='ascii.fixed_width',
                    formats=write_formats, overwrite=True)
        print('-----')
    return


def query_cluster(args, cosmo, eso, i, cluster):
    """Takes a single row from the redmapper catalog"""
    a2k = cosmo.arcsec_per_kpc_proper(cluster['z'])
    size = (args.search_radius * a2k).to(u.deg)
    column_filters = {
        'coord1': cluster['ra'], 'coord2': cluster['dec'],
        'box': size.value, 'format': 'decimal'}
    query = eso.query_instrument(
        'muse', column_filters=column_filters, columns=[])
    if query is None or 'Program ID' not in query.keys():
        return
    ic(query.colnames)
    query = query[query['DPR CATG'] == 'SCIENCE']
    clinfo = [cluster[key] for key in ('name','ra','dec','z','lambda')]
    clcoord = SkyCoord(ra=cluster['ra']*u.deg, dec=cluster['dec']*u.deg)
    qcoord = SkyCoord(ra=query['RA']*u.deg, dec=query['DEC']*u.deg)
    dist = (clcoord.separation(qcoord) / a2k).to(u.Mpc).value
    # information obtained from the query
    qinfo = \
        [','.join([str(i) for i in np.unique(query['Object'].value)])] \
        + [','.join([f'{ti:.0f}' for ti in query['EXPTIME [s]'].value])] \
        + [','.join([f'{i:.2f}' for i in dist])] \
        + [','.join(np.unique(query[key])) for key in ['Program ID', 'DP.ID']]
    info = [i] + clinfo + qinfo
    return info

#### Ancillary data ####


def add_masses(chances, cat, masscol, factor):
    suff = cat.label.split('-')[0]
    chances[f'm500_{suff}'] = -np.ones(chances['RA_J2000'].size)
    # ic(cat[masscol])
    # ic(np.sort(chances.colnames))
    # ic(np.sort(cat.colnames))
    mask = chances[f'{cat.label}_idx'] > -99
    chances[f'm500_{suff}'][mask] \
        = cat[masscol][chances[f'{cat.label}_idx'][mask]]
    chances[f'm500_{suff}'].format = '%.2f'
    if factor is not None:
        chances[f'm500_{suff}_corr'] = -np.ones(chances['RA_J2000'].size)
        chances[f'm500_{suff}_corr'][mask] \
            = factor * chances[f'm500_{suff}'][mask]
        chances[f'm500_{suff}_corr'].format = '%.2f'
    return chances


def load_catalog(args, chances, name):
    """This is for the ones I have in ``astro``"""
    cat = Catalog(name)
    ic(cat)
    goodz = (cat[cat.base_cols[3]] >= args.zrng[0]) \
        & (cat[cat.base_cols[3]] < args.zrng[1])
    gooddec = (cat[cat.base_cols[2]] >= -80) & (cat[cat.base_cols[2]] <= 5)
    cat = Catalog(name, catalog=cat[goodz & gooddec])
    ic(cat)
    match_catalog(chances, cat)
    return chances, cat


def load_rass(args, chances):
    filename = 'aux/xray/local_fof_insideSplus.cat'
    return


def load_spec(args, chances, base_cols=('index','RA','DEC','z')):
    filename = 'aux/spectroscopic/SpecZ_Catalogue_20221105.csv'
    spec = Catalog(
        'spec', ascii.read(filename, format='csv'), base_cols=base_cols)
    chances, spec = match_galaxy_catalog(args, chances, spec)
    return chances, spec

def load_xmm(args, chances):
    filename = 'aux/xray/local_fof_insideSplus_xmm.cat'
    xmm = Catalog(
        'XMM', ascii.read(filename, format='commented_header'),
        base_cols=('OBSERVATION.TARGET','RA_X','DEC_X','z'))
    chances, xmm = match_catalog(chances, xmm)
    return chances, xmm


def match_galaxy_catalog(args, chances, galcat, radius=5, unit='r200',
                         m500_if_missing=1e14, dz=0.03):
    """Match cluster catalog to external galaxy catalog
    
    ``unit`` must be either 'arcmin' or 'r200'"""
    assert unit in ('arcmin', 'r200')
    if unit == 'r200':
        missing = (chances['d200'] == -1)
        if missing.sum() > 0:
            massdata = calculate_m200_from_m500(
                m500_if_missing, chances['z'][missing])
            for col, name in zip(massdata, 'mcrd'):
                chances[f'{name}200'][missing] = col
            chances['m200'][missing] /= 1e14
    maxdist = (radius * chances['d200'] if unit == 'r200' else radius) * u.arcmin
    ic(galcat.z.shape, maxdist.shape)
    Nspec, Nspec_z = np.zeros((2,maxdist.size), dtype=int)
    for i, (cl, dmax) in tqdm(enumerate(zip(chances, maxdist))):
        ic(i, cl['Cluster Name','RA_J2000','Dec_J2000','z','m200','d200'])
        cosdec = np.cos(np.radians(cl['Dec_J2000']))
        nearby = (np.abs(cl['RA_J2000']-galcat.ra)*u.deg < 2*dmax/cosdec) \
            & (np.abs(cl['Dec_J2000']-galcat.dec)*u.deg < 2*dmax)
        sep = cl['coords'].separation(galcat.coords[nearby])
        ic(dmax, nearby.sum(), nearby.sum()/galcat.z.size)
        matches = (sep < dmax)
        Nspec[i] = matches.sum()
        zmatches = matches & (np.abs(galcat.z[nearby]-cl['z'])/(1+cl['z']) < dz)
        Nspec_z[i] = zmatches.sum()
        ic(Nspec[i], Nspec_z[i])
        clname = cl['Cluster Name'].replace(' ', '_')
        galcat.catalog[nearby][galcat.base_cols][zmatches].write(
            f'aux/spectroscopic/{args.sample}/{clname}.txt',
            format='ascii.fixed_width', overwrite=True)
    chances['Nspec'] = Nspec
    chances['Nspec_z'] = Nspec_z
    ic(chances['Cluster Name','z','m200','r200','d200','Nspec','Nspec_z'])
    return chances, galcat


def match_catalog(chances, cat, radius=5*u.arcmin, name=None):
    try:
        dist = chances['coords'].separation(cat.coords[:,None])
    except AttributeError:
        coords = SkyCoord(ra=cat['ra'], dec=cat['dec'], unit='deg')
        dist = chances['coords'].separation(coords[:,None])
    closest = np.min(dist, axis=0)
    matches = (closest < radius)
    ic(cat.name, matches.sum())
    idx = -99 * np.ones(matches.size, dtype=int)
    idx[matches] = np.argmin(dist, axis=0)[matches]
    try:
        name = cat.label
    except AttributeError:
        if name is None:
            raise ValueError('must provide a name')
    chances[f'{name}_idx'] = idx
    chances[name] \
        = [cat.obj[i] if i > -99 else '' for i in idx]
    return chances, cat


def meerkat(args, chances):
    mk = Catalog(
        'MeerKAT', ascii.read('aux/meerkat/meerkat_legacy.csv', format='csv'))
    #ic(mk)
    chances, mk = match_catalog(chances, mk, name='MeerKAT')
    return chances, mk


def meerkat_diffuse(args, chances):
    mkd = fits.open('aux/meerkat/Table4_MGCLS_diffuse.fits')[1].data
    mkd = Catalog(
        'MKDiffuse', mkd,
        base_cols=('ClusterName','R.A.J2000 (deg)','Dec.J2000 (deg)','z'))
    chances, mkd = match_catalog(chances, mkd, name='MKDiffuse')
    return chances, mkd


def query_ned(args, cat, path='aux/ned/spec'):
    os.makedirs(path, exist_ok=True)
    http = urllib3.PoolManager()
    for i, cl in enumerate(cat):
        name = cl['Cluster Name'].replace(" ", "_")
        output = os.path.join(path, f'{name}_ned.txt')
        if os.path.isfile(output):
            continue
        #query = Ned.query_region(cl['coords'], radius=cl['d200']*u.arcmin)
        #ic(cl, query['Object Name'].size)
        #query.write(output, format='ascii.fixed_width')
        ti = time()
        url = http.request('GET',
            'http://ned.ipac.caltech.edu/cgi-bin/objsearch' \
            '?search_type=Near+Position+Search&in_csys=Equatorial' \
            '&in_equinox=J2000.0&lon={:.6f}d&lat={:.6f}d&radius={:.2f}' \
            '&dot_include=ANY&in_objtypes1=GGroups&in_objtypes1=GClusters' \
            '&in_objtypes1=QSO&in_objtypes2=Radio&in_objtypes2=SmmS' \
            '&in_objtypes2=Infrared&in_objtypes2=Xray&nmp_op=ANY' \
            '&out_csys=Equatorial&out_equinox=J2000.0&obj_sort=RA+or+Longitude' \
            '&of=ascii_tab&zv_breaker=30000.0&list_limit=5&img_stamp=YES'.format(
                cl['RA_J2000'], cl['Dec_J2000'], 60*cl['d200']))
        ic(url)
        ic(url.data)
        ic(time()-ti)
        break
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


def summarize_ancillary(args, chances):
    cols = ['Cluster Name', 'RA_J2000', 'Dec_J2000', 'z',
            'MCXC', 'XMM', 'PSZ2', 'ACT-DR5', 'MeerKAT', 'Nspec', 'Nspec_z']
    if args.sample == 'lowz':
        cols.extend(['SPLUS', 'AAOzs', 'Lx'])
    tbl = chances[cols]
    # optical photometry
    decam = (chances['DECam'] != 'NO')
    if args.sample == 'lowz':
        vst = (chances['VST'] == 'Y')
        opt = (vst | decam)
        print(f'{(vst & decam).sum()} with both VST and DECam,' \
            f' {(1 - opt).sum()} without either:')
    else:
        opt = decam
        print(f'{decam.sum()} with DECam, {(1-decam).sum()} without:')
    print(tbl[~opt])
    if args.sample == 'lowz':
        print(f'  {vst.sum()} with VST')
    print(f'  {decam.sum()} with DECam')
    if args.sample == 'lowz':
        splus_observed = ~np.isin(chances['SPLUS'], ('', 'NO'))
        splus_upcoming = np.array(
            ['target' in comment for comment in chances['SPLUS comments']])
        splus = splus_observed | splus_upcoming
        print(f'{splus.sum()} with SPLUS')
        print(f'  {splus_observed.sum()} already with SPLUS')
        print(f'  {splus_upcoming.sum()} with upcoming SPLUS')
    #print()
    # spectroscopy
    if args.sample == 'lowz':
        aao = (chances['AAOzs'] > 0)
        print(f'{aao.sum()} with AAOzs')
    # SZ
    psz = (chances['PSZ2'] != '')
    act = (chances['ACT-DR5'] != '')
    sz = psz | act
    print(f'{sz.sum()} with SZ')
    print(f'  {psz.sum()} with PSZ2')
    print(f'  {act.sum()} with ACT-DR5')
    # X-rays
    xmm = (chances['XMM'] != '')
    mcxc = (chances['MCXC'] != '')
    xrays = xmm | mcxc
    print(f'{xrays.sum()} with X-rays')
    print(f'  {xmm.sum()} with XMM')
    print(f'  {mcxc.sum()} in MCXC')
    # radio
    mk = (chances['MeerKAT'] != '')
    print(f'{mk.sum()} with MeerKAT')
    print('## Combinations ##')
    print(f'{(sz & xrays).sum()} with SZ and X-rays')
    print(f'{(act & xmm).sum()} with ACT-DR5 and XMM')
    print(f'{(sz & mk).sum()} with SZ and MeerKAT')
    print(f'{(~sz & ~xrays).sum()} without SZ nor X-rays:')
    print(tbl[(~sz & ~xrays)])
    #print(f'')
    return


def query_tgss(args, cat):
    return

def parse_args():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('sample', choices=('evolution', 'lowz'), default='lowz')
    add('--debug', action='store_true')
    args = parser.parse_args()
    if not args.debug:
        ic.disable()
    args.zrng = (0, 0.07) if args.sample == 'lowz' else (0.07, 0.5)
    return args


main()
