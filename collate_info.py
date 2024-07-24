import argparse
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import ascii, fits
from astropy.table import Table, join, vstack
from astroquery.ipac.ned import Ned
from astroquery.sdss import SDSS
from colossus.cosmology import cosmology
from colossus.halo import concentration
from datetime import date
from icecream import ic
from matplotlib import pyplot as plt, ticker
from multiprocessing.pool import ThreadPool, Pool
import numpy as np
import os
from profiley.nfw import NFW
from scipy.ndimage import gaussian_filter
from scipy.special import erf
import sys
from time import time
from tqdm import tqdm
import urllib3

from astro.clusters import Catalog
from plottery.plotutils import savefig, update_rcParams
update_rcParams()


def main():
    args = parse_args()
    chances = chances_catalog(args)

    chances, others = load_ancillary(args, chances, 'chances')
    print(chances['name','ra','dec'][chances['z'] == 1])
    return
    psz, act, spt, codex, mcxc = others

    # print('\n\n*** Missing from CHANCES ***\n')
    # missing = review_missing(args, chances, *others)
    # missing = load_decam(args, missing)
    # missing, _ = load_ancillary(args, missing, 'missing')

    print('\n\n*** Fully mass-selected sample ***\n')
    most_massive_input \
        = [i for i in others if i.name == args.mass_selection][0]
    most_massive = get_most_massive(
        args, most_massive_input, chances)
    for catname in ('psz2', 'act-dr5', 'spt-sz', 'codex'):
        most_massive, _ = load_catalog(args, most_massive, catname)
    most_massive = load_decam(args, most_massive)
    most_massive = load_ancillary(args, most_massive, 'most_massive')[0]
    print(np.sort(most_massive.colnames))

    #print(np.sort(most_massive['Cluster Name']))
    most_massive_in_chances = np.isin(
        most_massive['name'], chances['name'])
    print(f'{most_massive_in_chances.sum()} of the' \
          f' {most_massive.size} most massive' \
          f' {most_massive_input.label} clusters in the range' \
          f' {args.zrng[0]} <= z <= {args.zrng[1]} are in CHANCES,')
    print(f'out of a total of {chances["name"].size} CHANCES' \
          f' {args.sample.capitalize()} clusters')
    cols = ['name','z','m500',most_massive.masscol,
            'PSZ2','ACT-DR5','N_spec','N_spec_z']
    print(most_massive[cols][most_massive_in_chances])
    #print(np.sort(most_massive.colnames))
    # most_massive_in_missing = np.isin(
    #     most_massive['Cluster Name'], missing['Cluster Name'])
    # print(f'Another {most_massive_in_missing.sum()} of these are' \
    #       ' in the "missing" sample')
    print(np.sort(most_massive_input.colnames))
    print(most_massive_input[np.isin(most_massive_input.obj,
                             most_massive[most_massive_input.label])])
    print(most_massive['name','z','CODEX-DECALS','m500','m500_CODEX'])
    plot_masses(args, chances, most_massive_input)

    return


### Evaluate selection function ###


def plot_masses(args, chances, cat):
    # mass correction factor -- ignoring as it's only important to do it
    # internally consistent
    mf = {'psz2': 10**0.13, 'act-dr5': 1, 'mcxc': 10**0.18, 'codex': 1,
          'spt-sz': 1}
    mf = mf[cat.name]
    zlabel = 'Redshift'
    mlabel = '$M_{500}$ ($10^{14}\,$M$_\odot$)'
    if args.sample == 'evolution':
        zbins = np.arange(0.05, 0.46, 0.02)
        mbins = np.arange(0, 20 if cat.name == 'psz2' else 30, 1)
        xlim = (0, 0.6)
    else:
        zbins = np.arange(0, 0.071, 0.005)
        mbins = np.arange(0, 10 if cat.name == 'psz2' else 22, 0.5)
        xlim = (0, 0.1)
    zx = (zbins[:-1]+zbins[1:]) / 2
    mx = (mbins[:-1]+mbins[1:]) / 2
    catmask = (cat.z > args.zrng[0]) & (cat.z < args.zrng[1]) \
            & (cat.dec > -80*u.deg) & (cat.dec < 5*u.deg) \
            & (np.abs(cat.b) > 20*u.deg)
    # use a mass definition that's consistent with the reference catalog
    catlabel = cat.label.split('-')[0]
    mass = chances[f'm500_{catlabel}']
    chances.sort(f'm500_{catlabel}')
    cat.catalog.sort(cat.masscol)
    catmatch = np.isin(cat.obj, chances[cat.label])
    print(np.sort(chances[cat.label].value))
    if cat.label == 'PSZ2':
        print(cat['name','z','ra','dec','GLON','GLAT','MSZ'][~catmatch][-5:])

    fig, axes = plt.subplots(
        2, 3, figsize=(22, 15), constrained_layout=True)
    axes = np.reshape(axes, -1)
    ax = axes[0]
    ax.scatter(cat.z[catmask], cat.mass[catmask], marker='.', c='C0',
               zorder=1, label=cat.label)
    ax.scatter(chances['z'], mass, marker='o', facecolor='none', s=100, c='C3',
               lw=2, zorder=-10, label=f'CHANCES x {catlabel}')
    ax.set(xlim=xlim, xlabel=zlabel, ylabel=f'{catlabel} {mlabel}')
    ax.legend()
    ax.set(ylim=(0,ax.get_ylim()[1]))
    # 2d contours
    ax = axes[1]
    #extent = (zbins[0])
    h2d = np.histogram2d(chances['z'], mass, (zbins,mbins))[0]
    ax.contour(zx, mx, gaussian_filter(h2d.T, 1), colors='C3')
    hact2d = np.histogram2d(cat.z[catmask], cat.mass[catmask], (zbins,mbins))[0]
    ax.contour(zx, mx, gaussian_filter(hact2d.T, 1) , colors='C0')
    ax.set(xlabel=zlabel, ylabel=f'{catlabel} {mlabel}')
    # mass histograms
    ax = axes[2]
    h = ax.hist(mass, mbins, color='0.4', histtype='stepfilled', lw=2,
                log=True, label=f'CHANCES x {catlabel}')[0]
    # h = ax.hist(mass[cmmask], mbins, color='0.2', histtype='stepfilled',
    #             zorder=-10, lw=2, log=True, label=f'CHANCES x {catlabel}')[0]
    h_ref = ax.hist(cat.mass[catmask], mbins, histtype='step', lw=4, color='C0',
                    log=True, zorder=100, label=cat.label)[0]
    ax.set(xlabel=f'{catlabel} {mlabel}', ylabel='$N(M_{500})$')
    ax.legend()
    ax = axes[3]
    n = chances['z'].size
    ic(h, h.shape)
    if cat.name == 'psz2':
        m0 = np.linspace(6, 8, 21) if args.sample == 'evolution' \
            else np.linspace(0.2, 3, 21)
    elif cat.name == 'codex':
        m0 = np.linspace(8, 12, 21) if args.sample == 'evolution' \
            else np.linspace(2, 5, 21)
    msigma = np.linspace(0.1, 2, 19)
    # in log space
    # m0 = np.linspace(0.7, 0.9, 21) if args.sample == 'evolution' \
    #     else np.linspace(-1, 0.2, 21)
    # msigma = np.linspace(0.03, 0.3, 19)
    a_m = np.linspace(-1, 0, 2)
    a_m = np.array([0])
    #m0 = np.array([8])
    #msigma = np.array([5])
    ic(m0, msigma)
    mbins_ks = np.linspace(0, mbins.max(), 100)
    #mbins_ks = np.logspace(-1, 1.3, 21)
    #h_ks = np.histogram(mass[cmmask], mbins_ks)[0]
    h_ks = np.histogram(mass, mbins_ks)[0]
    cdf = np.cumsum(h_ks)/h_ks.sum()
    ks, cdf_all = ks_mcut(
        args, axes[3], axes[5], cdf, mbins_ks, cat.mass[catmask], m0,
        msigma, a_m)
    print(ks.shape, cdf_all.shape)
    ax.set(xlabel=mlabel, ylabel='$p(<M_{500})$')
    ax.legend()
    ax = axes[4]
    extent = (m0[0],m0[-1],msigma[0],msigma[-1])
    im = ax.imshow(
        ks[0], origin='lower', aspect='auto', extent=extent)
    plt.colorbar(im, ax=ax, label='KS statistic')
    # using a fixed a_m=0 for now so we just need ks[0]
    ks_smooth = gaussian_filter(ks[0], 1)
    # find minimum - first interpolate
    # ic(m0.shape, msigma.shape, ks_smooth.shape)
    # f_ks_smooth = interp2d(m0, msigma, ks_smooth, kind='cubic')
    # ksmin = minimize(f_ks_smooth, x0=(7,0.8))
    # print(ksmin)
    #ax.scatter(*ksmin.x, marker='x', c='w', s=80, lw=2)
    ax.contour(ks_smooth, extent=extent, levels=(0.12,0.2,0.4), colors='w')
    ax.set(xlabel='$m_0$ ($10^{14}$M$_\odot$)',
           ylabel='$\sigma_m$ ($10^{14}$M$_\odot$)')
    #axes[5].axis('off')
    ax = axes[5]
    ax.set(xlabel='KS statistic', ylabel='N')
    output = f'plots/mass_z_{args.sample}_{cat.name}.png'
    savefig(output, fig=fig, tight=False)
    return


def ks_mcut(args, ax_ks, ax_pte, cdf, mbins, mass, m0, msigma, a_m, nsamples=1000):
    mcut = mass_selection(
        mass, m0[:,None], msigma[:,None,None], a_m[:,None,None,None])
    ic(mcut.shape)
    ti = time()
    h = np.array(
        [[[np.histogram(mass[mcut_ijk], mbins)[0]
            for mcut_ijk in mcut_ij] for mcut_ij in mcut_i] for mcut_i in mcut])
    cdf_params = np.cumsum(h, axis=-1)/(np.sum(h, axis=-1)[...,None])
    ks = np.max(np.abs(cdf_params - cdf), axis=-1)
    ic(cdf.shape, h.shape, ks.shape)
    mx = (mbins[1:]+mbins[:-1]) / 2
    ic(mbins, mx)
    [ax_ks.plot(mx, cdf_params[k,j,i])#,
                # label=f'({m0[i]:.1f},{msigma[j]:.1f},{a_m[k]:.1f})')
     for k in range(a_m.size) for j in range(msigma.size)
     for i in range(m0.size)]
    ax_ks.plot(mx, cdf, '-', color='k', lw=8, zorder=100)#, label='CHANCES subsample')
    ic(ks, ks.shape)
    print(f'KS stats in {time()-ti:.1f} s')
    # for every pair of m0,msigma, we create a detection probability as
    # the product of the mass histogram and the selection function,
    # draw randomly from that detection probability and get the KS
    # distribution
    # this is the true histogram
    nm_tot = np.histogram(mass, mbins)[0]
    nm_chances = np.sum(h, axis=-1)
    ic(h.shape, nm_chances.shape)
    # let's do it first for a hand-picked m0,msigma
    ic(m0.shape, msigma.shape, cdf.shape, cdf_params.shape, ks.shape)
    i_m0 = 9
    i_msigma = 7
    m0_i = m0[i_m0]
    msigma_i = msigma[i_msigma]
    pdetect = selection_function(mx, m0_i, msigma_i)
    pm = nm_tot * pdetect / (nm_tot*pdetect).sum()
    m_detected = np.random.choice(
            mx, size=(10000,nm_chances[0,i_msigma,i_m0]), p=pm)
    ic(m_detected.shape)
    nm_detected = np.array(
        [np.histogram(m_detected_i, mbins)[0] for m_detected_i in m_detected])
    ic(nm_detected.shape)
    cdf_i = np.cumsum(nm_detected, axis=1) / np.sum(nm_detected, axis=1)[:,None]
    ic(cdf_i.shape)
    ks_i = np.max(np.abs(cdf_i - cdf_params[0,i_msigma,i_m0]), axis=1)
    ic(ks_i)
    ic(ks_i.shape)
    ax_pte.hist(ks_i, 'doane', histtype='step')
    ax_pte.axvline(ks[0,i_msigma,i_m0], ls='--')
    pte = (ks_i > ks[0,i_msigma,i_m0]).sum() / ks_i.size
    ax_pte.annotate(
        f'PTE={pte:.2f}', xy=(0.95,0.95), xycoords='axes fraction',
        ha='right', va='top', fontsize=15)
    return ks, cdf_params

    m0_grid, msigma_grid, am_grid = np.meshgrid(m0, msigma, a_m)
    ic(msigma_grid.shape, m0_grid.shape, am_grid.shape, h.shape, ks.shape)
    ti = time()
    if args.ncores > 1:
        # to use this I need to return the indices that went into ks_mcut_pte too
        with Pool(args.ncores) as pool:
            results = [[[pool.apply_async(
                    ks_mcut_pte, args=(mass,m0_ijk,ms_ijk,am_ijk,mbins,h_ijk,ks_ijk),
                    kwds={'nsamples': nsamples})
                        for m0_ijk, ms_ijk, am_ijk, h_ijk, ks_ijk
                            in zip(m0_ij, ms_ij, am_ij, h_ij, ks_ij)]
                        for m0_ij, ms_ij, am_ij, h_ij, ks_ij
                            in zip(m0_i, ms_i, am_i, h, ks)]
                        for m0_i, ms_i, am_i, h_i, ks_i
                            in zip(m0_grid, msigma_grid, am_grid, h, ks)]
            pool.close()
            pool.join()
        results = np.reshape(results, -1)
    else:
        results \
            = [[[ks_mcut_pte(
                    mass, m0_ijk, ms_ijk, am_ijk, mbins, h_ijk, ks_ijk,
                    nsamples=nsamples)
                for m0_ijk, ms_ijk, am_ijk, h_ijk, ks_ijk
                    in zip(m0_ij, ms_ij, am_ij, h_ij, ks_ij)]
                for m0_ij, ms_ij, am_ij, h_ij, ks_ij
                    in zip(m0_i, ms_i, am_i, h_i, ks_i)]
                for m0_i, ms_i, am_i, h_i, ks_i
                    in zip(m0_grid, msigma_grid, am_grid, h, ks)]
        results = np.reshape(results, (-1,2))
    ic(results.shape)
    ks_mc = np.zeros((nsamples,ks.size))
    pte = np.zeros(results.size)
    ic(ks_mc.shape, pte.shape)
    for i, out in enumerate(results):
        if args.ncores > 1:
            out = out.get()
        ks_mc[:,i], pte[i] = out
    ic(pte)
    print(f'PTEs in {time()-ti:.1f} s')
    ic(pte.shape)
    return ks, pte


def ks_mcut_pte(mass, m0, msigma, a_m, mbins, h0, ks, mass_err=0.2,
                nsamples=100000):
    ones = np.ones(nsamples)[:,None]
    mcut_mc = mass_selection(mass, ones*m0, ones*msigma, ones*a_m)
    # add error
    # if mass_err > 0:
    #     mm = np.random.normal(mcut_mc, mass_err*mcut_mc)
    #     ic(mm, mm.shape)
    #     mcut_mc = mm
    # if (4 < m0 < 4.5) and (0.55 < msigma < 0.65):
    #     ic(mcut_mc[:2])
    #     ic(mcut_mc.shape, mcut_mc.sum())
    ic(m0, msigma, a_m)
    ic(mcut_mc.shape)
    h_mc = np.array([np.histogram(mass[mc], mbins)[0] for mc in mcut_mc])
    ic(h_mc.shape, h0.shape, ks.shape)
    # ic((np.cumsum(h_mc, axis=1)/(np.sum(h_mc, axis=1)[:,None])).shape)
    # ic((np.cumsum(h0, axis=1)/(np.sum(h0, axis=1)[:,None])).shape)
    ks_mc = np.max(
        np.abs(np.cumsum(h_mc, axis=1)/(np.sum(h_mc, axis=1)[:,None]) \
            - np.cumsum(h0)/h0.sum()),
        axis=1)
    ic(ks_mc, ks_mc.shape)
    ic(ks)
    ic()
    # if (4 < m0 < 4.5) and (0.55 < msigma < 0.65):
    #     ic(h_mc.shape, ks_mc.shape, ks, ks_mc, (ks_mc > ks).sum() / nsamples)
    return ks_mc, np.array((ks_mc > ks).sum() / nsamples, dtype=float)


def mass_selection(m, m0, sigma, a_m=-1):
    prob = selection_function(m, m0, sigma, a_m)
    ic(np.percentile(prob, [0,1,25,50,75,99,100]))
    det = np.random.random(size=prob.shape)
    ic(prob[:100])
    ic(det[:100])
    ic((prob > det).sum(), prob.size)
    return (prob > det)


def selection_function(m, m0, sigma, a_m=0):
    # this is normalized like shit
    s = 0.5 * (1 + m**a_m * erf((m - m0) / (sigma * 2**0.5)))
    return s


### External catalogs ###


def load_ancillary(args, catalog, catalog_name):
    # these are the ones from which I might get a mass
    catalog, psz = load_catalog(args, catalog, 'psz2')
    catalog, act = load_catalog(args, catalog, 'act-dr5')
    catalog, spt = load_catalog(args, catalog, 'spt-sz')
    catalog, mcxc = load_catalog(args, catalog, 'mcxc')
    catalog, redmapper = load_catalog(args, catalog, 'redmapper')
    if 'DECam' not in catalog.colnames:
        catalog = load_decam(args, catalog)
    # add indices in SZ+X-ray catalogs
    for szcat in (psz, act, spt, mcxc):
        idxcol = f'{szcat.label}_idx'
        rng = np.arange(szcat.obj.size, dtype=int)
        catalog[idxcol] \
            = [rng[szcat.obj == obj][0] if obj != '' else -99
               for obj in catalog[szcat.label]]
    #if 'SPLUS' not in catalog.colnames:
    catalog = load_splus(args, catalog)
    catalog, codex = load_codex3(args, catalog)
    ic(codex)
    # other catalogs
    catalog, mk = load_meerkat(args, catalog)
    catalog, mkd = load_meerkat_diffuse(args, catalog)
    #catalog, rass = load_rass(args, catalog)
    catalog, xmm = load_xmm(args, catalog)
    catalog, hiflugcs = load_hiflugcs(args, catalog)
    catalog, meneacs = load_meneacs(args, catalog)
    catalog, lovoccs = load_lovoccs(args, catalog)
    catalog, clashvlt = load_clashvlt(args, catalog)
    ic(catalog)

    # add masses
    catalog = add_masses(args, catalog, psz, 'MSZ', 10**0.13)
    catalog = add_masses(args, catalog, act, 'M500cCal', None)
    catalog = add_masses(args, catalog, spt, 'M500c', None)
    catalog = add_masses(args, catalog, mcxc, 'M500', 10**0.18)
    catalog = add_masses(args, catalog, codex, 'M500', None)
    #catalog = add_masses(args, catalog, codex, 'LX0124', None)
    if 'lowz' in args.sample:
        catalog['m500'] = catalog['m500_CODEX']
    else:
        catalog['m500'] = catalog['m500_ACT']
    catalog['m500'][catalog['m500'] == -1] \
        = catalog['m500_SPT'][catalog['m500'] == -1]
    catalog['m500'][catalog['m500'] == -1] \
        = catalog['m500_PSZ2_corr'][catalog['m500'] == -1]
    if 'evolution' in args.sample:
        catalog['m500'][catalog['m500'] == -1] \
            = catalog['m500_CODEX'][catalog['m500'] == -1]
    catalog['m500'][catalog['m500'] == -1] \
        = catalog['m500_MCXC_corr'][catalog['m500'] == -1]
    catalog = calculate_m200(catalog)
    ic(catalog)

    #query_muse(args, catalog)
    #query_ned(args, catalog)

    # match to galaxy catalogs
    catalog, first = load_gal_first(args, catalog)
    catalog, tgss = load_gal_tgss(args, catalog)
    catalog, spec = load_gal_spec(args, catalog)

    # the extent we want, in degrees
    catalog['5d200(deg)'] = (5/60) * catalog['d200']
    catalog['5d200(deg)'][catalog['d200'] == -1] = -1
    catalog.sort('name')

    today = date.today().strftime('%Y-%m-%d')
    catalog.write(f'catalogues/clusters_{catalog_name}_{args.sample}_{today}.txt',
              format='ascii.fixed_width', overwrite=True)
    catalog.write(f'catalogues/clusters_{catalog_name}_{args.sample}_{today}.csv',
              format='ascii.csv', overwrite=True)
    
    summarize_ancillary(args, catalog)

    summarize_ancillary(args, catalog)
    others = (psz, act, spt, codex, mcxc)
    return catalog, others


def get_most_massive(args, cat, chances, n=200):
    gal = cat.coords.transform_to('galactic')
    mask = (cat.z > args.zrng[0]) & (cat.z < args.zrng[1]) \
        & (cat.dec > -80*u.deg) & (cat.dec < 5*u.deg) \
        & (np.abs(gal.b) > 20*u.deg)
    print(np.sort(cat.colnames))
    jsort = np.argsort(cat.mass[mask])[-n:]
    most_massive = Table(
        {'name': cat.obj[mask][jsort],
         'ra': cat.ra[mask][jsort],
         'dec': cat.dec[mask][jsort],
         'z': cat.z[mask][jsort],
         cat.masscol: cat.mass[mask][jsort]})
    most_massive[cat.masscol].format = '.2f'
    most_massive.sort('name')
    most_massive = Catalog(
        'Most Massive', most_massive,
        base_cols=('name','ra','dec','z'),
        masscol=cat.masscol)
    for col in ('ra', 'dec', 'z'):
        most_massive.catalog[col].format = '%.3f'
    # compliance...
    most_massive.catalog['coords'] = most_massive.coords
    # match names with chances
    dist_chances = most_massive.coords.separation(
        chances['coords'][:,None])
    closest = np.argmin(dist_chances, axis=0)
    matches = np.min(dist_chances, axis=0) < 5*u.arcmin
    most_massive.catalog['name'][matches] \
        = chances['name'][closest[matches]]
    tbl = most_massive.catalog
    # additional attributes so that we can also use it
    # as a Catalog-like object
    tbl.masscol = most_massive.masscol
    tbl.name = most_massive.name
    tbl.ra = most_massive.ra
    tbl.dec = most_massive.dec
    tbl.z = most_massive.z
    tbl.mass = most_massive.mass
    tbl.size = most_massive.size
    return tbl


def review_missing(args, chances, psz, act, spt, codex, mcxc):
    # are we missing massive clusters?
    # these are the minimal constraints: redshift, declination and DECam
    # availability
    masks = [(cat.z > args.zrng[0]) & (cat.z < args.zrng[1]) \
                & (cat.dec > -80) & (cat.dec < 5) & (np.abs(cat.b) > 20*u.deg)
             for cat in (psz, act, spt, codex, mcxc)]
    psz_z, act_z, spt_z, codex_z, mcxc_z = masks

    massive = {
        'psz2': np.argsort(psz['MSZ'][psz_z].value)[-args.nmassive:], 
        'act-dr5': np.argsort(act['M500cCal'][act_z].value)[-args.nmassive:],
        'spt-sz': np.argsort(spt['M500c'][spt_z].value)[-args.nmassive:],
        'mcxc': np.argsort(mcxc['M500'][mcxc_z].value)[-args.nmassive:]}
    # merge the three samples -- giving them the same names as CHANCES
    # so I can use them easily
    szmassive = Table(
        {'name': np.hstack(
            [psz['NAME'][psz_z][massive['psz2']],
             act['name'][act_z][massive['act-dr5']],
             spt['SPT'][spt_z][massive['spt-sz']],
             mcxc['MCXC'][mcxc_z][massive['mcxc']]]),
         'ra': np.hstack(
            [psz['RA'][psz_z][massive['psz2']],
             act['RADeg'][act_z][massive['act-dr5']],
             spt['RAdeg'][spt_z][massive['spt-sz']],
             mcxc['RAdeg'][mcxc_z][massive['mcxc']]]),
         'dec': np.hstack(
            [psz['DEC'][psz_z][massive['psz2']],
             act['decDeg'][act_z][massive['act-dr5']],
             spt['DEdeg'][spt_z][massive['spt-sz']],
             mcxc['DEdeg'][mcxc_z][massive['mcxc']]]),
         'z': np.hstack(
            [psz['REDSHIFT'][psz_z][massive['psz2']],
             act['redshift'][act_z][massive['act-dr5']],
             spt['z'][spt_z][massive['spt-sz']],
             mcxc['z'][mcxc_z][massive['mcxc']]])
        })
    # Catalog object
    szmassive.sort('name')
    szmassive = Catalog(
        'Missing Massive', szmassive,
        base_cols=('name','ra','dec','z'))
    for col in ('ra', 'dec', 'z'):
        szmassive.catalog[col].format = '%.3f'
    # also for consistency with CHANCES
    szmassive.catalog['coords'] = szmassive.coords

    for szcat in (psz, act, spt, mcxc):
        sep = szmassive.coords.separation(szcat.coords[:,None])
        closest = np.argmin(sep, axis=0)
        matches = (np.min(sep, axis=0) < 5*u.arcmin)
        szmassive.catalog[szcat.label] = szcat.obj[closest]
        szmassive.catalog[szcat.label][~matches] = ''
    ic(szmassive[:20])
    ic(szmassive[20:40])
    ic(szmassive[40:])
    # add indices
    # for szcat in (psz, act, spt, mcxc):
    #     idxcol = f'{szcat.label}_idx'
    #     rng = np.arange(szcat.obj.size, dtype=int)
    #     szmassive.catalog[idxcol] \
    #         = [rng[szcat.obj == obj][0] if obj != '' else -99
    #            for obj in szmassive[szcat.label]]
    unique = []
    for cl in szmassive.catalog:
        if np.isin(cl['PSZ2','ACT-DR5','SPT-SZ','MCXC'].values, unique).sum() > 0:
            continue
        if cl['PSZ2'] != '':
            unique.append(cl['PSZ2'])
        elif cl['ACT-DR5'] != '':
            unique.append(cl['ACT-DR5'])
        elif cl['SPT-SZ'] != '':
            unique.append(cl['SPT-SZ'])
        elif cl['MCXC'] != '':
            unique.append(f"MCXC {cl['MCXC']}")
    szmassive.catalog['name'] = unique
    unique, unique_idx = np.unique(unique, return_index=True)
    szmassive.catalog = szmassive.catalog[unique_idx]
    print(f'There are {unique.size} unique "massive" clusters')
    # which are not in CHANCES?
    massive_in_chances = np.isin(
        szmassive['name'],
        np.reshape(
            [chances[col] for col in ['PSZ2','ACT-DR5','SPT-SZ','MCXC']], -1))
    ic(massive_in_chances, massive_in_chances.shape)
    missing = ~massive_in_chances
    ic(szmassive[missing]['name','z','PSZ2','ACT-DR5','MCXC'])
    return szmassive[missing]


def chances_catalog(args):
    use_final = True
    if args.sample == 'lowz':
        file = 'catalogues-ciria/S1501_clusters_final.csv'
        file_old = 'CHANCES low-z clusters.csv'
    else:
        file = 'catalogues-ciria/S1502_clusters_final.csv'
        file_old = 'CHANCES Evolution clusters.csv'
    cat = ascii.read(file, format='csv')
    cat_old = ascii.read(file_old, format='csv')
    cat.rename_column('m200', 'm200_listed')
    # happens in low-z
    if 'col12' in cat.colnames:
        ic(cat['col12'].value)
        cat.remove_column('col12')
    cols = ['Cluster', 'RA', 'Dec', 'z']
    cols_old = ['Cluster Name', 'RA_J2000', 'Dec_J2000', 'Z']
    cat.rename_columns(cols, ['name', 'ra', 'dec', 'z'])
    cat_old.rename_columns(cols_old, ['name', 'ra', 'dec', 'z'])
    # in the new files z means number of members, not redshift!
    cat['z'] = [cat_old['z'][cat_old['name'] == name][0]
                if name in cat_old['name'] else 1
                for name in cat['name']]
    z_manual = {'A0194': 0.018, 'A0548': 0.042, 'A1631': 0.046,
                'A2415': 0.058, 'A2457': 0.059, 'A2734': 0.062,
                'A2870': 0.024, 'A3341': 0.038, 'A3389': 0.027,
                'A3574': 0.016, 'AM2002': 0.023, 'AS560': 0.037,
                'IIZw108': 0.049, 'MZ00407': 0.022}
    for cl, z in z_manual.items():
        cat['z'][cat['name'] == cl] = z
    mass_manual = {'AM2002': 0.7, 'MZ00407': 0.3}
    for cl, m in mass_manual.items():
        cat['m200_listed'][cat['name'] == cl] = m
    cat['coords'] = SkyCoord(ra=cat['ra'], dec=cat['dec'], unit='deg')
    cat.sort('name')
    return cat


def calculate_m200(chances, cosmo=Planck18):
    m200, c200, r200, d200 = np.zeros((4,chances['m500'].size))
    miss = (chances['m500'] == 0)
    ic(chances['m500','z'][~miss])
    m200[~miss], c200[~miss], r200[~miss], d200[~miss] \
        = calculate_m200_from_m500(
            1e14*chances['m500'][~miss], chances['z'][~miss], cosmo=cosmo)
    if miss.sum() > 0:
        m200[miss], c200[miss], r200[miss], d200[miss] \
            = calculate_m200_from_m500(
                1e14, chances['z'][miss], cosmo=cosmo)
    with_listed_m200 = (chances['m200_listed'] > 0) & (chances['m500'] <= 0)
    ic(with_listed_m200.sum())
    nfw_with_listed_m200 = NFW(
        1e14*chances['m200_listed'][with_listed_m200], 5,
        chances['z'][with_listed_m200],
        overdensity=200, frame='physical', cosmo=cosmo)
    # these have a mass in Ciria's file but not in the catalogues we
    # review here
    m200[with_listed_m200] = 1e14 * chances['m200_listed'][with_listed_m200]
    r200[with_listed_m200], d200[with_listed_m200] \
        = radius_from_m200(nfw_with_listed_m200, cosmo=cosmo)
    ic(r200)
    chances['m200'] = m200 / 1e14
    chances['c200'] = c200
    chances['r200'] = r200
    chances['d200'] = d200
    no_mass = (chances['m500'] == -1) & (chances['m200_listed'] == 0)
    for col in ('m200', 'c200', 'r200', 'd200'):
        chances[col][no_mass] = -1
        chances[col].format = '%.2f'
        chances[col][np.isnan(chances[col])] = -1
    return chances


def calculate_m200_from_m500(m500, z, cosmo=Planck18, model='ishiyama21'):
    cosmology.fromAstropy(cosmo, sigma8=0.81, ns=0.966)
    c = concentration.concentration(m500, '500c', z, model=model)
    nfw = NFW(m500, c, z, overdensity=500, frame='physical', cosmo=cosmo)
    m200, c200 = nfw.mdelta(200)
    r200, d200 = radius_from_m200(nfw, cosmo)
    return m200, c200, r200, d200


def radius_from_m200(nfw, cosmo=Planck18):
    r200 = nfw.radius
    kpc2arcmin = cosmo.arcsec_per_kpc_proper(nfw.z)
    d200 = (r200*u.Mpc * kpc2arcmin).to('arcmin').value
    return r200, d200


### Matching routines ###


def match_galaxy_catalog(args, chances, galcat, radius=5, unit='r200',
                         dz=0.03):
    """Match cluster catalog to external galaxy catalog

    ``unit`` must be either 'arcmin' or 'r200'"""
    print(f'Matching galaxy catalog {galcat.name}')
    assert unit in ('arcmin', 'r200')
    # if unit == 'r200':
    #     missing = (chances['d200'] == -1)
    #     if missing.sum() > 0:
    #         massdata = calculate_m200_from_m500(
    #             m500_if_missing, chances['z'][missing])
    #         for col, name in zip(massdata, 'mcrd'):
    #             chances[f'{name}200'][missing] = col
    #         chances['m200'][missing] /= 1e14
    maxdist = (radius * chances['d200'] if unit == 'r200' else radius) * u.arcmin
    Ngal, Ngal_z = np.zeros((2,maxdist.size), dtype=int)
    path = os.path.join('aux', 'spectroscopic', args.sample.split('-')[0])
    os.makedirs(path, exist_ok=True)
    # this should never happen
    if 'Cluster Name' in chances.colnames:
        namecol, racol, deccol = ['Cluster Name', 'RA_J2000', 'Dec_J2000']
    else:
        namecol, racol, deccol = ['name', 'ra', 'dec']
    for i, (cl, dmax) in tqdm(enumerate(zip(chances, maxdist))):
        #ic(i, cl['Cluster Name','RA_J2000','Dec_J2000','z','m200','d200'])
        cosdec = np.cos(np.radians(cl[deccol]))
        nearby = (np.abs(cl[racol]*u.deg-galcat.ra) < 2*dmax/cosdec) \
            & (np.abs(cl[deccol]*u.deg-galcat.dec) < 2*dmax)
        if nearby.sum() == 0:
            continue
        sep = cl['coords'].separation(galcat.coords[nearby])
        ic(dmax, nearby.sum(), nearby.sum()/galcat.z.size)
        matches = (sep < dmax)
        Ngal[i] = matches.sum()
        if Ngal[i] == 0:
            continue
        clname = cl['Cluster Name'].replace(' ', '_')
        galcat.catalog[nearby][galcat.base_cols][matches].write(
            os.path.join(path, f'{clname}.txt'),
            format='ascii.fixed_width', overwrite=True)
        if dz is None:
            ic(Ngal[i])
            continue
        zmatches = matches & (np.abs(galcat.z[nearby]-cl['z'])/(1+cl['z']) < dz)
        Ngal_z[i] = zmatches.sum()
        ic(Ngal[i], Ngal_z[i])
    try:
        chances[f'N_{galcat.name.lower()}'] = Ngal
        if dz is not None:
            chances[f'N_{galcat.name.lower()}_z'] = Ngal_z
    except TypeError:
        chances.catalog[f'N_{galcat.name.lower()}'] = Ngal
        if dz is not None:
            chances.catalog[f'N_{galcat.name.lower()}_z'] = Ngal_z
    #ic(chances['Cluster Name','z','m200','r200','d200','Nspec','Nspec_z'])
    return chances, galcat


def match_catalog(chances, cat, radius=5*u.arcmin, name=None):
    try:
        dist = chances['coords'].separation(cat.coords[:,None])
    except AttributeError:
        coords = SkyCoord(ra=cat['ra'], dec=cat['dec'], unit='deg')
        dist = chances['coords'].separation(coords[:,None])
    if dist.size == 0:
        return chances, cat
    ic(np.sort(dist.to(u.arcmin)))
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
    try:
        chances[f'{name}_idx'] = idx
        chances[name] \
            = [cat.obj[i] if i > -99 else '' for i in idx]
    except TypeError:
        chances.catalog[f'{name}_idx'] = idx
        chances.catalog[name] \
            = [cat.obj[i] if i > -99 else '' for i in idx]
    return chances, cat



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


def add_masses(args, chances, cat, masscol, factor):
    suff = cat.label.split('-')[0]
    col = 'name' if 'name' in chances.colnames else 'Cluster Name'
    chances[f'm500_{suff}'] = -np.ones(chances[col].size)
    mask = chances[f'{cat.label}_idx'] > -99
    chances[f'm500_{suff}'][mask] \
        = cat[masscol][chances[f'{cat.label}_idx'][mask]]
    chances[f'm500_{suff}'].format = '%.2f'
    if factor is not None:
        chances[f'm500_{suff}_corr'] = -np.ones(chances[col].size)
        chances[f'm500_{suff}_corr'][mask] \
            = factor * chances[f'm500_{suff}'][mask]
        chances[f'm500_{suff}_corr'].format = '%.2f'
    return chances


def add_masses_manual(chances):
    chances['m500_other'] = -np.ones(chances['Cluster Name'].size)
    chances['m500_other_source'] \
        = [20*' ' for i in range(chances['Cluster Name'].size)]
    chances['m500_manual'].format = '%.2f'
    #clusters = {'A754': }
    return


def load_catalog(args, chances, name):
    """This is for the ones I have in ``astro``"""
    ic(name)
    cat = Catalog(name)
    goodz = (cat.z >= args.zrng[0]) & (cat.z < args.zrng[1])
    gooddec = (cat.dec >= -80*u.deg) & (cat.dec <= 5*u.deg)
    cat = Catalog(name, catalog=cat[goodz & gooddec], base_cols=('name','ra','dec','z'))
    # Simet et al. relation
    if name == 'redmapper':
        cat.catalog['M200m'] = 10**14.34 * (cat['LAMBDA']/40)**1.33
    chances, cat = match_catalog(chances, cat)
    return chances, cat


def load_decam(args, catalog):
    """Add whether each cluster is in DECaLS after looking manually
    at the sky viewer"""
    # this should be a pretty good approximation
    gal = catalog['coords'].transform_to('galactic')
    decam = ['NO' if abs(b) <= 20 else 'DR10' for b in gal.b.deg]
    try:
        catalog['DECam'] = decam
    except TypeError:
        catalog.catalog['DECam'] = decam
    return catalog


### Galaxy catalogs ###


def load_gal_first(args, chances):
    filename = 'aux/radio/first_14dec17.fits.gz'
    first = Catalog(
        'FIRST', fits.getdata(filename), base_cols=('INDEX', 'RA', 'DEC', 'FINT'))
    chances, first = match_galaxy_catalog(args, chances, first, dz=None)
    return chances, first


def load_gal_spec(args, chances, base_cols=('index','RA','DEC','z')):
    filename = 'aux/spectroscopic/SpecZ_Catalogue_20221105.csv'
    spec = Catalog(
        'spec', ascii.read(filename, format='csv'), base_cols=base_cols)
    chances, spec = match_galaxy_catalog(args, chances, spec)
    return chances, spec


def load_gal_tgss(args, chances):
    filename = 'aux/radio/TGSSADR1_7sigma_catalog.fits'
    tgss = Catalog('TGSS', fits.getdata(filename),
                   base_cols=('Source_name','RA','DEC','Total_flux'))
    chances, tgss = match_galaxy_catalog(args, chances, tgss, dz=None)
    return chances, tgss


### Cluster catalogs ###


def load_clashvlt(args, chances):
    filename = 'aux/spectroscopic/clashvlt_sample.txt'
    clashvlt = Catalog(
        'CLASH-VLT', ascii.read(filename, format='fixed_width'),
        base_cols=('Cluster', 'RA', 'DEC', 'z'),
        coord_unit=(u.hourangle, u.deg))
    chances, clashvlt = match_catalog(chances, clashvlt)
    return chances, clashvlt


def load_codex(args, chances):
    filename = 'aux/xray/codex10_eromapper_Xmass.fits'
    codex = Table(fits.open(filename)[1].data)
    #codex = codex[codex['codex50']]
    codex['M500'] = codex['M200c'] * codex['M500/M200']
    cols = ['id_cluster', 'RA_X-ray', 'Dec_X-ray', 'z_lambda', 'lambda',
            'Lx0124', 'M500', 'M200c']
    codex['id_cluster'] = np.array(codex['id_cluster'], dtype=str)
    codex['M500'] /= 1e14
    codex['M200c'] /= 1e14
    codex = Catalog(
        'codex', codex, cols=cols, base_cols=cols[:4],
        label='CODEX-DECALS', masscol='M500')
    chances, codex = match_catalog(chances, codex)
    return chances, codex


def load_codex3(args, chances):
    filename = 'aux/xray/codex3_lsdr10_info.fits'
    codex = Table(fits.open(filename)[1].data)
    codex['M200c'] /= 1e14
    codex['M500'] = codex['M200c'] * codex['M500/M200']
    cols = ['CODEX3', 'RA_X', 'Dec_X', 'z_best', 'lambda',
            'Lx0124', 'M500', 'M200c']
    codex = Catalog(
        'codex', codex, cols=cols, base_cols=cols[:4],
        label='CODEX3-LSDR10', masscol='M500')
    chances, codex = match_catalog(chances, codex)
    return chances, codex

def load_hiflugcs(args, chances):
    filename = 'aux/xray/hiflugcs_sample.txt'
    hiflugcs = ascii.read(filename, format='cds')
    hiflugcs = Catalog(
        'HIFLUGCS', hiflugcs[hiflugcs['Sample'] == 'Included'],
        base_cols=('CName', 'RAdeg', 'DEdeg', 'z'))
    chances, hiflugcs = match_catalog(chances, hiflugcs)
    return chances, hiflugcs


def load_lovoccs(args, chances):
    filename = 'aux/spectroscopic/lovoccs_sample.txt'
    lovoccs = Catalog(
        'LoVoCCS', ascii.read(filename, format='basic'),
        base_cols=('Name','ra','dec','z'))
    chances, lovoccs = match_catalog(chances, lovoccs)
    return chances, lovoccs


def load_meneacs(args, chances):
    filename = 'aux/optical/meneacs_cccp.txt'
    meneacs = Catalog('MENeaCS', ascii.read(filename, format='basic'),
                      base_cols=('Cluster','RA','Dec','z'),
                      coord_unit=(u.hourangle, u.deg))
    chances, meneacs = match_catalog(chances, meneacs)
    return chances, meneacs


def load_meerkat(args, chances):
    mk = Catalog(
        'MeerKAT', ascii.read('aux/meerkat/meerkat_legacy.csv', format='csv'))
    #ic(mk)
    chances, mk = match_catalog(chances, mk, name='MeerKAT')
    return chances, mk


def load_meerkat_diffuse(args, chances):
    mkd = fits.open('aux/meerkat/Table4_MGCLS_diffuse.fits')[1].data
    mkd = Catalog(
        'MKDiffuse', mkd,
        base_cols=('ClusterName','R.A.J2000 (deg)','Dec.J2000 (deg)','z'))
    chances, mkd = match_catalog(chances, mkd, name='MKDiffuse')
    return chances, mkd


def load_ned(args, cat, path='aux/ned/spec'):
    os.makedirs(path, exist_ok=True)
    http = urllib3.PoolManager()
    for i, cl in enumerate(cat):
        name = cl['name'].replace(" ", "_")
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
                cl['ra'], cl['dec'], 60*cl['d200']))
        ic(url)
        ic(url.data)
        ic(time()-ti)
        break
    return


def load_rass(args, chances):
    filename = 'aux/xray/local_fof_insideSplus.cat'
    return


def load_sdss(args, cat):
    output = f'aux/sdss/sdss_spec_{args.sample}.tbl'
    if os.path.isfile(output):
        return ascii.read(output, format='fixed_width')
    sdss = SDSS.query_region(cat['coords'], spectro=True, radius=1*u.deg)
    ic(sdss)
    if len(sdss.colnames) == 1 and 'htmlhead' in sdss.colnames:
        return
    sdss.write(output, format='ascii.fixed_width', overwrite=True)
    return sdss


def load_splus(args, cat):
    splus = ascii.read('../splus/S-PLUS_footprint.csv', format='csv')
    splus['coords'] = SkyCoord(
        ra=splus['RA'], dec=splus['DEC'], unit=(u.hourangle, u.deg))
    splus_dist = splus['coords'].separation(cat['coords'][:,None])
    in_splus = np.min(splus_dist, axis=1) < 1*u.deg
    splus = ['YES' if i else 'NO' for i in in_splus]
    try:
        cat['S-PLUS'] = splus
    except TypeError:
        cat.catalog['S-PLUS'] = splus
    return cat


def load_xmm(args, chances):
    filename = 'aux/xray/local_fof_insideSplus_xmm.cat'
    xmm = Catalog(
        'XMM', ascii.read(filename, format='commented_header'),
        base_cols=('OBSERVATION.TARGET','RA_X','DEC_X','z'))
    chances, xmm = match_catalog(chances, xmm)
    return chances, xmm


### Summarize ###


def summarize_ancillary(args, chances):
    print()
    ic(np.sort(chances.colnames))
    for col in ('m500', 'm200', 'd200'):
        chances[col].format = '%.2f'
    cols = ['name', 'z', 'm500', '5d200(deg)',
            'CODEX3-LSDR10', 'PSZ2', 'ACT-DR5', 'SPT-SZ',
            'MeerKAT', 'DECam', 'S-PLUS',
            'N_spec', 'N_spec_z', 'N_tgss', 'N_first']
    if args.sample == 'lowz':
        cols = cols[:4] + ['XMM'] + cols[4:]
        if 'AAOzs' in chances.colnames:
            cols.append('AAOzs')
        if 'Lx' in chances.colnames:
            cols.append('Lx')
    tbl = chances[cols]
    print(tbl)
    # optical photometry -- need to add for missing clusters
    if 'DECam' in chances.colnames:
        decam = (chances['DECam'] != 'NO')
        if 'VST' in chances.colnames:
            vst = (chances['VST'] == 'Y')
            opt = (vst | decam)
            print(f'{(vst & decam).sum()} with both VST and DECam,' \
                f' {(1 - opt).sum()} without either:')
        else:
            opt = decam
            print(f'{decam.sum()} with DECam, {(1-decam).sum()} without:')
        print(tbl[~opt])
        if 'vst' in chances.colnames:
            print(f'  {vst.sum()} with VST')
        print(f'  {decam.sum()} with DECam')
    if 'S-PLUS' in chances.colnames:
        splus_observed = ~np.isin(chances['S-PLUS'], ('', 'NO'))
        if 'SPLUS comments' in chances.colnames:
            splus_upcoming = np.array(
                ['target' in comment for comment in chances['SPLUS comments']])
            splus = splus_observed | splus_upcoming
        else:
            splus = splus_observed
        print(f'{splus.sum()} with SPLUS:')
        print(tbl[splus])
        if 'SPLUS comments' in chances.colnames:
            print(f'  {splus_observed.sum()} already with SPLUS')
            print(f'  {splus_upcoming.sum()} with upcoming SPLUS')
    
    #print()
    # spectroscopy
    if 'AAOzs' in chances.colnames:
        aao = (chances['AAOzs'] > 0)
        print(f'{aao.sum()} with AAOzs')
    meneacs = (chances['MENeaCS'] != '')
    print(f'{meneacs.sum()} in MENeaCS:')
    print(tbl[meneacs])
    lovoccs = (chances['LoVoCCS'] != '')
    print(f'{lovoccs.sum()} in LoVoCCS:')
    print(tbl[lovoccs])
    clashvlt = (chances['CLASH-VLT'] != '')
    print(f'{clashvlt.sum()} in CLASH-VLT:')
    print(tbl[clashvlt])
    # SZ
    psz = (chances['PSZ2'] != '')
    act = (chances['ACT-DR5'] != '')
    spt = (chances['SPT-SZ'] != '')
    sz = psz | act | spt
    print(f'{sz.sum()} with SZ, {(1-sz).sum()} without:')
    print(tbl[~sz])
    print(f'  {psz.sum()} with PSZ2')
    print(f'  {act.sum()} with ACT-DR5')
    print(f'  {spt.sum()} in SPT-SZ')
    print(f'  {(spt & ~act).sum()} in SPT-SZ not in ACT-DR5:')
    print(tbl[spt & ~act])
    # X-rays
    xmm = (chances['XMM'] != '')
    mcxc = (chances['MCXC'] != '')
    hiflugcs = (chances['HIFLUGCS'] != '')
    xrays = xmm | mcxc | hiflugcs
    print(f'{xrays.sum()} with X-rays')
    print(f'  {hiflugcs.sum()} in HIFLUGCS:')
    print(tbl[hiflugcs])
    print(f'  {xmm.sum()} with XMM')
    print(f'  {mcxc.sum()} in MCXC')
    #codex = (chances['CODEX-DECALS'] != '')
    codex = (chances['CODEX3-LSDR10'] != '')
    print(f'{codex.sum()} in CODEX:')
    print(tbl[codex])
    # radio
    mk = (chances['MeerKAT'] != '')
    first = (chances['N_first'] > 0)
    tgss = (chances['N_tgss'] > 0)
    print(f'{(mk | first | tgss).sum()} with radio')
    print(f'{(mk & first).sum()} with MeerKAT and FIRST')
    print(f'{(mk & tgss).sum()} with MeerKAT and TGSS')
    print(f'  {mk.sum()} with MeerKAT:')
    print(tbl[mk])
    print(f'  {first.sum()} in FIRST')
    print(f'  {tgss.sum()} in TGSS')
    print('## Combinations ##')
    print(f'{(decam & splus).sum()} with both DECam and SPLUS:')
    print(tbl[decam & splus])
    print(f'{(sz & xrays).sum()} with SZ and X-rays')
    print(f'{(act & xmm).sum()} with ACT-DR5 and XMM')
    print(f'{(sz & mk).sum()} with SZ and MeerKAT')
    print(f'{(sz & xrays & mk).sum()} with SZ and X-rays and MeerKAT')
    print(f'{(sz & xrays & mk).sum()} with SZ and X-rays and TGSS')
    print(f'{(~sz & ~xrays).sum()} without SZ nor X-rays:')
    print(tbl[(~sz & ~xrays)])
    print(f'{(lovoccs & mk).sum()} in both LoVoCCS and MeerKAT:')
    print(tbl[(lovoccs & mk)])
    print(f'{((lovoccs | meneacs) & mk).sum()} in LoVoCCS or MENeaCS and MeerKAT:')
    print(tbl[(lovoccs | meneacs) & mk])
    print(f'{(lovoccs | meneacs | mk).sum()} in LoVoCCS or MENeaCS or MeerKAT:')
    print(tbl[lovoccs | meneacs | mk])
    if 'S-PLUS' in chances.colnames:
        print(f'{(splus & lovoccs).sum()} in SPLUS and LoVoCCS:')
        print(tbl[splus & lovoccs])
    print(f'{(decam & (splus | lovoccs | mk | clashvlt)).sum()} in any of the above:')
    print(tbl[decam & (splus | lovoccs | mk | clashvlt)])
    print()
    return cols

def parse_args():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('sample',
        choices=('evolution', 'lowz', 'evolution-final', 'lowz-final'),
        default='lowz')
    add('--debug', action='store_true')
    add('-m', '--mass-selection', default='psz2')
    add('--ncores', default=1, type=int)
    add('--nmassive', default=20, type=int)
    args = parser.parse_args()
    if not args.debug:
        ic.disable()
    args.zrng = (0, 0.07) if 'lowz' in args.sample else (0.07, 0.45)
    return args


if __name__ == '__main__':
    main()
