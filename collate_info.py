import argparse
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import ascii, fits
from astropy.table import QTable, Table, join, vstack
from astroquery.ipac.ned import Ned
from astroquery.sdss import SDSS
from colossus.cosmology import cosmology
from colossus.halo import concentration
from datetime import date
from icecream import ic
from matplotlib import pyplot as plt, ticker
from matplotlib.colors import LogNorm, Normalize
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

from astro.clusters import ClusterCatalog
from plottery.plotutils import savefig, update_rcParams

update_rcParams()


def main():
    args = parse_args()
    chances = chances_catalog(args)
    print(chances)
    # sys.exit()

    chances, others = load_ancillary(args, chances, "chances")
    print(chances["name", "ra", "dec"][chances["z"] == 1])
    return
    meneacs, wings, locuss, psz, act, sptecs, sptsz, codex, mcxc = others

    # print('\n\n*** Missing from CHANCES ***\n')
    # missing = review_missing(args, chances, *others)
    # missing = load_decam(args, missing)
    # missing, _ = load_ancillary(args, missing, 'missing')

    print("\n\n*** Fully mass-selected sample ***\n")
    most_massive_input = [i for i in others if i.name == args.mass_selection][0]
    most_massive = get_most_massive(args, most_massive_input, chances)
    for catname in ("psz2", "act-dr5", "spt-ecs", "spt-sz", "codex"):
        most_massive, _ = load_catalog(args, most_massive, catname)
    most_massive = load_decam(args, most_massive)
    most_massive = load_ancillary(args, most_massive, "most_massive")[0]
    print(np.sort(most_massive.colnames))

    # print(np.sort(most_massive['Cluster Name']))
    most_massive_in_chances = np.isin(most_massive["name"], chances["name"])
    print(
        f"{most_massive_in_chances.sum()} of the"
        f" {most_massive.size} most massive"
        f" {most_massive_input.label} clusters in the range"
        f" {args.zrng[0]} <= z <= {args.zrng[1]} are in CHANCES,"
    )
    print(
        f'out of a total of {chances["name"].size} CHANCES'
        f" {args.sample.capitalize()} clusters"
    )
    cols = [
        "name",
        "z",
        "m500",
        most_massive.masscol,
        "PSZ2",
        "ACT-DR5",
        "N_spec",
        "N_spec_z",
    ]
    print(most_massive[cols][most_massive_in_chances])
    # print(np.sort(most_massive.colnames))
    # most_massive_in_missing = np.isin(
    #     most_massive['Cluster Name'], missing['Cluster Name'])
    # print(f'Another {most_massive_in_missing.sum()} of these are' \
    #       ' in the "missing" sample')
    print(np.sort(most_massive_input.colnames))
    print(
        most_massive_input[
            np.isin(most_massive_input.obj, most_massive[most_massive_input.label])
        ]
    )
    print(most_massive["name", "z", "CODEX-DECALS", "m500", "m500_CODEX"])
    plot_masses(args, chances, most_massive_input)

    return


### Evaluate selection function ###


def plot_masses(args, chances, cat):
    # mass correction factor -- ignoring as it's only important to do it
    # internally consistent
    mf = {
        "psz2": 10**0.13,
        "act-dr5": 1,
        "mcxc": 10**0.18,
        "codex": 1,
        "spt-ecs": 1,
        "spt-sz": 1,
    }
    mf = mf[cat.name]
    zlabel = "Redshift"
    mlabel = "$M_{500}$ ($10^{14}\,$M$_\odot$)"
    if args.sample == "evolution":
        zbins = np.arange(0.05, 0.46, 0.02)
        mbins = np.arange(0, 20 if cat.name == "psz2" else 30, 1)
        xlim = (0, 0.6)
    else:
        zbins = np.arange(0, 0.071, 0.005)
        mbins = np.arange(0, 10 if cat.name == "psz2" else 22, 0.5)
        xlim = (0, 0.1)
    zx = (zbins[:-1] + zbins[1:]) / 2
    mx = (mbins[:-1] + mbins[1:]) / 2
    catmask = (
        (cat.z > args.zrng[0])
        & (cat.z < args.zrng[1])
        & (cat.dec > -80 * u.deg)
        & (cat.dec < 5 * u.deg)
        & (np.abs(cat.b) > 20 * u.deg)
    )
    # use a mass definition that's consistent with the reference catalog
    catlabel = cat.label.split("-")[0]
    mass = chances[f"m500_{catlabel}"]
    chances.sort(f"m500_{catlabel}")
    cat.catalog.sort(cat.masscol)
    catmatch = np.isin(cat.obj, chances[cat.label])
    print(np.sort(chances[cat.label].value))
    if cat.label == "PSZ2":
        print(cat["name", "z", "ra", "dec", "GLON", "GLAT", "MSZ"][~catmatch][-5:])

    fig, axes = plt.subplots(2, 3, figsize=(22, 15), constrained_layout=True)
    axes = np.reshape(axes, -1)
    ax = axes[0]
    ax.scatter(
        cat.z[catmask], cat.mass[catmask], marker=".", c="C0", zorder=1, label=cat.label
    )
    ax.scatter(
        chances["z"],
        mass,
        marker="o",
        facecolor="none",
        s=100,
        c="C3",
        lw=2,
        zorder=-10,
        label=f"CHANCES x {catlabel}",
    )
    ax.set(xlim=xlim, xlabel=zlabel, ylabel=f"{catlabel} {mlabel}")
    ax.legend()
    ax.set(ylim=(0, ax.get_ylim()[1]))
    # 2d contours
    ax = axes[1]
    # extent = (zbins[0])
    h2d = np.histogram2d(chances["z"], mass, (zbins, mbins))[0]
    ax.contour(zx, mx, gaussian_filter(h2d.T, 1), colors="C3")
    hact2d = np.histogram2d(cat.z[catmask], cat.mass[catmask], (zbins, mbins))[0]
    ax.contour(zx, mx, gaussian_filter(hact2d.T, 1), colors="C0")
    ax.set(xlabel=zlabel, ylabel=f"{catlabel} {mlabel}")
    # mass histograms
    ax = axes[2]
    h = ax.hist(
        mass,
        mbins,
        color="0.4",
        histtype="stepfilled",
        lw=2,
        log=True,
        label=f"CHANCES x {catlabel}",
    )[0]
    # h = ax.hist(mass[cmmask], mbins, color='0.2', histtype='stepfilled',
    #             zorder=-10, lw=2, log=True, label=f'CHANCES x {catlabel}')[0]
    h_ref = ax.hist(
        cat.mass[catmask],
        mbins,
        histtype="step",
        lw=4,
        color="C0",
        log=True,
        zorder=100,
        label=cat.label,
    )[0]
    ax.set(xlabel=f"{catlabel} {mlabel}", ylabel="$N(M_{500})$")
    ax.legend()
    ax = axes[3]
    n = chances["z"].size
    ic(h, h.shape)
    if cat.name == "psz2":
        m0 = (
            np.linspace(6, 8, 21)
            if args.sample == "evolution"
            else np.linspace(0.2, 3, 21)
        )
    elif cat.name == "codex":
        m0 = (
            np.linspace(8, 12, 21)
            if args.sample == "evolution"
            else np.linspace(2, 5, 21)
        )
    msigma = np.linspace(0.1, 2, 19)
    # in log space
    # m0 = np.linspace(0.7, 0.9, 21) if args.sample == 'evolution' \
    #     else np.linspace(-1, 0.2, 21)
    # msigma = np.linspace(0.03, 0.3, 19)
    a_m = np.linspace(-1, 0, 2)
    a_m = np.array([0])
    # m0 = np.array([8])
    # msigma = np.array([5])
    ic(m0, msigma)
    mbins_ks = np.linspace(0, mbins.max(), 100)
    # mbins_ks = np.logspace(-1, 1.3, 21)
    # h_ks = np.histogram(mass[cmmask], mbins_ks)[0]
    h_ks = np.histogram(mass, mbins_ks)[0]
    cdf = np.cumsum(h_ks) / h_ks.sum()
    ks, cdf_all = ks_mcut(
        args, axes[3], axes[5], cdf, mbins_ks, cat.mass[catmask], m0, msigma, a_m
    )
    print(ks.shape, cdf_all.shape)
    ax.set(xlabel=mlabel, ylabel="$p(<M_{500})$")
    ax.legend()
    ax = axes[4]
    extent = (m0[0], m0[-1], msigma[0], msigma[-1])
    im = ax.imshow(ks[0], origin="lower", aspect="auto", extent=extent)
    plt.colorbar(im, ax=ax, label="KS statistic")
    # using a fixed a_m=0 for now so we just need ks[0]
    ks_smooth = gaussian_filter(ks[0], 1)
    # find minimum - first interpolate
    # ic(m0.shape, msigma.shape, ks_smooth.shape)
    # f_ks_smooth = interp2d(m0, msigma, ks_smooth, kind='cubic')
    # ksmin = minimize(f_ks_smooth, x0=(7,0.8))
    # print(ksmin)
    # ax.scatter(*ksmin.x, marker='x', c='w', s=80, lw=2)
    ax.contour(ks_smooth, extent=extent, levels=(0.12, 0.2, 0.4), colors="w")
    ax.set(
        xlabel="$m_0$ ($10^{14}$M$_\odot$)", ylabel="$\sigma_m$ ($10^{14}$M$_\odot$)"
    )
    # axes[5].axis('off')
    ax = axes[5]
    ax.set(xlabel="KS statistic", ylabel="N")
    output = f"plots/mass_z_{args.sample}_{cat.name}.png"
    savefig(output, fig=fig, tight=False)
    return


def ks_mcut(args, ax_ks, ax_pte, cdf, mbins, mass, m0, msigma, a_m, nsamples=1000):
    mcut = mass_selection(
        mass, m0[:, None], msigma[:, None, None], a_m[:, None, None, None]
    )
    ic(mcut.shape)
    ti = time()
    h = np.array(
        [
            [
                [np.histogram(mass[mcut_ijk], mbins)[0] for mcut_ijk in mcut_ij]
                for mcut_ij in mcut_i
            ]
            for mcut_i in mcut
        ]
    )
    cdf_params = np.cumsum(h, axis=-1) / (np.sum(h, axis=-1)[..., None])
    ks = np.max(np.abs(cdf_params - cdf), axis=-1)
    ic(cdf.shape, h.shape, ks.shape)
    mx = (mbins[1:] + mbins[:-1]) / 2
    ic(mbins, mx)
    [
        ax_ks.plot(mx, cdf_params[k, j, i])  # ,
        # label=f'({m0[i]:.1f},{msigma[j]:.1f},{a_m[k]:.1f})')
        for k in range(a_m.size)
        for j in range(msigma.size)
        for i in range(m0.size)
    ]
    ax_ks.plot(
        mx, cdf, "-", color="k", lw=8, zorder=100
    )  # , label='CHANCES subsample')
    ic(ks, ks.shape)
    print(f"KS stats in {time()-ti:.1f} s")
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
    pm = nm_tot * pdetect / (nm_tot * pdetect).sum()
    m_detected = np.random.choice(mx, size=(10000, nm_chances[0, i_msigma, i_m0]), p=pm)
    ic(m_detected.shape)
    nm_detected = np.array(
        [np.histogram(m_detected_i, mbins)[0] for m_detected_i in m_detected]
    )
    ic(nm_detected.shape)
    cdf_i = np.cumsum(nm_detected, axis=1) / np.sum(nm_detected, axis=1)[:, None]
    ic(cdf_i.shape)
    ks_i = np.max(np.abs(cdf_i - cdf_params[0, i_msigma, i_m0]), axis=1)
    ic(ks_i)
    ic(ks_i.shape)
    ax_pte.hist(ks_i, "doane", histtype="step")
    ax_pte.axvline(ks[0, i_msigma, i_m0], ls="--")
    pte = (ks_i > ks[0, i_msigma, i_m0]).sum() / ks_i.size
    ax_pte.annotate(
        f"PTE={pte:.2f}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=15,
    )
    return ks, cdf_params

    m0_grid, msigma_grid, am_grid = np.meshgrid(m0, msigma, a_m)
    ic(msigma_grid.shape, m0_grid.shape, am_grid.shape, h.shape, ks.shape)
    ti = time()
    if args.ncores > 1:
        # to use this I need to return the indices that went into ks_mcut_pte too
        with Pool(args.ncores) as pool:
            results = [
                [
                    [
                        pool.apply_async(
                            ks_mcut_pte,
                            args=(mass, m0_ijk, ms_ijk, am_ijk, mbins, h_ijk, ks_ijk),
                            kwds={"nsamples": nsamples},
                        )
                        for m0_ijk, ms_ijk, am_ijk, h_ijk, ks_ijk in zip(
                            m0_ij, ms_ij, am_ij, h_ij, ks_ij
                        )
                    ]
                    for m0_ij, ms_ij, am_ij, h_ij, ks_ij in zip(m0_i, ms_i, am_i, h, ks)
                ]
                for m0_i, ms_i, am_i, h_i, ks_i in zip(
                    m0_grid, msigma_grid, am_grid, h, ks
                )
            ]
            pool.close()
            pool.join()
        results = np.reshape(results, -1)
    else:
        results = [
            [
                [
                    ks_mcut_pte(
                        mass,
                        m0_ijk,
                        ms_ijk,
                        am_ijk,
                        mbins,
                        h_ijk,
                        ks_ijk,
                        nsamples=nsamples,
                    )
                    for m0_ijk, ms_ijk, am_ijk, h_ijk, ks_ijk in zip(
                        m0_ij, ms_ij, am_ij, h_ij, ks_ij
                    )
                ]
                for m0_ij, ms_ij, am_ij, h_ij, ks_ij in zip(m0_i, ms_i, am_i, h_i, ks_i)
            ]
            for m0_i, ms_i, am_i, h_i, ks_i in zip(m0_grid, msigma_grid, am_grid, h, ks)
        ]
        results = np.reshape(results, (-1, 2))
    ic(results.shape)
    ks_mc = np.zeros((nsamples, ks.size))
    pte = np.zeros(results.size)
    ic(ks_mc.shape, pte.shape)
    for i, out in enumerate(results):
        if args.ncores > 1:
            out = out.get()
        ks_mc[:, i], pte[i] = out
    ic(pte)
    print(f"PTEs in {time()-ti:.1f} s")
    ic(pte.shape)
    return ks, pte


def ks_mcut_pte(mass, m0, msigma, a_m, mbins, h0, ks, mass_err=0.2, nsamples=100000):
    ones = np.ones(nsamples)[:, None]
    mcut_mc = mass_selection(mass, ones * m0, ones * msigma, ones * a_m)
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
        np.abs(
            np.cumsum(h_mc, axis=1) / (np.sum(h_mc, axis=1)[:, None])
            - np.cumsum(h0) / h0.sum()
        ),
        axis=1,
    )
    ic(ks_mc, ks_mc.shape)
    ic(ks)
    ic()
    # if (4 < m0 < 4.5) and (0.55 < msigma < 0.65):
    #     ic(h_mc.shape, ks_mc.shape, ks, ks_mc, (ks_mc > ks).sum() / nsamples)
    return ks_mc, np.array((ks_mc > ks).sum() / nsamples, dtype=float)


def mass_selection(m, m0, sigma, a_m=-1):
    prob = selection_function(m, m0, sigma, a_m)
    ic(np.percentile(prob, [0, 1, 25, 50, 75, 99, 100]))
    det = np.random.random(size=prob.shape)
    ic(prob[:100])
    ic(det[:100])
    ic((prob > det).sum(), prob.size)
    return prob > det


def selection_function(m, m0, sigma, a_m=0):
    # this is normalized like shit
    s = 0.5 * (1 + m**a_m * erf((m - m0) / (sigma * 2**0.5)))
    return s


### External catalogs ###


def load_ancillary(args, catalog, catalog_name, cosmo=Planck18):
    use_axesls = True
    # these are the ones from which I might get a mass
    catalog, psz = load_catalog(args, catalog, "psz2")
    catalog, act = load_catalog(args, catalog, "act-dr5")
    catalog, sptecs = load_catalog(args, catalog, "spt-ecs")
    catalog, sptsz = load_catalog(args, catalog, "spt-sz")
    catalog, erass = load_catalog(args, catalog, "erass1")
    catalog, mcxc = load_catalog(args, catalog, "mcxc")
    catalog, redmapper = load_catalog(args, catalog, "redmapper")
    catalog, codex = load_codex(args, catalog)
    catalog, axes2mrs = load_axes2mrs(args, catalog)
    catalog, axesls = load_axesls(args, catalog)
    catalog, comalit = load_comalit(args, catalog)
    catalog = load_splus(args, catalog)
    if "DECam" not in catalog.colnames:
        catalog = load_decam(args, catalog)
    # add indices in SZ+X-ray catalogs
    for szcat in (psz, act, sptecs, sptsz, mcxc):
        idxcol = f"{szcat.label}_idx"
        rng = np.arange(szcat.obj.size, dtype=int)
        catalog[idxcol] = [
            rng[szcat.obj == obj][0] if obj != "" else -99
            for obj in catalog[szcat.label]
        ]
    ic(codex)
    # other catalogs
    catalog, mk = load_meerkat(args, catalog)
    catalog, mkd = load_meerkat_diffuse(args, catalog)
    # catalog, rass = load_rass(args, catalog)
    catalog, xmm = load_xmm(args, catalog)
    catalog, hiflugcs = load_hiflugcs(args, catalog)
    catalog, meneacs = load_meneacs(args, catalog)
    catalog, locuss = load_locuss(args, catalog)
    catalog, lovoccs = load_lovoccs(args, catalog)
    catalog, clashvlt = load_clashvlt(args, catalog)
    catalog, wings = load_wings(args, catalog)
    ic(catalog)

    # add masses
    catalog.masscols = {}
    catalog.factors = {}
    catalog = add_masses(catalog, psz, "MSZ", 1.10, massdef="500c", cosmo=cosmo)
    catalog = add_masses(catalog, act, "M500cCal", massdef="500c", cosmo=cosmo)
    catalog = add_masses(catalog, sptecs, "M500", massdef="500c", cosmo=cosmo)
    catalog = add_masses(catalog, sptsz, "M500", massdef="500c", cosmo=cosmo)
    catalog = add_masses(catalog, mcxc, "M500", 1.41, massdef="500c", cosmo=cosmo)
    catalog = add_masses(catalog, codex, "M200c", 0.70, massdef="200c", cosmo=cosmo)
    # catalog = add_masses(
    #     catalog, codex, "M200c_sigma", 1.13, massdef="200c", cosmo=cosmo
    # )
    catalog = add_masses(catalog, comalit, "M200c", massdef="200c", cosmo=cosmo)
    # here we fix factor=1 because the comparison to ACT is affected
    # by the merging of double clusters in AXES
    catalog = add_masses(catalog, axes2mrs, "M200c", 0.88, massdef="200c", cosmo=cosmo)
    catalog = add_masses(catalog, axesls, "M200c", 0.88, massdef="200c", cosmo=cosmo)
    # have not tested these
    catalog = add_masses(catalog, meneacs, "m200", 1, massdef="200c", cosmo=cosmo)
    catalog = add_masses(catalog, locuss, "M200", 1, massdef="200c", cosmo=cosmo)
    catalog = add_masses(catalog, erass, "M500", 1, massdef="500c", cosmo=cosmo)
    # this one obtained by comparing to PSZ - hence the 1.10
    # catalog = add_masses(
    #     catalog, wings, "m200", 0.69 * 1.10, massdef="200c", cosmo=cosmo
    # )
    # it has 2 clusters in common with ACT and the
    catalog = add_masses(catalog, wings, "m200", 0.76, massdef="200c", cosmo=cosmo)

    plot_codex_mass_ratio(args, catalog)

    # this should register m200
    ic(np.sort(catalog.colnames))
    catalog["source"] = [12 * " " for name in catalog["name"]]
    catalog["m200"] = -np.ones(catalog["name"].size)
    catalog["r200"] = -np.ones(catalog["name"].size)
    if args.sample == "lowz":
        mcatlist = [
            "MENeaCS",
            "CoMaLit",
            ("AXES-2MRS", 0, 0.04),
            ("AXES-LEGACY", 0, 0.04),
            # "eRASS1",
            "ACT-DR5",
            "SPT-ECS",
            "SPT-SZ",
            "WINGS",
            "MCXC",
            "PSZ2",
            "CODEX",
            ("AXES-2MRS", 0.04, 1),
            ("AXES-LEGACY", 0.04, 1),
        ]
    else:
        mcatlist = [
            "LoCuSS",
            "MENeaCS",
            "CoMaLit",
            "ACT-DR5",
            "SPT-ECS",
            "SPT-SZ",
            # "eRASS1",
            "PSZ2",
            "CODEX",
            # "AXES-2MRS",
            # "AXES-LEGACY",
            "MCXC",
        ]
    priorities = ["priorities:"]
    for mcatname in mcatlist:
        if not isinstance(mcatname, str):
            mcatname, zmin, zmax = mcatname
            priorities.append(f"{mcatname}({zmin:.2f}-{zmax:.2f})")
        else:
            zmax = -99
            priorities.append(mcatname)
        hasmass = catalog[f"m200_{mcatname}"] > -1
        if zmax > 0:
            hasmass = hasmass & (catalog["z"] > zmin) & (catalog["z"] <= zmax)
        update = hasmass & (catalog["m200"] == -1)
        # this one's a special case
        if mcatname == "PSZ2":
            update = update | (catalog["name"] == "Abell 3395")
        printcols = ["name", "ra", "dec", "z", "m200", f"m200_{mcatname}", "source"]
        for col in ("m200", "r200"):
            catalog[col][update] = catalog[f"{col}_{mcatname}"][update]
        catalog[f"m200_{mcatname}"].format = "%.2e"
        catalog[f"r200_{mcatname}"].format = "%.2f"
        catalog["source"][update] = mcatname
        print(f"{mcatname}: Total {hasmass.sum()}, Updated {update.sum()}")
        print(catalog[printcols][catalog[f"m200_{mcatname}"] > -1])
        print(catalog[printcols][update])
        print("---")
    catalog.meta["comments"].append(" ".join(priorities))
    catalog["d200"] = (
        (catalog["r200"] * u.Mpc * cosmo.arcsec_per_kpc_proper(catalog["z"]))
        .to(u.arcmin)
        .value
    )
    catalog["m200"].format = "%.1e"
    catalog["r200"].format = "%.2f"
    catalog["d200"].format = "%.2f"
    # this is to avoid google sheets messing things up
    for col in ("z", "ra", "dec"):
        catalog[col].format = "%.4f"
    ic(np.sort(catalog.colnames))
    # catalog = calculate_m200(catalog)
    ic(catalog)
    printcols = ["name", "ra", "dec", "z", "source", "m200", "r200", "d200"]
    print("Empty: ", (catalog["m200"] == -1).sum())
    print((catalog[printcols][catalog["m200"] == -1]))
    cl = catalog["name"] == "A1835"
    print(catalog[printcols])
    print(catalog["name"].size)
    printcols = ["name", "z"] + [col for col in catalog.colnames if "m200" in col]
    print(catalog[printcols])

    # query_muse(args, catalog)
    # query_ned(args, catalog)

    # match to galaxy catalogs
    catalog, first = load_gal_first(args, catalog)
    catalog, tgss = load_gal_tgss(args, catalog)
    catalog, spec = load_gal_spec(args, catalog)

    # the extent we want, in degrees
    catalog["5d200(deg)"] = (5 / 60) * catalog["d200"]
    catalog["5d200(deg)"][catalog["d200"] == -1] = -1
    catalog.sort("name")
    if args.sample == "evolution":
        catalog.remove_columns(["MergedCat", "TargetList", "Jan25upload"])

    # sexagesimal

    hmsdms = catalog["coords"].to_string(style="hmsdms", sep=":", pad=True, precision=1)
    catalog["hms"], catalog["dms"] = np.transpose([c.split() for c in hmsdms])
    # we don't want it in the table!
    catalog.remove_column("coords")

    today = date.today().strftime("%Y%m%d")
    output = f"catalogues/clusters_{catalog_name}_{args.sample}_{today}"
    # long-form tables with all the details
    for col in catalog.colnames:
        if "m200" in col.lower() or "m500" in col.lower():
            catalog[col].format = "%.2e"
    # comments
    catalog.meta["comments"].append(
        "masscols: "
        + " ".join([f"{name}:{col}" for name, col in catalog.masscols.items()])
    )
    catalog.meta["comments"].append(
        "factors: " + " ".join([f"{name}:{f}" for name, f in catalog.factors.items()])
    )
    catalog.write(
        f"{output}_large.txt", comment="#", format="ascii.fixed_width", overwrite=True
    )
    catalog.write(
        f"{output}_large.csv", comment="#", format="ascii.csv", overwrite=True
    )
    cols = ["name", "hms", "dms", "z", "m200", "r200", "d200", "source"]
    units = {
        "ra": "deg",
        "dec": "deg",
        "hms": "hh:mm:ss",
        "dms": "dd:mm:ss",
        "m200": "1e14 * Msun",
        "r200": "Mpc",
        "d200": "arcmin",
        "5d200(deg)": "deg",
    }
    # qcat = QTable(
    #     catalog[cols], units={key: unit for key, unit in units.items() if key in cols}
    # )
    qcat = Table(catalog[cols])
    qcat.meta["comments"] = catalog.meta["comments"] + [
        "Units: " + ",".join(units.get(key, "1") for key in cols)
    ]
    qcat["m200"] = qcat["m200"] / 1e14
    qcat["m200"].format = "%.1f"
    # qcat.write(f"{output}.ecsv", format="ascii.ecsv", overwrite=True)
    qcat.write(f"{output}.txt", format="ascii.fixed_width", comment="#", overwrite=True)
    qcat.write(f"{output}.csv", format="ascii.csv", comment="#", overwrite=True)
    # qcat.rename_column("hms", "ra")
    qcat["name"] = [
        name.replace("-", "$-$").replace(" 00", " ").replace(" 0", " ")
        for name in catalog["name"]
    ]
    qcat["dms"] = [f"${d[0]}${d[1:]}" for d in catalog["dms"]]
    qcat["z"].format = "%.3f"
    qcat.write(f"{output}.tex", format="ascii.latex", overwrite=True)
    print(f"Saved to {output} and {output}_large")

    summarize_ancillary(args, catalog)
    summarize_masses(args, catalog)
    print("\n----")
    for catname in ("CODEX", "PSZ2", "eRASS1", "CoMaLit"):
        print(catname)
        printcols = ["name", "z"] + [
            col for col in np.sort(catalog.colnames) if catname in col
        ]
        print(catalog[printcols][catalog[f"{catname}_idx"] > 0])
        print("---")
    printcols = ["name", "z", "m200", "source"] + [
        col for col in np.sort(catalog.colnames) if ("PSZ2" in col or "MCXC" in col)
    ]
    print(catalog[printcols])
    r = catalog["PSZ2_MSZ"] / catalog["MCXC_M500"]
    j = (catalog["PSZ2_MSZ"] > 0) & (catalog["MCXC_M500"] > 0)
    p = np.percentile(r[j], [16, 50, 84])
    fig, ax = plt.subplots()
    s = ax.scatter(catalog["m200"][j], r[j], c=catalog["z"][j])
    ax.axhline(1, ls="--", color="k")
    ax.axhline(p[0], color="0.4", ls=":")
    ax.axhline(p[2], color="0.4", ls=":")
    ax.axhline(p[1], color="k", ls="-")
    plt.colorbar(s, label="Redshift")
    ax.set(
        xlabel="CHANCES $M_{200}$",
        ylabel="$M_{500}^\\mathrm{PSZ2} / M_{500}^\\mathrm{MCXC}$",
    )
    output = "plots/m500_psz_mcxc.png"
    savefig(output, fig=fig)

    if args.sample == "lowz":
        w = catalog["m200_WINGS"] > -1
        print(
            catalog["name", "z", "m200_WINGS", "r200_WINGS", "m200", "r200", "source"][
                w
            ]
        )
        w = w & (catalog["source"] != "WINGS")
        rdiff = catalog["r200"][w] / catalog["r200_WINGS"][w]
        print(
            f"r200/r200_WINGS (excluding source=WINGS):\n{np.array(rdiff)}\nmedian = {np.median(rdiff):.2f}\nmean = {np.mean(rdiff):.2f}\nstd = {np.std(rdiff):.2f}"
        )

    others = (meneacs, wings, locuss, psz, act, sptecs, sptsz, codex, mcxc)
    return catalog, others


# def write_table_latex(cat, output):


def get_most_massive(args, cat, chances, n=200):
    gal = cat.coords.transform_to("galactic")
    mask = (
        (cat.z > args.zrng[0])
        & (cat.z < args.zrng[1])
        & (cat.dec > -80 * u.deg)
        & (cat.dec < 5 * u.deg)
        & (np.abs(gal.b) > 20 * u.deg)
    )
    print(np.sort(cat.colnames))
    jsort = np.argsort(cat.mass[mask])[-n:]
    most_massive = Table(
        {
            "name": cat.obj[mask][jsort],
            "ra": cat.ra[mask][jsort],
            "dec": cat.dec[mask][jsort],
            "z": cat.z[mask][jsort],
            cat.masscol: cat.mass[mask][jsort],
            "m200_listed": cat["m200_listed"][mask][jsort],
        }
    )
    most_massive[cat.masscol].format = ".2f"
    most_massive.sort("name")
    most_massive = ClusterCatalog(
        "Most Massive",
        most_massive,
        base_cols=("name", "ra", "dec", "z"),
        masscol=cat.masscol,
    )
    for col in ("ra", "dec", "z"):
        most_massive.catalog[col].format = "%.3f"
    # compliance...
    most_massive.catalog["coords"] = most_massive.coords
    # match names with chances
    dist_chances = most_massive.coords.separation(chances["coords"][:, None])
    closest = np.argmin(dist_chances, axis=0)
    matches = np.min(dist_chances, axis=0) < 5 * u.arcmin
    most_massive.catalog["name"][matches] = chances["name"][closest[matches]]
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


def plot_codex_mass_ratio(args, catalog, s=50):
    def annotate_cluster(row, x, z, ratio, j, log=True):
        name = catalog["name"][j]
        ic(name, z[j], ratio[j] - 1)
        dx = 0.1 * x[j] if log else 0.05
        row[1].text(x[j] + dx, ratio[j] - 1, name, ha="left", va="center", fontsize=14)
        row[2].text(
            z[j] + 0.002, ratio[j] - 1, name, ha="left", va="center", fontsize=14
        )
        return

    def plot_row(row, x, y, z, xlabel, ylabel, log=True):
        im_m = row[0].scatter(x, y, c=z, s=s)
        row[0].set(xlabel=xlabel, ylabel=ylabel)
        valid = (x > 0) & (y > 0)
        ratio = y / x
        r0 = np.median(ratio[valid])
        e = np.abs(np.percentile(ratio[valid], [16, 84]) - r0) / (valid.size - 1) ** 0.5
        label = rf"$\langle x\rangle = {r0:.3f}_{{{-e[0]:.3f}}}^{{+{e[1]:.3f}}}$"
        row[1].scatter(x[valid], ratio[valid] - 1, c=z[valid], s=s)
        row[1].axhline(0, ls="--", color="k", lw=2)
        row[1].set(xlabel=xlabel, ylabel=f"{ylabel} / {xlabel} $-1$")
        if log:
            row[0].set(xscale="log", yscale="log")
            row[1].set(xscale="log")
        norm = LogNorm(vmin=1e14, vmax=3e15) if log else Normalize()
        im_z = row[2].scatter(z[valid], ratio[valid] - 1, c=x[valid], s=s, norm=norm)
        row[2].axhline(0, ls="--", color="k", lw=2)
        row[2].set(xlabel="Redshift")
        mbar = plt.colorbar(
            im_z, ax=row[2], label=xlabel, orientation="vertical", aspect=10
        )
        # so that they are neither the maxima nor minima, for annotation
        ratio[np.isnan(ratio) | (ratio < 0)] = 1
        for j in np.argsort(ratio)[:3]:
            annotate_cluster(row, x, z, ratio, j, log=log)
        for j in np.argsort(ratio)[-3:]:
            annotate_cluster(row, x, z, ratio, j, log=log)
        return im_m, im_z

    ic(np.sort(catalog.colnames))
    ic(catalog[["name", "z", "m200_CODEX", "m200_AXES-LEGACY"]])
    codex_mass_ratio = catalog["m200_AXES-LEGACY"] / catalog["m200_CODEX"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), constrained_layout=True)
    # m500
    # m500_codex = 1e14 * catalog["m500_CODEX"]
    # m500_axesls = 1e14 * catalog["m500_AXES-LEGACY"]
    # im_m, im_z = plot_row(
    #     axes[0],
    #     m500_codex,
    #     m500_axesls,
    #     catalog["z"],
    #     "$M_{500}^\mathrm{CODEX}$",
    #     "$M_{500}^\mathrm{AXES-LEGACY}$",
    # )
    # m200
    # m200_codex, c200_codex, r200_codex, d200_codex = calculate_m200_from_m500(
    #     m500_codex, catalog["z"]
    # )
    # m200_axesls, c200_axesls, r200_axesls, d200_axesls = calculate_m200_from_m500(
    #     m500_axesls, catalog["z"]
    # )
    m200_codex = 1e14 * catalog["m200_CODEX"]
    m200_axesls = 1e14 * catalog["m200_AXES-LEGACY"]
    im_m, im_z = plot_row(
        axes[1],
        m200_codex,
        m200_axesls,
        catalog["z"],
        "$M_{200}^\mathrm{CODEX}$",
        "$M_{200}^\mathrm{AXES-LEGACY}$",
    )
    # r200
    # im_m, im_z = plot_row(
    #     axes[2],
    #     r200_codex,
    #     r200_axesls,
    #     catalog["z"],
    #     "$r_{200}^\mathrm{CODEX}$",
    #     "$r_{200}^\mathrm{AXES-LEGACY}$",
    #     log=False,
    # )
    # references
    m = np.logspace(14, 15.5, 10)
    for row in axes[:2]:
        row[0].plot(m, m, "k--", lw=2)
    r = np.linspace(1, 3, 10)
    axes[2, 0].plot(r, r, "k--", lw=2)
    # colorbars
    zbar = plt.colorbar(
        im_m, ax=axes[-1, :2], label="Redshift", orientation="horizontal", aspect=18
    )
    zbar.ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    output = f"plots/codex_mass_ratio_{args.sample}.png"
    savefig(output, fig=fig, tight=False)
    # sys.exit()
    return


def review_missing(args, chances, psz, act, sptecs, sptsz, codex, mcxc):
    # are we missing massive clusters?
    # these are the minimal constraints: redshift, declination and DECam
    # availability
    masks = [
        (cat.z > args.zrng[0])
        & (cat.z < args.zrng[1])
        & (cat.dec > -80)
        & (cat.dec < 5)
        & (np.abs(cat.b) > 20 * u.deg)
        for cat in (psz, act, sptecs, sptsz, codex, mcxc)
    ]
    psz_z, act_z, sptecs_z, sptsz_z, codex_z, mcxc_z = masks

    massive = {
        "psz2": np.argsort(psz["MSZ"][psz_z].value)[-args.nmassive :],
        "act-dr5": np.argsort(act["M500cCal"][act_z].value)[-args.nmassive :],
        "spt-ecs": np.argsort(sptecs["M500"][sptecs_z].value)[-args.nmassive :],
        "spt-sz": np.argsort(sptsz["M500"][sptsz_z].value)[-args.nmassive :],
        "mcxc": np.argsort(mcxc["M500"][mcxc_z].value)[-args.nmassive :],
    }
    # merge the three samples -- giving them the same names as CHANCES
    # so I can use them easily
    szmassive = Table(
        {
            "name": np.hstack(
                [
                    psz["NAME"][psz_z][massive["psz2"]],
                    act["name"][act_z][massive["act-dr5"]],
                    sptecs["SPT-ECS"][sptecs_z][massive["spt-ecs"]],
                    sptsz["SPT-SZ"][sptsz_z][massive["spt-sz"]],
                    mcxc["MCXC"][mcxc_z][massive["mcxc"]],
                ]
            ),
            "ra": np.hstack(
                [
                    psz["RA"][psz_z][massive["psz2"]],
                    act["RADeg"][act_z][massive["act-dr5"]],
                    sptecs["RAdeg"][sptecs_z][massive["spt-ecs"]],
                    sptsz["RAdeg"][sptsz_z][massive["spt-sz"]],
                    mcxc["RAdeg"][mcxc_z][massive["mcxc"]],
                ]
            ),
            "dec": np.hstack(
                [
                    psz["DEC"][psz_z][massive["psz2"]],
                    act["decDeg"][act_z][massive["act-dr5"]],
                    sptecs["RAdeg"][sptecs_z][massive["spt-ecs"]],
                    sptsz["DEdeg"][sptsz_z][massive["spt-sz"]],
                    mcxc["DEdeg"][mcxc_z][massive["mcxc"]],
                ]
            ),
            "z": np.hstack(
                [
                    psz["REDSHIFT"][psz_z][massive["psz2"]],
                    act["redshift"][act_z][massive["act-dr5"]],
                    sptecs["z"][sptecs_z][massive["spt-ecs"]],
                    sptsz["z"][sptsz_z][massive["spt-sz"]],
                    mcxc["z"][mcxc_z][massive["mcxc"]],
                ]
            ),
        }
    )
    # Catalog object
    szmassive.sort("name")
    szmassive = ClusterCatalog(
        "Missing Massive", szmassive, base_cols=("name", "ra", "dec", "z")
    )
    for col in ("ra", "dec", "z"):
        szmassive.catalog[col].format = "%.3f"
    # also for consistency with CHANCES
    szmassive.catalog["coords"] = szmassive.coords

    for szcat in (psz, act, sptecs, sptsz, mcxc):
        sep = szmassive.coords.separation(szcat.coords[:, None])
        closest = np.argmin(sep, axis=0)
        matches = np.min(sep, axis=0) < 5 * u.arcmin
        szmassive.catalog[szcat.label] = szcat.obj[closest]
        szmassive.catalog[szcat.label][~matches] = ""
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
        if (
            np.isin(
                cl["PSZ2", "ACT-DR5", "SPT-ECS", "SPT-SZ", "MCXC"].values, unique
            ).sum()
            > 0
        ):
            continue
        if cl["PSZ2"] != "":
            unique.append(cl["PSZ2"])
        elif cl["ACT-DR5"] != "":
            unique.append(cl["ACT-DR5"])
        elif cl["SPT-ECS"] != "":
            unique.append(cl["SPT-ECS"])
        elif cl["SPT-SZ"] != "":
            unique.append(cl["SPT-SZ"])
        elif cl["MCXC"] != "":
            unique.append(f"MCXC {cl['MCXC']}")
    szmassive.catalog["name"] = unique
    unique, unique_idx = np.unique(unique, return_index=True)
    szmassive.catalog = szmassive.catalog[unique_idx]
    print(f'There are {unique.size} unique "massive" clusters')
    # which are not in CHANCES?
    massive_in_chances = np.isin(
        szmassive["name"],
        np.reshape(
            [chances[col] for col in ["PSZ2", "ACT-DR5", "SPT-ECS", "SPT-SZ", "MCXC"]],
            -1,
        ),
    )
    ic(massive_in_chances, massive_in_chances.shape)
    missing = ~massive_in_chances
    ic(szmassive[missing]["name", "z", "PSZ2", "ACT-DR5", "MCXC"])
    return szmassive[missing]


def chances_catalog(args):
    use_final = True
    if args.sample == "lowz":
        # file = "catalogues-ciria/S1501_clusters_final.csv"
        file = "catalogues/S1501_clusters_20240410.csv"
        file_old = "CHANCES low-z clusters.csv"
    else:
        # file = "catalogues-ciria/S1502_clusters_final.csv"
        file = "catalogues/S1502_clusters_202403.csv"
        file_old = "CHANCES Evolution clusters.csv"
    cat = ascii.read(file, format="csv")
    print(np.sort(cat.colnames))
    cat_old = ascii.read(file_old, format="csv")
    cat.rename_column("m200", "m200_listed")
    # happens in low-z
    if "col12" in cat.colnames:
        ic(cat["col12"].value)
        cat.remove_column("col12")
    if args.sample == "lowz":
        cols = ["Cluster name", "RA (J2000)", "Dec (J2000)", "z"]
    else:
        cols = ["Cluster", "RA_J2000", "Dec_J2000", "z"]
    cols_old = ["Cluster Name", "RA_J2000", "Dec_J2000", "Z"]
    cat.rename_columns(cols, ["name", "ra", "dec", "z"])
    cat_old.rename_columns(cols_old, ["name", "ra", "dec", "z"])
    # if args.sample == "lowz":
    #     # in the new files z means number of members, not redshift!
    #     cat["z"] = [
    #         cat_old["z"][cat_old["name"] == name][0] if name in cat_old["name"] else 1
    #         for name in cat["name"]
    #     ]
    z_manual = {
        "A0194": 0.018,
        "A0548": 0.042,
        "A1631": 0.046,
        "A2415": 0.058,
        "A2457": 0.059,
        "A2734": 0.062,
        "A2870": 0.024,
        "A3341": 0.038,
        "A3389": 0.027,
        "A3574": 0.016,
        "AM2002": 0.023,
        "AS560": 0.037,
        "IIZw108": 0.049,
        "MZ00407": 0.022,
    }
    # for cl, z in z_manual.items():
    #     cat["z"][cat["name"] == cl] = z
    mass_manual = {"AM2002": 0.7, "MZ00407": 0.3}
    for cl, m in mass_manual.items():
        cat["m200_listed"][cat["name"] == cl] = m
    cat["coords"] = SkyCoord(ra=cat["ra"], dec=cat["dec"], unit="deg")
    cat.sort("name")
    for i, name in enumerate(cat["name"]):
        cat["name"][i] = (
            name.replace("Abell-", "Abell ")
            .replace("MACS-", "MACS ")
            .replace("PSZ2-", "PSZ2 ")
            .replace("RXC-", "RXC ")
        )
    cat["m200_listed"] *= 1e14
    cat["m200_listed"].format = ".2e"
    # additional info filled while collating
    cat.meta["comments"] = []
    ic(np.sort(cat.colnames))
    print(cat)
    return cat


def calculate_m200(chances, cosmo=Planck18):
    m200, c200, r200, d200 = np.zeros((4, chances["m500"].size))
    miss = chances["m500"] == 0
    ic(chances["m500", "z"][~miss])
    m200[~miss], c200[~miss], r200[~miss], d200[~miss] = calculate_m200_from_m500(
        1e14 * chances["m500"][~miss], chances["z"][~miss], cosmo=cosmo
    )
    if miss.sum() > 0:
        m200[miss], c200[miss], r200[miss], d200[miss] = calculate_m200_from_m500(
            chances["m200_listed"][miss], chances["z"][miss], cosmo=cosmo
        )
    ic(np.sort(chances.colnames))
    with_listed_m200 = (chances["m200_listed"] > 0) & (chances["m500"] <= 0)
    ic(with_listed_m200.sum())
    nfw_with_listed_m200 = NFW(
        1e14 * chances["m200_listed"][with_listed_m200],
        5,
        chances["z"][with_listed_m200],
        overdensity=200,
        frame="physical",
        cosmo=cosmo,
    )
    # these have a mass in Ciria's file but not in the catalogues we
    # review here
    ic(chances["m200_listed"][with_listed_m200] / m200[with_listed_m200])
    m200[with_listed_m200] = 1e14 * chances["m200_listed"][with_listed_m200]
    # need to make sure this is consistent with calculate_m200_from_m500
    r200[with_listed_m200], d200[with_listed_m200] = radius_from_m200(
        nfw_with_listed_m200, cosmo=cosmo
    )
    ic(r200)
    chances["m200"] = m200 / 1e14
    chances["c200"] = c200
    chances["r200"] = r200
    chances["d200"] = d200
    no_mass = (chances["m500"] == -1) & (chances["m200_listed"] == 0)
    for col in ("m200", "c200", "r200", "d200"):
        chances[col][no_mass] = -1
        chances[col].format = "%.2f"
        chances[col][np.isnan(chances[col])] = -1
    return chances


def calculate_m200_from_mdelta(mdelta, z, delta, cosmo=Planck18, model="ishiyama21"):
    cosmology.fromAstropy(cosmo, sigma8=0.81, ns=0.966)
    # try:
    c = concentration.concentration(mdelta, delta, z, model=model)
    # except Exception as e:
    #     j = np.argmin
    #     print(mdelta[-1], z[-1], concentration.concentration(mdelta, delta, z[-1]))
    #     print(mdelta[0], z[0])
    nfw = NFW(
        mdelta, c, z, overdensity=float(delta[:-1]), frame="physical", cosmo=cosmo
    )
    # print(nfw)
    m200, r200 = nfw.mdelta(200, "c")
    d200 = (r200 * u.Mpc * cosmo.arcsec_per_kpc_proper(nfw.z)).to(u.arcmin).value
    return m200, r200 / nfw.rs, r200, d200


def calculate_m200_from_m500(m500, z, cosmo=Planck18, model="ishiyama21"):
    return calculate_m200_from_mdelta(m500, z, "500c", cosmo=cosmo, model=model)


def radius_from_m200(nfw, cosmo=Planck18):
    r200 = nfw.radius
    kpc2arcmin = cosmo.arcsec_per_kpc_proper(nfw.z)
    d200 = (r200 * u.Mpc * kpc2arcmin).to("arcmin").value
    return r200, d200


### Matching routines ###


def match_galaxy_catalog(args, chances, galcat, radius=5, unit="r200", dz=0.03):
    """Match cluster catalog to external galaxy catalog

    ``unit`` must be either 'arcmin' or 'r200'"""
    print(f"Matching galaxy catalog {galcat.name}")
    assert unit in ("arcmin", "r200")
    # if unit == 'r200':
    #     missing = (chances['d200'] == -1)
    #     if missing.sum() > 0:
    #         massdata = calculate_m200_from_m500(
    #             m500_if_missing, chances['z'][missing])
    #         for col, name in zip(massdata, 'mcrd'):
    #             chances[f'{name}200'][missing] = col
    #         chances['m200'][missing] /= 1e14
    maxdist = (radius * chances["d200"] if unit == "r200" else radius) * u.arcmin
    Ngal, Ngal_z = np.zeros((2, maxdist.size), dtype=int)
    if "Cluster Name" in chances.colnames:
        namecol, racol, deccol = ["Cluster Name", "RA_J2000", "Dec_J2000"]
    else:
        namecol, racol, deccol = ["name", "ra", "dec"]
    ic(type(chances))
    ic(np.sort(chances.colnames))
    for i, (cl, dmax) in tqdm(enumerate(zip(chances, maxdist))):
        # ic(i, cl['Cluster Name','RA_J2000','Dec_J2000','z','m200','d200'])
        cosdec = np.cos(np.radians(cl[deccol]))
        nearby = (np.abs(cl[racol] * u.deg - galcat.ra) < 2 * dmax / cosdec) & (
            np.abs(cl[deccol] * u.deg - galcat.dec) < 2 * dmax
        )
        if nearby.sum() == 0:
            continue
        sep = cl["coords"].separation(galcat.coords[nearby])
        ic(dmax, nearby.sum(), nearby.sum() / galcat.z.size)
        matches = sep < dmax
        Ngal[i] = matches.sum()
        if Ngal[i] == 0:
            continue
        clname = cl[namecol].replace(" ", "_")
        galcat.catalog[nearby]["name", "ra", "dec", "z"][matches].write(
            f"aux/spectroscopic/{args.sample}/{clname}.txt",
            format="ascii.fixed_width",
            overwrite=True,
        )
        if dz is None:
            ic(Ngal[i])
            continue
        zmatches = matches & (np.abs(galcat.z[nearby] - cl["z"]) / (1 + cl["z"]) < dz)
        Ngal_z[i] = zmatches.sum()
        ic(Ngal[i], Ngal_z[i])
    try:
        chances[f"N_{galcat.name.lower()}"] = Ngal
        if dz is not None:
            chances[f"N_{galcat.name.lower()}_z"] = Ngal_z
    except TypeError:
        chances.catalog[f"N_{galcat.name.lower()}"] = Ngal
        if dz is not None:
            chances.catalog[f"N_{galcat.name.lower()}_z"] = Ngal_z
    # ic(chances['Cluster Name','z','m200','r200','d200','Nspec','Nspec_z'])
    return chances, galcat


def match_catalog(chances, cat, radius=5 * u.arcmin, dz=0.1, name=None):
    """Should match in redshift too"""
    try:
        dist = chances["coords"].separation(cat.coords[:, None])
    except AttributeError:
        coords = SkyCoord(ra=cat["ra"], dec=cat["dec"], unit="deg")
        dist = chances["coords"].separation(coords[:, None])
    if dist.size == 0:
        return chances, cat
    # print(np.sort(dist.to(u.arcmin)))
    jclosest = np.argmin(dist, axis=0)
    closest = np.min(dist, axis=0)
    matches = (closest < radius) & (
        np.abs(cat.z[jclosest] - chances["z"]) < (dz * (1 + chances["z"]))
    )
    # print(cat.name, matches.sum())
    idx = -99 * np.ones(matches.size, dtype=int)
    idx[matches] = np.argmin(dist, axis=0)[matches]
    # print(idx, idx[matches])
    if cat.label == "CODEX":
        print(cat.reference_value)
    try:
        name = cat.label
    except AttributeError:
        if name is None:
            raise ValueError("must provide a name")
    try:
        chances[f"{name}_idx"] = idx
        chances[name] = [cat.obj[i] if i > -99 else "" for i in idx]
        if hasattr(cat, "reference_value"):
            chances[f"{name}_{cat.reference_value}"] = cat[cat.reference_value][idx]
    except TypeError:
        chances.catalog[f"{name}_idx"] = idx
        chances.catalog[name] = [cat.obj[i] if i > -99 else "" for i in idx]
        if hasattr(cat, "reference_value"):
            chances.catalog[f"{name}_{cat.reference_value}"] = cat[cat.reference_value][
                idx
            ]
    return chances, cat


#### MUSE query ####


def run_query(args, catalog):
    names = [
        "index",
        "name",
        "ra",
        "dec",
        "z",
        "m200",
        "object",
        "texp(s)",
        "dist(Mpc)",
        "proposal",
        "dp_id",
    ]
    write_formats = {
        "index": "%5d",
        "ra": "%10.6f",
        "dec": "%9.5f",
        "z": "%.3f",
        "lambda": "%5.1f",
        **{
            name: "%s"
            for name in ("name", "proposal", "object", "dp_id", "texp(s)", "dist(Mpc)")
        },
    }
    ncl = catalog["name"].size
    nblocks = ncl // args.block_size + 1
    query = None
    eso = Eso()
    Eso.login("cjsifon", store_password=True)
    for iblock in range(nblocks):
        print(f"Block # {iblock+1:2d} / {nblocks}")
        start = iblock * args.block_size
        end = min((iblock + 1) * args.block_size, ncl)
        ic(catalog["ra", "dec"][start:end])
        if args.threads == 1:
            q = [
                query_cluster(args, cosmo, eso, i, cluster)
                for i, cluster in tqdm(
                    enumerate(catalog[start:end], iblock * args.block_size),
                    total=end - start,
                )
            ]
        else:
            ti = time()
            pool = ThreadPool(args.threads)
            q = [
                pool.apply_async(query_cluster, args=(args, cosmo, eso, i, cluster))
                for i, cluster in enumerate(
                    catalog[start:end], iblock * args.block_size
                )
            ]
            pool.close()
            pool.join()
            q = [i.get() for i in q]
            print(f"Done in {(time()-ti)/60:.2f} min")
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
        query.write("archive_info/programs.fits", format="fits", overwrite=True)
        query.write(
            "archive_info/programs.txt",
            format="ascii.fixed_width",
            formats=write_formats,
            overwrite=True,
        )
        print("-----")
    return


def query_cluster(args, cosmo, eso, i, cluster):
    """Takes a single row from the redmapper catalog"""
    a2k = cosmo.arcsec_per_kpc_proper(cluster["z"])
    size = (args.search_radius * a2k).to(u.deg)
    column_filters = {
        "coord1": cluster["ra"],
        "coord2": cluster["dec"],
        "box": size.value,
        "format": "decimal",
    }
    query = eso.query_instrument("muse", column_filters=column_filters, columns=[])
    if query is None or "Program ID" not in query.keys():
        return
    ic(query.colnames)
    query = query[query["DPR CATG"] == "SCIENCE"]
    clinfo = [cluster[key] for key in ("name", "ra", "dec", "z", "lambda")]
    clcoord = SkyCoord(ra=cluster["ra"] * u.deg, dec=cluster["dec"] * u.deg)
    qcoord = SkyCoord(ra=query["RA"] * u.deg, dec=query["DEC"] * u.deg)
    dist = (clcoord.separation(qcoord) / a2k).to(u.Mpc).value
    # information obtained from the query
    qinfo = (
        [",".join([str(i) for i in np.unique(query["Object"].value)])]
        + [",".join([f"{ti:.0f}" for ti in query["EXPTIME [s]"].value])]
        + [",".join([f"{i:.2f}" for i in dist])]
        + [",".join(np.unique(query[key])) for key in ["Program ID", "DP.ID"]]
    )
    info = [i] + clinfo + qinfo
    return info


#### Ancillary data ####


def add_masses(chances, cat, masscol, factor=1, cosmo=Planck18, massdef=None):
    # print("---")
    if massdef is None:
        massdef = masscol[1:5]
    name = cat.label  # .split("-")[0]
    col = "name" if "name" in chances.colnames else "Cluster Name"
    chances[f"{name}_{masscol}"] = -np.ones(chances[col].size)
    for q in ("m200", "r200"):
        chances[f"{q}_{name}"] = -np.ones(chances[col].size)
    mask = chances[f"{cat.label}_idx"] > -99
    if mask.sum() == 0:
        return chances
    catz = cat.z if hasattr(cat, "z") else cat[cat.colnames[3]]
    catm = cat[masscol]
    mdelta = catm[chances[f"{cat.label}_idx"][mask]]
    if mdelta.max() < 1000:
        mdelta = 1e14 * mdelta
    if cat.label == "eRASS1":
        mdelta /= 10
    m200, c200, r200, d200 = calculate_m200_from_mdelta(
        mdelta,
        catz[chances[f"{cat.label}_idx"][mask]],
        massdef,
        cosmo=cosmo,
    )
    chances[f"{name}_{masscol}"][mask] = mdelta
    chances[f"m200_{name}"][mask] = factor * m200
    chances[f"r200_{name}"][mask] = factor ** (1 / 3) * r200
    chances[f"m200_{name}"].format = "%.2e"
    chances[f"r200_{name}"].format = "%.2f"
    chances.masscols[name] = f"{masscol}"
    chances.factors[name] = factor
    return chances


def add_masses_manual(chances):
    chances["m500_other"] = -np.ones(chances["Cluster Name"].size)
    chances["m500_other_source"] = [
        20 * " " for i in range(chances["Cluster Name"].size)
    ]
    chances["m500_manual"].format = "%.2f"
    # clusters = {'A754': }
    return


def load_catalog(args, chances, name):
    """This is for the ones I have in ``astro``"""
    ic(name)
    cat = ClusterCatalog(name)
    goodz = (cat.z >= args.zrng[0]) & (cat.z < args.zrng[1])
    gooddec = (cat.dec >= -80 * u.deg) & (cat.dec <= 5 * u.deg)
    cat = ClusterCatalog(
        name, catalog=cat[goodz & gooddec], base_cols=("name", "ra", "dec", "z")
    )
    # Simet et al. relation
    if name == "redmapper":
        cat.catalog["M200m"] = 10**14.34 * (cat["LAMBDA"] / 40) ** 1.33
    if name == "eRASS1":
        cat.catalog = cat.catalog[(cat["M500"] / (cat["M500"] - cat["M500_L"])) > 3]
    chances, cat = match_catalog(chances, cat)
    return chances, cat


def load_decam(args, catalog):
    """Add whether each cluster is in DECaLS after looking manually
    at the sky viewer"""
    # this should be a pretty good approximation
    gal = catalog["coords"].transform_to("galactic")
    decam = ["NO" if abs(b) <= 20 else "DR10" for b in gal.b.deg]
    try:
        catalog["DECam"] = decam
    except TypeError:
        catalog.catalog["DECam"] = decam
    return catalog


### Galaxy catalogs ###


def load_gal_first(args, chances):
    filename = "aux/radio/first_14dec17.fits.gz"
    first = ClusterCatalog(
        "FIRST", fits.getdata(filename), base_cols=("INDEX", "RA", "DEC", "FINT")
    )
    chances, first = match_galaxy_catalog(args, chances, first, dz=None)
    return chances, first


def load_gal_spec(args, chances, base_cols=("index", "RA", "DEC", "z")):
    filename = "aux/spectroscopic/SpecZ_Catalogue_20221105.csv"
    spec = ClusterCatalog(
        "spec", ascii.read(filename, format="csv"), base_cols=base_cols
    )
    chances, spec = match_galaxy_catalog(args, chances, spec)
    return chances, spec


def load_gal_tgss(args, chances):
    filename = "aux/radio/TGSSADR1_7sigma_catalog.fits"
    tgss = ClusterCatalog(
        "TGSS",
        fits.getdata(filename),
        base_cols=("Source_name", "RA", "DEC", "Total_flux"),
    )
    chances, tgss = match_galaxy_catalog(args, chances, tgss, dz=None)
    return chances, tgss


### Cluster catalogs ###


def load_axes2mrs(args, chances):
    filename = "aux/xray/Xmass_BayesGroups_n3ext200kpc_nofake_info2.fits"
    cols = [
        "C3ID",
        "RA_X",
        "DEC_X",
        "z",
        "M200c",
        "eM200c",
        # "M500",
        # "eM500",
        "Lx",
        "eLx",
    ]
    axes2mrs = Table(fits.open(filename)[1].data)
    axes2mrs["M200c"] /= 1e14
    axes2mrs["eM200c"] /= 1e14
    # axes2mrs["M500"] = axes2mrs["M200c"] * axes2mrs["M500/M200"]
    # axes2mrs["eM500"] = axes2mrs["eM200c"] * axes2mrs["M500/M200"]
    axes2mrs = ClusterCatalog(
        "axes-2mrs",
        axes2mrs,
        cols=cols,
        base_cols=cols[:4],
        label="AXES-2MRS",
        masscol="M200c",
    )
    chances, axes2mrs = match_catalog(chances, axes2mrs)
    return chances, axes2mrs


def load_axesls(args, chances):
    # filename = "aux/xray/codex3_lsdr10_info.fits"
    # cols = ["CODEX3", "RA_X", "Dec_X", "z_best", "lambda", "Lx0124", "M500", "M200c"]
    filename = "aux/xray/Xmass_axes_legacy.cat"
    cols = ["CODEX3", "RA_X", "DEC_X", "zcmb", "Lx", "M200c", "eM200c", "sigma"]
    axesls = Table.read(filename, format="ascii.commented_header")
    axesls["M200c"] /= 1e14
    axesls["eM200c"] /= 1e14
    # axesls["M500"] = axesls["M200c"] * axesls["M500/M200"]
    axesls = ClusterCatalog(
        "axes-legacy",
        axesls,
        cols=cols,
        base_cols=cols[:4],
        label="AXES-LEGACY",
        masscol="M200c",
    )
    chances, axesls = match_catalog(chances, axesls)
    return chances, axesls


def load_clashvlt(args, chances):
    filename = "aux/spectroscopic/clashvlt_sample.txt"
    clashvlt = ClusterCatalog(
        "CLASH-VLT",
        ascii.read(filename, format="fixed_width"),
        base_cols=("Cluster", "RA", "DEC", "z"),
        coord_unit=(u.hourangle, u.deg),
    )
    chances, clashvlt = match_catalog(chances, clashvlt)
    return chances, clashvlt


def load_codex(args, chances):
    """this is point sources, good for evolution"""
    filename = "aux/xray/codex10_eromapper_Xmass.fits"
    codex = Table(fits.open(filename)[1].data)
    codex = codex[codex["lambda"] > 33.5 * 0.7 * (codex["z_lambda"] / 0.15) ** 0.8]
    # codex = codex[codex['codex50']]
    codex.rename_columns(["ra_opt", "dec_opt"], ["ra_bcg", "dec_bcg"])
    # M200 from velocity dispersion -- this comes out 3.8% lower than ACT-DR5
    Hz = Planck18.H(codex["z_lambda"]).value
    H0 = Hz / Planck18.H0.value
    Ez = Hz / H0
    # Lx-sigma relation from Damsted+2023 for SPIDERS sample (Table B.3)
    # first assign the z > 0.3 relation to all
    ln_sigmav = 6.584 + 0.220 * np.log(codex["Lx0124"] / 1e44 / Ez)
    # intermediate redshifts
    midz = (codex["z_lambda"] >= 0.15) & (codex["z_lambda"] < 0.3)
    ln_sigmav[midz] = 6.556 + 0.218 * np.log(codex["Lx0124"][midz] / 1e44 / Ez[midz])
    # now fix for low-z
    lowz = codex["z_lambda"] < 0.15
    ln_sigmav[lowz] = 6.529 + 0.147 * np.log(codex["Lx0124"][lowz] / 1e44 / Ez[lowz])
    codex["sigma_v"] = np.exp(ln_sigmav)
    codex["M200c_sigma"] = 1e15 * (codex["sigma_v"] / 1177) ** (1 / 0.364) / (Hz / 100)
    # codex["M500"] = codex["M200c"] * codex["M500/M200"]
    cols = [
        "id_cluster",
        "RA_X-ray",
        "Dec_X-ray",
        "z_lambda",
        "ra_bcg",
        "dec_bcg",
        "lambda",
        "Lx0124",
        # "M500",
        "M200c",
        "M200c_sigma",
    ]
    codex["id_cluster"] = np.array(codex["id_cluster"], dtype=str)
    # codex["M500"] /= 1e14
    # codex["M200c"] /= 1e14
    codex["M200c_sigma"] /= 1e14
    codex = ClusterCatalog(
        "codex",
        codex,
        cols=cols,
        base_cols=cols[:4],
        label="CODEX",
        masscol="M200c_sigma",
    )
    codex.reference_value = "M200c"
    chances, codex = match_catalog(chances, codex)
    return chances, codex


def load_comalit(args, chances):
    filename = "aux/lensing/comalit/J_MNRAS_450_3665_single.dat"
    tbl = {key: [] for key in ("name", "ra", "dec", "z", "M200c", "eM200c")}
    with open(filename) as f:
        _ = [f.readline() for i in range(6)]
        for line in f:
            if line[0] == "-":
                continue
            line = line.strip().split("|")
            tbl["name"].append(line[0].strip())
            tbl["ra"].append(":".join(line[1].split()[:3]))
            tbl["dec"].append(":".join(line[1].split()[3:]))
            tbl["z"].append(float(line[2]))
            tbl["M200c"].append(float(line[13]))
            tbl["eM200c"].append(float(line[14]))
    coord = SkyCoord(tbl["ra"], tbl["dec"], unit=(u.hourangle, u.deg))
    tbl["ra"] = coord.ra.deg
    tbl["dec"] = coord.dec.deg
    comalit = ClusterCatalog("CoMaLit", tbl, base_cols=("name", "ra", "dec", "z"))
    chances, comalit = match_catalog(chances, comalit)
    return chances, comalit


def load_hiflugcs(args, chances):
    filename = "aux/xray/hiflugcs_sample.txt"
    hiflugcs = ascii.read(filename, format="cds")
    hiflugcs = ClusterCatalog(
        "HIFLUGCS",
        hiflugcs[hiflugcs["Sample"] == "Included"],
        base_cols=("CName", "RAdeg", "DEdeg", "z"),
    )
    chances, hiflugcs = match_catalog(chances, hiflugcs)
    return chances, hiflugcs


def load_locuss(args, chances):
    filename = "aux/lensing/locuss/locuss-tab-mass-corr.csv"
    locuss = ascii.read(filename, format="csv")
    locuss["name"] = [
        name.replace("ABELL", "Abell ").replace("RXC", "RXC ")
        for name in locuss["name"]
    ]
    # easy way out
    locuss = join(
        locuss, chances["name", "ra", "dec", "z"], join_type="left", keys="name"
    )
    locuss = ClusterCatalog("LoCuSS", locuss, base_cols=("name", "ra", "dec", "z"))
    chances, locuss = match_catalog(chances, locuss)
    return chances, locuss


def load_lovoccs(args, chances):
    filename = "aux/spectroscopic/lovoccs_sample.txt"
    lovoccs = ClusterCatalog(
        "LoVoCCS",
        ascii.read(filename, format="basic"),
        base_cols=("Name", "ra", "dec", "z"),
    )
    chances, lovoccs = match_catalog(chances, lovoccs)
    return chances, lovoccs


def load_meneacs(args, chances):
    filename = "aux/lensing/meneacs_cccp.txt"
    meneacs = ClusterCatalog(
        "MENeaCS",
        ascii.read(filename, format="basic"),
        base_cols=("Cluster", "RA", "Dec", "z"),
        coord_unit=(u.hourangle, u.deg),
    )
    masses = ascii.read("aux/lensing/meneacs_cccp_masses.csv", format="csv")
    meneacs.catalog = join(
        meneacs.catalog,
        masses["name", "m200", "m200_err"],
        keys="name",
        join_type="left",
    )
    chances, meneacs = match_catalog(chances, meneacs)
    return chances, meneacs


def load_meerkat(args, chances):
    mk = ClusterCatalog(
        "MeerKAT", ascii.read("aux/meerkat/meerkat_legacy.csv", format="csv")
    )
    # ic(mk)
    chances, mk = match_catalog(chances, mk, name="MeerKAT")
    return chances, mk


def load_meerkat_diffuse(args, chances):
    mkd = fits.open("aux/meerkat/Table4_MGCLS_diffuse.fits")[1].data
    mkd = ClusterCatalog(
        "MKDiffuse",
        mkd,
        base_cols=("ClusterName", "R.A.J2000 (deg)", "Dec.J2000 (deg)", "z"),
    )
    chances, mkd = match_catalog(chances, mkd, name="MKDiffuse")
    return chances, mkd


def load_ned(args, cat, path="aux/ned/spec"):
    os.makedirs(path, exist_ok=True)
    http = urllib3.PoolManager()
    for i, cl in enumerate(cat):
        name = cl["name"].replace(" ", "_")
        output = os.path.join(path, f"{name}_ned.txt")
        if os.path.isfile(output):
            continue
        # query = Ned.query_region(cl['coords'], radius=cl['d200']*u.arcmin)
        # ic(cl, query['Object Name'].size)
        # query.write(output, format='ascii.fixed_width')
        ti = time()
        url = http.request(
            "GET",
            "http://ned.ipac.caltech.edu/cgi-bin/objsearch"
            "?search_type=Near+Position+Search&in_csys=Equatorial"
            "&in_equinox=J2000.0&lon={:.6f}d&lat={:.6f}d&radius={:.2f}"
            "&dot_include=ANY&in_objtypes1=GGroups&in_objtypes1=GClusters"
            "&in_objtypes1=QSO&in_objtypes2=Radio&in_objtypes2=SmmS"
            "&in_objtypes2=Infrared&in_objtypes2=Xray&nmp_op=ANY"
            "&out_csys=Equatorial&out_equinox=J2000.0&obj_sort=RA+or+Longitude"
            "&of=ascii_tab&zv_breaker=30000.0&list_limit=5&img_stamp=YES".format(
                cl["ra"], cl["dec"], 60 * cl["d200"]
            ),
        )
        ic(url)
        ic(url.data)
        ic(time() - ti)
        break
    return


def load_rass(args, chances):
    filename = "aux/xray/local_fof_insideSplus.cat"
    return


def load_sdss(args, cat):
    output = f"aux/sdss/sdss_spec_{args.sample}.tbl"
    if os.path.isfile(output):
        return ascii.read(output, format="fixed_width")
    sdss = SDSS.query_region(cat["coords"], spectro=True, radius=1 * u.deg)
    ic(sdss)
    if len(sdss.colnames) == 1 and "htmlhead" in sdss.colnames:
        return
    sdss.write(output, format="ascii.fixed_width", overwrite=True)
    return sdss


def load_splus(args, cat):
    splus = ascii.read("../splus/S-PLUS_footprint.csv", format="csv")
    splus["coords"] = SkyCoord(
        ra=splus["RA"], dec=splus["DEC"], unit=(u.hourangle, u.deg)
    )
    splus_dist = splus["coords"].separation(cat["coords"][:, None])
    in_splus = np.min(splus_dist, axis=1) < 1 * u.deg
    splus = ["YES" if i else "NO" for i in in_splus]
    try:
        cat["S-PLUS"] = splus
    except TypeError:
        cat.catalog["S-PLUS"] = splus
    return cat


def load_xmm(args, chances):
    filename = "aux/xray/local_fof_insideSplus_xmm.cat"
    xmm = ClusterCatalog(
        "XMM",
        ascii.read(filename, format="commented_header"),
        base_cols=("OBSERVATION.TARGET", "RA_X", "DEC_X", "z"),
    )
    chances, xmm = match_catalog(chances, xmm)
    return chances, xmm


def load_wings(args, chances):
    wings_positions = ascii.read("aux/xray/wings.txt", format="fixed_width")
    coords = SkyCoord(
        ra=wings_positions["hms"], dec=wings_positions["dms"], unit=(u.hourangle, u.deg)
    )
    wings_positions.rename_column("Cluster", "name")
    wings_positions["ra"] = coords.ra.deg
    wings_positions["dec"] = coords.dec.deg
    filename = "aux/spectroscopic/wings_r200.csv"
    wings = ascii.read(filename, format="csv")
    wings.rename_columns(["CLUSTER", "z_cl", "logM200"], ["name", "z", "m200"])
    wings["m200"] = 10 ** wings["m200"]
    wings = join(
        wings, wings_positions["name", "ra", "dec"], keys="name", join_type="left"
    )
    wings = ClusterCatalog("WINGS", wings, base_cols=("name", "ra", "dec", "z"))
    chances, wings = match_catalog(chances, wings)
    return chances, wings


### Summarize ###


def summarize_ancillary(args, chances):
    print()
    ic(np.sort(chances.colnames))
    for col in ("m500", "m200"):
        if col in chances.colnames:
            chances[col].format = "%.2e"
    chances["5d200(deg)"].format = "%.2f"
    cols = [
        "name",
        "z",
        "m200",
        "5d200(deg)",
        "CODEX",
        "AXES-LEGACY",
        "PSZ2",
        "ACT-DR5",
        "SPT-ECS",
        "SPT-SZ",
        "MeerKAT",
        "DECam",
        "S-PLUS",
        "N_spec",
        "N_spec_z",
        "N_tgss",
        "N_first",
    ]
    if "AXES-LEGACY" not in chances.colnames:
        cols.pop("AXES-LEGACY")
    if args.sample == "lowz":
        cols = cols[:4] + ["XMM"] + cols[4:]
        if "AAOzs" in chances.colnames:
            cols.append("AAOzs")
        if "Lx" in chances.colnames:
            cols.append("Lx")
    tbl = chances[cols]
    print(tbl)
    # optical photometry -- need to add for missing clusters
    if "DECam" in chances.colnames:
        decam = chances["DECam"] != "NO"
        if "VST" in chances.colnames:
            vst = chances["VST"] == "Y"
            opt = vst | decam
            print(
                f"{(vst & decam).sum()} with both VST and DECam,"
                f" {(1 - opt).sum()} without either:"
            )
        else:
            opt = decam
            print(f"{decam.sum()} with DECam, {(1-decam).sum()} without:")
        print(tbl[~opt])
        if "vst" in chances.colnames:
            print(f"  {vst.sum()} with VST")
        print(f"  {decam.sum()} with DECam")
    if "S-PLUS" in chances.colnames:
        splus_observed = ~np.isin(chances["S-PLUS"], ("", "NO"))
        if "SPLUS comments" in chances.colnames:
            splus_upcoming = np.array(
                ["target" in comment for comment in chances["SPLUS comments"]]
            )
            splus = splus_observed | splus_upcoming
        else:
            splus = splus_observed
        print(f"{splus.sum()} with SPLUS:")
        print(tbl[splus])
        if "SPLUS comments" in chances.colnames:
            print(f"  {splus_observed.sum()} already with SPLUS")
            print(f"  {splus_upcoming.sum()} with upcoming SPLUS")

    # print()
    # spectroscopy
    if "AAOzs" in chances.colnames:
        aao = chances["AAOzs"] > 0
        print(f"{aao.sum()} with AAOzs")
    meneacs = chances["MENeaCS"] != ""
    print(f"{meneacs.sum()} in MENeaCS:")
    print(tbl[meneacs])
    lovoccs = chances["LoVoCCS"] != ""
    print(f"{lovoccs.sum()} in LoVoCCS:")
    print(tbl[lovoccs])
    clashvlt = chances["CLASH-VLT"] != ""
    print(f"{clashvlt.sum()} in CLASH-VLT:")
    print(tbl[clashvlt])
    # SZ
    psz = chances["PSZ2"] != ""
    act = chances["ACT-DR5"] != ""
    sptecs = chances["SPT-ECS"] != ""
    sptsz = chances["SPT-SZ"] != ""
    sz = psz | act | sptecs | sptsz
    print(f"{sz.sum()} with SZ, {(1-sz).sum()} without:")
    print(tbl[~sz])
    print(f"  {psz.sum()} with PSZ2")
    print(f"  {act.sum()} with ACT-DR5")
    print(f"  {sptecs.sum()} in SPT-ECS")
    print(f"  {sptsz.sum()} in SPT-SZ")
    print(f"  {((sptecs | sptsz) & ~act).sum()} in SPT-SZ+ECS not in ACT-DR5:")
    print(tbl[(sptsz | sptecs) & ~act])
    # X-rays
    xmm = chances["XMM"] != ""
    mcxc = chances["MCXC"] != ""
    hiflugcs = chances["HIFLUGCS"] != ""
    xrays = xmm | mcxc | hiflugcs
    print(f"{xrays.sum()} with X-rays")
    print(f"  {hiflugcs.sum()} in HIFLUGCS:")
    print(tbl[hiflugcs])
    print(f"  {xmm.sum()} with XMM")
    print(f"  {mcxc.sum()} in MCXC")
    try:
        codex = chances["CODEX"] != ""
    except KeyError:
        codex = chances["AXES-LEGACY"] != ""
    codex = (chances["CODEX"] != "") | (chances["AXES-LEGACY"] != "")
    print(f"{codex.sum()} in CODEX:")
    print(tbl[codex])
    # radio
    mk = chances["MeerKAT"] != ""
    first = chances["N_first"] > 0
    tgss = chances["N_tgss"] > 0
    print(f"{(mk | first | tgss).sum()} with radio")
    print(f"{(mk & first).sum()} with MeerKAT and FIRST")
    print(f"{(mk & tgss).sum()} with MeerKAT and TGSS")
    print(f"  {mk.sum()} with MeerKAT:")
    print(tbl[mk])
    print(f"  {first.sum()} in FIRST")
    print(f"  {tgss.sum()} in TGSS")
    print("## Combinations ##")
    print(f"{(decam & splus).sum()} with both DECam and SPLUS:")
    print(tbl[decam & splus])
    print(f"{(sz & xrays).sum()} with SZ and X-rays")
    print(f"{(act & xmm).sum()} with ACT-DR5 and XMM")
    print(f"{(sz & mk).sum()} with SZ and MeerKAT")
    print(f"{(sz & xrays & mk).sum()} with SZ and X-rays and MeerKAT")
    print(f"{(sz & xrays & mk).sum()} with SZ and X-rays and TGSS")
    print(f"{(~sz & ~xrays).sum()} without SZ nor X-rays:")
    print(tbl[(~sz & ~xrays)])
    print(f"{(lovoccs & mk).sum()} in both LoVoCCS and MeerKAT:")
    print(tbl[(lovoccs & mk)])
    print(f"{((lovoccs | meneacs) & mk).sum()} in LoVoCCS or MENeaCS and MeerKAT:")
    print(tbl[(lovoccs | meneacs) & mk])
    print(f"{(lovoccs | meneacs | mk).sum()} in LoVoCCS or MENeaCS or MeerKAT:")
    print(tbl[lovoccs | meneacs | mk])
    if "S-PLUS" in chances.colnames:
        print(f"{(splus & lovoccs).sum()} in SPLUS and LoVoCCS:")
        print(tbl[splus & lovoccs])
    print(f"{(decam & (splus | lovoccs | mk | clashvlt)).sum()} in any of the above:")
    print(tbl[decam & (splus | lovoccs | mk | clashvlt)])
    print()
    return cols


def summarize_masses(args, chances):
    print("\n\n")
    cols = ["name", "z", "source"] + [
        col for col in chances.colnames if "500" in col or "200" in col
    ]
    for i in range(0, chances["name"].size, 30):
        print(chances[cols][i : i + 30])
    return


def parse_args():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add("sample", choices=("evolution", "lowz"), default="lowz")
    add("--debug", action="store_true")
    add("-m", "--mass-selection", default="psz2")
    add("--ncores", default=1, type=int)
    add("--nmassive", default=20, type=int)
    args = parser.parse_args()
    if not args.debug:
        ic.disable()
    args.zrng = (0, 0.1) if args.sample == "lowz" else (0.05, 0.50)
    return args


if __name__ == "__main__":
    main()
