import argparse
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
from astropy.io import ascii, fits
from astropy.table import Table, join, join_skycoord
import cmasher as cmr
from icecream import ic
from matplotlib import pyplot as plt, ticker
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle
import numpy as np
from numpy.polynomial import Polynomial, polynomial
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, root, root_scalar
from scipy.stats import binned_statistic as binstat

from plottery.plotutils import savefig, update_rcParams

update_rcParams()

from astro.clusters import ClusterCatalog, catalogs
from astro.footprint import Footprint

# local
from collate_info import calculate_m200_from_m500


def main():
    args = parse_args()
    sample = args.sample
    codex3 = Table(fits.open("aux/xray/codex3_lsdr10_info.fits")[1].data)
    codex3 = filter_sky(codex3, "ra", "dec")
    # this is the baseline catalog
    cat = codex3
    ic(np.sort(cat.colnames))
    ic((cat["M200c"] == 0).sum())
    # eromapper = Table(fits.open(
    #     'aux/optical/eromapper_optical_dr10_grz_catalog.fit.gz')[1].data)
    eromapper = Table(
        fits.open("aux/xray/codex_eromapper_catalog_fixedcodex.fit")[1].data
    )
    eromapper = filter_sky(eromapper, "ra", "dec")
    ic(np.sort(eromapper.colnames))
    # eromapper = eromapper[eromapper['codex10']]
    xmass = Table(
        fits.open("aux/xray/Xmass_BayesGroups_n3ext200kpc_nofake_info.fits")[1].data
    )
    ic(np.sort(xmass.colnames))
    xmass = filter_sky(xmass, "RA_X", "DEC_X")

    codex = Table(fits.open("aux/xray/codex_LSDR10_flux_info50.fits")[1].data)
    codex = filter_sky(codex, "RA_X", "Dec_X")
    codex = ClusterCatalog(
        "codex",
        codex,
        base_cols=("CODEX", "RA_X", "Dec_X", "z_best"),
        label="CODEX",
        masscol="lambda",
    )

    # match C3ID
    xmatches = np.isin(xmass["C3ID"], codex3["CODEX3"])
    xmass["lambda"] = [
        codex3["lambda"][codex3["CODEX3"] == c3] if c3 in codex3["CODEX3"] else 0
        for c3 in xmass["C3ID"]
    ]

    plot_xmass_codex(xmass, codex)

    z = cat["best_z"]
    zmask = z < 0.07
    mask = zmask
    ic(mask.size, mask.sum())

    fig, ax = plt.subplots(figsize=(6, 4))
    c = ax.scatter(
        cat["lambda"][mask],
        cat["M200c"][mask],
        marker=".",
        label="CODEX3",
        c=cat["best_z"][mask],
    )
    plt.colorbar(c, ax=ax, label="Redshift")
    # ax.scatter(eromapper['lambda'][matches_eromapper], eromapper['M200c'][matches_eromapper],
    #            marker='x', label='extended X-ray in eromapper-10')
    # ax.legend(fontsize=12)
    # ax.set(#xlabel='Redshift $z$', ylabel=r'Richness $\lambda$',
    #        xlim=(0, 0.08))
    ax.set(xscale="log", yscale="log", xlabel="Richness", ylabel="X-ray mass")
    output = f"plots/codex3_richness_{sample}.png"
    savefig(output, fig=fig)

    if sample == "lowz":
        zbins = np.linspace(0.03, 0.07, 4)
        lambins = np.logspace(1.4, 2, 5)
    else:
        zbins = np.linspace(0.07, 0.45, 5)
        lambins = np.array([1, 2, 3, 5, 8, 10, 20])
    ic(zbins, lambins)
    completeness(
        cat["best_z"],
        cat["lambda"],
        eromapper["best_z"],
        eromapper["lambda"],
        zbins,
        lambins,
        "codex3_lsdr10",
        "eromapper",
        sample,
    )

    # Original CHANCES catalog
    catalog_date = "20240725" if sample == "evolution" else "20241009"
    chances = ascii.read(
        f"catalogues/clusters_chances_{sample}_{catalog_date}.txt", format="fixed_width"
    )
    if "coords" in chances.colnames:
        chances.remove_column("coords")
    if "ra" not in chances.colnames:
        coords = SkyCoord(
            [f"{ra} {dec}" for ra, dec in chances["hms", "dms"]],
            unit=(u.hourangle, u.deg),
        )
        chances["ra"] = coords.ra.deg
        chances["dec"] = coords.dec.deg
    chances = filter_sky(chances, "ra", "dec")
    with_matches, matches, mindist = crossmatch(chances, codex3)
    chances["lambda"] = np.zeros(chances["coords"].size)
    chances["lambda"][with_matches] = codex3["lambda"][matches]
    completeness(
        chances["z"][with_matches],
        chances["lambda"][with_matches],
        codex3["best_z"],
        codex3["lambda"],
        zbins,
        lambins,
        f"chances_{sample}",
        "codex3",
        sample,
    )

    chances["lambda"].format = ".1f"
    codex3["best_z"].format = ".3f"
    codex3["lambda"].format = ".1f"
    cx = codex3["CODEX3", "ra", "dec", "best_z", "lambda"]
    if sample == "lowz":
        m = (chances["z"] < 0.045) & (chances["lambda"] > 40)
        print(chances["name", "ra", "dec", "z", "lambda"][m])
        m1 = (codex3["best_z"] < 0.045) & (codex3["lambda"] > 40)
        print(cx[m1])
        print(cx[matches][chances["name"][with_matches] == "A0536"])
        print(mindist[with_matches][chances["name"][with_matches] == "A0536"])
    else:
        m1 = (codex3["lambda"] > 200) | (
            (codex3["lambda"] > 150) & (codex3["best_z"] > 0.3)
        )
        print(cx[m1])

    # match CODEX3 to Abell
    abell = ClusterCatalog("abell")
    ic(abell.colnames)
    abell.catalog["coords"] = abell.coords
    matches = crossmatch(codex3, abell.catalog)
    codex3["abell"] = codex3["coords"].size * [11 * " "]
    codex3["abell"][matches[0]] = abell.obj[matches[1]]
    codex3["z_abell"] = np.zeros(codex3["coords"].size)
    codex3["z_abell"][matches[0]] = abell.z[matches[1]]
    cx = codex3["CODEX3", "abell", "ra", "dec", "best_z", "z_abell", "lit_z", "lambda"]
    print(cx)
    cl_test = "ABELL 3391"
    print(abell[abell.obj == cl_test])
    mask = codex3["abell"] == cl_test
    print(mask.size, mask.sum())
    print(cx[mask])

    # to Xmass
    matches = crossmatch(xmass, abell.catalog)
    xmass["abell"] = xmass["coords"].size * [11 * " "]
    xmass["abell"][matches[0]] = abell.obj[matches[1]]
    xmass["z_abell"] = np.zeros(xmass["coords"].size)
    xmass["z_abell"][matches[0]] = abell.z[matches[1]]
    xm = xmass[
        "C3ID", "abell", "RA_X", "DEC_X", "z", "z_abell", "Lx", "ngal", "sigma_v"
    ]
    mask = xmass["abell"] == cl_test
    print(mask.size, mask.sum())
    print(xm[mask])

    # some other catalogs of interest
    lovoccs = ascii.read("aux/spectroscopic/lovoccs_sample.txt", format="basic")
    lovoccs = filter_sky(lovoccs, "ra", "dec")
    lovoccs = ClusterCatalog(
        "lovoccs", lovoccs, label="LoVoCCS", base_cols=["Name", "ra", "dec", "z"]
    )
    meerkat = ascii.read("aux/meerkat/meerkat_legacy.csv", format="csv")
    meerkat = filter_sky(meerkat, "ra", "dec")
    meerkat = ClusterCatalog("meerkat", meerkat, label="MeerKAT")
    hiflugcs = ascii.read("aux/xray/hiflugcs_sample.txt", format="cds")
    hiflugcs = filter_sky(hiflugcs, "RAdeg", "DEdeg")
    hiflugcs = ClusterCatalog(
        "hiflugcs",
        hiflugcs,
        base_cols=("CName", "RAdeg", "DEdeg", "z"),
        label="HIFLUGCS",
    )
    if sample == "lowz":
        wings = ascii.read("aux/xray/wings.txt", format="fixed_width")
        wings["coords"] = SkyCoord(
            ra=wings["hms"], dec=wings["dms"], unit=(u.hourangle, u.deg)
        )
        wings["ra"] = wings["coords"].ra.deg
        wings["dec"] = wings["coords"].dec.deg
        wings = filter_sky(wings, "ra", "dec")
        wings = ClusterCatalog(
            "wings",
            wings,
            label="WINGS",
            base_cols=("Cluster", "ra", "dec", "z"),
            masscol="Lx_1e44",
        )

    if sample == "lowz":
        splus = load_splus(chances)
    else:
        splus = None

    act = ClusterCatalog("act-dr5")
    # act.catalog['coords'] = act.coords
    ic(np.sort(act.colnames))
    psz = ClusterCatalog("psz2")
    # psz.catalog['coords'] = psz.coords
    # codex.catalog['coords'] = codex.coords
    xmass = ClusterCatalog(
        "xmass",
        xmass,
        base_cols=("C3ID", "RA_X", "DEC_X", "z"),
        label="Xmass",
        masscol="M200c",
    )

    # dwellings on the CHANCES selection function
    ref_level = args.ref
    if np.iterable(ref_level) and "max" in ref_level:
        Nref = int(ref_level[3:])
    else:
        Nref = 30
    matchsep = (30 if sample == "lowz" else 10) * u.arcmin
    matchsep_chances = (60 if sample == "lowz" else 30) * u.arcmin
    kwargs = dict(
        matching=[
            ["CHANCES", chances, matchsep_chances],
            ["Abell", abell, matchsep],
            ["CODEX", codex, matchsep],
            ["ACT-DR5", act, matchsep],
            ["PSZ2", psz, matchsep],
            ["LoVoCCS", lovoccs, matchsep],
            ["HIFLUGCS", hiflugcs, matchsep],
            ["MeerKAT", meerkat, matchsep],
        ],
        splus=splus,
        Nref=Nref,
    )
    if sample == "lowz":
        # we don't need to match codex here
        kwargs["matching"][2] = kwargs["matching"][1]
        kwargs["matching"][1] = ["WINGS", wings, 10 * u.arcmin]
        kwargs["matching"].pop(3)
        # required for join
        xmass.catalog.remove_columns(["coords", "lambda"])
        codex.catalog.remove_column("coords")
        join_keys = ["name", "ra", "dec", "M200c", "z"]
        lowz = join(
            xmass.catalog,
            codex.catalog,
            join_type="outer",
            keys=join_keys,
            table_names=["xmass", "codex"],
        )
        lowz = filter_sky(lowz, "ra", "dec")
        lowz["z"] = [zb if zb > 0 else z for zb, z in lowz["best_z", "z"]]

        print(np.sort(lowz.colnames))
        #
        # lowz.remove_columns(['coords'])
        # codex3.remove_column('coords')
        # lowz = join(
        #     lowz, codex3, join_type='outer', keys_left=join_keys,
        #     keys_right=['CODEX3','ra','dec','M200c','z_best'],
        #     table_names=['lowz','codex3'])
        # for key in ('z','M200c'):
        #     lowz[key] \
        #         = [v1 if v1 > 0 else v2
        #            for v1, v2 in lowz[f'{key}_lowz',f'{key}_codex3']]
        #
        print(np.sort(lowz.colnames))
        lowz = ClusterCatalog(
            "lowz",
            lowz,
            label="Lowz",
            masscol="M200c",
            # base_cols=('name','ra_lowz','dec_lowz','z')
        )
        lowz.catalog["M200c"] /= 1e14
        selfunc(args, lowz, ref_level, **kwargs)

        # crossmatch a few that are missing information
        # ramiss = [336.968, 96.595, 327.970, 36.467, 40.337]
        # decmiss = [-30.570, -53.696, -15.717, -41.909, -28.657]
        # cmiss = SkyCoord(ra=ramiss, dec=decmiss, unit='deg')
        # idx, sep, _ = cmiss.match_to_catalog_sky(eromapper.coords)
        # print(idx)
        # print(sep.to(u.arcmin))
        # print(lowz[join_keys][idx])

    if sample == "evolution":
        psz_selected = selfunc(args, psz, ref_level, **kwargs)
        # act_selected = selfunc(args, act, ref_level, **kwargs)
        # codex_selected = selfunc(args, codex, ref_level, **kwargs)

    return
    # to PSZ
    psz.catalog = filter_sky(psz.catalog, "RA", "DEC")
    ic(np.sort(psz.colnames))
    matches = crossmatch(chances, psz.catalog)
    chances["psz"] = chances["coords"].size * [25 * " "]
    chances["psz"][matches[0]] = psz.obj[matches[1]]
    chances["z_psz"] = np.zeros(chances["coords"].size)
    chances["z_psz"][matches[0]] = psz.z[matches[1]]
    chances["MSZ"] = np.zeros(chances["coords"].size)
    chances["MSZ"][matches[0]] = psz["MSZ"][matches[1]]
    if sample == "evolution":
        # lambins = np.array([1, 2, 3, 5, 8, 10, 20])
        mbins = np.logspace(0.3, 1.5, 6)
        ylim = (1, 30)
    else:
        mbins = np.logspace(-0.7, 1, 4)
        ylim = (0.5, 10)
    completeness(
        chances["z"][matches[0]],
        chances["MSZ"][matches[0]],
        psz["REDSHIFT"],
        psz["MSZ"],
        zbins,
        mbins,
        f"chances_{sample}",
        "psz",
        sample,
        ylim=ylim,
        ylabel="MSZ",
    )
    mask = (chances["MSZ"] > 0) & (chances["z"] > 0.16) & (chances["z"] < 0.17)
    print(chances[mask])

    # msk = (codex['best_z'] > 0.02) & (codex['best_z'] < 0.025) \
    #     & (codex['lambda'] > 20)
    # codex['best_z'].format = '.4f'
    # codex['lambda'].format = '.1f'
    # codex['ra'].format = '.5f'
    # codex['dec'].format = '.5f'
    # test = codex[msk]
    # test['sep'] = test['coords'].separation(
    #     SkyCoord(ra=np.mean(test['ra']), dec=np.mean(test['dec']), unit='deg'))
    # test['sep'] = test['sep'].to(u.arcmin)
    # test['sep'].format = '.2f'
    # print(test['id_cluster','ra','dec','sep','best_z','lambda'])

    return


def completeness(
    z,
    richness,
    z_ref,
    rich_ref,
    zbins,
    lambins,
    catlabel,
    reflabel,
    sample,
    output="",
    ylabel="Richness $\lambda$",
    xlim=None,
    ylim=None,
):
    z0 = (zbins[:-1] + zbins[1:]) / 2
    lam0 = (lambins[:-1] + lambins[1:]) / 2
    n = np.histogram2d(z, richness, (zbins, lambins))[0]
    n_ref = np.histogram2d(z_ref, rich_ref, (zbins, lambins))[0]
    completeness = n / n_ref
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(z_ref, rich_ref, c="C0", marker="o", label=reflabel)
    ax.scatter(z, richness, c="C1", marker="x", label=catlabel, zorder=2)
    if xlim is None:
        xlim = (0.01, 0.07) if sample == "lowz" else (0.05, 0.5)
    if ylim is None:
        ylim = (8, 100) if sample == "lowz" else (20, 250)
    ax.set(xlabel="Redshift $z$", ylabel=ylabel, yscale="log", xlim=xlim, ylim=ylim)
    # specify binning in the right panel
    fillkwargs = dict(color="0.6", alpha=0.5)
    if lambins[0] > ylim[0]:
        ax.fill_between(xlim, 2 * [ylim[0]], 2 * [lambins[0]], **fillkwargs)
    if lambins[-1] < ylim[1]:
        ax.fill_between(xlim, 2 * [lambins[-1]], 2 * [ylim[1]], **fillkwargs)
    if zbins[0] > xlim[0]:
        ax.fill_betweenx(ylim, 2 * [xlim[0]], 2 * [zbins[0]], **fillkwargs)
    if zbins[-1] < xlim[1]:
        ax.fill_betweenx(ylim, 2 * [zbins[-1]], 2 * [xlim[1]], **fillkwargs)
    for x in zbins:
        ax.axvline(x, ls="-", color="0.6")
    for y in lambins:
        ax.axhline(y, ls="-", color="0.6")
    if reflabel == "psz":
        x = np.linspace(0.07, 0.45, 10)
        y = 2 + 8 * x
        ax.plot(x, y, ls="-", lw=3, color="C9")
    ax.legend(fontsize=14, loc="upper left" if sample == "lowz" else "lower right")
    ax = axes[1]
    im = ax.pcolormesh(zbins, lambins, completeness.T, cmap="Greens", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label=f"{catlabel} / {reflabel}")
    for i, (zi, ni) in enumerate(zip(z0, n_ref)):
        for j, (lamj, nij) in enumerate(zip(lam0, ni)):
            cij = completeness[i][j]
            if np.isnan(cij):
                continue
            ax.text(
                zi,
                lamj,
                f"{cij:.2f}\n({nij:.0f})",
                ha="center",
                va="center",
                fontsize=12,
                color="w" if cij > 0.4 else "k",
            )
    ax.set(xlabel="Redshift $z$", yscale="log")
    for ax in axes:

        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    if not output:
        output = f"plots/completeness-zlam-{catlabel}-{reflabel}-{sample}.png"
    savefig(output, fig=fig)
    return completeness


def crossmatch(ref, cat, maxsep=30 * u.arcmin):
    dist = ref["coords"].separation(cat["coords"][:, None])
    mindist = np.min(dist, axis=0)
    closest = np.argmin(dist, axis=0)
    matches = mindist < maxsep
    return matches, closest[matches], mindist


def filter_sky(cat, racol, deccol):
    ra = cat[racol]
    dec = cat[deccol]
    if "coords" not in cat.colnames:
        cat["coords"] = SkyCoord(ra=ra, dec=dec, unit="deg")
    gal = cat["coords"].transform_to("galactic")
    cat["l"] = gal.l.value
    cat["b"] = gal.b.value
    mask = (dec > -80) & (dec < 5) & (np.abs(cat["b"]) > 20)
    # exclude superclusters
    shapley, horologium = supercluster_mask(ra, dec)
    ic(mask.size, mask.sum())
    mask = mask & ~shapley & ~horologium
    ic(mask.size, mask.sum())
    return cat[mask]


def load_splus(chances):
    splus = ascii.read("../splus/S-PLUS_footprint.csv", format="csv")
    splus.rename_columns(["RA", "DEC"], ["hms", "dms"])
    splus["coords"] = SkyCoord(
        ra=splus["hms"], dec=splus["dms"], unit=(u.hourangle, u.deg)
    )
    splus["ra"] = splus["coords"].ra.deg
    splus["dec"] = splus["coords"].dec.deg
    print(splus[:10])
    print(splus.colnames)
    # load targeted observations
    for name, ra, dec, in_splus, notes in chances[
        "name", "ra", "dec", "SPLUS", "SPLUS comments"
    ]:
        if in_splus == "NO" and "target" not in notes:
            continue
        try:
            in_splus[:3]
        except IndexError:
            continue
        if in_splus[:3] == "202":
            semester = in_splus
        else:
            try:
                semester = notes.split()[-2]
            except AttributeError:
                continue
        c = SkyCoord(ra=ra, dec=dec, unit=u.deg)
        # print(c.ra.to(u.hourangle)[:-3].replace('h', ':').replace('m', ':'))
        splus.add_row([f"{name}_{semester}", "--", "--", 2000, c, ra, dec])
    return splus


def plot_xmass_codex(xmass, codex):
    # compare CODEX and Xmass for low-z
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), width_ratios=(3, 2))
    axes = np.reshape(axes, -1)
    xlim = (0, 0.072)
    xmass["RA_X"][xmass["RA_X"] > 180] = xmass["RA_X"][xmass["RA_X"] > 180] - 360
    codex["ra"][codex["ra"] > 180] = codex["ra"][codex["ra"] > 180] - 360
    # Xmass
    kwargs = dict(
        c=xmass["lambda"],
        s=50,
        edgecolors="k",
        cmap="inferno_r",
        norm=LogNorm(vmax=100),
        alpha=0.8,
    )
    ax = axes[0]
    ax.scatter(xmass["RA_X"], xmass["DEC_X"], **kwargs, lw=2)
    ax.set(xlabel="RA", ylabel="DEC")
    ax = axes[1]
    c = ax.scatter(xmass["z"], xmass["M200c"], **kwargs, lw=2, label="Xmass")
    cbar = plt.colorbar(c, ax=ax, label="Richness")
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.set(xlabel="Redshift", ylabel="X-ray M200c", xlim=xlim, yscale="log")
    # CODEX
    j = (codex["best_z"] < 0.07) & (codex["dec"] <= 5)
    kwargs["c"] = codex["lambda"][j]
    kwargs["marker"] = "D"
    kwargs["s"] = 45
    kwargs["zorder"] = -1
    ax = axes[0]
    ax.scatter(codex["ra"][j], codex["dec"][j], **kwargs)
    axes[1].scatter(codex["z"][j], codex["M200c"][j], **kwargs, label="CODEX")
    axes[1].legend(fontsize=14, loc="lower right")
    output = "plots/xmass_with_richness.png"
    savefig(output, fig=fig)
    return


def selfunc(
    args,
    cat,
    ref_level=0.9,
    Nref=100,
    matching=None,
    footprints=None,
    splus=None,
    cmap="cmr.rainforest_r",
    cmin=0,
    cmax=1,
):
    """We first calculate a "selection function" using the entire sample
    over the full sky and then we register only those within the
    4MOST sky"""
    sky = (cat.dec > -80 * u.deg) & (cat.dec < 5 * u.deg) & (np.abs(cat.b) > 20 * u.deg)
    ic(sky.size, sky.sum())
    if args.sample == "evolution":
        zbins = np.linspace(0, 0.5, 12)
        zbins_analysis = np.array([0.07, 0.15, 0.25, 0.35, 0.45])
        zbins_analysis = np.linspace(0.07, 0.45, 6)
    else:
        zbins = np.linspace(0, 0.07, 12)
        zbins_analysis = np.array([0, 0.01, 0.03, 0.05, 0.07])
        # zbins_analysis = n
    z0 = (zbins[:-1] + zbins[1:]) / 2
    z0_analysis = (zbins_analysis[:-1] + zbins_analysis[1:]) / 2
    zmask = (cat.z >= zbins_analysis[0]) & (cat.z <= zbins_analysis[-1])
    ic(zmask.size, zmask.sum())
    cmap = cmr.get_sub_cmap(cmap, cmin, cmax)
    # let's first try to get the selection function
    if cat.name in ("act-dr5", "psz2"):
        logmbins = np.linspace(-0.7, 1.4, 50)
    elif cat.name == "codex":
        logmbins = np.linspace(0.7, 2.7, 50)
    elif cat.name == "lowz":
        logmbins = np.linspace(-1.3, 1.7, 50)
    logm0 = (logmbins[:-1] + logmbins[1:]) / 2
    mbins = 10**logmbins
    m0 = 10**logm0
    # maxima
    maxima = binstat(cat.z, np.log10(cat.mass), statistic=np.max, bins=zbins).statistic
    ic(maxima)
    n_zm = np.histogram2d(cat.z, np.log10(cat.mass), (zbins, logmbins))[0]
    ncum_zm = np.cumsum(n_zm, axis=1) / np.sum(n_zm, axis=1)[:, None]
    # fit an error function to each redshift bin
    # fit = [curve_fit(generalized_erf, logm0, ncum_z, p0=(0.1,0.4))
    #        for ncum_z in ncum_zm]
    # fitcov = [i[1] for i in fit]
    # fit = [i[0] for i in fit]
    # ic(fit)
    # model = np.array([generalized_erf(logm0, *x) for x in fit])
    # ic(model.shape)
    # froot = lambda x, *args: generalized_erf(x, *args) - ref_level
    # #ref = [root_scalar(froot, args=tuple(f), x0=0.3, x1=0.4) for f in fit]
    # logmsel = [root(froot, 0.3, tuple(f)).x[0] for f in fit]
    # ic(logmsel)
    # for now... this should be the same as the above but the original
    # logmsel fails for ref_level >~0.5
    if np.iterable(ref_level) and "max" in ref_level:
        ref_has_max = True
        ic(zbins_analysis)
        ic(np.histogram(cat.z[sky], zbins_analysis)[0])
        # I don't understand why binned_statistic doesn't work
        logmsel = np.zeros(z0_analysis.size)
        for i in range(zbins_analysis.size - 1):
            j = (cat.z[sky] >= zbins_analysis[i]) & (cat.z[sky] < zbins_analysis[i + 1])
            logmsel[i] = np.log10(np.sort(cat.mass[sky][j])[-Nref // 4])
    else:
        ref_has_max = False
        # raw percentiles
        per = lambda x: np.percentile(x, 100 * ref_level)
        per = binstat(cat.z, np.log10(cat.mass), statistic=per, bins=zbins).statistic
        ic(per)
        logmsel = per
    msel = 10 ** np.array(logmsel)
    ic(msel)
    # let's try to fit a polynomial to our "mass limit"
    zfit = z0_analysis if ref_has_max else z0
    pfit = np.squeeze(polynomial.polyfit(zfit, logmsel, 3))
    ic(pfit)

    def p(x):
        """log-space polynomial fit to the poor-man's selection function"""
        p = Polynomial(pfit)
        return 10 ** p(x)

    # let's find a cut that will give us Nref clusters in the desired
    # redshift range
    msk = sky & zmask
    cref = 0.5
    ic(Nref)
    while (msk & (cat.mass > cref * p(cat.z))).sum() > Nref:
        cref *= 1.01
    cref_lo = 0.5 * cref if args.sample == "evolution" else 0.2 * cref
    massmask = cat.mass > cref * p(cat.z)
    massmask_low = ~massmask & (cat.mass > cref_lo * p(cat.z))
    ic(cref, massmask.sum())
    # plot!
    # fig, axes = plt.subplots(
    #     1, 3, figsize=(18,5), constrained_layout=True)
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    spec = fig.add_gridspec(2, 2, width_ratios=[0.4, 0.6])
    axes = [
        fig.add_subplot(spec[0, 0]),
        fig.add_subplot(spec[0, 1]),
        fig.add_subplot(spec[1, :]),
    ]
    ic(axes)
    # percentiles
    ax = axes[0]
    mlabel = "SZ" if args.sample == "evolution" else "X"
    ax.set(ylabel=f"$M_\mathrm{{{mlabel}}}$ ($10^{{14}}$ M$_\odot$)")
    im = ax.pcolormesh(zbins, mbins, ncum_zm.T, vmin=0, vmax=1, cmap=cmap)
    plt.colorbar(
        im, ax=ax, label=f"$N(<M_\mathrm{{{mlabel}}}|z)$ / $N(z)$", fraction=0.05, pad=0
    )
    if ref_has_max:
        c = "w"
    else:
        c = "k" if ref_level < 0.5 else "w"
    ax.plot(zfit, msel, f"{c}-", lw=4)
    ax.plot(z0, p(z0), f"{c}--", lw=4)
    # data points
    good = sky & zmask & massmask
    ra = cat.ra.copy()
    # if cat.name == 'act-dr5':
    ra[ra > 180 * u.deg] -= 360 * u.deg
    ic(ra)
    for ax, x, y in zip(axes[1:], (cat.z, ra), (cat.mass, cat.dec)):
        # ax.scatter(x[~good], y[~good], marker=".", c="0.6", s=3, zorder=-2)
        ax.scatter(
            x[sky & zmask & ~massmask],
            y[sky & zmask & ~massmask],
            marker=".",
            c="0.6",
            s=3,
            zorder=-2,
        )
        ax.plot(x[good], y[good], "o", mec="C0", mfc="none", mew=3, zorder=0)
        msk = sky & zmask & massmask_low
        ax.scatter(x[msk], y[msk], marker="o", c="C9", s=4, zorder=-1)
        msk = sky & zmask & ~(massmask | massmask_low)
        ax.scatter(x[msk], y[msk], marker="o", c="C6", s=3, zorder=-1)
    ax = axes[1]
    ax.plot(zfit, msel, "C3-", lw=3, zorder=0)
    ax.plot(z0, p(z0), "C3--", lw=3, zorder=0)
    ax.plot(z0, cref * p(z0), "C3-", lw=2, zorder=0)
    ax.plot(z0, cref_lo * p(z0), "C3-", lw=1, zorder=0)
    if args.sample == "lowz":
        axes[0].xaxis.set_major_locator(ticker.MultipleLocator(0.02))
        axes[1].set(xlim=(0, 0.075))
    # for reference
    msk_lo = (cat.mass > cref * p(cat.z)) & (cat.z > 0) & (cat.z < 0.07)
    # number of clusters selected in each bin
    for zb in zbins_analysis:
        ax.axvline(zb, ls="--", color="k", lw=1)
    Nz = np.histogram(cat.z[sky & zmask & massmask], zbins_analysis)[0]
    # Nz = np.histogram(cat.z[sky & zmask], zbins_analysis)[0]
    ytext = 20 + 5 * (args.sample == "lowz")
    for zi, Nz_i in zip(z0_analysis, Nz):
        ax.annotate(
            f"{Nz_i:.0f}", xy=(zi, ytext), ha="center", va="center", fontsize=12
        )
    if args.sample == "evolution":
        ax.set(xlim=(0, 0.5))
    if args.sample == "evolution":
        ylim = (5, 300) if cat.name == "codex" else (1, 24)
    else:
        ylim = (0.05, 40)
    for ax in axes[:2]:
        ax.set(xlabel="Redshift", yscale="log", ylim=ylim)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    shapley, horologium = supercluster_mask(
        what="patches",
        shapley_kwargs=dict(
            edgecolor="C9", facecolor="none", lw=3, zorder=-10, alpha=1
        ),
        horologium_kwargs=dict(edgecolor="C6", facecolor="none", lw=3, zorder=-10),
    )
    ic(shapley, horologium)
    # axes[2].add_collection(shapley)
    # axes[2].add_collection(horologium)
    for sh in shapley:
        axes[2].add_patch(sh)
    for ho in horologium:
        axes[2].add_patch(ho)
    axes[2].set(xlabel="RA (deg)", ylabel="Dec (deg)", ylim=(-90, 25))

    tbl = Table(
        {
            "name": cat.obj,
            "ra": cat.ra,
            "dec": cat.dec,
            "z": cat.z,
            "gal_l": cat.l,
            "mass": cat.mass,
            "complete": massmask,
        }
    )
    # tbl = tbl[msk_ev | msk_lo]
    tbl = tbl[sky & zmask]
    N = tbl["ra"].size
    tbl["ra"].format = ".5f"
    tbl["dec"].format = ".5f"
    tbl["z"].format = ".3f"
    tbl["gal_l"].format = ".2f"
    tbl["mass"].format = ".2f"
    tbl.sort("z")
    tbl["lowmass"] = (tbl["mass"] >= cref_lo * p(tbl["z"])) & ~tbl["complete"]
    if args.sample == "lowz":
        # with_zmass = (tbl['mass'] > 0) & (tbl['z'] > 0)
        tbl["r200"] = (
            3
            * 1e14
            * tbl["mass"]
            * u.Msun
            / (800 * np.pi * cosmo.critical_density(tbl["z"]))
        ) ** (1 / 3)
        tbl["r200"] = tbl["r200"].to(u.Mpc)
        tbl["d200"] = tbl["r200"] / cosmo.kpc_proper_per_arcmin(tbl["z"])
        tbl["d200"] = tbl["d200"].to(u.arcmin)
        for key in "rd":
            tbl[f"{key}200"].format = ".2f"
    tra = np.array(tbl["ra"])
    # if cat.name == 'act-dr5':
    tra[tra > 180] -= 360
    coords = SkyCoord(ra=tbl["ra"], dec=tbl["dec"], unit="deg")
    # for the plot
    labels = []
    lines = []
    # match SPLUS
    if splus:
        sep = coords.separation(splus["coords"][:, None])
        tbl["N_SPLUS"] = np.sum(sep < 5 * tbl["d200"], axis=0)
        d200 = tbl["d200"].to(u.deg).value
        # this assumes 1 sq.deg. per SPLUS field
        tbl["f_SPLUS"] = tbl["N_SPLUS"] / (np.pi * (5 * d200) ** 2)
        tbl["f_SPLUS"].format = ".2f"
        minsep = np.min(sep, axis=0)
        closest = np.argmin(sep, axis=0)
        ic(coords.shape, sep.shape, minsep.shape)
        ic(minsep)
        tbl["S-PLUS"] = [
            splus["NAME"][cl] if ms < size else ""
            for cl, ms, size in zip(closest, minsep, 5 * d200 * u.deg)
        ]
        in_splus = tbl["S-PLUS"] != ""
        print(f"{in_splus.sum()}/{N} clusters in S-PLUS")
        sample_masks = [
            (in_splus & tbl["complete"]),
            (in_splus & tbl["lowmass"]),
            (in_splus & ~tbl["complete"] & ~tbl["lowmass"]),
        ]
        n0, n1, n2 = [m.sum() for m in sample_masks]
        kwargs = dict(marker=".", zorder=10)
        if args.sample == "lowz":
            spluspts = []
            for i, (m, c, s) in enumerate(
                zip(sample_masks, ("C3", "C3", "C2"), (25, 35, 25))
            ):
                kwargs["c"] = c
                kwargs["s"] = s
                if i >= 1:
                    kwargs["label"] = "_none_"
                pts = axes[1].scatter(tbl["z"][m], tbl["mass"][m], **kwargs)
                if i > 0:
                    spluspts.append(pts)
                axes[2].scatter(tra[m], tbl["dec"][m], **kwargs)
            lines.append(tuple(spluspts))
            labels.append(f"in S-PLUS (N={n0}/{n1}/{n2})")
    if matching is not None:
        matches = {}
        for (cname, mcat, maxsep), marker in zip(matching, "osx^v<>8D"):
            if cname.lower() == cat.label:
                continue
            sep = coords.separation(mcat["coords"][:, None])
            pairs = sep < maxsep
            minsep = np.min(sep, axis=0)
            closest = np.argmin(sep, axis=0)
            tbl[cname] = [
                mcat["name"][cl] if ms < maxsep else ""
                for cl, ms in zip(closest, minsep)
            ]
            matches[cname] = tbl[cname] != ""
            if cname == "Abell":
                continue
            n0 = (matches[cname] & tbl["complete"]).sum()
            n1 = (matches[cname] & tbl["lowmass"]).sum()
            n2 = (matches[cname] & ~(tbl["complete"] | tbl["lowmass"])).sum()
            kwargs = dict(ms=12, mfc="none", mew=2, zorder=0)
            labels.append(f"in {cname} (N={n0}/{n1}/{n2})")
            if cname == "WINGS":
                n0 = (matches[cname] & tbl["complete"]).sum()
                n1 = (matches[cname] & tbl["lowmass"]).sum()
                n2 = (matches[cname] & ~(tbl["complete"] | tbl["lowmass"])).sum()
                kwargs = dict(ms=15, mfc="none", mew=2, zorder=0)
                labels.append(f"in {cname} (N={n0}/{n1}/{n2})")
                lines.append(
                    axes[1].plot(
                        tbl["z"][matches[cname]],
                        tbl["mass"][matches[cname]],
                        f"C1{marker}",
                        **kwargs,
                    )[0]
                )
            if cname != "CHANCES":
                continue
            lines.append(
                axes[1].plot(
                    tbl["z"][matches[cname]],
                    tbl["mass"][matches[cname]],
                    f"k{marker}",
                    **kwargs,
                )[0]
            )
            axes[2].plot(
                tra[matches[cname]], tbl["dec"][matches[cname]], f"k{marker}", **kwargs
            )
            p0 = np.sum(pairs, axis=0)
            p1 = np.sum(pairs, axis=1)
            ic(pairs.shape)
            ic(p0.size, np.unique(p0, return_counts=True))
            ic(np.unique(p1, return_counts=True))
            # any CHANCES clusters not in this list?
            minsep2 = np.min(sep, axis=1)
            ic(mcat["name"].size, minsep2.size)
            ic(np.sort(minsep2))
            closest2 = np.argmin(sep, axis=1)
            missing = minsep2 > maxsep
            ic(missing.size, missing.sum())
            mcat["dist"] = minsep2
            print(f"{missing.sum()} missing CHANCES clusters:")
            if args.sample == "lowz":
                miss = mcat[
                    "name",
                    "z",
                    "ra",
                    "dec",
                    "Lambda",
                    "S-PLUS",
                    "Catalogue",
                    "CODEXID",
                    "dist",
                ][missing]
                print(miss)
                miss.write("lowz_chances_missing.csv", format="csv", overwrite=True)
            a3301 = tbl[
                "name",
                "ra",
                "dec",
                "z",
            ]
            mra = mcat["ra"].copy()
            mra[mra > 180] -= 360
            axes[2].plot(
                mra[missing],
                mcat["dec"][missing],
                "kD",
                ms=10,
                mfc="none",
                label="Additional CHANCES",
            )
            axes[2].legend(loc="lower left", fontsize=14)
            print(tbl[np.isin(tbl["name"], ["93192518"])])
            sc = SkyCoord(ra=139.50507, dec=-12.14024, unit="deg")
            g = sc.transform_to("galactic")
            print(g)
            # print(tbl['name'][minsep2[:,]])
            # for i in np.arange(mcat['name'].size)[missing]:
            #     print(i, mcat['name'][i], closest2[i])
        cluster_names = tra.size * [""]
        for i in range(tra.size):
            cluster_names[i] = [
                tbl[cname][i] for cname, _, _ in matching if tbl[cname][i] != ""
            ]
            cluster_names[i] = (
                cluster_names[i][0] if len(cluster_names[i]) > 0 else tbl["name"][i]
            )
        tbl.add_column(cluster_names, name="Cluster Name", index=1)
    legend_kwargs = dict(fontsize=12, loc="lower right")
    if args.sample == "lowz":
        legend_kwargs["handler_map"] = {tuple: HandlerTuple(ndivide=None)}
    axes[1].legend(lines, labels, **legend_kwargs)
    if footprints is not None:
        for fp in footprints:
            tbl[f"in_{fp.name}"] = fp.in_footprint(tbl["ra"], tbl["dec"])
    # save figure and table
    output = f"masscounts_{args.sample}_{cat.name}"
    if ref_has_max:
        output = f"{output}_{ref_level}"
    else:
        output = f"{output}_{100*ref_level:.0f}"
    savefig(f"plots/{output}.png", fig=fig, tight=False)
    # from previous analyses
    if cat.name in ("act-dr5", "psz2"):
        if cat.name == "psz2":
            tbl["mass"] *= 10**0.13
        for key in "mcrd":
            tbl[f"{key}200"] = np.zeros(tbl["name"].size)
        with_zmass = (tbl["mass"] > 0) & (tbl["z"] > 0)
        (
            tbl["m200"][with_zmass],
            tbl["c200"][with_zmass],
            tbl["r200"][with_zmass],
            tbl["d200"][with_zmass],
        ) = calculate_m200_from_m500(
            1e14 * tbl["mass"][with_zmass], tbl["z"][with_zmass], cosmo=cosmo
        )
        tbl["m200"] /= 1e14
        for key in "mcrd":
            tbl[f"{key}200"].format = ".2f"
    tbl.write(f"output/{output}.txt", format="ascii.fixed_width", overwrite=True)
    return tbl


def supercluster_mask(
    ra=None,
    dec=None,
    what="masks",
    shapley_kwargs={},
    horologium_kwargs={},
    wrap_ra=180,
):
    """Return masks or lines to plot.

    If ``what==lines`` the output can be used as follows:

    .. code-block::

        shapley, horologium = supercluster_mask(ra, dec, what='lines')
        for sh in shapley:
            plt.plot(*sh)
        for ho in horologium:
            plt.plot(*ho)

    if ``what==patches`` then:

    .. code-block::

        shapley, horologium = supercluster_mask(ra, dec, what='patches')
        ax.add_collection(shapley)
        ax.add_collection(horologium)
    """
    assert what in ("lines", "masks", "patches")
    if what == "masks":
        err = ""
        if ra is None or dec is None:
            err = "both ``ra`` and ``dec`` must be specified if ``what==masks``"
        elif ra.size != dec.size:
            err = "both ``ra`` and ``dec`` must have the same size"
        if err:
            raise ValueError(err)
    ra_shapley = [192, 207]
    dec_shapley = [-36, -26]
    ra_horo = [[47, 70], [44, 55], [44, 60]]
    dec_horo = [[-62, -53], [-53, -40], [-40, -35]]
    if wrap_ra != 0:
        ra_shapley = [x - 2 * wrap_ra if x > wrap_ra else x for x in ra_shapley]
        ra_horo = [
            [xi - 2 * wrap_ra if xi > wrap_ra else xi for xi in x] for x in ra_horo
        ]
    if what == "masks":
        shapley = (
            (ra > ra_shapley[0])
            & (ra < ra_shapley[1])
            & (dec > dec_shapley[0])
            & (dec < dec_shapley[1])
        )
        horologium = np.array(
            np.max(
                [
                    (ra > ra_i[0])
                    & (ra < ra_i[1])
                    & (dec > dec_i[0])
                    & (dec < dec_i[1])
                    for ra_i, dec_i in zip(ra_horo, dec_horo)
                ],
                axis=0,
            ),
            dtype=bool,
        )
    elif what == "lines":
        shapley = [
            [ra_shapley[0], ra_shapley[1], ra_shapley[1], ra_shapley[0]],
            [dec_shapley[0], dec_shapley[1], dec_shapley[0], dec_shapley[1]],
        ]
        horologium = [
            [
                [ra_i[0], ra_i[1], ra_i[1], ra_i[0]],
                [dec_i[0], dec_i[1], dec_i[0], dec_i[1]],
            ]
            for ra_i, dec_i in zip(ra_horo, dec_horo)
        ]
    elif what == "patches":
        shapley = [
            Rectangle(
                (ra_shapley[0], dec_shapley[0]),
                ra_shapley[1] - ra_shapley[0],
                dec_shapley[1] - dec_shapley[0],
                **shapley_kwargs,
            )
        ]
        horologium = [
            Rectangle(
                (ra_i[0], dec_i[0]),
                ra_i[1] - ra_i[0],
                dec_i[1] - dec_i[0],
                **horologium_kwargs,
            )
            for ra_i, dec_i in zip(ra_horo, dec_horo)
        ]
        # shapley = PatchCollection(shapley)
        # horologium = PatchCollection(horologium)
    return shapley, horologium


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", choices=("evolution", "lowz"))
    parser.add_argument(
        "--catalog", type=str, default="aux/xray/codex_eromapper_catalog_fixedcodex.fit"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ref", default="0.9")
    args = parser.parse_args()
    if not args.debug:
        ic.disable()
    if "max" not in args.ref:
        args.ref = args.ref = float(args.ref)
    return args


main()
