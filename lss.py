from argparse import ArgumentParser
from astropy import units as u, constants as c
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
from astropy.table import Table, vstack
import cmasher as cmr
from colossus.cosmology import cosmology
from colossus.halo.mass_adv import changeMassDefinitionCModel
from datetime import date
from glob import glob
from icecream import ic
from matplotlib import pyplot as plt, ticker
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle
import numpy as np
import os
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
import seaborn as sns
from time import time
from tqdm import tqdm

import ligo.skymap.plot
from profiley.nfw import NFW
from plottery.plotutils import savefig, update_rcParams
from profiley.helpers.spherical import radius_from_mass

from astro.clusters import ClusterCatalog
from astro.footprint import Footprint

from tools import scalebar_label

cosmology.setCosmology("planck18")
update_rcParams()

light = (c.c).to(u.km / u.s).value


def main():
    args = parse_args()
    if args.sample == "all":
        lowz = ClusterCatalog("chances-lowz")
        evol = ClusterCatalog("chances-evol")
        chances = ClusterCatalog(
            "chances", vstack([lowz.catalog, evol.catalog]), masscol="m200"
        )
    else:
        chances = ClusterCatalog(f"chances-{args.sample[:4]}")
    chances.catalog["sigma"] = velocity_dispersion_from_mass(chances)
    chances.catalog["sigma"].format = "%.0f"
    chances, cat = find_subclusters(args, chances)
    print(chances)
    cat = add_r200(cat)
    print(cat)
    if args.sample == "all":
        infall_mass_function(args, chances, cat, plottype="hist")
        infall_mass_function(args, chances, cat, plottype="hist", sigma_clip=3)
        infall_mass_function(args, chances, cat, plottype="points", sigma_clip=3)
        phase_space(args, chances, cat, yaxis="sigma")
        phase_space(args, chances, cat, yaxis="km/s")
    else:
        wrap_plot_sky(args, chances, cat, show_neighbors=False)
        wrap_plot_sky(args, chances, cat, show_neighbors=True, suffix="neighbors")
    chances.catalog.sort(f"N_{args.catalog}")
    # print(chances)


def infall_mass_function(
    args, chances, cat, hide_main=True, plottype="kde", sigma_clip=0
):
    z_main = np.array([chances["z"][chances["name"] == cl][0] for cl in cat["chances"]])
    m_main = 1e14 * np.array(
        [chances["m200"][chances["name"] == cl][0] for cl in cat["chances"]]
    )
    sigma_main = np.array(
        [chances["sigma"][chances["name"] == cl] for cl in cat["chances"]]
    )[:, 0]
    fig, ax = plt.subplots(layout="constrained")
    logmubins = np.arange(-2.6, 0.4, 0.2)
    mubins = 10**logmubins
    mu = cat["m200"] / m_main
    # kde = gaussian_kde(np.log10(cat["m200"] / m_main), bw_method=0.1)
    # logx = np.arange(-3, 0.3, 0.1)
    # ax.plot(10**logx, kde(logx), "k-")
    if hide_main:
        mask = cat["is_main"] == 0
    else:
        mask = np.ones(cat["is_main"].size, dtype=bool)
    masks = [mask, mask & (z_main <= 0.07), mask & (z_main > 0.07)]
    labels = ["All", "Low-z", "Evolution"]
    colors = ["k", "C0", "C3"]
    markers = ["^", "s", "o"]
    masks_main = [chances["z"] < 1, chances["z"] <= 0.07, chances["z"] > 0.07]
    if plottype in ("hist", "points"):
        for i, (mask, label, c, m) in enumerate(zip(masks, labels, colors, masks_main)):
            if sigma_clip:
                vcut = np.abs(cat["vpec (km/s)"]) < sigma_clip * sigma_main
                mask = mask & vcut
            # normalized by the number of CHANCES clusters
            N = np.histogram(mu[mask], mubins)[0]
            n = N / m.sum()
            # draw a normalized histogram of data that have already been binned
            if plottype == "hist" or label == "All":
                ax.hist(
                    mubins[:-1],
                    mubins,
                    histtype="step",
                    weights=n,
                    ls="-",
                    lw=2,
                    color=c,
                )
                ax.plot([], [], ls="-", lw=3, color=c, label=label)
            else:
                nerr = N**0.5 / m.sum()
                nerr = n * (1 / N + 1 / m.sum()) ** 0.5
                uplims = N == 0
                n[uplims] = 1 / m.sum()
                nerr[uplims] = n[uplims] - 0.009
                print(label, nerr, uplims)
                xmu = 10 ** ((logmubins[:-1] + logmubins[1:]) / 2)
                xmu = xmu * (1 + 0.03 * (-1) ** i)
                ax.errorbar(
                    xmu[~uplims],
                    n[~uplims],
                    nerr[~uplims],
                    color=c,
                    fmt=markers[i],
                    ms=5 + i,
                    mew=2,
                    label=label,
                )
                if uplims.sum():
                    ax.errorbar(
                        xmu[uplims],
                        n[uplims],
                        nerr[uplims],
                        color=c,
                        fmt=",",
                        mew=2,
                        uplims=True,
                    )
    elif plottype == "smooth":
        logmu = (logmubins[:-1] + logmubins[1:]) / 2
        n = CubicSpline(logmu, np.histogram(mu[mask], bins=mubins)[0])
        nlo = CubicSpline(
            logmu, np.histogram(mu[mask & (z_main <= 0.07)], bins=mubins)[0]
        )
        nev = CubicSpline(
            logmu, np.histogram(mu[mask & (z_main > 0.07)], bins=mubins)[0]
        )
        x = np.logspace(logmu[0], logmu[-1], 1000)
        ax.plot(x, n(np.log(x)), "k-", label="All")
        ax.plot(x, nlo(np.log(x)), "C0", label="Low-z")
        ax.plot(x, nev(np.log(x)), "C3", label="Evolution")
    elif plottype == "kde":
        kw_kde = dict(ax=ax, log_scale=True, bw_method=0.1)
        sns.kdeplot(mu[mask], color="k", label="All", **kw_kde)
        sns.kdeplot(mu[mask & (z_main <= 0.07)], color="C0", label="Low-z", **kw_kde)
        sns.kdeplot(mu[mask & (z_main > 0.07)], color="C3", label="Evolution", **kw_kde)
    # opaque borders for both histograms
    # ax.hist(mu[mask & (z_main <= 0.07)], mubins, color="C0", lw=0.5, histtype="step")
    # ax.hist(mu[mask & (z_main > 0.07)], mubins, color="C3", lw=0.5, histtype="step")
    ax.legend(fontsize=12)
    ax.set(
        xlabel="$\mu\equiv M_{200}^\mathrm{infalling}/M_{200}^\mathrm{main}$",
        ylabel="$N(\mu)d\mu$ per main cluster",
        xscale="log",
        yscale="log",
    )
    ax.set_xlim(2e-3, 2)
    if plottype == "points":
        ax.set_ylim(0.007, 4)
    ax.set_xticks(np.logspace(-2, 0, 3), ["0.01", "0.1", "1"])
    ax.set_yticks(np.logspace(-2, 0, 3), ["0.01", "0.1", "1"])
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    output = "plots/lss/infall_mass_function"
    if sigma_clip:
        output = f"{output}_{plottype}_{sigma_clip}vclip"
    savefig(f"{output}.pdf", fig=fig, tight=False)


def phase_space(
    args, chances, cat, hide_main=True, sigma_clip=0, yaxis="sigma", show_histogram=True
):
    cat.sort("best_z")
    # sort by something random so the figure is not dominated by any single redshift
    cat.sort("dist (r200)")
    z_main = np.array([chances["z"][chances["name"] == cl] for cl in cat["chances"]])[
        :, 0
    ]
    sigma_main = np.array(
        [chances["sigma"][chances["name"] == cl] for cl in cat["chances"]]
    )[:, 0]
    r200_main = np.array(
        [chances["r200"][chances["name"] == cl] for cl in cat["chances"]]
    )[:, 0]
    m200_main = np.array(
        [chances["m200"][chances["name"] == cl] for cl in cat["chances"]]
    )[:, 0]
    if show_histogram:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9, 7),
            sharey=True,
            width_ratios=(4, 1.2),
            layout="constrained",
        )
        ax, hax = axes
    else:
        fig, ax = plt.figure(figsize=(9, 6), layout="constrained")
        axes = ax
    if hide_main:
        mask = cat["is_main"] == 0
    else:
        mask = np.ones(cat["is_main"].size, dtype=bool)
    if sigma_clip > 0:
        mask = mask & (
            light * np.abs(cat["best_z"] - z_main) / (1 + z_main)
            < sigma_clip * sigma_main
        )
    yval = cat["vpec (km/s)"]
    if yaxis == "sigma":
        yval = yval / sigma_main
    else:
        yval = yval / 1e3
    # note that changing axes limits or figure size will change the symbol size
    s_factor = 34000
    im = ax.scatter(
        cat["dist (r200)"][mask],
        yval[mask],
        # s=cat["m200"][mask] / 1e12,
        s=s_factor * (cat["r200"][mask] / r200_main[mask]) ** 2 / 30,
        c=z_main[mask],
        alpha=0.5,
    )
    # escape velocities
    i = np.argmin(np.abs(m200_main - 10))
    nfw = NFW(1e14 * m200_main[i], 5, z_main[i], overdensity=200)
    r = np.linspace(0, 6, 1000)
    q = nfw.cosmo.Om0 / 2 - nfw.cosmo.Ode0
    # print(nfw.mass)
    # r_eq = ((c.G * nfw.mass * u.Msun / q * cosmo.H(nfw.z) ** 2) ** (1 / 3)).to(u.Mpc)
    # print(r_eq)
    vesc = nfw.escape_velocity(r * r200_main[i], log_rmax=np.log10(r_eq))
    print(i, 1e14 * m200_main[i], z_main[i])
    print(np.squeeze(vesc), sigma_main[i])
    ax.plot(r, vesc / sigma_main[i], "k-", lw=1.5)
    ax.plot(r, -vesc / sigma_main[i], "k-", lw=1.5)
    # use this to find the size that gives a radius of 1 in data units
    # ax.scatter(1, -3, s=34000, c="k")
    # ax.grid()
    # circles = [Circle((d, y), s, color=)]
    if show_histogram:
        # beware this should change if axis limits change
        if yaxis == "sigma":
            vbins = np.arange(-6, 6, 1e-3)
        else:
            vbins = np.arange(-10, 10, 1e-3)
        vx = (vbins[1:] + vbins[:-1]) / 2
        hmask = mask & (yval >= vx[0]) & (yval <= vx[-1])
        smooth = 0.2
        hist = np.histogram(yval[hmask], vbins)[0]
        kde = gaussian_kde(yval[hmask], smooth)
        hax.plot(kde(vx), vx, "k-")
        # evolution
        evol = z_main > 0.07
        kde_evol = gaussian_kde(yval[hmask & evol], smooth)
        hax.plot(kde_evol(vx), vx, "C0-", label="Evolution")
        kde_lowz = gaussian_kde(yval[hmask & ~evol], smooth)
        hax.plot(kde_lowz(vx), vx, "C3-", label="Low-z")
        hax.set_xticks([])
        hax.legend(loc="lower right", fontsize=14)
        cbar = plt.colorbar(im, ax=axes, orientation="horizontal", location="top")
        cbar.ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    else:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    cbar.set_label("Main cluster redshift")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set(
        xlabel="$r_\mathrm{proj}/r_{200}$",
        xlim=(0, 5.1),
    )
    output = "plots/lss/phase_space"
    if yaxis == "sigma":
        ax.set(ylabel="$v_\\mathrm{pec}/\\sigma_\\mathrm{main}$", ylim=(-6, 6))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        output = f"{output}_sigma"
    else:
        ax.set(ylabel="$v_\\mathrm{pec}$ ($10^3$ km s$^{-1}$)")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    if show_histogram:
        output = f"{output}_hist"

    savefig(f"{output}.pdf", fig=fig, tight=False)
    return


def add_r200(cat):
    """Use the McClintock+18 relation for M200m and convert to M200c with colossus"""
    rich = 1.21 * cat["lambda"]
    z = cat["best_z"]
    # McClintock+18 eq 52 (M200m)
    M0 = 10**14.49
    l0 = 40
    z0 = 0.35
    F = 1.36
    G = -0.30
    m200m = M0 * (rich / l0) ** F * ((1 + z) / (1 + z0)) ** G
    m200 = [
        changeMassDefinitionCModel(
            cosmo.h * m, zi, "200m", "200c", c_model="ishiyama21"
        )
        for m, zi in zip(m200m, z)
    ]
    m200, r200, c200 = np.transpose(m200)
    m200 = m200 / cosmo.h
    r200 = r200 / (1000 * cosmo.h)
    d200 = ((r200 * u.Mpc) * cosmo.arcsec_per_kpc_proper(z)).to(u.deg).value
    cat["m200"] = m200
    cat["r200"] = r200
    cat["d200"] = d200
    cat["m200"].format = ".2e"
    cat["r200"].format = ".2f"
    cat["d200"].format = ".4f"
    return cat


def wrap_plot_sky(
    args, chances, cat, fsize=1.5, show_main=True, show_neighbors=True, suffix=""
):
    """fsize is the size of the figure in units of 5r200"""

    # colors = [[0, 0, 1], [0, 0, 0.5], [0.1, 0, 0.1], [0.5, 0, 0], [1, 0, 0]]
    # nodes = [0, 0.25, 0.5, 0.75, 1]
    # colors = [[]]
    # cmap = LinearSegmentedColormap.from_list("redshift", list(zip(nodes, colors)))
    # this is the same as used in the superclusters, adapted from coolwarm
    colors = [[0, 0, 1], [0.4, 0.4, 0.4], [1, 0, 0]]
    nodes = [0, 0.5, 1]
    cmap = LinearSegmentedColormap.from_list("vmap", list(zip(nodes, colors)))
    vmax = 8000

    chances.catalog["coord"] = SkyCoord(
        ra=chances["ra"], dec=chances["dec"], unit="deg"
    )

    # overlapping clusters are handled individually
    if args.sample == "lowz":
        plot_sky_a119_a147_a168(args, chances, cat, cmap)
        plot_sky_a3651_a3667(args, chances, cat, cmap)
        return

    for i, cl in tqdm(enumerate(chances), total=chances.size):
        center = get_center(cl)
        radius = get_radius(cl, fsize * 5)
        plot_sky(
            args,
            chances,
            cl,
            cat,
            center,
            radius,
            cmap,
            show_neighbors=show_neighbors,
            suffix=suffix,
        )
    fig, ax = plt.subplots(figsize=(20, 1.2), constrained_layout=True)
    fig.colorbar(
        ScalarMappable(norm=Normalize(-vmax, vmax), cmap=cmap),
        cax=ax,
        label="$v_\\mathrm{pec}$ (km s$^{-1}$)",
        orientation="horizontal",
    )
    output = "plots/lss/lss_colorbar"
    for ext in ("pdf", "png"):
        savefig(f"{output}.{ext}", fig=fig, tight=False)
    return


def plot_sky_a119_a147_a168(args, chances, cat, cmap):
    fig, ax = plot_sky(
        args,
        chances,
        chances[chances["name"] == "Abell 0119"][0],
        cat,
        "17d 0d",
        "6 deg",
        cmap,
        z0=0.044,
        annotate=False,
        show_neighbors=True,
        hide_coordinates=False,
        sigma_clip=False,
        save=False,
    )
    # ax.text_coord("Abell 168", xy=())
    bar = (10 * u.Mpc * cosmo.arcsec_per_kpc_comoving(0.056)).to(u.deg)
    bar = ax.scalebar((0.1, 0.9), bar, lw=4, color="C1", capstyle="butt")
    scalebar_label(bar, "10 cMpc", fontsize=13, color="C1", fontweight="bold", pad=0.01)
    output = "plots/lss/overlapping/overlapping_A119_A147_A1168.pdf"
    savefig(output, fig=fig, tight=False)
    return


def plot_sky_a3651_a3667(args, chances, cat, cmap):
    fig, ax = plot_sky(
        args,
        chances,
        chances[chances["name"] == "Abell 3651"][0],
        cat,
        "301.5d -56.3d",
        "4 deg",
        cmap,
        z0=0.056,
        annotate=False,
        show_neighbors=True,
        hide_coordinates=False,
        sigma_clip=False,
        save=False,
    )
    # Dietl et al.
    contourfile = (
        "aux/xray/erosita/contours_a3667/A3667_Dietl_filament_contours_filtered.ctr"
    )
    if os.path.isfile(contourfile):
        # choose a few levels to show
        show_levels = np.array([0, 3, 6, 8])
        show_levels = show_levels[:1]
        with open(contourfile) as cf:
            contours = []
            levels = []
            start = False
            for line in cf:
                line = line.strip()
                if line.startswith("level"):
                    level = float(line.split("=")[1])
                    levels.append(level)
                if line == "(":
                    start = True
                    contours.append([])
                elif line == ")":
                    start = False
                    i = len(levels) - 1
                    c = np.array(contours[-1], dtype=float)
                    # color = f"C{len(levels)}"
                    color = "C2"
                    if i in show_levels:
                        if (
                            i == 0 and ((max(c[:, 0]) > 303) and (min(c[:, 0]) < 298))
                        ) or i > 0:
                            ax.plot_coord(
                                SkyCoord(
                                    ra=c[:, 0],
                                    dec=c[:, 1],
                                    unit="deg",
                                ),
                                color=color,
                                lw=1.2,
                                zorder=1010,
                            )
                            if i == 0:
                                bridge = Footprint("bridge", footprint=c)
                elif start:
                    contours[-1].append(line.split())
    print(bridge)
    #     for i, level in enumerate(levels):
    #         ax.plot([], [], label=str(level), color=f"C{i}")
    # Alexis' contours
    contourfiles = glob("aux/xray/erosita/contours_a3667/cl*.seg")
    for i, contourfile in enumerate(contourfiles):
        seg = Table.read(contourfile, format="ascii.commented_header")
        # for contour in np.unique(seg["col5"]):
        #     j = seg["col5"] == contour
        #     ax.plot_coord(
        #         SkyCoord(ra=seg["col1"][j], dec=seg["col2"][j], unit="deg"),
        #         f"C{i+4}-",
        #         lw=1,
        #         zorder=1010,
        #     )
        for s in seg:
            ax.plot_coord(
                SkyCoord(
                    ra=[s["col1"], s["col3"]], dec=[s["col2"], s["col4"]], unit="deg"
                ),
                f"C{2*i}-",
                lw=2,
                zorder=1000,
            )
    # ax.legend(fontsize=12)
    levels = np.array(levels)
    print(len(contours))
    print(levels)
    print(levels[show_levels])
    ax.annotate(
        "Abell 3667 (z=0.053)",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=16,
    )
    ax.annotate(
        "Abell 3651 (z=0.060)",
        xy=(0.95, 0.9),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=16,
    )
    bar = (5 * u.Mpc * cosmo.arcsec_per_kpc_comoving(0.056)).to(u.deg)
    bar = ax.scalebar((0.75, 0.08), bar, lw=4, color="C1", capstyle="butt", zorder=100)
    scalebar_label(bar, "5 cMpc", fontsize=13, color="C1", fontweight="bold", pad=0.01)
    output = "plots/lss/overlapping/overlapping_A3651_A3667.pdf"
    savefig(output, fig=fig, tight=False)
    return


def plot_sky(
    args,
    chances,
    cl,
    cat,
    center,
    radius,
    cmap,
    fig=None,
    z0=None,
    vmax=8000,
    annotate=True,
    show_main=True,
    show_neighbors=False,
    hide_coordinates=True,
    sigma_clip=True,
    suffix="",
    save=True,
):
    name = cl["name"]
    if args.clusters is not None and name not in args.clusters:
        return
    mask = cat["chances"] == name
    # print(cl)
    if fig is None:
        fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax = plt.axes(projection="astro zoom", center=center, radius=radius)
    ax.mark_inset_circle(ax, get_center(cl), get_radius(cl, 1), lw=0.8, zorder=1000)
    ax.mark_inset_circle(ax, get_center(cl), get_radius(cl, 5))
    if show_neighbors:
        cldist = cl["coord"].separation(chances["coord"])
        neighbors = (cldist < 5 * (cl["d200"] + chances["d200"]) * u.arcmin) & (
            cldist > 1 * u.arcmin
        )
        # print(f"{neighbors.sum()} neighbors:")
        # print(chances[neighbors])
        # neighbors = neighbors & (np.abs(chances["z"] - cl["z"]))
        for n in chances[neighbors]:
            ax.mark_inset_circle(ax, get_center(n), get_radius(n, 1), lw=0.8, zorder=10)
            ax.mark_inset_circle(ax, get_center(n), get_radius(n, 5), zorder=10)
            ax.plot(
                n["ra"],
                n["dec"],
                "kx",
                transform=ax.get_transform("world"),
                mew=2,
                ms=8,
                zorder=1001,
            )
            if annotate:
                ax.text_coord(
                    SkyCoord(ra=n["ra"], dec=n["dec"], unit="deg"),
                    f'{n["name"]}\n(z={n["z"]:.3f})',
                    ha="center",
                    va="center",
                    color="k",
                    fontsize=10,
                )
            mask = mask | (cat["chances"] == n["name"])
    subcl = cat[mask]
    subcl.sort("lambda")
    subcl = subcl[::-1]
    if z0 is None:
        z0 = cl["z"]
    for i, scl in enumerate(subcl):
        radius = scl["d200"]
        dv = light * (scl["best_z"] - z0) / (1 + z0)
        dvc = (dv + vmax) / (2 * vmax)
        color = list(cmap(dvc))
        # add transparency - doing it this way to also have transparency in the edge color
        color[-1] = 0.7
        if (np.abs(dv) < 3 * cl["sigma"]) or not sigma_clip:
            kwds = dict(lw=0, facecolor=color, edgecolor=color, zorder=9 + i)
        else:
            # to somewhat account for the thicker edge
            radius = 0.98 * radius
            kwds = dict(lw=2, facecolor="none", edgecolor=color, zorder=99 + i)
        ax.mark_inset_circle(ax, get_center(scl), f"{radius} deg", **kwds)
    if show_main and mask.sum() > 0:
        is_main = subcl["is_main"] == 1
        if is_main.sum() > 0:
            ax.plot_coord(
                SkyCoord(subcl["ra"][is_main], subcl["dec"][is_main], unit="deg"),
                "*",
                color="w",
                ms=12,
                mew=0,
                zorder=1000,
            )
    ax.plot(
        cl["ra"],
        cl["dec"],
        "kx",
        transform=ax.get_transform("world"),
        mew=2,
        ms=8,
        zorder=1001,
    )
    if annotate:
        ax.annotate(
            f"N={mask.sum()}",
            xy=(0.94, 0.94),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=16,
        )
        label = cl["name"].replace("-", "$-$").replace(" 00", " ").replace(" 0", " ")
        ax.annotate(
            f'{label}\n(z={cl["z"]:.3f}, $\\theta_{{200}}={cl["d200"]:.1f}\', \\sigma={cl["sigma"]:.0f}$ km s$^{{-1}}$)',
            xy=(0.06, 0.04),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=16,
        )
    # if not show_neighbors:
    #     ax.annotate(
    #         "$5r_{200}$",
    #         xy=(0.5, 0.85),
    #         xycoords="axes fraction",
    #         ha="center",
    #         va="bottom",
    #         fontsize=16,
    #         fontweight="heavy",
    #     )
    if annotate:
        bar = (5 * u.Mpc * cosmo.arcsec_per_kpc_comoving(cl["z"])).to(u.deg)
        bar = ax.scalebar((0.1, 0.88), bar, lw=4, color="C1", capstyle="butt")
        scalebar_label(
            bar, "5 cMpc", fontsize=13, color="C1", fontweight="bold", pad=0.01
        )
    ax.grid(True)
    ax.set_xlabel("Right Ascension")
    ax.set_ylabel("Declination")
    if hide_coordinates:
        for key in ("ra", "dec"):
            ax.coords[key].set_ticklabel_visible(False)
    if not save:
        return fig, ax
    if "Hydra" in name:
        outname = "Hydra"
    else:
        outname = name.replace(" ", "_").replace("+", "p").replace("-", "m")
    output = f"plots/lss/{args.sample}"
    if suffix:
        output = os.path.join(output, suffix)
    output = os.path.join(output, f"lss_{outname}.png")
    if suffix:
        output = output.replace(".png", f"_{suffix}.png")
    os.makedirs(os.path.split(output)[0], exist_ok=True)
    savefig(output, fig=fig, tight=False, verbose=False)
    return fig, ax


def get_center(item):
    return f'{item["ra"]}d {item["dec"]}d'


def get_radius(item, f=1):
    return f'{f * item["d200"]/60} deg'


def find_subclusters(
    args, chances, lambdacol="lambda", dz_frac_max=0.1, dv_max=10000 * u.km / u.s
):
    today = date.today().strftime("%Y%m%d")
    output = f"catalogues/eromapper/lss_{args.sample}_{args.catalog}.txt"
    ti = time()
    # this should depend on args.catalog
    cat = ClusterCatalog(
        args.catalog,
        Table.read("aux/optical/eromapper_optical_dr10_grz_catalog.fit"),
        cols=[
            "mem_match_id",
            "ra",
            "dec",
            lambdacol,
            f"{lambdacol}_e",
            "best_z",
            "best_zerr",
            "best_z_type",
        ],
        base_cols=("mem_match_id", "ra", "dec", "best_z"),
        masscol="lambda",
    )
    cat.catalog = cat[cat["z"] > 0.001]
    print(f"Loaded eromapper in {time()-ti:.1f} s")
    # eromapper.catalog.rename_column("name", "index")
    cat.catalog["name"] = np.arange(cat.size, dtype=int)
    # match_name = np.chararray(eromapper.size, itemsize=100)
    cols = [
        "chances",
        "eromapper_idx",
        "ra",
        "dec",
        "z",
        "best_zerr",
        "best_z_type",
        lambdacol,
        f"{lambdacol}_e",
        "dist (arcmin)",
        "dist (Mpc)",
        "dist (r200)",
        "vpec (km/s)",
        "is_main",
    ]
    matches = {key: [] for key in cols}
    chances.catalog[f"N_{args.catalog}"] = np.zeros(chances.size, dtype=int)
    # fixed
    dz_phot = dz_frac_max * cat["z"]
    for i, (cl, coord, z, d200, r200, sigma) in tqdm(
        enumerate(
            zip(
                chances.obj,
                chances.coords,
                chances.z,
                chances["d200"],
                chances["r200"],
                chances["sigma"],
            )
        ),
        total=chances.size,
    ):
        dz_int = (dv_max / c.c * (1 + z)).to(1)
        # dz_max = (dz_int**2 + dz_phot**2) ** 0.5
        # dz_max = dz_phot
        dz_max = dz_int
        jz = np.abs(cat.z - z) < dz_max
        sep = coord.separation(cat.coords)
        close = jz & (sep < 5 * d200 * u.arcmin)
        matches["chances"].extend([cl] * close.sum())
        matches["eromapper_idx"].extend(cat["name"][close])
        for col in cols[2:9]:
            matches[col].extend(cat[col][close])
        rich = cat[lambdacol][close]
        arcmin = sep[close].to("arcmin").value
        dist_r200 = arcmin / d200
        dist = r200 * dist_r200
        matches["dist (arcmin)"].extend(arcmin)
        matches["dist (r200)"].extend(dist_r200)
        matches["dist (Mpc)"].extend(dist)
        matches["vpec (km/s)"].extend(299792.5 * (cat["z"][close] - z) / (1 + z))
        main_candidates = dist_r200 < 1
        if close.sum() > 0 and np.any(main_candidates):
            is_main = main_candidates & (rich == rich[main_candidates].max())
        else:
            is_main = np.zeros(close.sum())
        matches["is_main"].extend(is_main.astype(int))
        chances.catalog[f"N_{args.catalog}"][i] = close.sum()
    matches = Table(matches)
    matches.rename_column("z", "best_z")
    matches.sort(["chances", "dist (arcmin)"])
    for col in ("ra", "dec"):
        matches[col].format = ".5f"
    for col in (lambdacol, f"{lambdacol}_e"):
        matches[col].format = ".1f"
    for col in matches.colnames:
        if "dist" in col:
            matches[col].format = ".2f"
    for col in ("best_z", "best_zerr"):
        matches[col].format = ".4f"
    matches["vpec (km/s)"].format = ".0f"
    print(matches)
    print(matches[matches[lambdacol] - matches[f"{lambdacol}_e"] >= 5])
    # main cluster statistics
    chances.catalog.sort("z")
    with_main = np.isin(chances["name"], matches["chances"][matches["is_main"] == 1])
    print(f"{with_main.sum()}/{chances.obj.size} main clusters identified")
    print(chances[with_main])
    print(chances[~with_main])
    chances.catalog.sort("name")
    matches.write(output, format="ascii.fixed_width", overwrite=True)
    return chances, matches


def velocity_dispersion_from_mass(chances):
    """Munari et al. (2013), M200c. Note that the"""
    A = 1177
    alpha = 0.364
    hz = cosmo.H(chances["z"]).value / 100
    return A * (hz * 1e14 * chances["m200"] / 1e15) ** alpha


def parse_args():
    parser = ArgumentParser()
    add = parser.add_argument
    add("sample", choices=("evolution", "lowz", "all"), default="all")
    add("-c", "--catalog", default="eromapper")
    add("--clusters", nargs="+")
    add("--overwrite", action="store_true")
    args = parser.parse_args()
    return args


main()
