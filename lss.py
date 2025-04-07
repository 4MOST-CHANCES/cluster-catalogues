from argparse import ArgumentParser
from astropy import units as u, constants as c
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
from astropy.table import Table, join, vstack
from astropy.visualization import hist
import cmasher as cmr
from colossus.cosmology import cosmology
from colossus.halo.mass_adv import changeMassDefinitionCModel
from datetime import date
from glob import glob
from matplotlib import cm, pyplot as plt, ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import os
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from scipy.stats import (
    binned_statistic as binstat,
    binned_statistic_2d as binstat2d,
    gaussian_kde,
)
import seaborn as sns
from time import time
from tqdm import tqdm

import ligo.skymap.plot
from profiley.nfw import NFW
from plottery.plotutils import savefig, update_rcParams

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
        print(lowz["name"].value)
        print(evol["name"].value)
        print(lowz["name"].size, evol["name"].size)
        print()
        chances = ClusterCatalog(
            "chances", vstack([lowz.catalog, evol.catalog]), masscol="m200"
        )
    else:
        chances = ClusterCatalog(f"chances-{args.sample[:4]}")
    chances.catalog["sigma"] = velocity_dispersion_from_mass(
        chances["m200"], chances["z"]
    )
    chances.catalog["sigma"].format = "%.0f"
    chances, cat = find_subclusters(args, chances)
    compare_redmapper_masses(args, chances, cat)
    print(chances)
    print(cat)
    print(np.unique(cat["best_z_type"], return_counts=True))
    ztype = cat.group_by(["best_z_type", "is_main"])
    for key, group in zip(ztype.groups.keys, ztype.groups):
        name = key["best_z_type"].replace("_", "\\_")
        c = "main" if key["is_main"] == 1 else "groups"
        key = f"{name} ({c})"
        print_redshift_uncertainties(chances, group, f"\\texttt{{{key}}}")
    ztype = cat.group_by("best_z_type")
    for key, group in zip(ztype.groups.keys, ztype.groups):
        name = key["best_z_type"].replace("_", "\\_")
        print_redshift_uncertainties(chances, group, f"\\texttt{{{name}}}")
    print_redshift_uncertainties(chances, cat, "All")
    if args.sample == "all":
        infall_mass_function(args, chances, cat, plottype="hist")
        infall_mass_function(args, chances, cat, plottype="hist", sigma_clip=3)
        infall_mass_function(args, chances, cat, plottype="points", sigma_clip=3)
        phase_space(args, chances, cat, yaxis="sigma", max_yerr_sigma=1)
        phase_space(args, chances, cat, yaxis="km/s")
    else:
        wrap_plot_sky(args, chances, cat, show_neighbors=False)
        wrap_plot_sky(args, chances, cat, show_neighbors=True, suffix="neighbors")
    chances.catalog.sort(f"N_{args.catalog}")
    # print(chances)
    return


def compare_redmapper_masses(args, chances, redmapper):
    combined = join(
        chances.catalog,
        redmapper[redmapper["is_main"] == 1],
        keys_left="name",
        keys_right="chances",
        join_type="inner",
        table_names=("cl", "gr"),
    )
    print(combined)
    diff = np.log10(combined["m200_gr"] / (1e14 * combined["m200_cl"]))
    stats = np.percentile(diff, [16, 50, 84]).value
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    ax = axes[0]
    ax.plot(1e14 * combined["m200_cl"], combined["m200_gr"], "ko")
    if args.sample == "evolution":
        t = np.logspace(14, 15.7, 10)
        xylim = (2e14, 5e15)
        hxlim = (-0.7, 0.7)
    else:
        t = np.logspace(13, 15.3)
        xylim = (5e13, 2e15)
        hxlim = (-1.4, 1.4)
    ax.plot(t, t, "k-")
    ax.plot(t, (1 + stats[1]) * t, "C0--")
    ax.plot(t, (1 + stats[0]) * t, "C0:")
    ax.plot(t, (1 + stats[2]) * t, "C0:")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="CHANCES mass",
        ylabel="redMaPPer mass",
        xlim=xylim,
        ylim=xylim,
    )
    ax.set_title(args.sample.capitalize(), fontsize=14)
    ax = axes[1]
    hist(diff, "knuth", ax=ax, histtype="stepfilled", color="0.5")
    ax.annotate(
        f"median = ${stats[1]:.2f}$ dex\nrange = $[{stats[0]:.2f}, {stats[2]:.2f}]$ dex",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=14,
    )
    ax.axvline(0, color="k", ls="-")
    ax.axvline(stats[1], color="C0", ls="--")
    ax.axvline(stats[0], color="C0", ls=":")
    ax.axvline(stats[2], color="C0", ls=":")
    ax.set(
        xlabel="$\log M_{200}^\\mathrm{rm} - \log M_{200}^\\mathrm{CHANCES}$",
        xlim=hxlim,
    )
    output = f"plots/lss/compare_redmapper_masses_{args.sample}.png"
    savefig(output, fig=fig, tight=False)
    return


def print_redshift_uncertainties(chances, group, key):
    sigma_main = np.array(
        [chances["sigma"][chances["name"] == cl] for cl in group["chances"]]
    )[:, 0]
    z_main = np.array([chances["z"][chances["name"] == cl] for cl in group["chances"]])[
        :, 0
    ]
    ngr = group["best_z_type"].size
    # percentage error
    zerr = 100 * group["best_zerr"] / group["best_z"]
    zerr_med = np.median(zerr)
    zerr_range = np.percentile(zerr, [16, 84])
    # vpec uncertainty
    # verr = c.c.to("km/s").value * group["best_zerr"] / (1 + z_main)
    # testing updated cg_specz errors
    verr = group["vpec_err"]
    verr_med = 100 * np.round(np.median(verr) / 100, 0)
    verr_range = 100 * np.round(np.percentile(verr / 100, [16, 84]), 0)
    # verr_med = np.median(verr)
    # verr_range = np.percentile(verr, [16, 84])
    print(verr_med, verr_range)
    # uncertainty in vpec/sigma_main
    vserr = verr / sigma_main
    vserr_med = np.median(vserr)
    vserr_range = np.percentile(vserr, [16, 84])
    print("= v1 =")
    if zerr_range[0] < 1:
        if zerr_range[1] < 2:
            zp_range = f"$\\multicolumn{{2}}{{c}}{{<{zerr_range[1]:.1f}}}$"
        else:
            zp_range = f"$<{zerr_range[1]:.0f}$"
    else:
        zp_range = f"{zerr_range[0]:.0f}--{zerr_range[1]:.0f}"
    print(f"{key} & {ngr} & {zp_range} & {verr_med:.0f} & {vserr_med:.2f}\\\\")
    print("= v2 =")
    zerr_range = np.abs(zerr_med - zerr_range)
    verr_range = np.abs(verr_med - verr_range)
    vserr_range = np.abs(vserr_med - vserr_range)
    zstr = f"${zerr_med:.1f}$ & $_{{-{zerr_range[0]:.1f}}}^{{+{zerr_range[1]:.1f}}}$"
    vstr = f"${verr_med:.0f}$ & $_{{-{verr_range[0]:.0f}}}^{{+{verr_range[1]:.0f}}}$"
    vs_str = (
        f"${vserr_med:.2f}$ & $_{{-{vserr_range[0]:.2f}}}^{{+{vserr_range[1]:.2f}}}$"
    )
    print(f"{key} & {ngr} & {zstr} & {vstr} & {vs_str}\\\\")
    return


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
    specz = cat["best_z_type"] != "photo_z"
    mask = mask & specz
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
        ax.set_ylim(0.007, 1)
    ax.set_xticks(np.logspace(-2, 0, 3), ["0.01", "0.1", "1"])
    ax.set_yticks(np.logspace(-2, 0, 3), ["0.01", "0.1", "1"])
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    output = "plots/lss/infall_mass_function"
    if sigma_clip:
        output = f"{output}_{plottype}_{sigma_clip}vclip"
    savefig(f"{output}.pdf", fig=fig, tight=False)


def phase_space(
    args,
    chances,
    cat,
    hide_main=True,
    sigma_clip=0,
    max_yerr_sigma=0,
    yaxis="sigma",
    show_histogram=True,
    cmap="viridis",
):
    rdm = np.random.default_rng(seed=1)
    # include photo-z somehow later too
    specz = cat["best_z_type"] != "photo_z"
    photz = cat["best_z_type"] == "photo_z"
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
    best = mask & specz
    yval = cat["vpec (km/s)"]
    yerr = cat["vpec_err"]
    if max_yerr_sigma > 0:
        small_err = yerr < max_yerr_sigma * sigma_main
        best = best & small_err
    else:
        small_err = np.ones(yval.size, dtype=bool)
    if yaxis == "sigma":
        yval = yval / sigma_main
        yerr = yerr / sigma_main
    else:
        yval = yval / 1e3
        yerr = yerr / 1e3
    # note that changing axes limits or figure size will change the symbol size
    s_factor = 34000
    # need to convert z_main into colors using a normalized cmap for this to work
    cnorm = Normalize(vmin=0, vmax=0.5)
    colors = cnorm(z_main)
    print("colors =", np.sort(colors))
    print()
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    colors = cmap(colors)
    ax.scatter(
        cat["dist (r200)"][mask & photz],
        yval[mask & photz],
        # s=s_factor * (cat["r200"] / r200_main)[mask & photz] ** 2 / 30,
        # edgecolors=colors[mask & photz],
        # facecolors="None",
        # lw=1,
        # alpha=0.3,
        s=4,
        c=colors[mask & photz],
        alpha=0.5,
    )
    if small_err.sum() > 0:
        m = mask & specz & ~small_err
        ax.scatter(
            cat["dist (r200)"][m],
            yval[m],
            s=s_factor * (cat["r200"] / r200_main)[m] ** 2 / 30,
            edgecolors=colors[m],
            facecolors="None",
            lw=1,
            alpha=0.5,
        )
        # show uncertainties for a small fraction of these
        m = m & (np.abs(yval) < 6)
        for d, v, ve, color in zip(cat["dist (r200)"][m], yval[m], yerr[m], colors[m]):
            if rdm.uniform() > 0.9:
                ax.errorbar(d, v, ve, fmt=",", color=color, alpha=0.5, lw=2)
    im = ax.scatter(
        cat["dist (r200)"][best],
        yval[best],
        s=s_factor * (cat["r200"] / r200_main)[best] ** 2 / 30,
        c=colors[best],
        lw=0,
        alpha=0.5,
    )
    # show uncertainties for the "best" sample
    # for d, v, ve, color in zip(
    #     cat["dist (r200)"][best], yval[best], yerr[best], colors[best]
    # ):
    #     if rdm.uniform() < 1:
    #         ax.errorbar(d, v, ve, fmt=",", color=color, alpha=0.5, lw=2)

    # escape velocities
    r = np.linspace(0, 10, 1001)[1:]
    for m200_ref, z_ref, color in zip(
        np.array([[3e14], [1e15]]), np.array([[0.05], [0.25]]), ("C3", "C0")
    ):
        sigma_ref = velocity_dispersion_from_mass(m200_ref / 1e14, z_ref)
        nfw = NFW(m200_ref, 4, z_ref, overdensity=200, background="c")
        r200_ref = nfw.rdelta(200, "c")
        rx = r200_ref * r
        # for comoving coordinates
        q = nfw.cosmo.Om0 / 2 - nfw.cosmo.Ode(z_ref)
        r_eq = (-c.G * nfw.mdelta(200, "m")[0] * u.Msun / q / cosmo.H(z_ref) ** 2).to(
            "Mpc^3"
        ).value ** (1 / 3)
        print("sigma_ref =", sigma_ref)
        print("r200_ref =", r200_ref)
        print("r_eq =", r_eq, rx.max())
        # this should be incorporated in profiley
        #

        def vesc(R):
            f = q * nfw.cosmo.H(z_ref) ** 2 * (R**2 - r_eq**2) * u.Mpc**2 / 2
            punit = u.Mpc**2 / u.s**2
            vesc = (
                ((2 * ((nfw.potential(R) - nfw.potential(r_eq)) * punit - f)) ** 0.5)
                .to("km/s")
                .value
            )
            # do I need this?
            # vesc[np.isnan(vesc)] = 0
            return vesc

        ax.plot(r, vesc(rx[:, None]) / sigma_ref, f"{color}-", lw=1.5)
        ax.plot(r, -vesc(rx[:, None]) / sigma_ref, f"{color}-", lw=1.5)
    # use this to find the size that gives a radius of 1 in data units
    # ax.scatter(1, -3, s=34000, c="k")
    # ax.grid()
    # circles = [Circle((d, y), s, color=)]
    if show_histogram:
        use_errors = False
        smooth = 0.2
        evol = z_main > 0.07
        # beware this should change if axis limits change
        if yaxis == "sigma":
            vbins = np.arange(-6, 6, 1e-3)
        else:
            vbins = np.arange(-10, 10, 1e-3)
        if use_errors:
            ye = np.max([yerr, smooth * np.ones(yerr.size)], axis=0)
            h = np.exp(-((vbins[:, None] - yval) ** 2) / (2 * ye**2)) / (
                (2 * np.pi) ** 0.5 * ye
            )
            print("h =", h.shape)
            h_lowz = np.mean(h[:, best & ~evol], axis=1)
            hax.plot(h_lowz, vbins, "C3-", label="Low-z")
            h_evol = np.mean(h[:, best & evol], axis=1)
            hax.plot(h_evol, vbins, "C0-", label="Evolution")
            h_all = np.mean(h[:, best], axis=1)
            hax.plot(h_all, vbins, "k--", label="All")
            hax.legend(loc="lower right", fontsize=14)
            hmax = h_lowz.max()
        else:
            vx = (vbins[1:] + vbins[:-1]) / 2
            for ztype, ls in zip((photz, specz), ("--", "-")):
                if ls == "--":
                    continue
                hmask = mask & (yval >= vx[0]) & (yval <= vx[-1]) & ztype
                hist = np.histogram(yval[hmask], vbins)[0]
                kde = gaussian_kde(yval[hmask], smooth)
                (l3,) = hax.plot(kde(vx), vx, "k--")
                kde_evol = gaussian_kde(yval[hmask & evol], smooth)
                (l1,) = hax.plot(kde_evol(vx), vx, "C0", ls=ls)
                kde_lowz = gaussian_kde(yval[hmask & ~evol], smooth)
                (l2,) = hax.plot(kde_lowz(vx), vx, "C3-", ls=ls)
            legend = hax.legend(
                (l2, l1, l3),
                ("Low-z", "Evolution", "All"),
                loc="lower right",
                fontsize=14,
            )
            hmax = kde_lowz(vx).max()
        hax.xaxis.set_major_locator(ticker.MultipleLocator(hmax / 2))
        hax.set_xticklabels([])
        hax.set_xlim(0, hax.get_xlim()[1])
        cbar = plt.colorbar(
            cm.ScalarMappable(cnorm, cmap),
            ax=axes,
            orientation="horizontal",
            location="top",
        )
        cbar.solids.set(alpha=0.5)
        cbar.ax.set_xlim(0, 0.45)
        cbar.ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    else:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylim(0, 0.45)
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

    savefig(f"{output}_v5.pdf", fig=fig, tight=False)
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
    d200 = ((r200 * u.Mpc) * cosmo.arcsec_per_kpc_proper(z)).to(u.arcmin).value
    cat["m200"] = m200
    cat["r200"] = r200
    cat["d200"] = d200
    cat["m200"].format = ".2e"
    cat["r200"].format = ".2f"
    cat["d200"].format = ".2f"
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
        plot_sky_hydra_antlia(args, chances, cat, cmap)
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


def plot_sky_hydra_antlia(args, chances, cat, cmap):
    fig, ax = plot_sky(
        args,
        chances,
        chances[chances["name"] == "Hydra (A1060)"][0],
        cat,
        "158d -31d",
        "12 deg",
        cmap,
        z0=0.012,
        annotate=False,
        show_neighbors=True,
        hide_coordinates=False,
        sigma_clip=False,
        save=False,
    )
    # ax.text_coord("Abell 168", xy=())
    bar = (5 * u.Mpc * cosmo.arcsec_per_kpc_comoving(0.012)).to(u.deg)
    bar = ax.scalebar((0.7, 0.08), bar, lw=4, color="C1", capstyle="butt")
    scalebar_label(bar, "5 cMpc", fontsize=13, color="C1", fontweight="bold", pad=0.01)
    ax.annotate(
        "Hydra",
        xy=(0.75, 0.75),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=16,
    )
    ax.annotate(
        "Antlia",
        xy=(0.35, 0.1),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize=16,
    )
    output = "plots/lss/overlapping/overlapping_Hydra_Antlia.pdf"
    savefig(output, fig=fig, tight=False)
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
        # remember we're storing this in arcmin
        radius = scl["d200"] / 60
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
    cat.catalog = cat[(cat["z"] > 0.001) & (cat["z"] < 0.8)]
    cat.catalog.sort("z")
    print(cat)
    print(f"Loaded eromapper in {time()-ti:.1f} s")
    # photoz uncertaintes
    zbins = np.arange(0, 0.51, 0.02)
    lambins = np.logspace(0.3, 2.3, 15)
    zerr = binstat2d(
        cat[lambdacol],
        cat["z"],
        cat["best_zerr"],
        statistic="median",
        bins=(lambins, zbins),
    ).statistic
    zerr = zerr / ((zbins[:-1] + zbins[1:]) / 2)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(zbins, lambins, zerr, cmap="Reds")
    ax.set(yscale="log", xlabel="Redshift", ylabel="$\lambda$")
    plt.colorbar(im, ax=ax, label="$\Delta z_\mathrm{cl}/z_\mathrm{cl}$")
    plot_output = "plots/redmapper_zerr.png"
    savefig(plot_output, fig=fig)
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
        "vpec_err",
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
        dz_max = dz_int
        jz = np.abs(cat.z - z) < dz_max
        # if z < 0.05:
        #     fig, ax = plt.subplots(layout="constrained")
        #     ax.plot(cat["z"], cat.z - z, "k.")
        #     ax.plot(cat["z"][jz], cat.z[jz] - z, "C1,")
        #     ax.axhline(dz_max, ls="--")
        #     ax.axvline(z, ls=":")
        #     ax.set(xlabel="Redshift", ylabel="z - z_cl")
        #     fig.savefig("plots/test_dzmax.png")
        #     plt.close(fig=fig)
        #     sys.exit()
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
        matches["vpec (km/s)"].extend(light * (cat["z"][close] - z) / (1 + z))
        matches["vpec_err"].extend(light * cat["best_zerr"][close] / (1 + z))
        main_candidates = dist_r200 < 1
        if close.sum() > 0 and np.any(main_candidates):
            is_main = main_candidates & (rich == rich[main_candidates].max())
        else:
            is_main = np.zeros(close.sum())
        matches["is_main"].extend(is_main.astype(int))
        chances.catalog[f"N_{args.catalog}"][i] = close.sum()
    matches = Table(matches)
    matches.rename_column("z", "best_z")
    matches = add_r200(matches)
    matches["vdisp (km/s)"] = np.exp(
        6.46
        + 0.365
        * np.log(matches["lambda"] * cosmo.H(matches["best_z"]) / cosmo.H0 / 47.2)
    )
    # updating vpec errors for these. z_boot and photo_z already have this contribution
    cgz = matches["best_z_type"] == "cg_spec_z"
    test = matches["vpec_err"][cgz]
    # these have only one spec-z so sqrt(N) = 1
    matches["vpec_err"][cgz] = (
        matches["vpec_err"][cgz] ** 2 + matches["vdisp (km/s)"][cgz] ** 2
    ) ** 0.5
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
    for col in [col for col in matches.colnames if "vpec" in col or "vdisp" in col]:
        matches[col].format = ".0f"
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


def velocity_dispersion_from_mass(m200, z):
    """Munari et al. (2013), M200c. Mass must be in 1e14 Msun"""
    A = 1177
    alpha = 0.364
    hz = cosmo.H(z).value / 100
    return A * (hz * 1e14 * m200 / 1e15) ** alpha


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
