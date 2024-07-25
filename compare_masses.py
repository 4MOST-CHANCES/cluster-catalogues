from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
from astropy.io import ascii, fits
from astropy.table import Table, join, hstack
from datetime import date
from icecream import ic
from matplotlib import pyplot as plt, ticker
import numpy as np
from scipy.optimize import curve_fit
import sys

from plottery.plotutils import savefig, update_rcParams

from linmix import LinMix

from astro.clusters import ClusterCatalog
from lnr import bces, kelly, to_linear, to_log, mcmc

update_rcParams()


def clustercat(name, suffix="", mcol="m200"):
    cat = ClusterCatalog(name)
    m200 = Table.read(f"aux/m200_converted/{name}_m200{suffix}.csv")
    cat.catalog["m200"] = -np.ones(cat.size)
    cat.catalog["m200"][cat.z > 0] = m200["m200"]
    cat.catalog[np.isnan(cat["m200"])] = -1
    print(name, cat["m200"].max())
    if cat["m200"].max() < 1e10:
        cat.catalog["m200"][cat.z > 0] = 1e14 * cat["m200"][cat.z > 0]
    # these are in units of 1e13 Msun
    # if name == "erass1":
    #     cat.catalog["m200"] /= 10
    cat.catalog["m200"].format = "%.2e"
    print(name, cat["m200", "z"])
    return cat


def main():
    # lowz = Table.read("")

    act = clustercat("act-dr5", "cal")
    psz = clustercat("psz2")
    erass = clustercat("erass1")
    # return
    mcxc = clustercat("mcxc")
    for suf in ["", "Cal", "Uncorr"]:
        act.catalog[f"M500c{suf}_err"] = 0.5 * (
            act[f"M500c{suf}_errMinus"] + act[f"M500c{suf}_errPlus"]
        )
    act.catalog["m200_err"] = act["M500cCal_err"] / act["M500cCal"] * act["m200"]
    print(np.sort(erass.colnames))
    psz.catalog["MSZ_err"] = 0.5 * (psz["MSZ_ERR_LOW"] + psz["MSZ_ERR_UP"])
    psz.catalog["m200_err"] = psz["MSZ_err"] / psz["MSZ"] * psz["m200"]
    # erass.catalog["M500_err"] = 0.5 * (erass["M500_L"] + erass["M500_H"])
    # upper uncertainties are usually larger than the mass estimate!
    erass.catalog["M500_err"] = erass["M500_L"]
    erass.catalog["m200_err"] = erass["M500_err"] / erass["M500"] * erass["m200"]
    # mass columns
    # mcol_act = "M500cCal"
    mcol_act = "m200"
    merrcol_act = f"{mcol_act}_err"
    # mcol_erass = "m200"
    mcol_codex = "M500c" if "M500" in mcol_act else "M200c"
    merrcol_codex = f"e{mcol_codex}"
    if act[merrcol_act].max() < 1000:
        act.catalog[merrcol_act] = 1e14 * act[merrcol_act]
    codex = load_codex()
    meneacs = load_meneacs()
    wings = load_wings()
    axes2mrs = Table(
        fits.open("aux/xray/Xmass_BayesGroups_n3ext200kpc_nofake_info2.fits")[1].data
    )
    axesls = Table.read(
        "aux/xray/Xmass_axes_legacy.cat", format="ascii.commented_header"
    )
    axesls.rename_column("zcmb", "z")
    print(np.sort(axes2mrs.colnames))
    print(np.sort(axesls.colnames))
    # axes2mrs = axes2mrs[axes2mrs["z"] <= 0.04]
    # axesls = axesls[axesls["zcmb"] <= 0.04]
    axes2mrs["coords"] = SkyCoord(axes2mrs["RA_X"], axes2mrs["DEC_X"], unit="deg")
    axesls["coords"] = SkyCoord(axesls["RA_X"], axesls["DEC_X"], unit="deg")
    mcol_axes2mrs = "M200c"
    mcol_axesls = "M200c"
    merrcol_axes2mrs = "eM200c"
    merrcol_axesls = "eM200c"
    ## match catalogs
    max_radius = 5 * u.arcmin
    act_codex_closest, act_in_codex = match_catalogs(act, codex, radius=max_radius)
    print(f"{act_in_codex.sum()}/{act_in_codex.size} ACT-CODEX matches")
    act_axes2mrs_closest, act_in_axes2mrs = match_catalogs(
        act, axes2mrs, radius=max_radius
    )
    print(f"{act_in_axes2mrs.sum()}/{act_in_axes2mrs.size} ACT-AXES2MRS matches")
    act_axesls_closest, act_in_axesls = match_catalogs(act, axesls, radius=max_radius)
    print(f"{act_in_axesls.sum()}/{act_in_axesls.size} ACT-AXESLS matches")
    axesls_axes2mrs_closest, axesls_in_axes2mrs = match_catalogs(
        axesls, axes2mrs, radius=max_radius
    )
    print(
        f"{axesls_in_axes2mrs.sum()}/{axesls_in_axes2mrs.size} AXESLS-AXES2MRS matches"
    )
    axes2mrs_codex_closest, axes2mrs_in_codex = match_catalogs(
        axes2mrs, codex, radius=max_radius
    )
    print(f"{axes2mrs_in_codex.sum()}/{axes2mrs_in_codex.size} CODEX-AXES2MRS matches")
    axesls_codex_closest, axesls_in_codex = match_catalogs(
        axesls, codex, radius=max_radius
    )
    print(f"{axesls_in_codex.sum()}/{axesls_in_codex.size} CODEX-AXESLS matches")
    act_erass_closest, act_in_erass = match_catalogs(act, erass, radius=max_radius)
    erass_act_closest, erass_in_act = match_catalogs(erass, act, radius=max_radius)
    print(f"{act_in_erass.sum()}/{act_in_erass.size} ACT-eRASS1 matches")
    act_mcxc_closest, act_in_mcxc = match_catalogs(act, mcxc, radius=max_radius)
    print(f"{act_in_mcxc.sum()}/{act_in_mcxc.size} ACT-MCXC matches")
    act_psz_closest, act_in_psz = match_catalogs(act, psz, radius=max_radius)
    print(f"{act_in_psz.sum()}/{act_in_psz.size} ACT-PSZ matches")
    act_wings_closest, act_in_wings = match_catalogs(act, wings, radius=max_radius)
    print(f"{act_in_wings.sum()}/{act_in_wings.size} ACT-WINGS matches")
    psz_wings_closest, psz_in_wings = match_catalogs(psz, wings, radius=max_radius)
    print(f"{psz_in_wings.sum()}/{psz_in_wings.size} PSZ-WINGS matches")
    erass_wings_closest, erass_in_wings = match_catalogs(
        erass, wings, radius=max_radius
    )
    print(np.sort(wings.colnames))
    print(np.sort(psz.colnames))
    pw = hstack(
        [
            wings["name", "z", "m200"][psz_wings_closest[psz_in_wings]],
            psz["name", "z", "m200"][psz_in_wings],
        ],
        table_names=["wings", "psz"],
    )
    pw["zdiff"] = (pw["z_psz"] - pw["z_wings"]) / pw["z_wings"]
    pw["m_psz/m_wings"] = pw["m200_psz"] / pw["m200_wings"]
    pw["m_wings/m_psz"] = pw["m200_wings"] / pw["m200_psz"]
    pw["zdiff"].format = "%.2e"
    pw["m_psz/m_wings"].format = "%.2f"
    pw["m_wings/m_psz"].format = "%.2f"
    pw.sort("m_psz/m_wings")
    print(pw)
    # print(wings["name", "z", "m200"][psz_wings_closest[psz_in_wings]])
    # print(psz["name", "z", "m200"][psz_in_wings])
    # return
    print(f"{erass_in_wings.sum()}/{erass_in_wings.size} eRASS1-WINGS matches")
    erass_codex_closest, erass_in_codex = match_catalogs(
        erass, codex, radius=max_radius
    )
    print(f"{erass_in_codex.sum()}/{erass_in_codex.size} eRASS1-CODEX matches")
    erass_axes2mrs_closest, erass_in_axes2mrs = match_catalogs(
        erass, axes2mrs, radius=max_radius
    )
    print(f"{erass_in_axes2mrs.sum()}/{erass_in_axes2mrs.size} eRASS1-AXES2MRS matches")
    erass_meneacs_closest, erass_in_meneacs = match_catalogs(
        erass, meneacs, radius=max_radius
    )
    print(f"{erass_in_meneacs.sum()}/{erass_in_meneacs.size} eRASS1-MENEACS matches")
    mcxc_wings_closest, mcxc_in_wings = match_catalogs(mcxc, wings, radius=max_radius)
    print(f"{mcxc_in_wings.sum()}/{mcxc_in_wings.size} MCXC-WINGS matches")
    psz_axes2mrs_closest, psz_in_axes2mrs = match_catalogs(
        psz, axes2mrs, radius=max_radius
    )
    print(f"{psz_in_axes2mrs.sum()}/{psz_in_axes2mrs.size} PSZ-AXES2MRS matches")

    fit_and_plot(
        "CODEX",
        "eRASS1",
        codex,
        erass,
        mcol_codex,
        "m200",
        "m200_err",
        xmask=erass_codex_closest[erass_in_codex],
        ymask=erass_in_codex,
    )
    fit_and_plot(
        "AXES-2MRS",
        "PSZ2",
        axes2mrs,
        psz,
        "M200c",
        "m200",
        "m200_err",
        xmask=psz_axes2mrs_closest[psz_in_axes2mrs],
        ymask=psz_in_axes2mrs,
    )
    fit_and_plot(
        "WINGS",
        "MCXC",
        wings,
        mcxc,
        "m200",
        "m200",
        xmask=mcxc_wings_closest[mcxc_in_wings],
        ymask=mcxc_in_wings,
    )
    fit_and_plot(
        "MENEACS",
        "eRASS1",
        meneacs,
        erass,
        "m200",
        "m200",
        "m200_err",
        xmask=erass_meneacs_closest[erass_in_meneacs],
        ymask=erass_in_meneacs,
    )
    fit_and_plot(
        "AXES-2MRS",
        "eRASS1",
        axes2mrs,
        erass,
        mcol_axes2mrs,
        "m200",
        "m200_err",
        xmask=erass_axes2mrs_closest[erass_in_axes2mrs],
        ymask=erass_in_axes2mrs,
    )
    fit_and_plot(
        "CODEX",
        "ACT-DR5",
        codex,
        act,
        "M200c",
        mcol_act,
        merrcol_act,
        xmask=act_codex_closest[act_in_codex],
        ymask=act_in_codex,
    )
    fit_and_plot(
        "CODEX",
        "ACT-DR5",
        codex,
        act,
        "M200c_sigma",
        mcol_act,
        merrcol_act,
        xmask=act_codex_closest[act_in_codex],
        ymask=act_in_codex,
        suffix="sigma",
    )
    fit_and_plot(
        "WINGS",
        "eRASS1",
        wings,
        erass,
        "m200",
        "m200",
        "m200_err",
        xmask=erass_wings_closest[erass_in_wings],
        ymask=erass_in_wings,
    )
    fit_and_plot(
        "WINGS",
        "PSZ2",
        wings,
        psz,
        "m200",
        "m200",
        "m200_err",
        xmask=psz_wings_closest[psz_in_wings],
        ymask=psz_in_wings,
    )
    fit_and_plot(
        "PSZ2",
        "WINGS",
        psz,
        wings,
        "m200",
        "m200",
        "m200_err",
        xmask=psz_in_wings,
        ymask=psz_wings_closest[psz_in_wings],
    )
    fit_and_plot(
        "WINGS",
        "ACT-DR5",
        psz,
        act,
        "m200",
        mcol_act,
        merrcol_act,
        xmask=act_wings_closest[act_in_wings],
        ymask=act_in_wings,
    )
    fit_and_plot(
        "PSZ2",
        "ACT-DR5",
        psz,
        act,
        "m200",
        mcol_act,
        merrcol_act,
        xmask=act_psz_closest[act_in_psz],
        ymask=act_in_psz,
    )
    fit_and_plot(
        "ACT-DR5",
        "eRASS1",
        act,
        erass,
        mcol_act,
        "m200",
        "m200_err",
        xmask=erass_act_closest[erass_in_act],
        ymask=erass_in_act,
    )
    fit_and_plot(
        "eRASS1",
        "ACT-DR5",
        erass,
        act,
        "m200",
        mcol_act,
        merrcol_act,
        xmask=act_erass_closest[act_in_erass],
        ymask=act_in_erass,
    )
    fit_and_plot(
        "MCXC",
        "ACT-DR5",
        mcxc,
        act,
        "m200",
        mcol_act,
        merrcol_act,
        xmask=act_mcxc_closest[act_in_mcxc],
        ymask=act_in_mcxc,
    )
    sys.exit()
    fit_and_plot(
        "CODEX",
        "AXES-2MRS",
        codex,
        axes2mrs,
        mcol_codex,
        mcol_axes2mrs,
        merrcol_axes2mrs,
        xmask=axes2mrs_codex_closest[axes2mrs_in_codex],
        ymask=axes2mrs_in_codex,
    )
    fit_and_plot(
        "CODEX",
        "AXES-LS",
        codex,
        axesls,
        mcol_codex,
        mcol_axesls,
        merrcol_axesls,
        xmask=axesls_codex_closest[axesls_in_codex],
        ymask=axesls_in_codex,
    )
    fit_and_plot(
        "AXES-2MRS",
        "AXES-LS",
        axes2mrs,
        axesls,
        mcol_axes2mrs,
        mcol_axesls,
        merrcol_axesls,
        xmask=axesls_axes2mrs_closest[axesls_in_axes2mrs],
        ymask=axesls_in_axes2mrs,
    )
    fit_and_plot(
        "AXES-LS",
        "AXES-2MRS",
        axesls,
        axes2mrs,
        mcol_axesls,
        mcol_axes2mrs,
        merrcol_axes2mrs,
        xmask=axesls_in_axes2mrs,
        ymask=axesls_axes2mrs_closest[axesls_in_axes2mrs],
    )
    fit_and_plot(
        "AXES-LS",
        "ACT-DR5",
        axesls,
        act,
        mcol_axesls,
        mcol_act,
        merrcol_act,
        xmask=act_axesls_closest[act_in_axesls],
        ymask=act_in_axesls,
    )


def fit_and_plot(
    xlabel,
    ylabel,
    x,
    y,
    xcol,
    ycol,
    yerrcol=None,
    xerrcol=None,
    xmask=None,
    ymask=None,
    suffix=None,
):
    """Add a mask to exclude objects from the fit. This will allow to mask AXES-2MRS_PSZ2 clusters based on redshift rather than mass comparison, for instance."""
    if xmask is None:
        xmask = np.ones(x.shape, dtype=bool)
    if ymask is None:
        ymask = np.ones(y.shape, dtype=bool)
    if yerrcol is None:
        fit_norm, fit_normcov = fit_mass_ratio(x[xcol][xmask], y[ycol][ymask], p0=(1,))
    else:
        fit_norm, fit_normcov = fit_mass_ratio(
            x[xcol][xmask],
            y[ycol][ymask],
            y[yerrcol][ymask],
            p0=(1,),
        )
    xrng = np.linspace(14, 15.7, 100)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    if yerrcol is None:
        ax.plot(x[xcol][xmask], y[ycol][ymask], "o", color="k")
    else:
        ax.errorbar(
            x[xcol][xmask], y[ycol][ymask], y[yerrcol][ymask], fmt="o", color="k"
        )
    # need redshifts
    # ax.scatter(x[xcol][xmask], y[ycol][ymask], c=x[], zorder=10)
    if xlabel == "AXES-2MRS" and ylabel == "PSZ2":
        j = y[ycol][ymask] < 2 * x[xcol][xmask]
        norm2, normcov2 = fit_mass_ratio(x[xcol][xmask][j], y[ycol][ymask][j], p0=(1,))
        ax.plot(
            10**xrng,
            10 ** line(xrng, *norm2),
            "C1-",
            lw=3,
            zorder=10,
            label=f"y = {10**norm2[0]:.2f}x",
        )
        ax.plot(
            x[xcol][xmask][~j], y[ycol][ymask][~j], "o", mfc="w", mec="k", zorder=10
        )
    else:
        ax.plot(
            10**xrng,
            10 ** line(xrng, *fit_norm),
            "C1-",
            lw=3,
            zorder=10,
            label=f"y = {10**fit_norm[0]:.2f}x",
        )
    if xlabel == "MCXC" and ylabel == "ACT-DR5":
        mask = y[ycol][ymask] > 10 * x[xcol][xmask]
        print(x["name", "OName", "AName", "z", "m200"][xmask][mask])
        print(y["name", "SNR", "z", "M500cCal", "m200"][ymask][mask])
    ax.plot(10**xrng, 10**xrng, "-", color="0.5", lw=2, zorder=10, label="y = x")
    ax.legend(loc="upper left", fontsize=15)
    xsub = xcol[1:].replace("_", ",")
    ysub = ycol[1:].replace("_", ",")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=f"$M_\\mathrm{{{xsub}}}^\\mathrm{{{xlabel}}}$ (M$_\odot$)",
        ylabel=f"$M_\\mathrm{{{ysub}}}^\\mathrm{{{ylabel}}}$ (M$_\odot$)",
        xlim=(3e13, 5e15),
        ylim=(8e13, 5e15),
    )
    output = f"plots/compare_masses/masses_{xlabel.lower()}_{ylabel.lower()}"
    if suffix:
        output = f"{output}_{suffix}"
    for ext in ("png", "pdf"):
        savefig(f"{output}.{ext}", fig=fig, tight=False)


def fit_mass_ratio(
    mass_x, mass_y, emass_y=None, emass_x=None, p0=(1, 1), use_linmix=False
):
    x = np.log10(mass_x)
    y = np.log10(mass_y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    # ex = (to_log(mass_x, which="lower")[0] + to_log(emass_x, which="upper")[0]) / 2
    if emass_y is None:
        ey = None
    else:
        ey = (
            to_log(mass_y, emass_y, which="lower")[1]
            + to_log(mass_y, emass_y, which="upper")[1]
        ) / 2
        ey = ey[mask]
    if emass_x is None:
        ex = None
    else:
        ex = (
            to_log(mass_x, emass_x, which="lower")[1]
            + to_log(mass_x, emass_x, which="upper")[1]
        ) / 2
    # print(ey)
    if use_linmix:
        lm = LinMix(x, y, xsig=ex, ysig=ey)
    fit, fitcov = curve_fit(line, x, y, p0=p0, sigma=ey, absolute_sigma=True)
    # fit = mcmc(mass_x, mass_y, x2err=emass_y, output=[50, 16, 84])
    # fit = kelly(mass_x, mass_y, x2err=emass_y, output="percentiles")
    print(fit)
    return fit, fitcov
    fit, fiterr = np.transpose(bces(mass_x, mass_y, x2err=emass_y, full_output=False))
    print(fit)
    return fit, fiterr


def line(x, a, b=1):
    return a + x * b


def load_codex():
    codex = Table(fits.open("aux/xray/codex10_eromapper_Xmass.fits")[1].data)
    # M200 from velocity dispersion
    Hz = cosmo.H(codex["z_lambda"]).value
    Ez = Hz / cosmo.H0.value
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
    m200c = 1e15 * (codex["sigma_v"] / 1177) ** (1 / 0.364) / (Hz / 100)
    # correct uncertainties too
    codex["eM200c"] = codex["eM200c"] * (m200c / codex["M200c"])
    codex["M200c_sigma"] = m200c
    # codex["M500c"] = codex["M200c"] * codex["M500/M200"]
    # codex["eM500c"] = codex["eM200c"] * codex["M500/M200"]
    codex = codex[codex["lambda"] > 33.5 * 0.7 * (codex["z_lambda"] / 0.15) ** 0.8]
    codex["coords"] = SkyCoord(codex["RA_X-ray"], codex["Dec_X-ray"], unit="deg")
    codex.rename_column("z_lambda", "z")
    return codex


def load_meneacs():
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
    meneacs.catalog["m200"] *= 1e14
    meneacs.catalog["m200_err"] *= 1e14
    return meneacs


def load_wings():
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
    wings["m200_err"] = (
        wings["m200"]
        * (wings["sigma_cl_err_min"] + wings["sigma_cl_err_max"])
        / wings["sigma_cl"]
    )
    wings["m200_err"][wings["m200_err"] < 1] = (
        0.1 * wings["m200"][wings["m200_err"] < 1]
    )
    wings = join(
        wings, wings_positions["name", "ra", "dec"], keys="name", join_type="left"
    )
    wings = ClusterCatalog("WINGS", wings, base_cols=("name", "ra", "dec", "z"))
    wings.catalog["m200"].format = "%.2e"
    return wings


def match_catalogs(ref, cat, radius=5 * u.arcmin, dz=0.1):
    dist = ref["coords"].separation(cat["coords"][:, None])
    closest = np.argmin(dist, axis=0)
    matches = (np.min(dist, axis=0) < radius) & (
        np.abs(ref["z"] - cat["z"][closest]) / ref["z"] <= dz
    )
    return closest, matches


if __name__ == "__main__":
    main()
