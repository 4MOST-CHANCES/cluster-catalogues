"""Mass redshift plots"""

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from icecream import ic
from matplotlib import pyplot as plt
import numpy as np

from astro.clusters import ClusterCatalog
from plottery.plotutils import savefig, update_rcParams

from compare_masses import match_catalogs

update_rcParams()


def main():
    filename = f"catalogues/clusters_chances_<sample>_<date>_large.csv"
    lowz = Table.read(
        filename.replace("<sample>", "lowz").replace("<date>", "20241009"),
        format="csv",
        comment="#",
    )
    evol = Table.read(
        filename.replace("<sample>", "evolution").replace("<date>", "20240725"),
        format="csv",
        comment="#",
    )
    plot_sample("lowz", lowz, other=evol, color_other="C0", south=True)
    plot_sample("evolution", evol, color="C0", south=True)
    plot_sample("lowz", lowz, other=None, color_other="C0")
    plot_sample("evolution", evol, color="C0")
    return


def clustercat(name, suffix="", mcol="m200"):
    cat = ClusterCatalog(name)
    m200 = Table.read(f"aux/m200_converted/{name}_m200{suffix}.csv")
    cat.catalog["m200"] = -np.ones(cat.size)
    cat.catalog["m200"][cat.z > 0] = m200["m200"]
    cat.catalog[np.isnan(cat["m200"])] = -1
    if cat["m200"].max() < 1e10:
        cat.catalog["m200"][cat.z > 0] = 1e14 * cat["m200"][cat.z > 0]
    cat.catalog["m200"].format = "%.2e"
    if "coords" not in cat.colnames:
        cat.catalog["coords"] = SkyCoord(ra=cat["ra"], dec=cat["dec"], unit="deg")
    return cat


def load_axes2mrs():
    filename = "aux/xray/Xmass_BayesGroups_n3ext200kpc_nofake_info2.fits"
    filename = "aux/xray/20240928/Xmass_2mrs_erass1_c89b022_n3_2ways_nh_full.fits"
    cols = ["C3ID", "RA_X", "DEC_X", "zcmb", "M200c", "eM200c", "Lx", "eLx"]
    axes2mrs = Table(fits.open(filename)[1].data)
    axes2mrs = ClusterCatalog(
        "axes-2mrs",
        axes2mrs,
        cols=cols,
        base_cols=cols[:4],
        label="AXES-2MRS",
        masscol="M200c",
    )
    axes2mrs.catalog["coords"] = SkyCoord(
        ra=axes2mrs["ra"], dec=axes2mrs["dec"], unit="deg"
    )
    axes2mrs["M200c"].format = ".2e"
    return axes2mrs


def load_codex():
    """this is point sources, good for evolution"""
    filename = "aux/xray/codex10_eromapper_Xmass.fits"
    codex = Table(fits.open(filename)[1].data)
    codex = codex[codex["lambda"] > 33.5 * 0.7 * (codex["z_lambda"] / 0.15) ** 0.8]
    # codex = codex[codex['codex50']]
    codex.rename_columns(["ra_opt", "dec_opt"], ["ra_bcg", "dec_bcg"])
    cols = [
        "id_cluster",
        "RA_X-ray",
        "Dec_X-ray",
        "z_lambda",
        "ra_bcg",
        "dec_bcg",
        "M200c",
    ]
    codex["id_cluster"] = np.array(codex["id_cluster"], dtype=str)
    codex = ClusterCatalog(
        "codex",
        codex,
        cols=cols,
        base_cols=cols[:4],
        label="CODEX",
        masscol="M200c",
    )
    return codex


def plot_sample(
    sample,
    chances,
    color="C3",
    other=None,
    color_other="k",
    south=False,
    match_radius=5 * u.arcmin,
):
    label = "Low-z" if sample == "lowz" else "Evolution"
    psz = clustercat("psz2")
    act = clustercat("act-dr5")
    if sample == "lowz":
        axes2mrs = load_axes2mrs()
    codex = load_codex()
    erass = clustercat("erass1")
    if south:
        mindec, maxdec = -80, +5
        psz = psz[(psz["dec"] > mindec) & (psz["dec"] < maxdec)]
        act = act[(act["dec"] > mindec) & (act["dec"] < maxdec)]
        erass = erass[(erass["dec"] > mindec) & (erass["dec"] < maxdec)]
        codex = codex[(codex["dec"] > mindec) & (codex["dec"] < maxdec)]
        if sample == "lowz":
            axes2mrs = axes2mrs[(axes2mrs["dec"] > mindec) & (axes2mrs["dec"] < maxdec)]
    chances["coords"] = SkyCoord(ra=chances["ra"], dec=chances["dec"], unit="deg")
    # remove duplicates. Note that I only implemented this for the catalogues in the current version of the plot!
    chances_psz_closest, chances_in_psz = match_catalogs_d200(psz, chances)
    # chances_psz_closest, chances_in_psz = match_catalogs(
    #     psz, chances, radius=match_radius
    # )
    if sample == "lowz":
        # axes2mrs_psz_closest, axes2mrs_in_psz = match_catalogs(
        #     psz, axes2mrs, radius=match_radius
        # )
        # chances_axes2mrs_closest, chances_in_axes2mrs = match_catalogs(
        #     axes2mrs, chances, radius=match_radius
        # )
        axes2mrs_psz_closest, axes2mrs_in_psz = match_catalogs_d200(psz, axes2mrs)
        chances_axes2mrs_closest, chances_in_axes2mrs = match_catalogs_d200(
            axes2mrs, chances
        )
        print(
            f"PSZ: {psz['name'].size}, AXES-2MRS matches: {axes2mrs_in_psz.sum()}, CHANCES matches: {chances_in_psz.sum()}"
        )
        print(
            f"AXES-2MRS: {axes2mrs['name'].size}, CHANCES matches: {chances_in_axes2mrs.sum()}"
        )
        psz = psz[~axes2mrs_in_psz & ~chances_in_psz]
        # Abell 3627 which is also in AXES-2MRS but farther than 5'
        psz = psz[psz["name"] != "PSZ2 G325.17-07.05"]
        axes2mrs = axes2mrs[~chances_in_axes2mrs]
        print(np.sort(psz.colnames))
        print(
            psz["name", "z", "ra", "dec", "MSZ", "ACT", "MCXC", "COMMENT"][
                (psz["z"] > 0) & (psz["z"] < 0.02)
            ]
        )
    else:
        act_psz_closest, act_in_psz = match_catalogs(psz, act, radius=match_radius)
        chances_act_closest, chances_in_act = match_catalogs(
            act, chances, radius=match_radius
        )
        print(
            f"PSZ: {psz['name'].size}, ACT-DR5 matches: {act_in_psz.sum()}, CHANCES matches: {chances_in_psz.sum()}"
        )
        print(f"ACT-DR5: {act['name'].size}, CHANCES  matches: {chances_in_act.sum()}")
        psz = psz[~act_in_psz & ~chances_in_psz]
        act = act[~chances_in_act]
    #
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.plot(
        chances["z"],
        chances["m200"],
        "*",
        color=color,
        ms=8,
        zorder=100,
        label=f"CHANCES {label}",
    )
    if other is not None:
        ax.plot(
            other["z"], other["m200"], "*", mec=color_other, mfc="none", ms=8, zorder=99
        )
    ax.plot(psz["z"], psz["m200"], "C4o", ms=4, mew=0, alpha=1, label="PSZ2")
    if sample == "evolution":
        plot_evolution(ax, chances, act)
    else:
        plot_lowz(ax, chances, axes2mrs)
    ax.legend(loc="lower right", fontsize=12)
    ax.set(xlabel="Redshift", ylabel="$M_{200}$ (M$_\odot$)", yscale="log")
    output = f"plots/sample_{sample}.pdf"
    if south:
        output = output.replace(".pdf", "_south.pdf")
    savefig(output, fig=fig, tight=False)
    return


def match_catalogs_d200(ref, cat, f=0.5, min_radius=5 * u.arcmin):
    """Like compare_masses.match_catalogs but using each cluster's r200 as a reference"""
    dist = ref["coords"].separation(cat["coords"][:, None])
    match_radius = min_radius * np.ones(cat["coords"].size)
    if "d200" in cat.colnames:
        match_radius = (
            np.max([f * cat["d200"] * u.arcmin, match_radius], axis=0) * u.arcmin
        )
    print(dist, match_radius)
    closest = np.argmin(dist, axis=0)
    matches = np.any(dist < match_radius[:, None], axis=0)
    return closest, matches


def plot_evolution(ax, chances, act):
    ax.plot(
        act["z"], act["m200"], "C1x", ms=4, mew=1, alpha=0.8, zorder=-1, label="ACT-DR5"
    )
    chances["name"] = [
        name.replace(" 0", " ").replace("bell ", "") for name in chances["name"]
    ]
    # notable clusters
    kw = {"fontsize": 10, "fontweight": "heavy"}
    jsort = list(np.argsort(chances["m200"]))
    for i, j in enumerate(jsort[:3]):
        ax.annotate(
            chances["name"][j],
            (2e-3 + chances["z"][j], 0.98 * chances["m200"][j]),
            ha="left",  # if i % 2 == 0 else "center",
            va="top",
            **kw,
        )
    for i, j in enumerate(jsort[-3::2]):
        ax.annotate(
            chances["name"][j],
            (2e-3 + chances["z"][j], 1.02 * chances["m200"][j]),
            ha="right",
            va="bottom",
            **kw,
        )
    ax.annotate(
        chances["name"][jsort[-2]],
        (chances["z"][jsort[-2]], 1.02 * chances["m200"][jsort[-2]]),
        xytext=(chances["z"][jsort[-2]], 4e15),
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="->"),
        **kw,
    )
    # added manually
    jcl = chances["name"] == "MACS J0329.7-0211"
    ax.annotate(
        chances["name"][jcl][0],
        (chances["z"][jcl][0], 0.98 * chances["m200"][jcl][0]),
        ha="center",
        xytext=(0.47, 5e14),
        arrowprops=dict(arrowstyle="->"),
        **kw,
    )
    # for z in (0.07, 0.45):
    #     ax.axvline(z, ls="--", lw=1, color="k")
    ax.fill_between([0, 0.07], 1e13, 5e15, color="0.9", zorder=-100)
    ax.fill_between([0.45, 0.8], 1e13, 5e15, color="0.9", zorder=-100)
    ax.set(xlim=(0, 0.6), ylim=(1e14, 5e15))


def plot_lowz(ax, chances, axes2mrs):
    # ax.plot(
    #     erass["z"], erass["m200"], "C2+", ms=3, mew=1, zorder=-1, label="eRASS1"
    # )
    chances["name"] = [
        name.replace(" 0", " ").replace("bell ", "") for name in chances["name"]
    ]
    ax.plot(
        axes2mrs["z"],
        axes2mrs["M200c"],
        "C2+",
        ms=4,
        mew=1.5,
        alpha=1,
        zorder=101,
        label="AXES-2MRS",
    )
    j = (axes2mrs["z"] < 0.02) & (axes2mrs["M200c"] > 1e14)
    ax.plot(
        axes2mrs["z"][j], axes2mrs["M200c"][j], "o", mec="k", mfc="none", ms=8, mew=1
    )
    print(
        axes2mrs["name", "z", "ra", "dec", "M200c"][
            (axes2mrs["z"] < 0.02) & (axes2mrs["M200c"] > 1e14)
        ]
    )
    # least and most massive
    kw = {"fontsize": 10, "fontweight": "bold", "zorder": 1000}
    jsort = list(np.argsort(chances["m200"]))
    jshow = jsort[:5] + jsort[-1:]
    for j in jshow:
        if chances["name"][j] == "Antlia":
            continue
        ax.annotate(
            chances["name"][j],
            (1e-3 + chances["z"][j], 0.99 * chances["m200"][j]),
            ha="left",
            va="top",
            **kw,
        )
    # most massive at low z
    show = (chances["z"] < 0.04) & (chances["m200"] > 3e14)
    for cl in chances[show]:
        ax.annotate(
            cl["name"],
            (cl["z"], 1.01 * cl["m200"]),
            ha="right",
            va="bottom",
            **kw,
        )
    # this is only one cluster but I need an arrow
    # show = (chances["z"] < 0.02) & (chances["m200"] > 3e14)
    # for cl in chances[show]:
    #     ax.annotate(
    #         cl["name"],
    #         (cl["z"], 1.01 * cl["m200"]),
    #         ha="left",
    #         va="bottom",
    #         xytext=(0.002, 1.6e15),
    #         arrowprops=dict(arrowstyle="->"),
    #         **kw,
    #     )
    # nearest clusters
    j = chances["name"] == "Hydra (A1060)"
    ax.annotate(
        "Hydra",
        (chances["z"][j][0] - 0.001, 0.99 * chances["m200"][j][0]),
        ha="right",
        va="top",
        # ha="left",
        # va="center",
        # xytext=(0.01, 6e14),
        # arrowprops=dict(arrowstyle="->"),
        **kw,
    )
    j = chances["name"] == "A3526"
    ax.annotate(
        "A3526",
        xy=(chances["z"][j], 1.01 * chances["m200"][j]),
        ha="right",
        va="bottom",
        **kw,
    )
    j = chances["name"] == "Antlia"
    ax.annotate(
        "Antlia",
        (0.008, 0.94 * chances["m200"][j][0]),
        ha="center",
        va="top",
        # ha="left",
        # va="bottom",
        # xytext=(1e-3, 4e14),
        # arrowprops=dict(arrowstyle="->"),
        **kw,
    )
    # j = chances["WINGS_idx"] != -99
    # ax.plot(
    #     chances["z"][j],
    #     chances["m200"][j],
    #     "C9o",
    #     ms=4,
    #     mew=0,
    #     zorder=11,
    #     label="in WINGS",
    # )
    # low-z not in CHANCES
    # kw = {"fontsize": 8, "fontweight": "normal", "zorder": 1000}
    # ax.annotate("NGC 5044", xy=(0.001, 1.7e14), ha="left", va="center", **kw)
    # ax.annotate("A3627", xy=(0.019, 3.2e14), ha="left", va="center", **kw)
    # ax.plot([0.017, 0.0188, 0.017], [2.79e14, 3.2e14, 3.45e14], "k-", lw=0.6)
    # ax.axvline(0.07, lw=0.8, dashes=(6, 6), color="k")
    ax.fill_between([0.07, 0.1], 1e13, 5e15, color="0.9", zorder=-100)
    ax.set(xlim=(0, 0.1), ylim=(1e13, 2.5e15), xticks=np.arange(0, 0.11, 0.02))


main()
