"""Mass redshift plots"""

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astroquery.ipac.ned import Ned
from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from astro.clusters import ClusterCatalog
from astro.footprint import Footprint
from plottery.plotutils import savefig, update_rcParams

from compare_masses import match_catalogs

update_rcParams()


def main():
    filename = f"catalogues/clusters_chances_<sample>_<date>_large.csv"
    lowz = Table.read(
        filename.replace("<sample>", "lowz").replace("<date>", "20250120"),
        format="csv",
        comment="#",
    )
    evol = Table.read(
        filename.replace("<sample>", "evolution").replace("<date>", "20240725"),
        format="csv",
        comment="#",
    )
    plot_sample("lowz", lowz, other=None, south=True)
    # plot_sample("evolution", evol, color="C0", south=True)
    # plot_sample("lowz", lowz, other=None, color_other="C0")
    # plot_sample("evolution", evol, color="C0")
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
    # filename = "aux/xray/20240928/Xmass_2mrs_erass1_c89b022_n3_2ways_nh_full.fits"
    cols = [
        "C3ID",
        "RA_X",
        "DEC_X",
        "z",
        "M200c",
        "eM200c",
        "R200_deg",
        "Lx",
        "eLx",
        "ngal",
        "glat",
    ]
    axes2mrs = Table(fits.open(filename)[1].data)
    # axes2mrs = axes2mrs[(np.abs(axes2mrs["glat"]) > 20)]  # & (axes2mrs["ngal"] > 5)]
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
    # mark very nearby massive systems
    j = (
        ((axes2mrs["z"] < 0.03) & (axes2mrs["M200c"] > 2e14))
        | (axes2mrs["M200c"] > 1e15)
    ) & (axes2mrs["dec"] < 5)
    nearby = axes2mrs["name", "ra", "dec", "glat", "z", "ngal", "M200c", "R200_deg"][j]
    matches = ["" for i in nearby]
    print("Querying NED...")
    for i, row in tqdm(enumerate(axes2mrs[j]), total=j.sum()):
        try:
            query = Ned.query_region(
                SkyCoord(ra=row["ra"], dec=row["dec"], unit="deg"), radius=10 * u.arcmin
            )
        except TimeoutError:
            matches[i] = "Timeout Error"
        else:
            clusters = query["Type"] == "GClstr"
            if clusters.sum() > 0:
                matches[i] = " / ".join(query["Object Name"][clusters].value)
    nearby.add_column(matches, name="NED matches")
    print("AXES2MRS nearby:")
    print(nearby)
    # AXES2MRS nearby (dec<+5):
    # name       ra       dec       glat     z    ngal  M200c                                             NED matches
    # -------- --------- --------- --------- ------ ---- -------- -----------------------------------------------------------------------------
    # 93282301 243.59511 -60.90638  -7.31817 0.0131   95 3.93e+14                    PSZ2 G325.17-07.05 / SRGA J161415.5-605124 / Norma Cluster <-- low b
    # 93222625 159.19658 -27.51791  26.31679  0.014   76 2.52e+14                                                  Hydra CLUSTER / [AAA2014] 15 <-- CHANCES
    # 93242801 192.22251 -41.31311  21.80524 0.0108   58 3.25e+14       Centaurus Cluster / PSZ2 G302.41+21.60 / RXGCC 465 / PSZ2 G302.49+21.53 <-- Abell 3526
    # 93223401 207.27802 -30.32663  30.79858  0.016   32 3.16e+14                                                                    ABELL 3574 <-- CHANCES
    # 93203508  198.8426 -16.39033  46.19601 0.0069   22 1.28e+14                                                 RXGCC 494 / MCXC J1315.3-1623 <-- NGC 5044, no r-band
    # 93171208  64.90373   2.40608 -31.98677 0.0121   14 1.63e+14                            NSC J041905+022732 / RXGCC 168 / MCXC J0419.6+0224 <-- NGC 1550, missing r-band patches
    # 93233201 204.01947 -34.22526  28.09796 0.0134    9 2.15e+14                                                                               <-- 10' from IC 4296, missing 2 tiny r-band patches
    # 93272201 201.12287 -57.62385   5.34737  0.018    8 3.53e+14                                     SRGA J132436.0-573255 / CIZA J1324.7-5736 <-- low b
    # 93193506 193.03026 -13.44804  49.54785 0.0156    6 1.10e+14                                                                               <-- NGC 4748, missing r-band patches
    # 93311301  280.1562 -77.18339 -25.74794 0.0182    6 1.31e+14                                                             MCXC J1840.6-7709 <-- high dust region?
    # 93300501  74.48485 -75.44926 -33.21634 0.0187    5 1.41e+14                                                                               <-- Decals images screwed by bright star HD32440
    # 93171301  69.44657  -2.53387  -31.1901 0.0147    4 1.19e+14                                                                               <-- ~5 arcmin from bright star HD29391, ~10 deg from LSDR10 edge
    # 93183401 189.93338  -5.35906  57.56812 0.0101    4 1.84e+14                                                                               <-- Nothing obviously wrong, coincides with spiral NGC 4593
    # 93252706 198.76982 -42.61432  20.17052 0.0116    3 1.19e+14                                                                               <-- 1 deg from LSDR10 edge
    # 93171413  74.57499   0.45213 -24.71935 0.0138    3 1.31e+14                                                          WHL J045754.8+002152 <-- Coincides exactly with bright star HD31738, ~10 deg from LSDR10 edge
    # 93254501 332.33778 -47.17533 -52.76214 0.0051    3 1.65e+14                                                                               <-- 10 arcmin from bright star Alnair (HD209952)
    # M200c > 1e15 but no so nearby (dec<+5):
    # 93214401 258.12451 -23.39679   9.38415 0.0296   41 1.52e+15                                     Ophiuchus CLUSTER / SRGA J171226.3-232138 <-- low b
    # 93233220 206.83605 -32.86714  28.45098 0.0394   17 1.01e+15                          ABELL 3571 / PSZ2 G316.31+28.53 / 2MASSCL J1347-3257 <-- CHANCES
    # 93282302 249.55788 -64.36029 -11.53301 0.0517    8 1.32e+15                       SRGA J163813.2-642121 / CIZA J1638.2-6420 / TrA CLUSTER <-- low b
    # 93192510 137.23423   -9.6581  24.80082 0.0559    8 1.07e+15              ABELL 0754 / 2MASSCL J0908-0938 / PSZ2 G239.29+24.75 / RXGCC 309 <-- CHANCES
    # 93292021 248.08389 -67.46555 -12.86244 0.0462    3 1.18e+15                                                                               <-- low b
    # 93273325 303.10736 -56.79502 -33.37944 0.0553    3 1.03e+15 ABELL S0854 / PSZ2 G340.88-33.36 / RXGCC 816 / SPT-CL J2012-5649 / ABELL 3667 <-- CHANCES
    # 93233116 202.18538 -31.54277  30.76226 0.0478    3 1.00e+15                      RXGCC 506 / 2MASSCL J1328-3132 / ABELL 3558:[RBP2007] B1 <-- Shapley
    axes2mrs = axes2mrs[np.abs(axes2mrs["glat"]) > 20]
    insc = in_superclusters(axes2mrs)
    print(f"Excluding {insc.sum()} AXES-2MRS clusters in the supercluster regions")
    axes2mrs = axes2mrs[~insc]
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
    insc = in_superclusters(codex)
    print(f"Excluding {insc.sum()} CODEX clusters in the supercluster regions")
    codex = codex[~insc]
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
    psz.catalog = psz[
        (psz["name"] != "PSZ2 G302.49+21.53") & (np.abs(psz["GLAT"]) > 20)
    ]
    psz_insc = in_superclusters(psz)
    print(f"Excluding {psz_insc.sum()} PSZ2 clusters in the supercluster regions")
    # psz.catalog = psz[~psz_insc]
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
        axes2mrs_psz_closest, axes2mrs_in_psz = match_catalogs_d200(
            psz, axes2mrs, debug=True
        )
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

        printcols = ["name", "z", "ra", "dec", "MSZ", "m200", "ACT", "MCXC", "COMMENT"]
        print("Nearby missed PSZ2:")
        print(psz[printcols][(psz["z"] > 0) & (psz["z"] < 0.02)])
    else:
        act_psz_closest, act_in_psz = match_catalogs(psz, act, radius=match_radius)
        chances_act_closest, chances_in_act = match_catalogs(
            act, chances, radius=match_radius
        )
        print(
            f"PSZ: {psz['name'].size}, ACT-DR5 matches: {act_in_psz.sum()}, CHANCES matches: {chances_in_psz.sum()}"
        )
        print(f"ACT-DR5: {act['name'].size}, CHANCES  matches: {chances_in_act.sum()}")
        # Last one is a duplicate of A3526
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
        plot_lowz(ax, chances, axes2mrs, psz)
    ax.legend(loc="lower right", fontsize=12)
    ax.set(xlabel="Redshift", ylabel="$M_{200}$ (M$_\odot$)", yscale="log")
    output = f"plots/sample_{sample}.pdf"
    if south:
        output = output.replace(".pdf", "_south.pdf")
    savefig(output, fig=fig, tight=False)
    return


def match_catalogs_d200(ref, cat, f=0.5, min_radius=5 * u.arcmin, debug=False):
    """Like compare_masses.match_catalogs but using each cluster's r200 as a reference"""
    dist = ref["coords"].separation(cat["coords"][:, None])
    if debug:
        j = cat["z"] < 0.02
    match_radius = min_radius * np.ones(cat["coords"].size)
    if "d200" in cat.colnames:
        match_radius = (
            np.max([f * cat["d200"] * u.arcmin, match_radius], axis=0) * u.arcmin
        )
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


def plot_lowz(ax, chances, axes2mrs, psz):
    # ax.plot(
    #     erass["z"], erass["m200"], "C2+", ms=3, mew=1, zorder=-1, label="eRASS1"
    # )
    chances["name"] = [
        name.replace(" 0", " ").replace("bell ", "") for name in chances["name"]
    ]
    j = axes2mrs["ngal"] >= 5
    ax.plot(
        axes2mrs["z"][j],
        axes2mrs["M200c"][j],
        "C2+",
        ms=4,
        mew=1.5,
        alpha=1,
        label="AXES-2MRS ($N_\\mathrm{gal}\geq5$)",
    )
    # j = axes2mrs["ngal"] < 5
    # ax.plot(
    #     axes2mrs["z"][j],
    #     axes2mrs["M200c"][j],
    #     "C0+",
    #     ms=4,
    #     mew=1.5,
    #     alpha=1,
    #     label="AXES-2MRS (n<5)",
    # )
    # mark very nearby massive systems missed by CHANCES
    # j = (axes2mrs["z"] < 0.02) & (axes2mrs["M200c"] > 1e14)
    # ax.plot(
    #     axes2mrs["z"][j], axes2mrs["M200c"][j], "o", mec="k", mfc="none", ms=8, mew=1
    # )
    # j = (psz["z"] > 0) & (psz["z"] < 0.02) & (psz["m200"] > 1e14)
    # ax.plot(psz["z"][j], psz["m200"][j], "o", mec="k", mfc="none", ms=8, mew=1)
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
        (chances["z"][j][0] - 0.001, 1.01 * chances["m200"][j][0]),
        ha="right",
        va="bottom",
        # ha="left",
        # va="center",
        # xytext=(0.01, 6e14),
        # arrowprops=dict(arrowstyle="->"),
        **kw,
    )
    # j = chances["name"] == "A3526"
    # ax.annotate(
    #     "A3526",
    #     xy=(chances["z"][j], 1.01 * chances["m200"][j]),
    #     ha="right",
    #     va="bottom",
    #     **kw,
    # )
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


def in_superclusters(cat):
    shapley = Footprint(
        "Shapley",
        footprint=np.array(
            [[[192, -36], [192, -26], [207, -26], [207, -36], [192, -36]]]
        ),
    )
    horologium = Footprint(
        "Horologium-Reticulum",
        footprint=np.array(
            [
                [
                    [46.5, -51],
                    [49, -60.7],
                    [66, -60.7],
                    [66, -51],
                    [55, -51],
                    [55, -47.5],
                    [58.5, -47.5],
                    [58.5, -37],
                    [46.5, -37],
                    [46.5, -51],
                ]
            ]
        ),
    )
    mask = shapley.in_footprint(cat["ra"], cat["dec"]) | horologium.in_footprint(
        cat["ra"], cat["dec"]
    )
    return mask


main()
