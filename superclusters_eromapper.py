from astropy import units as u
from astropy.constants import c as clight
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
from astropy.table import Table, join, join_skycoord, hstack
import cmasher as cmr
from datetime import date
from icecream import ic
from matplotlib import cm, pyplot as plt, ticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Ellipse
import multiprocessing as mp
import numpy as np
from time import time
import sys

import ligo.skymap.plot
from ligo.skymap.plot.poly import subdivide_vertices
from plottery.plotutils import savefig, update_rcParams
from profiley.helpers.spherical import radius_from_mass

from astro.clusters import ClusterCatalog
from astro.clusters.catalog import Abell
from astro.footprint import Footprint

from collate_info import parse_args
from tools import scalebar_label

update_rcParams()

clight = clight.to("km/s").value


def main():
    args = parse_args()
    args.sample = "lowz"
    chances = ClusterCatalog(f"chances-{args.sample[:4]}")
    print(chances)

    shapley = Footprint(
        "Shapley",
        footprint=np.array(
            [[[192, -36], [192, -26], [207, -26], [207, -36], [192, -36]]]
        ),
    )
    shapley.z = 0.048
    shapley.ra = 200
    shapley.dec = -31
    shapley.xlim = (190, 208)
    shapley.ylim = (-37.5, -25)

    horologium = Footprint(
        "Horologium-Reticulum",
        # original
        # footprint=np.array([[[44, -60.4], [60, -60.4], [60, -62], [63, -62], [63, -60.4],
        # [64.5, -60.4], [64.5, -59], [66, -59], [66, -50], [55, -50], [55, -40], [44, -40]]]))
        # new
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
    horologium.z = 0.060
    horologium.ra = 55.5
    horologium.dec = -50.5
    horologium.xlim = (40, 70)
    horologium.ylim = (-64, -35)

    cat = load_eromapper()
    # we don't need the rest
    cat.catalog = cat[(cat["best_z"] > 0.002) & (cat["best_z"] < 0.1)]
    print(np.sort(cat["best_z"]))
    print(np.sort(cat["z"]))
    return

    # need this first pass to find clusters in the SCs
    colors = [[0, 0, 1], [0.4, 0.4, 0.4], [1, 0, 0]]
    nodes = [0, 0.5, 1]
    cmap = LinearSegmentedColormap.from_list("vmap", list(zip(nodes, colors)))
    kwargs = dict(cmap=cmap, alpha=0.8, vmax=8000, show_chances=False)

    ncores = 1
    if ncores == 1:
        shapley.clusters = plot_supercluster(
            cat, shapley, chances, show_primary=False, **kwargs
        )
        horologium.clusters = plot_supercluster(
            cat, horologium, chances, show_primary=False, **kwargs
        )
        vmax = kwargs.get("vmax", 8000)
        shapley.primary, shapley.secondary = primaries(shapley, vmax=vmax)
        horologium.primary, horologium.secondary = primaries(horologium, vmax=vmax)
        # to add primaries to the figure
        shapley.clusters = plot_supercluster(cat, shapley, chances, **kwargs)
        horologium.clusters = plot_supercluster(cat, horologium, chances, **kwargs)
    else:
        with mp.Pool(2) as pool:
            pool.apply_async(run_sc, args=(cat, shapley, chances), kwds=kwargs)
            pool.apply_async(run_sc, args=(cat, horologium, chances), kwds=kwargs)
            pool.close()
            pool.join()

    fig, ax = plt.subplots(figsize=(20, 1.2), constrained_layout=True)
    cbar = fig.colorbar(
        cm.ScalarMappable(
            cmap=kwargs["cmap"],
            norm=Normalize(vmin=-kwargs["vmax"], vmax=kwargs["vmax"]),
        ),
        cax=ax,
        label="$v_\\mathrm{pec}$ (km s$^{-1}$)",
        orientation="horizontal",
    )
    # cbar.set_label("$\Delta z$", fontsize=20)
    # cbar.ax.xaxis.set_major_locator(ticker.MultipleLocator(0.002))
    output = "plots/superclusters/eromapper_sc_colorbar"
    for ext in ("pdf", "png"):
        savefig(f"{output}.{ext}", fig=fig, tight=False)

    return


def run_sc(cat, sc, chances, **kwargs):
    sc.clusters = plot_supercluster(cat, sc, chances, show_primary=False, **kwargs)
    vmax = kwargs.get("vmax", 8000)
    print("vmax =", vmax)
    sc.primary, sc.secondary = primaries(sc, vmax=vmax)
    sc.clusters = plot_supercluster(cat, sc, chances, **kwargs)
    return sc


def primaries(sc, vmax=8000):
    """Sift through the catalogue to find primaries"""
    sc.clusters["coords"] = SkyCoord(
        ra=sc.clusters["ra"], dec=sc.clusters["dec"], unit="deg"
    )
    print(sc.name)
    sc.clusters.sort("m200")
    sc.clusters.reverse()
    # print(sc.clusters)
    primary = []
    secondary = []
    dist = sc.clusters["coords"].separation(sc.clusters["coords"][:, None])
    within_5r200 = dist < 5 * sc.clusters["d200"] * u.deg
    within_z = (
        clight
        * np.abs(sc.clusters["z"] - sc.clusters["z"][:, None])
        / (1 + sc.clusters["z"][:, None])
        < vmax
    )
    for i, cl in enumerate(sc.clusters):
        bigger = (sc.clusters["m200"] > cl["m200"]) & ~np.isin(
            sc.clusters["name"], secondary
        )
        is_inside_primary = np.any(within_5r200[i] & within_z[i] & bigger)
        # is_inside_primary
        primary_is_inside = np.any(within_5r200[:, i] & within_z[:, i] & bigger)
        if is_inside_primary:  # or primary_is_inside:
            secondary.append(cl["name"])
            continue
        primary = np.append(primary, cl["name"])
    primary = np.array(primary)
    secondary = np.unique(secondary)
    print(f"{primary.size} primaries")
    # print(primary)
    print(sc.clusters["name"][~np.isin(sc.clusters["name"], secondary)])
    print(f"{secondary.size} secondaries")
    # print(secondary)
    print(sc.clusters["m200"].size == primary.size + secondary.size)
    for tbl, tbl_name in zip((primary, secondary), ("primary", "secondary")):
        np.savetxt(
            f"catalogues/eromapper/{sc.name.lower()}_eromapper_{tbl_name}.tbl",
            tbl,
            fmt="%d",
        )
    return primary, secondary


def plot_supercluster(
    cat,
    sc,
    chances,
    show_primary=True,
    show_abell=True,
    show_chances=True,
    barsize=10 * u.Mpc,
    cmap="cmr.wildfire",
    alpha=0.8,
    dz=0.01,
    vmax=5000,
):
    """barsize in Mpc"""
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    # how many clusters are in the Shapley region?
    in_sc = sc.in_footprint(cat["ra"], cat["dec"])
    # added by hand as they are just at the boundary
    # if sc.name == "Shapley":
    #     in_sc = in_sc | (cat["name"] == "Abell 3571")
    # if sc.name == "Horologium-Reticulum":
    #     a3266 = (
    #         (cat["ra"] > 67) & (cat["dec"]) & (cat["dec"] > -62) & (cat["dec"] < -61)
    #     )
    #     in_sc = in_sc | a3266
    in_sc_extended = (
        (cat["ra"] > sc.xlim[0])
        & (cat["ra"] < sc.xlim[1])
        & (cat["dec"] > sc.ylim[0])
        & (cat["dec"] < sc.ylim[1])
    )
    # cat.catalog.sort("m200")
    # cat.catalog = cat.catalog[::-1]
    zmin = sc.z - dz
    zmax = sc.z + dz
    vpec = clight * (cat["z"] - sc.z) / (1 + sc.z)
    zmin = sc.z - vmax * (1 + sc.z) / clight
    zmax = sc.z + vmax * (1 + sc.z) / clight
    # in_sc_z = (cat["z"] >= zmin) & (cat["z"] <= zmax)
    in_sc_z = np.abs(vpec) < vmax
    print("z_rng = ", zmin, zmax, cat["z"][in_sc_z].min(), cat["z"][in_sc_z].max())
    print("in_sc:", in_sc.size, in_sc.sum(), (in_sc & in_sc_z).sum())

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    plotradius = 10 if sc.name == "Shapley" else 15
    print(sc.ra, sc.dec, plotradius, get_circle_center(sc.ra, sc.dec))
    ax = plt.axes(
        projection="astro zoom",
        center=get_circle_center(sc.ra, sc.dec),
        radius=f"{plotradius} deg",
        # rotate="0 deg",
    )
    ax.set_title(f"{sc.name} (z={sc.z:.3f})")
    _ = [
        ax.mark_inset_circle(
            ax,
            get_circle_center(ra, dec),
            radius=get_circle_radius(d200),
            # facecolor=cmap((vp - vmin) / (vmax - vmin)),
            facecolor=cmap((z - zmin) / (zmax - zmin)),
            lw=0,
            alpha=0.5,
        )
        for ra, dec, d200, z, vp in zip(
            cat["ra"][in_sc_z],
            cat["dec"][in_sc_z],
            cat["d200"][in_sc_z],
            cat["z"][in_sc_z],
            vpec[in_sc_z],
        )
    ]
    ax.set(xlabel="Right Ascension", ylabel="Declination")

    if sc.name == "Shapley":
        special = chances[np.isin(chances["name"], ["Abell 3571"])]
    elif sc.name == "Horologium-Reticulum":
        special = chances[np.isin(chances["name"], ["Abell 3266"])]
    for cl in special:
        # remember that the CHANCES d200 are in arcmin!
        ax.plot(
            cl["ra"],
            cl["dec"],
            "C0+",
            mew=2,
            ms=18,
            transform=ax.get_transform("world"),
        )
        ax.mark_inset_circle(
            ax,
            get_circle_center(cl["ra"], cl["dec"]),
            get_circle_radius(5 * cl["d200"] / 60),
            facecolor="none",
            edgecolor="C0",
            lw=2,
        )
        ax.text_coord(
            SkyCoord(ra=cl["ra"] + 0.1, dec=cl["dec"] - 0.1, unit="deg"),
            cl["name"],
            ha="right",
            va="top",
            color="C0",
            fontsize=14,
            fontweight="bold",
        )
    if show_abell and hasattr(sc, "primary"):
        # these selected by hand
        i = 0
        sccat = cat[in_sc & in_sc_z]
        sccat.sort("m200")
        sccat.reverse()
        shown = []
        ha_dict = dict(
            right=("A3089", "A3110", "A3164", "A3535", "AS442", "AS733"),
            center=("A3128", "AS726"),
        )
        va_dict = dict(
            top=("A3111", "A3158", "A3225", "A3532", "AS726"),
            center=("A3110", "A3528", "A3530", "A3535", "A3564"),
        )
        for cl in sccat:
            if cl["name"] in special["name"]:  # or (cl["name"] not in sc.primary):
                continue
            abell = Abell.query(
                ra=np.array([cl["ra"]]),
                dec=np.array([cl["dec"]]),
                radius=20 * u.arcmin,
            )
            print(abell)
            if abell["name"].size == 0:
                continue
            aname = abell["name"][0].replace("bell ", "")
            if aname in shown:
                continue
            shown.append(aname)
            coord = SkyCoord(ra=cl["ra"], dec=cl["dec"], unit="deg")
            ax.plot_coord(coord, "kx", ms=6, zorder=1000)
            # text alignment
            ha = [key for key in ha_dict if aname in ha_dict.get(key)]
            ha = ha[0] if len(ha) == 1 else "left"
            va = [key for key in va_dict if aname in va_dict.get(key)]
            va = va[0] if len(va) == 1 else "bottom"
            if va == "center" and ha != "center":
                dra = 20 * u.arcmin * (-1) ** (ha == "left")
            else:
                dra = 0 * u.arcmin
            if va == "top" and ha == "center":
                ddec = -20 * u.arcmin
            else:
                ddec = 0 * u.arcmin
            ax.text_coord(
                coord.spherical_offsets_by(dra, ddec),
                aname,
                ha=ha,
                va=va,
                color="k",
                fontsize=14,
                fontweight="bold",
            )
            i += 1
            if (sc.name == "Shapley" and i == 7) or (
                sc.name == "Horologium-Reticulum" and i == 9
            ):
                break
    # if sc.name == "Horologium-Reticulum":
    # fornax = Circle((54.71135, -35.40093), 12.5, facecolor="none", edgecolor="0.5", lw=2)
    # fornax = Ellipse(
    #     (54.71135, -35.40093),
    #     2 * 12.5 / np.cos(np.pi / 180 * 35.4),
    #     2 * 12.5,
    #     facecolor="none",
    #     edgecolor="0.5",
    #     lw=2,
    # )
    # if use_skymap:
    #     ax.mark_inset_circle(
    #         ax,
    #         center=get_circle_center(*fornax.center),
    #         radius="12.5 deg",
    #         facecolor="none",
    #         edgecolor="0.5",
    #         lw=1,
    #     )
    # else:
    #     ax.add_patch(fornax)
    # ax.plot(
    #     *fornax.center,
    #     color="0.5",
    # )
    # additional = np.isin(Abell.obj, ["A3074", "A3078", "A3133"])
    # print(Abell.catalog[additional])
    # for cl in Abell.catalog[additional]:
    # ax.annotate(cl["name"], xy=(cl["ra"], cl["dec"]), ha="left", va="bottom", color="C3", fontsize=16)
    if show_primary:
        p = np.isin(cat["name"], sc.primary)
        for cl in cat[p]:
            # if cl["name"] in sc.primary:
            #     ec = "k"
            # elif cl["name"] in sc.secondary:
            #     ec = "none"
            ax.mark_inset_circle(
                ax,
                center=get_circle_center(cl["ra"], cl["dec"]),
                radius=get_circle_radius(cl["d200"]),
                fc="none",
                ec="k",
                lw=1,
            )
            ax.mark_inset_circle(
                ax,
                center=get_circle_center(cl["ra"], cl["dec"]),
                radius=get_circle_radius(5 * cl["d200"]),
                fc="none",
                ec="k",
                lw=1,
            )
    if show_chances:
        chances_color = (0.5, 0, 0.5)
        mask = (
            (chances["ra"] > sc.xlim[0])
            & (chances["ra"] < sc.xlim[1])
            & (chances["dec"] > sc.ylim[0])
            & (chances["dec"] < sc.ylim[1])
            # & (chances["z"])
            # & ~(sc.in_footprint(chances["ra"], chances["dec"]))
        )
        for cl in chances[mask]:
            ax.mark_inset_circle(
                ax,
                get_circle_center(cl["ra"], cl["dec"]),
                get_circle_radius(5 * cl["d200"] / 60),
                fc="none",
                ec=chances_color,
                lw=0.5,
            )
        for cl in chances[mask]:
            ax.text_coord(
                SkyCoord(cl["ra"] + 0.02, cl["dec"] - 0.05, unit="deg"),
                cl["name"],
                ha="left",
                va="top",
                color=chances_color,
                fontsize=12,
                # transform=ax.transAxes,
            )
        kw = dict(color=chances_color, ms=12, mew=3)
        kw["transform"] = ax.get_transform("world")
        ax.plot(chances[mask]["ra"], chances[mask]["dec"], "+", **kw)
    xy = subdivide_vertices(sc.footprint[0], 100)
    # print(xy, len(xy))
    ax.plot(
        # *np.transpose(sc.footprint[0]),
        *np.transpose(xy),
        "-",
        color=(0.2, 0.7, 0),
        lw=4,
        zorder=-100,
        transform=ax.get_transform("world"),
    )
    # b = (barsize * cosmo.arcsec_per_kpc_comoving(sc.z)).to(u.deg).value
    # xb = sc.xlim[0] + 0.1 * (sc.xlim[1] - sc.xlim[0])
    # yb = sc.ylim[0] + 0.1 * (sc.ylim[1] - sc.ylim[0])
    # ax.plot((xb, xb + b), (yb, yb), "C0-", lw=4)
    # ax.annotate(
    #     f"{barsize.value:.0f} cMpc",
    #     xy=(xb + b / 2, yb + 0.01 * (sc.ylim[1] - sc.ylim[0])),
    #     fontsize=15,
    #     ha="center",
    #     va="bottom",
    #     transform=ax.transAxes,
    # )
    b = (barsize * cosmo.arcsec_per_kpc_comoving(sc.z)).to(u.deg)
    bar = ax.scalebar(
        (0.84 - 0.05 * (sc.name == "Shapley"), 0.12),
        b,
        color="C1",
        lw=4,
        capstyle="butt",
        zorder=100,
    )
    scalebar_label(
        bar,
        f"{barsize.value:.0f} cMpc",
        color="C1",
        fontsize=13,
        fontweight="bold",
        pad=0.01,
    )
    ax.grid(True)
    name = sc.name.split("-")[0].lower()
    output = f"plots/superclusters/eromapper_sc_{name}"
    for ext in ("pdf", "png"):
        savefig(f"{output}.{ext}", fig=fig, tight=False)

    return cat[in_sc & in_sc_z]


def get_circle_center(ra, dec):
    return f"{ra}d {dec}d"


def get_circle_radius(r_deg):
    return f"{r_deg} deg"


def load_axes2mrs():
    filename = "aux/xray/Xmass_BayesGroups_n3ext200kpc_nofake_info.fits"
    cols = [
        "C3ID",
        "ra",
        "dec",
        "z",
        "Lx",
        "eLx",
        "M200c",
        "eM200c",
        "R200_deg",
    ]
    axes2mrs = Table(fits.open(filename)[1].data)[cols]
    axes2mrs.rename_columns(["C3ID", "R200_deg"], ["AXES", "d200"])
    axes2mrs["2MRS"] = 1
    return axes2mrs


def load_axesls():
    filename = "aux/xray/Xmass_c3_LSDR10updated_info_nofilters_clean.fits"
    cols = [
        "C3",
        "ra",
        "dec",
        "z_best",
        "LAMBDA",
        "Lx0124",
        "eLx",
        "M200c",
        "eM200c",
        "d200",
    ]
    axesls = Table(fits.open(filename)[1].data)[cols]
    axesls.rename_columns(["C3", "dec", "z_best", "Lx0124"], ["AXES", "dec", "z", "Lx"])
    axesls["LEGACY"] = 1
    return axesls


def load_eromapper():
    ti = time()
    cat = ClusterCatalog(
        "eromapper",
        Table.read("aux/optical/eromapper_optical_dr10_grz_catalog.fit"),
        cols=[
            "mem_match_id",
            "ra",
            "dec",
            "lambda",
            "lambda_e",
            "best_z",
            "best_zerr",
            "best_z_type",
        ],
        base_cols=("mem_match_id", "ra", "dec", "best_z"),
        masscol="lambda",
    )
    cat.catalog["best_z"] = cat["z"]
    rich = 1.21 * cat["lambda"]
    z = cat["best_z"]
    # McClintock+18 eq 52 (actually M200m!)
    M0 = 10**14.49
    l0 = 40
    z0 = 0.35
    F = 1.36
    G = -0.30
    m200 = M0 * (rich / l0) ** F * ((1 + z) / (1 + z0)) ** G
    # m = 4*pi/3 * r**3 * rho
    rho = cosmo.critical_density(z).to(u.Msun / u.Mpc**3).value
    r200 = radius_from_mass(m200, 200, rho)
    d200 = ((r200 * u.Mpc) * cosmo.arcsec_per_kpc_proper(z)).to(u.deg).value
    cat.catalog["m200"] = m200
    cat.catalog["r200"] = r200
    cat.catalog["d200"] = d200
    cat.catalog["m200"].format = ".2e"
    cat.catalog["r200"].format = ".2f"
    cat.catalog["d200"].format = ".4f"
    print(f"Loaded eromapper in {time()-ti:.1f} s")
    # look for A3571
    j = Abell.obj == "A3571"
    print(Abell[j])
    q = cat.query(
        ra=Abell.ra[j], dec=Abell.dec[j], radius=30 * u.arcmin, z=0.1, z_width=0.1
    )
    q = q[q["z"] < 0.2]
    q["arcmin"] = (
        Abell.coords[j]
        .separation(SkyCoord(ra=q["ra"], dec=q["dec"], unit="deg"))
        .to("arcmin")
    )
    q.sort("arcmin")
    print(q)
    return cat


if __name__ == "__main__":
    main()
