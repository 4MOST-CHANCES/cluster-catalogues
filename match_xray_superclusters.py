from astropy import units as u
from astropy.constants import c as clight
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
from astropy.table import Table, join, join_skycoord, hstack
from datetime import date
from icecream import ic
from matplotlib import cm, pyplot as plt, ticker
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Ellipse
import numpy as np

import ligo.skymap.plot
from plottery.plotutils import savefig, update_rcParams
from profiley.helpers.spherical import radius_from_mass

from astro.clusters import ClusterCatalog
from astro.clusters.catalog import Abell
from astro.footprint import Footprint

from collate_info import chances_catalog, parse_args

update_rcParams()

clight = clight.to("km/s").value


def main():
    args = parse_args()
    args.sample = "lowz"
    chances = chances_catalog(args)

    shapley = Footprint(
        "Shapley",
        footprint=np.array(
            [[[192, -36], [192, -26], [207, -26], [207, -36], [192, -36]]]
        ),
    )
    shapley.z = 0.048
    shapley.ra = 200
    shapley.dec = -31
    shapley.xlim = (188, 215)
    shapley.ylim = (-40, -22)

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
                ]
            ]
        ),
    )
    horologium.z = 0.060
    horologium.ra = 55
    horologium.dec = -50
    horologium.xlim = (40, 70)
    horologium.ylim = (-64, -35)

    # copied over from match_xray_primary_cleaned.py
    axes2mrs = load_axes2mrs()
    axesls = load_axesls()
    # contaminated in X-rays - lambda=12
    axes2mrs = axes2mrs[axes2mrs["AXES"] != "93191202"]
    axesls = axesls[axesls["AXES"] != "93191202"]
    axes = join(axes2mrs, axesls, join_type="outer", keys="AXES")
    for key in ("z", "RA_X", "DEC_X", "Lx", "eLx", "M200c", "eM200c", "R200c_deg"):
        axes[key] = [
            x1 if x1 else x2 for x1, x2 in zip(axes[f"{key}_1"], axes[f"{key}_2"])
        ]
        axes.remove_column(f"{key}_1")
        if key != "z":
            axes.remove_column(f"{key}_2")
    axes.rename_column("z_2", "z_LEGACY")
    # rescale masses and sizes (Alexis used H0=70)
    for col in ("M200c", "eM200c"):
        axes[col] = axes[col] / (cosmo.H0.value / 70)
    eromapper = load_eromapper()

    # need this first pass to find clusters in the SCs
    shapley.clusters = plot_supercluster(axes, shapley, chances, show_primary=False)
    horologium.clusters = plot_supercluster(
        axes, horologium, chances, show_primary=False
    )

    shapley.primary, shapley.secondary = primaries(axes, shapley)
    horologium.primary, horologium.secondary = primaries(axes, horologium)

    # to add primaries to the figure
    kwargs = dict(cmap="viridis", alpha=0.8, vmin=-20000, vmax=20000)
    shapley.clusters = plot_supercluster(axes, shapley, chances, **kwargs)
    horologium.clusters = plot_supercluster(axes, horologium, chances, **kwargs)
    fig, ax = plt.subplots(figsize=(20, 1.2), constrained_layout=True)
    fig.colorbar(
        cm.ScalarMappable(
            cmap=kwargs["cmap"],
            norm=Normalize(vmin=kwargs["vmin"], vmax=kwargs["vmax"]),
        ),
        cax=ax,
        label="$v_\\mathrm{pec}$ (km s$^{-1}$)",
        orientation="horizontal",
    )
    output = "plots/axes_sc_colorbar"
    for ext in ("pdf", "png"):
        savefig(f"{output}.{ext}", fig=fig, tight=False)

    return


def primaries(cat, sc, vmax=8000):
    """Sift through the catalogue to find primaries"""
    for col in ("M200c", "eM200c"):
        sc.clusters[col].format = "%.2e"
    sc.clusters["coords"] = SkyCoord(
        ra=sc.clusters["RA_X"], dec=sc.clusters["DEC_X"], unit="deg"
    )
    print(sc.name)
    sc.clusters.sort("M200c")
    sc.clusters.reverse()
    print(sc.clusters)
    primary = []
    secondary = []
    # curved sky
    dist = sc.clusters["coords"].separation(sc.clusters["coords"][:, None])
    # flat sky
    # dist = ((sc.clusters["RA_X"] - sc.clusters["RA_X"][:, None])**2 \
    # + (sc.clusters["DEC_X"] - sc.clusters["DEC_X"][:, None])**2)**0.5 * u.deg
    within_5r200 = dist < 5 * sc.clusters["R200c_deg"] * u.deg
    within_z = (
        clight
        * np.abs(sc.clusters["z"] - sc.clusters["z"][:, None])
        / (1 + sc.clusters["z"][:, None])
        < vmax
    )
    for i, cl in enumerate(sc.clusters):
        print(cl)
        # if cl["AXES"] in secondary:
        # continue
        bigger = sc.clusters["M200c"] > cl["M200c"]
        print(f"{bigger.sum()} bigger")
        is_inside_primary = np.any(within_5r200[i] & within_z[i] & bigger)
        primary_is_inside = np.any(within_5r200[:, i] & within_z[:, i] & bigger)
        # this is one of the ones we don't understand
        if bigger.sum() == 5:
            print(within_5r200[i])
            print(within_5r200[i] & bigger)
            print(sc.clusters[within_5r200[i] & bigger])
            print(dist[i][within_5r200[i] & bigger])

        print("***")
        # within_5r200 = cl["coords"].separation(sc.clusters["coords"]) < 5 * cl["R200c_deg"] * u.deg
        # within_z = clight * np.abs(sc.clusters["z"] - cl["z"]) / (1 + cl["z"]) < vmax
        # close = (within_5r200 & within_z)
        # if np.any(sc.clusters["M200c"][close] > cl["M200c"]):
        if is_inside_primary or primary_is_inside:
            secondary.append(cl["AXES"])
            continue
        primary.append(cl["AXES"])
        # secondary.extend(list(sc.clusters["AXES"][close]))
    primary = np.array(primary)
    secondary = np.unique(secondary)
    print("Primary:")
    print(primary)
    print(sc.clusters["AXES"][~np.isin(sc.clusters["AXES"], secondary)])
    print("Secondary:")
    print(secondary)
    print(sc.clusters["M200c"].size == primary.size + secondary.size)
    for tbl, tbl_name in zip((primary, secondary), ("primary", "secondary")):
        np.savetxt(f"final-catalogues/{sc.name.lower()}_{tbl_name}.tbl", tbl, fmt="%d")
    return primary, secondary


def plot_supercluster(
    cat,
    sc,
    chances,
    velmax=8000,
    show_primary=True,
    show_abell=False,
    show_chances=True,
    barsize=10 * u.Mpc,
    cmap="viridis",
    alpha=0.8,
    vmin=-20000,
    vmax=20000,
    use_skymap=True,
):
    """barsize in Mpc"""
    # how many AXES sources are in the Shapley region?
    in_sc = sc.in_footprint(cat["RA_X"], cat["DEC_X"])
    vpec = clight * (cat["z"] - sc.z) / (1 + sc.z)
    in_sc_z = np.abs(vpec) < velmax
    print(in_sc.size, in_sc.sum(), (in_sc & in_sc_z).sum())

    aspect = np.abs((sc.xlim[1] - sc.xlim[0]) / (sc.ylim[1] - sc.ylim[0]))
    fig = plt.figure(figsize=(8 * aspect, 8), constrained_layout=True)
    # ax = fig.add_subplot()  # projection=ccrs.Mollweide())
    if use_skymap:
        plotradius = 1.5 * max([sc.xlim[1] - sc.xlim[0], sc.ylim[1] - sc.ylim[0]])
        ax = plt.axes(
            projection="astro zoom",
            center=get_circle_center(sc.ra, sc.dec),
            radius=f"{plotradius} deg",
            # rotate="0 deg",
        )
    else:
        ax = fig.add_subplot()  # projection=ccrs.Mollweide())
    ax.set_title(f"{sc.name} (z={sc.z:.3f})")
    s = (cat["M200c"]) / 1e12
    im = ax.scatter(
        cat["RA_X"],
        cat["DEC_X"],
        c=vpec,
        s=s,
        marker="o",
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set(
        xlabel="RA (deg)",
        ylabel="Dec (deg)",
        xlim=sc.xlim,
        ylim=sc.ylim,
        aspect="equal",
    )
    for cl in cat[in_sc_z]:
        if (
            (cl["RA_X"] > min(sc.xlim))
            & (cl["RA_X"] < max(sc.xlim))
            & (cl["DEC_X"] > min(sc.ylim))
            & (cl["DEC_X"] < max(sc.ylim))
        ):
            if use_skymap:
                ax.mark_inset_circle(
                    ax,
                    get_circle_center(cl["RA_X"], cl["DEC_X"]),
                    get_circle_radius(5 * cl["R200c_deg"]),
                    facecolor="none",
                    edgecolor="k",
                )
            else:
                patch = Ellipse(
                    (cl["RA_X"], cl["DEC_X"]),
                    2 * 5 * cl["R200c_deg"] / np.cos(np.pi / 180 * cl["DEC_X"]),
                    2 * 5 * cl["R200c_deg"],
                    facecolor="none",
                    edgecolor="k",
                )
                ax.add_patch(patch)
        if (cl["M200c"] > 1e14 or True) and show_abell:
            abell = Abell.query(
                ra=np.array([cl["RA_X"]]),
                dec=np.array([cl["DEC_X"]]),
                radius=10 * u.arcmin,
            )
            if abell["name"].size == 0:
                continue
            ax.annotate(
                abell["name"][0],
                xy=(cl["RA_X"], cl["DEC_X"]),
                ha="left",
                va="bottom",
                color="C3",
                fontsize=16,
                fontweight="bold",
            )
    if sc.name == "Horologium-Reticulum":
        # fornax = Circle((54.71135, -35.40093), 12.5, facecolor="none", edgecolor="0.5", lw=2)
        fornax = Ellipse(
            (54.71135, -35.40093),
            2 * 12.5 / np.cos(np.pi / 180 * 35.4),
            2 * 12.5,
            facecolor="none",
            edgecolor="0.5",
            lw=2,
        )
        if use_skymap:
            ax.mark_inset_circle(
                ax,
                center=get_circle_center(*fornax.center),
                radius="12.5 deg",
                facecolor="none",
                edgecolor="0.5",
                lw=2,
            )
        else:
            ax.add_patch(fornax)
        ax.plot(
            *fornax.center,
            color="0.5",
        )
        # additional = np.isin(Abell.obj, ["A3074", "A3078", "A3133"])
        # print(Abell.catalog[additional])
        # for cl in Abell.catalog[additional]:
        # ax.annotate(cl["name"], xy=(cl["ra"], cl["dec"]), ha="left", va="bottom", color="C3", fontsize=16)
    if show_abell:
        abell_mask = (np.abs(clight * (Abell.z - sc.z) / (1 + sc.z)) < 10000) | (
            Abell.z < 0
        )
        print(abell_mask.size, abell_mask.sum())
        ax.plot(Abell.ra[abell_mask], Abell.dec[abell_mask], "C3o", ms=3)
    if show_primary:
        p = np.isin(cat["AXES"], sc.primary)
        ax.plot(cat["RA_X"][p], cat["DEC_X"][p], "kx", ms=6, mew=3)
    if show_chances:
        mask = (
            (chances["ra"] > sc.xlim[0])
            & (chances["ra"] < sc.xlim[1])
            & (chances["dec"] > sc.ylim[0])
            & (chances["dec"] < sc.ylim[1])
            # & ~(sc.in_footprint(chances["ra"], chances["dec"]))
        )
        print(mask.size, mask.sum())
        ax.plot(chances[mask]["ra"], chances[mask]["dec"], "C0+", ms=6, mew=2.5)
        for cl in chances[mask]:
            ax.annotate(
                cl["name"],
                xy=(cl["ra"], cl["dec"]),
                ha="left",
                va="top",
                color="C3",
                fontsize=16,
                fontweight="bold",
            )
    if use_skymap:
        ...
    else:
        sc.plot(
            ax=ax,
            facecolor="none",
            edgecolor="k",
            lw=3,
            transform=ax.get_transform("world"),
        )
    b = (barsize * cosmo.arcsec_per_kpc_comoving(sc.z)).to(u.deg).value
    xb = sc.xlim[0] + 0.1 * (sc.xlim[1] - sc.xlim[0])
    yb = sc.ylim[0] + 0.1 * (sc.ylim[1] - sc.ylim[0])
    ax.plot((xb, xb + b), (yb, yb), "C0-", lw=4)
    ax.annotate(
        f"{barsize.value:.0f} cMpc",
        xy=(xb + b / 2, yb + 0.01 * (sc.ylim[1] - sc.ylim[0])),
        fontsize=15,
        ha="center",
        va="bottom",
    )
    ax.invert_xaxis()
    if use_skymap:
        ax.grid(True)
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    # plt.colorbar(im, ax=ax, label="$v_\\mathrm{pec}$ (km/s)")
    name = sc.name.split("-")[0].lower()
    output = f"plots/axes_sc_{name}"
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
        "RA_X",
        "DEC_X",
        "z",
        "Lx",
        "eLx",
        "M200c",
        "eM200c",
        "R200_deg",
    ]
    axes2mrs = Table(fits.open(filename)[1].data)[cols]
    axes2mrs.rename_columns(["C3ID", "R200_deg"], ["AXES", "R200c_deg"])
    axes2mrs["2MRS"] = 1
    return axes2mrs


def load_axesls():
    filename = "aux/xray/Xmass_c3_LSDR10updated_info_nofilters_clean.fits"
    cols = [
        "C3",
        "RA_X",
        "Dec_X",
        "z_best",
        "LAMBDA",
        "Lx0124",
        "eLx",
        "M200c",
        "eM200c",
        "R200c_deg",
    ]
    axesls = Table(fits.open(filename)[1].data)[cols]
    axesls.rename_columns(
        ["C3", "Dec_X", "z_best", "Lx0124"], ["AXES", "DEC_X", "z", "Lx"]
    )
    axesls["LEGACY"] = 1
    return axesls


def load_eromapper():
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
    cat.rename_column("z", "best_z")
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
    cat["m200"] = m200
    cat["r200"] = r200
    cat["d200"] = d200
    cat["m200"].format = ".2e"
    cat["r200"].format = ".2f"
    cat["d200"].format = ".4f"
    return cat


if __name__ == "__main__":
    main()
