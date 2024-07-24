from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.visualization.wcsaxes import Quadrangle
from icecream import ic
from matplotlib import pyplot as plt, ticker
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle
import numpy as np
import os

import ligo.skymap.plot
from ligo.skymap.plot.poly import subdivide_vertices
from plottery.plotutils import savefig, update_rcParams

from astro.clusters import ClusterCatalog
from astro.footprint import Footprint

update_rcParams({"xtick.minor.size": 0})

shapley = Footprint(
    "Shapley",
    footprint=np.array([[[192, -36], [192, -26], [207, -26], [207, -36], [192, -36]]]),
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


def main():
    axcenter = 6  # hours
    axstartdeg = 15 * (axcenter - 12)
    lowz = ClusterCatalog("chances-lowz")
    evol = ClusterCatalog("chances-evol")
    lowz.catalog = lowz[~np.isin(lowz["name"], ["Abell 3389", "Abell 4038"])]
    lowz.catalog.sort("d200")
    evol.catalog.sort("d200")
    print(lowz)
    print(evol)
    # evol_near = evol["z"] <= 0.1
    # sample = vstack([lowz.catalog, evol.catalog[evol_near]])
    sample = vstack([lowz.catalog, evol.catalog])
    # this way smaller points will always show on top of larger points, easing visibility
    sample.sort("d200")
    sample = sample[::-1]
    for col in ("ra", "dec"):
        sample[col].format = ".5f"
    print(sample)
    sample["coord"] = SkyCoord(ra=sample["ra"], dec=sample["dec"], unit="deg")

    l = np.linspace(-179.99, 179.99, 200)
    b = np.ones(l.size)
    ra = np.linspace(0, 360, b.size)
    galaxy = [
        SkyCoord(l, -20 * b, unit="deg", frame="galactic"),
        SkyCoord(l, 20 * b, unit="deg", frame="galactic"),
    ]
    gal_radec = []
    for i, gal in enumerate(galaxy):
        gal_radec.append(gal.transform_to("fk5"))
        j = [i for i, x in enumerate(gal_radec[-1].ra.deg) if x > 15 * (axcenter + 12)][
            0
        ]
        galaxy[i] = np.roll(gal, -j)

    cmap = "plasma"
    cmap = plt.get_cmap(cmap)
    vmin = 0
    vmax = 0.45
    colors = cmap((sample["z"] - vmin) / (vmax - vmin))
    print(colors[:3])
    fig = plt.figure(figsize=(14, 6), layout="constrained")
    # gs = fig.add_gridspec(1, 2, width_ratios=())
    ax = plt.axes(projection="astro hours mollweide", center="6h 0d")
    ax.grid()
    for gal in galaxy:
        ax.plot_coord(gal, color="k", ls="-")
    # poor man's fill_between
    brng = np.linspace(-20, 20, 100)
    for bi in brng:
        g = SkyCoord(l, bi * b, unit="deg", frame="galactic")
        radec = g.transform_to("fk5")
        j = [i for i, x in enumerate(radec.ra.deg) if x > 15 * (axcenter + 12)][0]
        g = np.roll(g, -j)
        ax.plot_coord(g, color="0.8", zorder=-2, alpha=0.5, lw=2)
    # also show SO-LAT and CCAT wide surveys
    txt = ax.text_coord(
        SkyCoord(ra=3.2 * 15, dec=10, unit="deg"),
        "SO-LAT / CCAT-WFS / LSST",
        color="k",
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        bbox=dict(facecolor="w", alpha=0.5, linewidth=0),
    )
    ax.plot_coord(
        SkyCoord(ra=ra, dec=-19 * b, unit="deg"),
        lw=210,
        color="C1",
        alpha=0.3,
        zorder=-1,
    )
    # eROSITA-DE
    kwargs_erosita = dict(color="C3", lw=3)
    ell = SkyCoord(180, np.linspace(-90, 90, 181), unit="deg", frame="galactic")
    ax.plot_coord(ell, **kwargs_erosita)
    ell = SkyCoord(0, np.linspace(-90, 90, 181), unit="deg", frame="galactic")
    radec = ell.transform_to("fk5")
    mask = (radec.ra.deg > 15 * axcenter) & (radec.ra.deg < 15 * (axcenter + 12))
    ax.plot_coord(ell[mask], **kwargs_erosita)
    ax.plot_coord(ell[~mask], **kwargs_erosita)
    ax.text_coord(
        SkyCoord(14 * 15, 8, unit="deg"),
        "eROSITA-DE",
        color=kwargs_erosita.get("color", "C3"),
        ha="left",
        va="bottom",
        fontsize=14,
        rotation=32,
        fontweight="bold",
    )
    # superclusters
    for sc in (shapley, horologium):
        xy = subdivide_vertices(sc.footprint[0], 100)
        ax.plot(
            *np.transpose(xy),
            "-",
            color=(0.2, 0.6, 0),
            transform=ax.get_transform("world"),
            lw=3,
            zorder=1000,
        )
    area_total = 0
    area_lowz = 0
    for i, cl in enumerate(sample):
        circle = Circle(
            (cl["ra"], cl["dec"]),
            max(5 * cl["d200"] / 60, 1),
            ec="k",
            fc=colors[i],
            alpha=0.8,
            lw=1,
            zorder=99 + i,
            transform=ax.get_transform("world"),
        )
        ax.add_patch(circle)
        area = np.pi * circle.radius**2 * np.cos(np.radians(circle.center[1]))
        if i == 0 or cl["name"] in ("Fornax", "Hydra", "Antlia"):
            print(cl)
            print(
                circle.radius,
                np.radians(circle.center[1]),
                np.cos(np.radians(circle.center[1])),
                area,
            )
            print()
        area_total += area
        if cl["z"] < 0.1:
            area_lowz += area
    print(area_total, area_lowz)
    # legend
    # ref = Circle((14*15, 40), 5, ec="k", fc="none", lw=1.5, transform=ax.get_transform("world"))
    # ax.add_patch(ref)
    # ax.text_coord(SkyCoord(ra=14*15+3, dec=43, unit="deg"), "$5r_{200}$",
    #                color="k", fontsize=12, ha="left", va="bottom")
    cbar = fig.colorbar(
        ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax)),
        ax=ax,
        label="Redshift",
    )
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    for ext in ("png", "pdf"):
        output = f"plots/sky_distribution.{ext}"
        savefig(output, fig=fig, tight=False, close=False)
    return


def get_center(item):
    return f'{item["ra"]}d {item["dec"]}d'


def get_radius(item, f=1):
    return f'{f * item["d200"]/60} deg'


main()
