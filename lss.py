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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Ellipse
import numpy as np
import os
import skymapper as skm
from time import time
from tqdm import tqdm

import ligo.skymap.plot
from profiley.nfw import NFW
from plottery.plotutils import savefig, update_rcParams
from profiley.helpers.spherical import radius_from_mass

from astro.clusters import ClusterCatalog

cosmology.setCosmology("planck18")
update_rcParams()


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
    chances, cat = find_subclusters(args, chances)
    cat = add_r200(cat)
    print(cat)
    phase_space(args, chances, cat)
    return
    wrap_plot_sky(args, chances, cat, fsize=3, suffix="neighbors")
    # wrap_plot_sky(args, chances, cat)
    chances.catalog.sort(f"N_{args.catalog}")
    print(chances)


def phase_space(args, chances, cat):
    # fig, ax = plt.subplots(layout="constrained")
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


def wrap_plot_sky(args, chances, cat, fsize=1.5, show_neighbors=True, suffix=""):
    """fsize is the size of the figure in units of 5r200"""

    def get_center(item):
        return f'{item["ra"]}d {item["dec"]}d'

    def get_radius(item, f=1):
        return f'{f * item["d200"]/60} deg'

    light = (c.c).to(u.km / u.s).value
    # colors = [[0, 0, 1], [0, 0, 0.5], [0.1, 0, 0.1], [0.5, 0, 0], [1, 0, 0]]
    # nodes = [0, 0.25, 0.5, 0.75, 1]
    # colors = [[]]
    # cmap = LinearSegmentedColormap.from_list("redshift", list(zip(nodes, colors)))
    # this is the same as used in the superclusters, adapted from coolwarm
    colors = [[0, 0, 1], [0.4, 0.4, 0.4], [1, 0, 0]]
    nodes = [0, 0.5, 1]
    cmap = LinearSegmentedColormap.from_list("vmap", list(zip(nodes, colors)))
    vmax = 8000

    if show_neighbors:
        coord = SkyCoord(ra=chances["ra"], dec=chances["dec"], unit="deg")
        cldist = coord.separation(coord[:, None])
    for i, cl in tqdm(enumerate(chances), total=chances.size):
        name = cl["name"]
        if args.clusters is not None and name not in args.clusters:
            continue
        mask = cat["chances"] == name
        print(cl)
        fig = plt.figure(figsize=(6, 6), constrained_layout=True)
        ax = plt.axes(
            projection="astro zoom",
            center=get_center(cl),
            radius=get_radius(cl, fsize * 5),
        )
        ax.mark_inset_circle(ax, get_center(cl), get_radius(cl, 5), zorder=10)
        if show_neighbors:
            neighbors = (cldist[i] < 5 * (cl["d200"] + chances["d200"]) * u.arcmin) & (
                cldist[i] > 1 * u.arcmin
            )
            print(f"{neighbors.sum()} neighbors:")
            print(chances[neighbors])
            # neighbors = neighbors & (np.abs(chances["z"] - cl["z"]))
            for n in chances[neighbors]:
                ax.mark_inset_circle(ax, get_center(n), get_radius(n, 5), zorder=10)
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
        print(subcl)
        for scl in subcl:
            radius = scl["d200"]
            dv = light * (scl["best_z"] - cl["z"]) / (1 + cl["z"])
            dv = (dv + vmax) / (2 * vmax)
            color = cmap(dv)
            ax.mark_inset_circle(
                ax,
                get_center(scl),
                f"{radius} deg",
                facecolor=color,
                alpha=0.7,
                lw=0,
                zorder=9,
            )
        ax.plot(
            cl["ra"],
            cl["dec"],
            "kx",
            transform=ax.get_transform("world"),
            mew=2,
            ms=8,
            zorder=11,
        )
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
            f'{label} (z={cl["z"]:.3f}, $\\theta_{{200}}={cl["d200"]:.1f}\'$)',
            xy=(0.06, 0.06),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=14,
        )
        if not show_neighbors:
            ax.annotate(
                "$5r_{200}$",
                xy=(0.5, 0.85),
                xycoords="axes fraction",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="heavy",
            )
        ax.grid(True)
        ax.set_xlabel("Right Ascension")
        ax.set_ylabel("Declination")
        for key in ("ra", "dec"):
            ax.coords[key].set_ticklabel_visible(False)
        # ax.set(xlim=(x - 1, x + 1), ylim=(y - 1, y + 1))
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
        # if name == "Abell 0500":
        #     break
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
    ]
    matches = {key: [] for key in cols}
    chances.catalog[f"N_{args.catalog}"] = np.zeros(chances.size, dtype=int)
    # fixed
    dz_phot = dz_frac_max * cat["z"]
    for i, (cl, coord, z, d200, r200) in tqdm(
        enumerate(
            zip(
                chances.obj, chances.coords, chances.z, chances["d200"], chances["r200"]
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
        arcmin = sep[close].to("arcmin").value
        matches["dist (arcmin)"].extend(arcmin)
        matches["dist (r200)"].extend(arcmin / d200)
        matches["dist (Mpc)"].extend(r200 * arcmin / d200)
        matches["vpec (km/s)"].extend(299792.5 * (cat["z"][close] - z) / (1 + z))
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
    matches.write(output, format="ascii.fixed_width", overwrite=True)
    return chances, matches


def parse_args():
    parser = ArgumentParser()
    add = parser.add_argument
    add("sample", choices=("evolution", "lowz", "all"), default="lowz")
    add("-c", "--catalog", default="eromapper")
    add("--clusters", nargs="+")
    add("--overwrite", action="store_true")
    args = parser.parse_args()
    return args


main()
