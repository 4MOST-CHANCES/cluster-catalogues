from astropy import units as u
from astropy.constants import c as clight
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
from astropy.table import Table, join, join_skycoord, hstack
from datetime import date
from icecream import ic
from matplotlib import pyplot as plt, ticker
import numpy as np

from collate_info import chances_catalog, parse_args


def main():
    args = parse_args()
    chances = Table.read(
        #f"catalogues/clusters_chances_{args.sample}_{date.today()}.csv",
        f"final-catalogues/lowz_axes_primary_{date.today()}.csv",
        format="ascii.csv",
    )
    print(np.sort(chances.colnames))
    axes2mrs = load_axes2mrs()
    axesls = load_axesls()
    # contaminated in X-rays - lambda=12
    axes2mrs = axes2mrs[axes2mrs["AXES"] != "93191202"]
    axesls = axesls[axesls["AXES"] != "93191202"]
    print(axes2mrs["RA_X"].size, axesls["RA_X"].size)
    chances["coords"] = SkyCoord(chances["RA_X"], chances["DEC_X"], unit="deg")
    # probably need to filter redshifts
    axes = join(axes2mrs, axesls, join_type="outer", keys="AXES")
    for key in ("z", "RA_X", "DEC_X", "Lx", "eLx", "M200c", "eM200c", "R200c_deg"):
        axes[key] = [
            x1 if x1 else x2 for x1, x2 in zip(axes[f"{key}_1"], axes[f"{key}_2"])
        ]
        axes.remove_column(f"{key}_1")
        if key != "z":
            axes.remove_column(f"{key}_2")
    axes.rename_column("z_2", "z_LEGACY")
    print(axes)
    axes2mrs["coords"] = SkyCoord(axes2mrs["RA_X"], axes2mrs["DEC_X"], unit="deg")
    axesls["coords"] = SkyCoord(axesls["RA_X"], axesls["DEC_X"], unit="deg")
    axes["coords"] = SkyCoord(axes["RA_X"], axes["DEC_X"], unit="deg")
    # for cat, name in zip((axes2mrs, axesls, axes), ("AXES-2MRS", "AXES-LS", "AXES")):
    #     print(f"** {name} **")
    #     match_catalogs(chances, axes2mrs, radius=chances["5d200(deg)"] * u.deg)
    match_catalogs(args, chances, axes, radius=5 * chances["R200c_deg"] * u.deg)

    return


def load_axes2mrs():
    filename = "aux/xray/Xmass_BayesGroups_n3ext200kpc_nofake_info2.fits"
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
    filename = "aux/xray/Xmass_axes_legacy.cat"
    cols = ["CODEX3", "RA_X", "DEC_X", "zcmb", "Lx", "M200c", "eM200c", "sigma"]
    axesls = Table.read(filename, format="ascii.commented_header")
    axesls["LEGACY"] = 1
    return axesls


def match_catalogs(args, chances, cat, radius):
    # For every AXES cluster, record CHANCES name, ra, dec, M200, R200, distance
    dist = chances["coords"].separation(cat["coords"][:, None])
    within_5r200 = dist < radius
    chances_cols = [
        "CHANCES_FIELD",
        "CHANCES_z",
        "CHANCES_M200c",
        "CHANCES_R200c_deg",
    ]
    distance_cols = [
        "CHANCES_DIST_deg",
        "CHANCES_DIST_R200c",
        "VPEC_kms",
    ]
    new = Table(names=chances_cols, dtype=["S20", "f8", "f8", "f8"])
    tdist = Table(names=distance_cols, dtype=["f8", "f8", "f8"])
    new = hstack([new, cat[:0], tdist])
    new.remove_column("coords")
    print(new)
    for i, cl in enumerate(cat):
        if not np.any(within_5r200[i]):
            continue
        for ch in chances["CHANCES_FIELD", "z", "M200c", "R200c_deg", "coords"][
            within_5r200[i]
        ]:
            sep = cl["coords"].separation(ch["coords"]).to("deg").value
            if sep == 0 * u.deg:
                continue
            vpec = (clight * (cl["z"] - ch["z"]) / (1 + ch["z"])).to("km/s").value
            # excluding coords
            row = list(ch)[:-1] + list(cl)[:-1] + [sep, sep / ch["R200c_deg"]] + [vpec]
            row = [i if i != "masked" else -1 for i in row]
            new.add_row(row)
    for col in ("LEGACY", "2MRS"):
        new[col] = np.max([new[col], np.zeros(new[col].size, dtype=int)], axis=0)
    for col in (
        "z_LEGACY",
        "z",
        "CHANCES_z",
        "CHANCES_R200c_deg",
        "CHANCES_DIST_deg",
        "R200c_deg",
    ):
        new[col].format = "%.4f"
    for col in ("RA_X", "DEC_X"):
        new[col].format = "%.5f"
    for col in ("Lx", "eLx", "M200c", "eM200c", "CHANCES_M200c"):
        new[col].format = "%.3e"
    for col in ("LAMBDA", "CHANCES_DIST_R200c"):
        new[col].format = "%.2f"
    new["VPEC_kms"].format = "%.0f"
    print(new)
    new.sort("M200c")
    new.reverse()
    new.sort("CHANCES_FIELD")
    output = f"final-catalogues/{args.sample}_axes_infalling_{date.today()}"
    new.write(f"{output}.csv", format="ascii.csv", overwrite=True)
    new.write(f"{output}.txt", format="ascii.fixed_width", overwrite=True)
    print(new[new["AXES"] == "93223401"])
    # for i, cl in enumerate(chances):
    #     if not np.any(within_5r200[:, i]):
    #         continue
    #     print(cl["name", "z"], cat["AXES"][within_5r200[:, i]].value)
    #     print()
    return


if __name__ == "__main__":
    main()
