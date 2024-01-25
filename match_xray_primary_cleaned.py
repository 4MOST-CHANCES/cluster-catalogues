from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits
from astropy.table import Table, join, join_skycoord, hstack
from datetime import date
from icecream import ic
from matplotlib import pyplot as plt, ticker
import numpy as np

from profiley.helpers.spherical import radius_from_mass

from collate_info import chances_catalog, parse_args

print(cosmo)


def main():
    args = parse_args()

    primary = Table(
        fits.open("xray_field_corrections/current_primary_clean.fits")[1].data
    )
    primary = primary[primary["AXES"] != -1]
    # we also decided to remove Alessia's groups -- note the additional space in the name
    for f in primary["CHANCES_FIELD"]:
        print(f'--{f}--')
    # print(primary[primary["CHANCES_FIELD"] == "AM2002 "])
    #primary = primary[primary["CHANCES_FIELD"] != "AM2002 "]
    primary = primary[~np.isin(primary["CHANCES_FIELD"], ("AM2002 ", "A2877  "))]
    print(primary)

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

    chances_xray = join(primary, axes, join_type="left", keys="AXES")
    for col in ("2MRS", "LEGACY"):
        chances_xray[col].fill_value = 0
    for col in ("z_LEGACY", "LAMBDA"):
        chances_xray[col].fill_value = -1
    chances_xray = chances_xray.filled()

    for col in ("M200c", "eM200c", "Lx", "eLx"):
        chances_xray[col].format = "%.2e"
    chances_xray["R200c_Mpc"] = radius_from_mass(
        chances_xray["M200c"],
        200,
        cosmo.critical_density(chances_xray["z"]).to(u.Msun / u.Mpc**3).value,
    )
    chances_xray["R200c_Mpc"].format = "%.2f"
    chances_xray["R200c_deg_Cristobal"] = (
        # chances_xray["R200c_deg"] = (
        (
            chances_xray["R200c_Mpc"]
            * u.Mpc
            * cosmo.arcsec_per_kpc_proper(chances_xray["z"])
        )
        .to(u.deg)
        .value
    )
    rdiff = np.abs(
        ((chances_xray["R200c_deg"] - chances_xray["R200c_deg_Cristobal"]) * u.deg)
        .to(u.arcmin)
        .value
    )
    plt.hist(rdiff, np.arange(0, 2, 0.1), histtype="step")
    plt.axvline(np.median(rdiff))
    plt.xlabel("R200c difference (arcmin)")
    plt.show()
    chances_xray.remove_column("R200c_deg")
    chances_xray.rename_column("R200c_deg_Cristobal", "R200c_deg")
    for col in ("z_LEGACY", "z", "R200c_deg"):
        chances_xray[col].format = "%.4f"
    chances_xray["LAMBDA"].format = "%.1f"
    chances_xray.sort("z")
    chances_xray["INDEX"] = np.arange(chances_xray["AXES"].size, dtype=int)
    cols = list(chances_xray.colnames)
    chances_xray = chances_xray[cols[-1:] + cols[:-1]]
    print(chances_xray)

    output = f"final-catalogues/lowz_axes_primary_{date.today()}"
    chances_xray.write(f"{output}.csv", format="ascii.csv", overwrite=True)
    chances_xray.write(f"{output}.txt", format="ascii.fixed_width", overwrite=True)

    return


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


if __name__ == "__main__":
    main()
