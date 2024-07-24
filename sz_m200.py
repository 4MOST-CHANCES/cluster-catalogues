from astropy import units as u
from astropy.cosmology import Planck18
from astropy.table import Table
from colossus.cosmology import cosmology
from colossus.halo import concentration
import multiprocessing as mp
import numpy as np
from profiley.nfw import NFW
from time import time

from astro.clusters import ClusterCatalog


def calculate_m200_from_m500(m500, z, cosmo=Planck18, model="ishiyama21"):
    cosmology.fromAstropy(cosmo, sigma8=0.81, ns=0.966)
    c = concentration.concentration(m500, "500c", z, model=model)
    nfw = NFW(m500, c, z, overdensity=500, frame="physical", cosmo=cosmo)
    m200, r200 = nfw.mdelta(200, "c")
    d200 = (r200 * u.Mpc * cosmo.arcsec_per_kpc_proper(nfw.z)).to(u.arcmin).value
    return m200, r200 / nfw.rs, r200, d200


def run(name, mcol, f=1):
    cat = ClusterCatalog(name)
    cat.catalog = cat.catalog[cat["z"] > 0]
    to = time()
    (
        cat.catalog["m200"],
        cat.catalog["c200"],
        cat.catalog["r200"],
        cat.catalog["d200"],
    ) = calculate_m200_from_m500(1e14 * f * cat[mcol], cat.z, cosmo=Planck18)
    cat.catalog["m200"] = cat["m200"] / 1e14
    cat["z"].format = "%.3f"
    for col in ("m200", "c200", "r200", "d200"):
        cat[col].format = "%.2f"
    print(f"{name} in {(time()-to)/60:.2f} min")
    output = f"aux/m200_converted/{name}_m200.csv"
    if mcol == "M500cCal":
        output = output.replace(".csv", "cal.csv")
    if mcol == "M500cUncorr":
        output = output.replace(".csv", "uncorr.csv")
    cat["name", "z", "m200", "r200", "d200"].write(
        output, format="ascii.csv", overwrite=True
    )
    return


cats = ["mcxc", "erass1", "act-dr5", "act-dr5", "act-dr5", "spt-sz", "spt-ecs", "psz2"]
cols = ["M500", "M500", "M500c", "M500cCal", "M500cUncorr", "M500", "M500", "MSZ"]
idx = np.s_[1:2]
cats = cats[idx]
cols = cols[idx]
with mp.Pool(len(cats)) as pool:
    for cat, col in zip(cats, cols):
        # eRASS1 masses are in units of 1e13 Msun
        f = 0.1 if cat == "erass1" else 1
        pool.apply_async(run, args=(cat, col, f))
    pool.close()
    pool.join()

# run("act-dr5", "M500cCal")
# run("spt-sz", "M500")
# run("spt-ecs", "M500")
# run("psz2", "MSZ", 10**0.13)
