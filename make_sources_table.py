from astropy.io import ascii
import numpy as np


def main(sample):
    if sample == "lowz":
        date = "20240503"
    elif sample == "evolution":
        date = "20240503"
    filename = f"catalogues/clusters_chances_{sample}_{date}_large.csv"
    cat = ascii.read(filename, format="csv", comment="#")
    n = cat["name"].size
    print(cat.meta["comments"])
    catalogs = [
        col.split("_")[1]
        for col in cat.colnames
        if col.startswith("m200_") and col != "m200_listed"
    ]
    print(cat.meta["comments"])
    jp, jf = [
        [i for i, com in enumerate(cat.meta["comments"]) if com.startswith(key)][0]
        for key in ("priorities", "factors")
    ]
    priorities = [name for name in cat.meta["comments"][jp].split()[1:]]
    factors = {
        fac.split(":")[0]: fac.split(":")[1]
        for fac in cat.meta["comments"][jf].split()[1:]
    }
    rows = []
    for i, p in enumerate(priorities):
        if "(" in p:
            p, zlim = p[:-1].split("(")
            zmin, zmax = zlim.split("-")
            priorities[i] = f"{p} (${zmin} < z \leq {zmax}$)"
        matches = len([j for j, m in enumerate(cat[f"m200_{p}"]) if m > 0])
        primary = (cat["source"] == p).sum()
        row = f"{p:<12s} & {factors.get(p, '1.0')} & {matches} & {primary} \\\\"
        print(row)
    print("-----")


main("lowz")
main("evolution")
