"""Compare radii from different source catalogues"""

from astropy.table import Table
from matplotlib import pyplot as plt
import numpy as np

from plottery.plotutils import savefig, update_rcParams

update_rcParams()


def main(sample="lowz"):
    if sample == "lowz":
        date = "20240429"
    elif sample == "evolution":
        date = "20240429"
    filename = f"catalogues/clusters_chances_{sample}_{date}_large.csv"
    tbl = Table.read(filename, format="ascii.csv", comment="#")
    for col in tbl.colnames:
        if "m200" in col:
            tbl[col].format = ".1e"

    ncl = tbl["r200"].size
    print(f"{ncl} clusters")

    rcols = [col for col in tbl.colnames if "r200_" in col]
    ncol = len(rcols)
    print(rcols, ncol)

    bins = np.linspace(-1, 1, 20)
    nax = 2
    fig, axes = plt.subplots(1, nax, figsize=(nax * 4.5, 5), constrained_layout=True)
    print(ncol // nax)
    for j, ax in enumerate(axes):
        for i in range(ncol // nax):
            idx = j * ncol // nax + i
            color = f"C{idx}"
            if idx >= ncol:
                continue
            print("###")
            col = rcols[idx]
            label = col.split("_")[1]
            used = np.array(
                [
                    i
                    for i in range(ncl)
                    if (tbl["source"][i] != label and tbl[col][i] > 0)
                ]
            )
            print(label, used, used.size)
            if used.size == 0:
                continue
            diff = (tbl[col][used] / tbl["r200"][used]) - 1
            print(diff.data)
            print(
                tbl[
                    "name",
                    "z",
                    f"m200_{label}",
                    f"r200_{label}",
                    "m200",
                    "r200",
                    "source",
                ][used]
            )
            # if diff.size <= 3:
            #     for d in diff:
            #         ax.arrow(d, 3, 0, -2.8, color=color, width=0.01, head_width=0.05)
            #     ax.plot([], [], "-", color=color, label=f"{label} ({used.size})")
            # else:
            ax.hist(
                diff,
                histtype="step",
                bins=bins,
                color=color,
                lw=3 - 0.5 * i,
                density=False,
                label=f"{label} ({used.size})",
            )
        ax.legend(fontsize=12)
        ax.set_xlabel("$r_{200}^\\mathrm{survey}/r_{200} - 1$")
    output = f"plots/compare_r200_{sample}.png"
    savefig(output, fig=fig, tight=False)
    return


# main("lowz")
main("evolution")
