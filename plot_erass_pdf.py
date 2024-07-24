from astropy.table import Table
from matplotlib import pyplot as plt
import numpy as np

from plottery.plotutils import savefig, update_rcParams

update_rcParams()


lowz = Table.read(
    "catalogues/clusters_chances_lowz_20240503_large.csv", format="csv", comment="#"
)
erass = Table.read(
    "/data2/cristobal/catalogs/erosita/erass1cl_primary_v3.2.fits.gz", format="fits"
)


def main():
    j = lowz["source"] == "eRASS1"
    # for i, cl in enumerate(lowz["name"][j][:3]):
    for i, cl in enumerate(["Abell 3395"]):
        fig, ax = plt.subplots()
        plot_cluster_pdf(ax, cl, color=f"C0")
        ax.legend(fontsize=14, loc="upper left", frameon=False)
        ax.set(
            xscale="log",
            # xlim=(0, 3e14),
            xlabel="$M_{500}$ (M$_\odot$)",
            ylabel="pdf",
            ylim=(-0.05, 0.9),
            title=cl,
        )
        output = f"plots/erass_pdf_{cl.replace(' ', '')}.png"
        savefig(output, fig=fig)


def plot_cluster_pdf(ax, cluster, color="C0"):
    j = lowz["name"] == cluster
    k = erass["NAME"] == lowz["eRASS1"][j]
    name = erass["NAME"][k]
    mx = erass["M500_PDF_array"][k].value[0]
    my = erass["M500_PDF"][k].value[0]
    ax.plot(mx, my, color=color, lw=2)
    print(
        cluster,
        1e13 * erass["M500"][k].value,
        1e13 * erass["M500_L"][k].value,
        1e13 * erass["M500_H"][k].value,
    )
    j = (mx >= 1e13 * (erass["M500"][k] - erass["M500_L"][k]).value[0]) & (
        mx <= 1e13 * (erass["M500"][k] + erass["M500_H"][k]).value[0]
    )
    ax.fill_between(mx[j], np.zeros(j.sum()), my[j], color=color, lw=0, alpha=0.4)
    ax.plot(
        [],
        [],
        ls="-",
        color=color,
        alpha=0.4,
        lw=10,
        label="$[M_{500}-M_\\mathrm{500,L};M_{500}+M_\\mathrm{500,H}]$",
    )
    j = (mx >= 1e13 * erass["M500_L"][k].value[0]) & (
        mx <= 1e13 * erass["M500_H"][k].value[0]
    )
    ax.fill_between(mx[j], np.zeros(j.sum()), my[j], color=color, lw=0, alpha=0.8)
    ax.plot(
        [],
        [],
        ls="-",
        color=color,
        alpha=0.8,
        lw=5,
        label="$[M_\\mathrm{500,L};M_\\mathrm{500,H}]$",
    )
    ax.plot(
        2 * [1e13 * erass["M500"][k].value[0]],
        [0, my.max()],
        color="k",
        ls="--",
        lw=1,
    )


main()
