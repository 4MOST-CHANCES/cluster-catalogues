from astropy.cosmology import Planck18 as cosmo
from astropy.table import Table
from matplotlib import pyplot as plt
import numpy as np

from plottery.plotutils import savefig, update_rcParams

update_rcParams()


def main(sample="lowz", date="20240419"):
    chances = Table.read(f"catalogues/clusters_chances_{sample}_{date}_large.csv")
    zranges = [2, 0.3, 0.15, 0]
    # Damsted et al. Table 4 [z > 0.3, 0.15 < z < 0.3, z < 0.15]
    scgf = [[6.599, 0.172], [6.552, 0.195], [6.526, 0.148]]
    # Table B.3
    spiders = [[6.584, 0.220], [6.556, 0.218], [6.529, 0.147]]

    Hz = cosmo.H(chances["z"]).value
    hz = Hz / 100
    Ez = Hz / cosmo.H0.value

    chances["m200_spiders"] = np.zeros(chances["name"].size)
    chances["m200_scfg"] = np.zeros(chances["name"].size)
    for i, (name, params) in enumerate(zip(["scfg", "spiders"], (scgf, spiders))):
        for j, (a, b) in enumerate(params):
            zmask = (chances["z"] < zranges[j]) & (chances["z"] > zranges[j + 1])
            if zmask.sum() == 0:
                continue
            sigmav = np.exp(a + b * np.log(chances["CODEX_Lx0124"] / 1e44 / Ez))
            chances[f"m200_{name}"][zmask] = (1e15 / hz) * (sigmav / 1177) ** (
                1 / 0.364
            )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    ax = axes[0]
    ax.scatter(
        chances["z"], chances["m200_spiders"] / chances["m200_scfg"] - 1, c=chances["z"]
    )
    ax.set(
        xlabel="Redshift",
        ylabel="$M_{200}^\\mathrm{SPIDERS}/M_{200}^\\mathrm{SCFG} - 1$",
    )
    ax = axes[1]
    ax.scatter(
        chances["z"],
        (chances["m200_spiders"] / chances["m200_scfg"]) ** (1 / 3) - 1,
        c=chances["z"],
    )
    ax.set(
        xlabel="Redshift",
        ylabel="$r_{200}^\\mathrm{SPIDERS}/r_{200}^\\mathrm{SCFG} - 1$",
    )
    output = "plots/compare_codex_m200_r200.png"
    savefig(output, fig=fig, tight=False)
    return


main()
