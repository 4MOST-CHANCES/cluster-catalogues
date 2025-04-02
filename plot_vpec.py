from astropy.io import fits
from astropy.table import Table
from astropy.visualization import hist
from matplotlib import pyplot as plt, ticker
import pandas as pd
from plottery.plotutils import update_rcParams

update_rcParams()

chances = Table.read(
    "aux/peculiar_velocities/chances_cat_data.txt", format="ascii.basic"
)
reflex = Table.read("aux/peculiar_velocities/reflex_cat_data.txt", format="ascii.basic")
abell = Table.read("aux/peculiar_velocities/abell_cat_data.txt", format="ascii.basic")

fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
hist(
    chances["v_value"],
    bins="knuth",
    density=True,
    color="C1",
    histtype="stepfilled",
    lw=0,
    alpha=0.6,
    label="CHANCES",
)
hist(
    reflex["v_value"],
    bins="knuth",
    density=True,
    color="C2",
    histtype="step",
    lw=2.5,
)
hist(
    abell["v_value"],
    bins="knuth",
    density=True,
    color="C0",
    histtype="step",
    lw=2,
)
ax.plot([], [], "C2-", lw=2.5, label="REFLEX")
ax.plot([], [], "C0-", lw=2, label="Abell")
ax.legend(loc="upper right", fontsize=14)
ax.set(xlabel="velocity norm (km/s)", ylabel="pdf", yticklabels=[])
output = "plots/vpec_chances_abell_reflex.pdf"
fig.savefig(output)
plt.close(fig)
