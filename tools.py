def scalebar_label(bar, label=None, pad=0, **kwargs):
    """Adapted from AutoScaledWCSAxes.ScaleBar.label()"""
    (x0, y), (x1, _) = bar._posA_posB
    if label is None:
        label = " {0.value:g}{0.unit:unicode}".format(bar._length)
    return bar._ax.text(
        0.5 * (x0 + x1),
        y + pad,
        label,
        ha="center",
        va="bottom",
        transform=bar._ax.transAxes,
        **kwargs,
    )
