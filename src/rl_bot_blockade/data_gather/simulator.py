import numpy as np
from matplotlib import colormaps, animation


def simulate(fig, ax, states):
    cmap = colormaps['Pastel2']
    im = ax.imshow(states[0], cmap=cmap, origin="upper")
    row = states.shape[1]
    col = states.shape[2]
    ax.set_xticks(ticks=np.arange(-0.5, row, 1), labels=[])
    ax.set_yticks(ticks=np.arange(-0.5, col, 1), labels=[])
    ax.grid(color="white", linestyle="-", linewidth=1)
    ax.tick_params(axis="both", which="both", length=0)
    anim = animation.FuncAnimation(fig=fig, func=lambda f: im.set_data(f), frames=states, interval=200,
                                   repeat=False)
    return anim


