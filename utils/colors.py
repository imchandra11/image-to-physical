# import matplotlib._color_data  # Colors
from random import randrange, sample

TABLEAU_COLORS = {
    'tab:blue': '#1f77b4',
    'tab:orange': '#ff7f0e',
    'tab:green': '#2ca02c',
    'tab:red': '#d62728',
    'tab:purple': '#9467bd',
    'tab:brown': '#8c564b',
    'tab:pink': '#e377c2',
    'tab:gray': '#7f7f7f',
    'tab:olive': '#bcbd22',
    'tab:cyan': '#17becf',
}

BASE_COLORS = {
    'b': (0, 0, 1),           # blue
    'g': (0, 0.5, 0),         # green
    'r': (1, 0, 0),           # red
    'c': (0, 0.75, 0.75),     # cyan
    'm': (0.75, 0, 0.75),     # magenta
    'y': (0.75, 0.75, 0),     # yellow
    'k': (0, 0, 0),           # black
    # 'w': (1, 1, 1),         # white
}


def getRandomColor(max=255):
    return (randrange(max), randrange(max), randrange(max))


def getRandomBASEColors(count=1):
    r"""
    Returns (r, g, b) from 0 to 255
    """
    # return sample(list(BASE_COLORS.values()), count)
    return sample(
        [tuple(e * 255 for e in t) for t in list(BASE_COLORS.values())], 
        count
    )


def getRandomTABLEAUColors(count=1):
    r"""
    Returns HEX!
    """
    return sample(list(TABLEAU_COLORS.values()), count)


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))