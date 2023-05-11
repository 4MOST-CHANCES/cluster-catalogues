from astropy.io import ascii, fits
from astroquery.eso import Eso
from matplotlib import pyplot as plt
import numpy as np

from astro.footprint import Footprint


def main():
    viking = np.transpose(
        [[[0, 0, 2.5, 52.5], [-36, -26, -26, -36]],
         [[330, 330, 360, 360], [-36, -26, -26, -36]],
         [[150, 150, 232.5, 232.5], [-5, 4, 4, -5]],
         [[129, 129, 141, 141], [-2, 3, 3, -2]]],
         axes=(0,2,1))
    viking = Footprint('VIKING', footprint=viking)
    
    #eso = Eso()
    return


main()
