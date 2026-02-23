from ase.io import read, write
from ase.build import fcc111, add_adsorbate, molecule
import numpy as np

slab = fcc111('Au', size=(6, 6, 6), vacuum=10.0)
A1 = slab.cell[0]
A2 = slab.cell[1]
a1 = A1/6.0
a2 = A2/6.0
#central ontop position
central_xy = (3.0*a1 + 3.0*a2)[:2]
#another ontop position
another_xy = (2.0*a1 + 2.0*a2)[:2]
add_adsorbate(slab, molecule("CO"), 3.0, position=another_xy)
add_adsorbate(slab, molecule("H2O"), 3.0, position=central_xy)

write("POSCAR_Au_with_water_CO", slab, vasp5=True)
