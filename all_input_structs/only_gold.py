from ase.io import read, write
import numpy as np

atoms = read("./POSCAR_Au_Nafion_Water")
symbols = np.array(atoms.get_chemical_symbols())

# keep only gold
atoms_au = atoms[symbols == "Au"]
atoms_au.set_cell(atoms.cell, scale_atoms=False)
atoms_au.set_pbc(atoms.pbc)

write("./POSCAR_Au_only", atoms_au, vasp5=True)
print("Wrote ./POSCAR_Au_only natoms=", len(atoms_au))