# tools_sim.py
from __future__ import annotations

import os
from typing import Optional, Dict, Any, List

import numpy as np
from pydantic import BaseModel, Field

from ase.io import read, write
from ase import units
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
from ase.io import Trajectory as AseTrajectory  # optional; or use ase.io.trajectory.Trajectory

# -------------------------
# Shared in-memory "state"
# -------------------------
_STATE: Dict[str, Any] = {
    "atoms": None,
    "calc_attached": False,
    "last_paths": {},
}


# -------------------------
# Helpers
# -------------------------
def _require_atoms():
    if _STATE["atoms"] is None:
        raise ValueError("No structure loaded. Run load_structure first.")
    return _STATE["atoms"]


def _infer_orthogonal_cell_xy(atoms):
    """Return (Lx, Ly) if cell is set; else None."""
    cell = atoms.cell.array
    if np.allclose(cell[0, 1:], 0, atol=1e-8) and np.allclose(cell[1, [0, 2]], 0, atol=1e-8):
        return float(cell[0, 0]), float(cell[1, 1])
    return None

def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

# -------------------------
# Tool schemas
# -------------------------
class LoadStructureArgs(BaseModel):
    path: str = Field(..., description="Path to POSCAR/CONTCAR/EXTXYZ/etc.")


class SetPBCArgs(BaseModel):
    path: Optional[str] = Field(None, description="Optional structure path. If provided, loads it first.")
    pbc: List[bool] = Field(..., description="PBC flags, e.g. [True, True, True] or [True, True, False].")
    ensure_cell_z: Optional[float] = Field(
        None, description="If provided, ensure cell z length is at least this value (Å)."
    )


class FreezeBottomLayersArgs(BaseModel):
    element: str = Field("Au", description="Element to consider for bottom-layer detection.")
    n_layers: int = Field(2, description="Number of bottom layers to freeze.")
    layer_tol: float = Field(0.5, description="Tolerance (Å) to cluster z into layers.")


class AttachUMAArgs(BaseModel):
    model_name: str = Field("uma-s-1p1", description="UMA model name, e.g. uma-s-1p1")
    task_name: str = Field("oc20", description="FAIRChem task name, e.g. oc20")
    device: str = Field("cpu", description="cuda or cpu")
    hf_cache: Optional[str] = Field(None, description="Optional HF cache dir.")


class RunMDArgs(BaseModel):
    steps: int = Field(2000, description="Number of MD steps.")
    temperature_K: float = Field(500.0, description="Target temperature in Kelvin.")
    timestep_fs: float = Field(1.0, description="MD timestep in fs.")
    friction_per_fs: float = Field(0.01, description="Langevin friction in 1/fs.")
    traj_path: str = Field("md.traj", description="Output trajectory path.")
    log_path: str = Field("md.log", description="Output log path.")
    log_interval: int = Field(50, description="Log every N steps.")
    traj_interval: int = Field(10, description="Write trajectory every N steps.")


class WriteOutputsArgs(BaseModel):
    poscar_path: str = Field("POSCAR_out", description="POSCAR output filename.")
    extxyz_path: str = Field("out.extxyz", description="EXTXYZ output filename.")
    group_elements_in_poscar: bool = Field(True, description="Sort atoms by atomic number before POSCAR write.")


# -------------------------
# Tools (functions)
# -------------------------
def load_structure(path: str) -> str:
    atoms = read(path)
    _STATE["atoms"] = atoms
    _STATE["calc_attached"] = False
    _STATE["last_paths"]["loaded"] = path
    return f"Loaded structure from {path}. natoms={len(atoms)} pbc={atoms.pbc.tolist()} cell={atoms.cell.lengths()} Å"

def make_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return f"Ensured directory exists: {path}"

def set_pbc(pbc: List[bool], ensure_cell_z: Optional[float] = None, path: Optional[str] = None) -> str:
    if isinstance(path, str) and path.strip().lower() in {"none", "null", ""}:
        path = None
        
    if path is not None:
        load_structure(path)

    atoms = _require_atoms()
    atoms.set_pbc(pbc)

    if ensure_cell_z is not None:
        cell = atoms.cell.array.copy()
        if cell[2, 2] < ensure_cell_z:
            cell[2, 2] = ensure_cell_z
            atoms.set_cell(cell, scale_atoms=False)

    _STATE["atoms"] = atoms
    return f"Set PBC to {atoms.pbc.tolist()}. cell={atoms.cell.lengths()} Å"



def freeze_bottom_layers(element: str = "Au", n_layers: int = 2, layer_tol: float = 0.5) -> str:
    atoms = _require_atoms()

    symbols = np.array(atoms.get_chemical_symbols())
    z = atoms.get_positions()[:, 2]

    mask_el = (symbols == element)
    if not mask_el.any():
        return f"No atoms with element={element} found. No constraints applied."

    z_el = z[mask_el]
    z_sorted = np.sort(z_el)

    # cluster z positions into "layers" using tolerance
    layers = []
    current = [z_sorted[0]]
    for val in z_sorted[1:]:
        if abs(val - current[-1]) <= layer_tol:
            current.append(val)
        else:
            layers.append(np.mean(current))
            current = [val]
    layers.append(np.mean(current))

    layers = np.array(sorted(layers))
    if len(layers) < n_layers:
        chosen = layers
    else:
        chosen = layers[:n_layers]  # bottom-most layers

    # mark atoms whose z is close to one of chosen layer z's
    idx_fix = []
    for i, (sym, zi) in enumerate(zip(symbols, z)):
        if sym != element:
            continue
        if np.min(np.abs(chosen - zi)) <= layer_tol:
            idx_fix.append(i)

    atoms.set_constraint(FixAtoms(indices=idx_fix))
    _STATE["atoms"] = atoms
    return f"Applied FixAtoms to {len(idx_fix)} atoms (bottom {n_layers} {element} layers)."

def attach_emt() -> str:
    """Attach ASE EMT calculator (fast fallback, no ML model download)."""
    atoms = _require_atoms()
    print("[attach_emt] start", flush=True)
    atoms.calc = EMT()
    print("[attach_emt] calculator attached", flush=True)
    _STATE["atoms"] = atoms
    _STATE["calc_attached"] = True
    return "Attached EMT calculator."

def attach_uma(model_name: str = "uma-s-1p1", task_name: str = "oc20", device: str = "cpu", hf_cache: Optional[str] = None) -> str:
    atoms = _require_atoms()
    print("[attach_uma] start", flush=True)

    print("[attach_uma] importing fairchem...", flush=True)
    from fairchem.core import pretrained_mlip, FAIRChemCalculator  # type: ignore
    print("[attach_uma] imported fairchem", flush=True)

    kwargs = {}
    if hf_cache:
        kwargs["cache_dir"] = str(hf_cache)

    print(f"[attach_uma] get_predict_unit model={model_name} device={device}", flush=True)
    predictor = pretrained_mlip.get_predict_unit(model_name, device=device, **kwargs)
    print("[attach_uma] predictor ready", flush=True)

    atoms.calc = FAIRChemCalculator(predictor, task_name=task_name)
    print("[attach_uma] calculator attached", flush=True)

    _STATE["atoms"] = atoms
    _STATE["calc_attached"] = True
    return f"Attached UMA calculator: model={model_name}, task={task_name}, device={device}."



def run_md(
    steps: int = 50,
    temperature_K: float = 700.0,
    timestep_fs: float = 1.0,
    friction_per_fs: float = 0.01,
    traj_path: str = "md.traj",
    log_path: str = "md.log",
    log_interval: int = 10,
    traj_interval: int = 10,
) -> str:
    atoms = _require_atoms()
    if atoms.calc is None:
        raise ValueError("No calculator attached. Run attach_uma first.")
    _ensure_parent_dir(traj_path)
    _ensure_parent_dir(log_path)
    #Clear old log if it exists
    if os.path.exists(log_path):
        os.remove(log_path)
    print("[run_md] using structure:", _STATE["last_paths"].get("loaded"), flush=True)
    print("[run_md] cell lengths (Å):", atoms.cell.lengths(), "pbc:", atoms.pbc.tolist(), flush=True)
    # initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    Stationary(atoms)
    ZeroRotation(atoms)

    dt = timestep_fs * units.fs
    friction = friction_per_fs / units.fs

    dyn = Langevin(atoms, timestep=dt, temperature_K=temperature_K, friction=friction)

    traj = Trajectory(traj_path, "w", atoms)
    dyn.attach(traj.write, interval=traj_interval)

    def log():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        line = (
        f"step={dyn.nsteps:6d}  Epot={epot: .6f} eV  "
        f"Ekin={ekin: .6f} eV  Etot={epot+ekin: .6f} eV"
        )
        with open(log_path, "a") as f:
            f.write(line + "\n")
        print("[run_md] " + line, flush=True)

    dyn.attach(log, interval=log_interval)
    dyn.run(steps)

    _STATE["atoms"] = atoms
    _STATE["last_paths"]["traj"] = traj_path
    _STATE["last_paths"]["log"] = log_path
    return f"MD finished: steps={steps}, T={temperature_K} K. Wrote {traj_path} and {log_path}."


def write_outputs(poscar_path: str = "POSCAR_out", extxyz_path: str = "out.extxyz", group_elements_in_poscar: bool = True) -> str:
    atoms = _require_atoms()
    _ensure_parent_dir(extxyz_path)
    _ensure_parent_dir(poscar_path)

    write(extxyz_path, atoms, format="extxyz")

    if group_elements_in_poscar:
        order = np.argsort(atoms.get_atomic_numbers())
        atoms_sorted = atoms[order]
        write(poscar_path, atoms_sorted, vasp5=True)
    else:
        write(poscar_path, atoms, vasp5=True)

    _STATE["last_paths"]["poscar"] = poscar_path
    _STATE["last_paths"]["extxyz"] = extxyz_path
    return f"Wrote {extxyz_path} and {poscar_path} (group_elements_in_poscar={group_elements_in_poscar})."


def export_traj_to_xyz(traj_path: str, xyz_path: str) -> str:
    # ensure parent dir exists
    parent = os.path.dirname(xyz_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # read all frames from the .traj and write as multi-frame xyz
    frames = read(traj_path, index=":")
    write(xyz_path, frames, format="extxyz")

    return f"Exported trajectory: {traj_path} -> {xyz_path} (nframes={len(frames)})"