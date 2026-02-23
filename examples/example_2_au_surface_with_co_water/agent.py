# agent.py
from __future__ import annotations
from typing import Any
import os
import argparse
import re
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from tools_sim import (
    load_structure, make_dir, set_pbc, freeze_bottom_layers, attach_uma, attach_emt, run_md, write_outputs, export_traj_to_xyz
)

# --------
# Wrap tools with @tool so LangChain can call them
# --------

@tool
def t_load_structure(path: str) -> str:
    """Load an atomic structure file (POSCAR/CONTCAR/EXTXYZ/etc) into agent state and report basic info."""
    return load_structure(path)
@tool
def t_make_dir(path: str) -> str:
    """Create a directory if it doesn't exist."""
    return make_dir(path)

@tool
def t_set_pbc(pbc: list[bool], ensure_cell_z: float | None = None, path: str | None = None) -> str:
    """Set periodic boundary conditions; optionally load structure first via path and ensure z length (Å)."""
    return set_pbc(pbc=pbc, ensure_cell_z=ensure_cell_z, path=path)

@tool
def t_freeze_bottom_layers(element: str = "Au", n_layers: int = 2, layer_tol: float = 0.5) -> str:
    """Freeze bottom n_layers of a given element by adding an ASE FixAtoms constraint (layer clustering in z)."""
    return freeze_bottom_layers(element=element, n_layers=n_layers, layer_tol=layer_tol)

@tool
def t_attach_uma(model_name: str = "uma-s-1p1", task_name: str = "oc20", device: str = "cpu", hf_cache: Any = None) -> str:
    """Attach UMA. model_name must be 'uma-s-1p1'. task_name must be 'oc20'."""
    # Coerce weird model outputs
    if isinstance(hf_cache, dict) or isinstance(hf_cache, list):
        hf_cache = None
    if hf_cache is not None and not isinstance(hf_cache, str):
        hf_cache = str(hf_cache)
    if " " in model_name and task_name == "oc20":
        # common model mistake: "uma-s-1p1 oc20"
        model_name = model_name.split()[0]
    return attach_uma(model_name=model_name, task_name=task_name, device=device, hf_cache=hf_cache)

@tool
def t_attach_emt() -> str:
    """Attach a lightweight fallback calculator (ASE EMT). Use if UMA is unavailable."""
    return attach_emt()

@tool
def t_run_md(
    steps: int = 50,
    temperature_K: float = 700.0,
    timestep_fs: float = 1.0,
    friction_per_fs: float = 0.01,
    traj_path: str = "md.traj",
    log_path: str = "md.log",
    log_interval: int = 10,
    traj_interval: int = 10,
) -> str:
    """Run MD. timestep_fs default 1.0; friction_per_fs default 0.01; traj_path should end with .traj."""
    if timestep_fs > 2.0:
        timestep_fs = 1.0
    if friction_per_fs > 0.1:
        friction_per_fs = 0.01
    if not traj_path.endswith(".traj"):
        traj_path = "md.traj"
    if not log_path.endswith(".log"):
        log_path = "md.log"
    return run_md(
        steps=steps,
        temperature_K=temperature_K,
        timestep_fs=timestep_fs,
        friction_per_fs=friction_per_fs,
        traj_path=traj_path,
        log_path=log_path,
        log_interval=log_interval,
        traj_interval=traj_interval,
    )

@tool
def t_write_outputs(poscar_path: str = "POSCAR_out", extxyz_path: str = "out.extxyz", group_elements_in_poscar: bool = True) -> str:
    """Write final structure to EXTXYZ and POSCAR (optionally grouped by element) and report filenames."""
    return write_outputs(poscar_path=poscar_path, extxyz_path=extxyz_path, group_elements_in_poscar=group_elements_in_poscar)

@tool
def t_export_traj_to_xyz(traj_path: str, xyz_path: str) -> str:
    """Convert an ASE .traj trajectory into a multi-frame .xyz file."""
    return export_traj_to_xyz(traj_path=traj_path, xyz_path=xyz_path)

TOOLS = [
    t_load_structure,
    t_make_dir,
    t_set_pbc,
    t_freeze_bottom_layers,
    t_attach_uma,
    t_attach_emt, 
    t_run_md,
    t_write_outputs,
    t_export_traj_to_xyz,
]

SYSTEM = """You are a simulation workflow agent. You MUST execute tools to complete the request.

Rules:
When you need to use a tool, you MUST call the tool (function calling). 
Do NOT print tool calls as code blocks or JSON text.
If you cannot call tools, respond with exactly: TOOL_CALLING_FAILED

- Do NOT write or output tool calls as JSON or as a plan.
- Always call tools using the tool calling mechanism.
- If a required file path is missing, ask for the exact path.
- If t_attach_uma fails for any reason (missing package, gated model, auth error), immediately call t_attach_emt and continue the workflow.
- Never run MD unless a calculator is attached.
- Your final response must ONLY summarize the tool RESULTS (the returned strings) and list the output files.
You must continue calling tools until ALL required outputs exist:
- output_dir/md.traj
- output_dir/md.xyz (multi-frame XYZ converted from the traj)
- output_dir/md.log
- output_dir/final.extxyz
- output_dir/POSCAR_final
Do not stop after partial steps like loading/setting PBC.
"""

def run_once(llm, tools, system_prompt, user_request):
    tool_map = {t.name: t for t in tools}
    llm = llm.bind_tools(tools)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_request)]
    ai = llm.invoke(messages)

    tool_calls = getattr(ai, "tool_calls", None) or []
    executed = []

    if not tool_calls:
        return f"No tool calls. Model said:\n{ai.content}"

    for call in tool_calls:
        name = call["name"]
        args = call.get("args", {}) or {}
        try:
            result = tool_map[name].invoke(args)
        except Exception as e:
            result = f"ERROR running {name}: {type(e).__name__}: {e}"
        executed.append((name, args, str(result)))

        # HARD STOP on any error (prevents infinite loops)
        if str(result).startswith("ERROR"):
            break

    lines = ["Executed tools (verbatim outputs):"]
    for n, a, r in executed:
        lines.append(f"- {n} args={a}\n  -> {r}")
    return "\n".join(lines)

def load_prompt_from_md(md_path: str, prompt_id: str) -> str:
    text = Path(md_path).read_text()
    # capture section content under "## prompt_id" until next "## " or end
    pattern = rf"^##\s+{re.escape(prompt_id)}\s*$([\s\S]*?)(?=^##\s+|\Z)"
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"Prompt id '{prompt_id}' not found in {md_path}")
    return m.group(1).strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="prompts.md", help="Path to prompts markdown file")
    parser.add_argument("--prompt-id", default="au_md_smoke", help="Prompt section id in prompts.md (e.g., au_md_smoke)")
    args = parser.parse_args()

    user_request = load_prompt_from_md(args.prompts, args.prompt_id)
    llm = ChatOllama(model="llama3.1:8b", temperature=0)  # must support tools
    final = run_once(llm, TOOLS, SYSTEM, user_request)
    print("\n=== FINAL ===")
    print(final)
