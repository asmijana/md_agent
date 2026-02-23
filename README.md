# md_agent   

An LLM-driven **simulation workflow agent** for running small ASE molecular dynamics (MD) workflows on surfaces and adsorbates. The agent plans and executes a workflow using **tool calls** (load structure → set PBC/cell → freeze layers → attach calculator → run MD → write outputs). It supports:   

- **UMA (FAIRChem) MLIP** (advanced mode; requires Hugging Face access to gated weights)   
- **ASE EMT** fallback (runs everywhere; good for demos / plumbing)   

This repo is designed to be a **GitHub-friendly demo**: it produces trajectories, logs, and final structures into an output directory, and can also export the full trajectory to a multi-frame `.xyz` (and/or `.extxyz`) for viewing   

## Example MD simulation  
![Demo](/demo.gif)  
<img src="assets/demo.gif" width="800" />   

## What it does   

Given a natural-language prompt like:   

> “Load POSCAR, set PBC, freeze bottom Au layers, attach calculator, run MD, write outputs into output_dir, convert trajectory to XYZ”   

…the agent will select tools and arguments at runtime, return your requested outputs, and report **the tools it used**   

The tool kit includes ones to load structure, set PBC, freeze layers, create directories, attach fairchem calculation, attach EMT calculator, run MD, write output files, and export trajectory files to xyz format. This can be expanded by the user     

---
## Requirements   
Note: I used ollama to access the LLM. I used a local LLM: llama3.1:8b via Ollama   
You can download Ollama at ollama.com/download, use ```ollama pull llama3.1:8b``` to get the llama3.1:8b model, and use ```ollama serve``` to run the framework.  

Core: langchain, langchain-ollama, pydantic, ase, numpy    
Optional: fairchem-core, huggingface_hub   
Note: The optional dependencies require auth+access to gated repo. See https://github.com/facebookresearch/fairchem for more information      

## Install   
### Activate a virtual environment and install dependencies   
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
ollama serve
```
### Running the agent   
The agent requires two inputs:   
- Initial structure: provided as a POSCAR file in the folder `all_input_structs/`   
- Prompt: provided in the `prompts.md` file under the corresponding prompt-id   

An example prompt following a prompt-id:   
> au_md_smoke     
> Load ./all_input_structs/POSCAR_Au_only, set PBC to [True, True, True] with at least 40 Å in z, freeze bottom 2 Au layers, make sure that all elements are grouped together. Create a directory called output_dir in the current directory if there is no such directory named output_dir in the current directory. attach the EMT calculator by running the tool t_attach_emt(), run 200 steps MD at 500 K in steps of 10, and write outputs, including the trajectory(.traj), log (.log), and other final structure files (.extxyz) in the directory called output_dir. Make sure that one of the outputs is the trajectory file (.traj) converted to a multi-frame .xyz file.   

Then run    
```bash
python agent.py --prompts prompts.md --prompt-id au_md_smoke
(OR)
python agent.py #default: prompts.md and au_md_smoke
```
### Outputs
By default, the workflow writes into output_dir/:
- `md.traj`: ASE trajectory   
- `md.log`: text log of energies   
- `md.xyz`: trajectory in extended xyz format
- `final.extxyz`: final structure in extended xyz format
- `POSCAR_final`: final structure in POSCAR/VASP format   

---
## Repo structure

```bash
md_agent/
├─ agent.py                                # LLM + tool execution
├─ tools_sim.py                            # All tools defined here
├─ prompts.md                              # Add your prompts here
├─ README.md
├─ demo.gif                                # Example MD simulation
├─ requirements.txt           
├─ .gitignore
├─ all_input_structs/
│  ├─ POSCAR_Au_only
│  ├─ POSCAR_Au_with_water_CO
│  └─ ...                                  # Add your input POSCAR here
├─ examples/
│  ├─ example_1_au_surface.                # Au(111) using EMT
│  └─ example_2_au_surface_with_co_water   # Au(111) with CO and H2O adsorbed
└─ output_dir/                             # generated (gitignored)
   ├─ md.traj
   ├─ md.xyz
   ├─ md.log
   ├─ final.extxyz
   └─ POSCAR_final
```