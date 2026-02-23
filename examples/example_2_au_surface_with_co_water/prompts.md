# md_agent prompts

## au_md_smoke
Load ./all_input_structs/POSCAR_Au_with_water_CO, set PBC to [True, True, True] with at least 40 Å in z, freeze bottom 2 Au layers, make sure that all elements are grouped together. Create a directory called output_dir if there is no such directory. attach UMA uma-s-1p1 oc20 on cpu, run 200 steps MD at 500 K in steps of 10, and write outputs, including the trajectory(.traj), log (.log), and other final structure files (.extxyz) in the directory called output_dir. Make sure that one of the outputs is the trajectory file (.traj) converted to a multi-frame .xyz file.
