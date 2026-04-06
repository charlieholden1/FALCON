with open("falcon_gui.py", "r") as f:
    lines = f.readlines()

with open("falcon_gui.py", "w") as f:
    for line in lines:
        if "n_depth += 1" in line:
            f.write(line)
            f.write("            if getattr(t, 'using_radar', False):\n")
            f.write("                occ_label += f' [RADAR {t.radar_confidence:.0%}]'\n")
        elif "f\"Depth: {n_depth}/{len(tracks)}\"," in line:
            f.write(line)
            f.write("            f\"Radar Fusion: {sum(1 for t in tracks if getattr(t, 'using_radar', False))}/{len(tracks)}\",\n")
        else:
            f.write(line)
