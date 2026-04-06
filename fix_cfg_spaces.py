import re
with open("iwr6843_config.cfg", "r") as f:
    lines = f.readlines()

out = []
for line in lines:
    line = line.strip()
    if line and not line.startswith("%"):
        # ensure perfectly single spaced
        line = re.sub(r'\s+', ' ', line)
    out.append(line)

with open("iwr6843_config.cfg", "w") as f:
    f.write("\n".join(out) + "\n")
