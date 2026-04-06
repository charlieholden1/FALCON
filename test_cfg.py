with open("iwr6843_config.cfg", "r") as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    if line.startswith("chirpCfg"):
        import re
        print(repr(line))
        # ensure perfectly 1 space between words
        clean_line = re.sub(r'\s+', ' ', line)
        print(repr(clean_line))
