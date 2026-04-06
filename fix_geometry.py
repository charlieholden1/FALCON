with open("falcon_gui.py", "r") as f:
    lines = f.readlines()

with open("falcon_gui.py", "w") as f:
    for line in lines:
        if "self.root.minsize(960, 540)" in line:
            f.write(line)
            f.write("        self.root.geometry('1280x720')\n")
            f.write("        self.root.resizable(False, False)\n")
        else:
            f.write(line)
