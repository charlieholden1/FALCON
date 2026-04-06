with open("camera_stream.py", "r") as f:
    lines = f.readlines()

with open("camera_stream.py", "w") as f:
    for line in lines:
        if "    _RS_COLOR_W: int = 640" in line:
            f.write("    _RS_COLOR_W: int = 848\n")
        elif "    _RS_COLOR_H: int = 480" in line:
            f.write("    _RS_COLOR_H: int = 480\n")
        elif "    _RS_DEPTH_W: int = 640" in line:
            f.write("    _RS_DEPTH_W: int = 848\n")
        elif "    _RS_DEPTH_H: int = 480" in line:
            f.write("    _RS_DEPTH_H: int = 480\n")
        else:
            f.write(line)
