This directory holds the minimum vendored Qt runtime pieces needed for the
PySide6 wheel to launch on this Ubuntu/Orin environment without root-level apt
installs.

Included:
- `lib/libdouble-conversion.so.3*`
- `lib/libxcb-cursor.so.0*`

The main `radar_viewer.py` entrypoint preloads these libraries before importing
PySide6 so the Qt + PyQtGraph backend can start cleanly.
