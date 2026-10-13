import os

# The script calls plt.show(); use a non-interactive backend so tests never open a window.
os.environ.setdefault("MPLBACKEND", "Agg")
