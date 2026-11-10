"""Entry point (for running from source and for PyInstaller)."""
import multiprocessing

from photo_dupes.gui import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
