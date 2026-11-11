"""Точка входа: и для запуска из исходников, и для PyInstaller."""
import multiprocessing

from photo_dupes.gui import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
