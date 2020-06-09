#!/usr/bin/env python

__version__ = "0.0.1"


import sys
from PyQt5.QtWidgets import QApplication
from .gui.MainWindow import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationDisplayName("EvoBridge")

    window = MainWindow()
    app.installEventFilter(window)
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()