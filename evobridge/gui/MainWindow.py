from .AppWidget import AppWidget
from PyQt5.QtWidgets import (
    QAction, QLabel, QMainWindow, QFileDialog, QActionGroup)
from PyQt5.QtGui import (QIcon)
from PyQt5.QtCore import (Qt, pyqtSlot)

from .State import State
import os
from ..lib.lsearch import LocalSearchOptimizer

moduleDir = os.path.dirname(__file__)


class MainWindow(QMainWindow):
    def __init__(self, file=None):
        QMainWindow.__init__(self)

        self.app = AppWidget()
        statusbar = self.statusBar()

        newAction = QAction("&New File", self)
        newAction.setShortcut("Ctrl+N")
        newAction.triggered.connect(self.newFile)

        openAction = QAction("&Open File...", self)
        openAction.setShortcut("Ctrl+O")
        openAction.triggered.connect(self.openFile)

        saveAsAction = QAction("&Save As...", self)
        saveAsAction.setShortcut("Shift+Ctrl+S")
        saveAsAction.triggered.connect(self.saveAsFile)

        newNodeAction = QAction(
            QIcon(os.path.join(moduleDir, 'icons/new-node.png')),
            "Create Node", self)
        newNodeAction.triggered.connect(self.app.drawing.addNewNode)

        newRockAction = QAction(
            QIcon(os.path.join(moduleDir, 'icons/new-rock.png')),
            "Create Rock", self)
        newRockAction.triggered.connect(self.app.drawing.addNewRock)

        toggleWoodAction = QAction(QIcon(), "Wood", self)
        toggleStreetAction = QAction(QIcon(), "Street", self)
        toggleSteelAction = QAction(QIcon(), "Steel", self)

        toggleWoodAction.setCheckable(True)
        toggleStreetAction.setCheckable(True)
        toggleSteelAction.setCheckable(True)

        toggleWoodAction.triggered.connect(self.app.drawing.toggleWood)
        toggleSteelAction.triggered.connect(self.app.drawing.toggleSteel)
        toggleStreetAction.triggered.connect(self.app.drawing.toggleStreet)

        materialActionGroup = QActionGroup(self)
        materialActionGroup.addAction(toggleStreetAction)
        materialActionGroup.addAction(toggleWoodAction)
        materialActionGroup.addAction(toggleSteelAction)
        materialActionGroup.setExclusive(True)

        plotAction = QAction(QIcon(), "Plot", self)
        plotAction.triggered.connect(self.plotBridge)

        optimizeAction = QAction(QIcon(), "Optimize", self)
        optimizeAction.triggered.connect(self.optimizeBridge)

        toolbar = self.addToolBar("Tools")
        toolbar.addAction(newNodeAction)
        toolbar.addAction(newRockAction)
        toolbar.addSeparator()
        toolbar.addActions(materialActionGroup.actions())
        toolbar.addSeparator()
        toolbar.addAction(plotAction)
        toolbar.addAction(optimizeAction)

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("&File")
        fileMenu.addAction(newAction)
        fileMenu.addAction(openAction)
        fileMenu.addAction(saveAsAction)

        self.objectInfoLabel = QLabel()
        self.objectInfoLabel.setText("0 objects selected")
        self.objectInfoLabel.setAlignment(Qt.AlignRight)

        self.cursorInfoLabel = QLabel()
        self.cursorInfoLabel.setText("")
        self.cursorInfoLabel.setAlignment(Qt.AlignRight)

        self.gridSizeLabel = QLabel()
        self.gridSizeLabel.setText("Grid Size: 1")
        self.gridSizeLabel.setAlignment(Qt.AlignRight)

        statusbar.addWidget(self.objectInfoLabel)
        statusbar.addPermanentWidget(self.cursorInfoLabel)
        statusbar.addPermanentWidget(self.gridSizeLabel)

        self.setCentralWidget(self.app)

        self.app.drawing.onGridSizeChange.connect(self.setGridSize)
        self.app.drawing.onCursorChange.connect(self.setCursorInfo)
        self.app.drawing.onObjectChange.connect(self.setObjectInfo)
        self.app.prop.onValueChanged.connect(self.app.drawing.update)

        toggleSteelAction.activate(QAction.Trigger)

        if file:
            try:
                self.app.drawing.state = State.loadState(file)
                self.setWindowFilePath(file)
                self.update()
            except:
                pass

    @pyqtSlot()
    def plotBridge(self):
        optimizer = LocalSearchOptimizer(self.app.drawing.state)
        try:
            optimizer.plot()
        except (ArithmeticError, AssertionError) as e:
            pass

    @pyqtSlot()
    def optimizeBridge(self):
        optimizer = LocalSearchOptimizer(
            self.app.drawing.state)
        try:
            optimizer.run()
            optimizer.plot()
        except (ArithmeticError, AssertionError) as e:
            print(e)

    @ pyqtSlot()
    def newFile(self):
        self.app.drawing.setState(State())
        self.setWindowFilePath(None)
        self.update()

    @ pyqtSlot()
    def openFile(self):
        filepath, ext = QFileDialog.getOpenFileName(
            self, 'Open bridge file', os.getcwd(), "Bridge files (*.bridge)")
        if len(filepath) > 0:
            self.app.drawing.setState(State.loadState(filepath))
            self.setWindowFilePath(filepath)
            self.update()

    @ pyqtSlot()
    def saveAsFile(self):
        filepath, ext = QFileDialog.getSaveFileName(
            self, 'Save bridge file', os.getcwd(), "Bridge files (*.bridge)")
        if len(filepath) > 0:
            self.app.drawing.state.saveState(filepath)
            self.setWindowFilePath(filepath)
            self.update()

    @ pyqtSlot(int)
    def setGridSize(self, val):
        self.gridSizeLabel.setText("Grid Size: {}".format(val))
        self.update()

    @ pyqtSlot(float, float)
    def setCursorInfo(self, x, y):
        self.cursorInfoLabel.setText("Position: ({:.1f}, {:.1f})".format(x, y))
        self.update()

    @ pyqtSlot(list)
    def setObjectInfo(self, objects):
        self.objectInfoLabel.setText(
            "{} object{} selected".format(len(objects), "" if len(objects) == 1 else "s"))

        if len(objects) == 1:
            self.app.prop.setActiveObject(objects[0])
        else:
            self.app.prop.setActiveObject(None)

        self.update()
