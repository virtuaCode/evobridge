from .AppWidget import AppWidget
from PyQt5.QtWidgets import (
    QAction, QLabel, QMainWindow, QFileDialog, QActionGroup, QProgressBar, QSpacerItem, QAbstractButton, QMessageBox, QSpinBox, QWidget)
from PyQt5.QtGui import (QIcon, QGuiApplication)
from PyQt5.QtCore import (Qt, pyqtSlot)
from .TestWidget import TestWidget
from .State import State
import os
import sys
from ..lib.optimize import LocalSearchOptimizer
from .Objects import Mutation, ObjectiveFunction
from ..lib.functions import create_onebit_mutate, create_threshold_accept, create_probbit_mutate

from ..lib.optimize import OptimizerFactory

from enum import Enum
import traceback

moduleDir = os.path.dirname(__file__)


class MainWindow(QMainWindow):
    def __init__(self, file=None):
        super().__init__()
        self.factory = OptimizerFactory()

        self.app = AppWidget(self.factory)
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

        self.plotAction = QAction(QIcon(), "Plot", self)
        self.plotAction.triggered.connect(self.plotBridge)

        self.optimizeAction = QAction(QIcon(), "Optimize", self)
        self.optimizeAction.triggered.connect(self.optimizeBridge)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(100)
        self.progress.setEnabled(False)

        self.iterations = QSpinBox()
        self.iterations.setSingleStep(1)
        self.iterations.setMinimum(1)
        self.iterations.setMaximum(10000)
        self.iterations.setMinimumWidth(90)
        self.iterations.setValue(300)

        toolbar = self.addToolBar("Tools")
        toolbar.addAction(newNodeAction)
        toolbar.addAction(newRockAction)
        toolbar.addSeparator()
        toolbar.addActions(materialActionGroup.actions())
        toolbar.addSeparator()
        toolbar.addAction(self.plotAction)
        toolbar.addAction(self.optimizeAction)
        toolbar.addWidget(self.progress)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Iterations: "))
        toolbar.addWidget(self.iterations)

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
            except BaseException as e:
                traceback.print_exc()
                #print(e, file=sys.stderr)

    @pyqtSlot()
    def plotBridge(self):

        try:
            optimizer = self.factory.createOptimizer(self.app.drawing.state)
            w = optimizer.plot(self)
            w.showMaximized()

        except BaseException as e:
            traceback.print_exc()
            #print(e.__traceback__, file=sys.stderr)
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("Failed to plot bridge")
            msgBox.setMinimumWidth(500)
            msgBox.setInformativeText(str(e))
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setDefaultButton(QMessageBox.Ok)
            msgBox.exec()

    @pyqtSlot()
    def optimizeBridge(self):
        self.optimizeAction.setEnabled(False)
        self.plotAction.setEnabled(False)
        self.iterations.setEnabled(False)
        self.app.tabs.setEnabled(False)

        try:
            optimizer = self.factory.createOptimizer(self.app.drawing.state)
            optimizer.run(progress=self.progress,
                          max_iter=int(self.iterations.value()))
            w = optimizer.plot(self)
            w.showMaximized()

        except BaseException as e:
            traceback.print_exc()
            #print(e.__traceback__, file=sys.stderr)
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("Failed to optimize bridge")
            msgBox.setMinimumWidth(500)
            msgBox.setInformativeText(str(e))
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setDefaultButton(QMessageBox.Ok)
            msgBox.exec()
        finally:
            self.optimizeAction.setEnabled(True)
            self.plotAction.setEnabled(True)
            self.iterations.setEnabled(True)
            self.app.tabs.setEnabled(True)

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

    @ pyqtSlot(float)
    def setGridSize(self, val):
        self.gridSizeLabel.setText("Grid Size: {}".format(max(1, val)*0.25))
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
