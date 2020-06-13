from PyQt5.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QFormLayout, QHBoxLayout, QSpinBox, QCheckBox
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt

from .Objects import StateObject, Node, Rock
import sys


class PropertyWidget(QWidget):
    onValueChanged = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Setup Rock Property Group
        self.rockGroup = QGroupBox("Rock Properties")

        rockPropLayout = QFormLayout(self)

        self.xRockSpin = QSpinBox()
        self.xRockSpin.setMinimum(-1000)
        self.xRockSpin.setMaximum(1000)
        self.xRockSpin.setMinimumWidth(90)

        self.yRockSpin = QSpinBox()
        self.yRockSpin.setMinimum(-1000)
        self.yRockSpin.setMaximum(1000)
        self.yRockSpin.setMinimumWidth(90)

        self.wRockSpin = QSpinBox()
        self.wRockSpin.setMinimum(1)
        self.wRockSpin.setMaximum(255)
        self.wRockSpin.setMinimumWidth(90)

        self.hRockSpin = QSpinBox()
        self.hRockSpin.setMinimum(1)
        self.hRockSpin.setMaximum(255)
        self.hRockSpin.setMinimumWidth(90)

        rockPropLayout.addRow("X Position:", self.xRockSpin)
        rockPropLayout.addRow("Y Position:", self.yRockSpin)
        rockPropLayout.addRow("Width:", self.wRockSpin)
        rockPropLayout.addRow("Height:", self.hRockSpin)

        self.rockGroup.setLayout(rockPropLayout)
        self.rockGroup.setVisible(False)

        # Setup Node Property Group
        self.nodeGroup = QGroupBox("Node Properties")

        nodePropLayout = QFormLayout(self)

        self.xNodeSpin = QSpinBox()
        self.xNodeSpin.setMinimum(0)
        self.xNodeSpin.setMaximum(255)
        self.xNodeSpin.setMinimumWidth(90)

        self.yNodeSpin = QSpinBox()
        self.yNodeSpin.setMinimum(0)
        self.yNodeSpin.setMaximum(255)
        self.yNodeSpin.setMinimumWidth(90)

        self.hSupportNodeCheck = QCheckBox()
        self.vSupportNodeCheck = QCheckBox()

        nodePropLayout.addRow("X Position:", self.xNodeSpin)
        nodePropLayout.addRow("Y Position:", self.yNodeSpin)
        nodePropLayout.addRow("H Support:", self.hSupportNodeCheck)
        nodePropLayout.addRow("V Support:", self.vSupportNodeCheck)

        self.nodeGroup.setLayout(nodePropLayout)
        self.nodeGroup.setVisible(False)

        # Setup Property Widget

        layout = QVBoxLayout(self)
        layout.addWidget(self.nodeGroup)
        layout.addWidget(self.rockGroup)
        layout.addStretch()

        self.setVisible(False)

        self.setLayout(layout)

    @pyqtSlot(int)
    def setXValue(self, val):
        self.activeObject.x = val
        self.onValueChanged.emit()

    @pyqtSlot(int)
    def setYValue(self, val):
        self.activeObject.y = val
        self.onValueChanged.emit()

    @pyqtSlot(int)
    def setWValue(self, val):
        self.activeObject.w = val
        self.onValueChanged.emit()

    @pyqtSlot(int)
    def setHValue(self, val):
        self.activeObject.h = val
        self.onValueChanged.emit()

    @pyqtSlot(int)
    def setHSupportValue(self, val):
        self.activeObject.h_support = val == Qt.Checked
        self.onValueChanged.emit()

    @pyqtSlot(int)
    def setVSupportValue(self, val):
        self.activeObject.v_support = val == Qt.Checked
        self.onValueChanged.emit()

    def setActiveObject(self, obj):
        self.activeObject = obj

        self.rockGroup.setVisible(False)
        self.nodeGroup.setVisible(False)

        if isinstance(obj, Node):
            self.setVisible(True)
            self.nodeGroup.setVisible(True)
            self.xNodeSpin.setValue(obj.x)
            self.yNodeSpin.setValue(obj.y)
            self.hSupportNodeCheck.setCheckState(
                Qt.Checked if obj.h_support else Qt.Unchecked)
            self.vSupportNodeCheck.setCheckState(
                Qt.Checked if obj.v_support else Qt.Unchecked)

            self.xNodeSpin.valueChanged.connect(self.setXValue)
            self.yNodeSpin.valueChanged.connect(self.setYValue)
            self.hSupportNodeCheck.stateChanged.connect(self.setHSupportValue)
            self.vSupportNodeCheck.stateChanged.connect(self.setVSupportValue)

        elif isinstance(obj, Rock):
            self.setVisible(True)
            self.rockGroup.setVisible(True)
            self.xRockSpin.setValue(obj.x)
            self.yRockSpin.setValue(obj.y)
            self.wRockSpin.setValue(obj.w)
            self.hRockSpin.setValue(obj.h)

            self.xRockSpin.valueChanged.connect(self.setXValue)
            self.yRockSpin.valueChanged.connect(self.setYValue)
            self.wRockSpin.valueChanged.connect(self.setWValue)
            self.hRockSpin.valueChanged.connect(self.setHValue)

        else:
            self.setVisible(False)
