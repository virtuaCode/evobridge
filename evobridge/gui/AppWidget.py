from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QAbstractButton, QRadioButton)
from PyQt5.Qt import pyqtSlot
from .DrawWidget import DrawWidget
from .PropertyWidget import PropertyWidget
from .TabWidget import TabWidget
from ..lib.optimize import OptimizerFactory


class AppWidget(QWidget):

    def __init__(self, factory: OptimizerFactory):
        QWidget.__init__(self)

        self.drawing = DrawWidget()
        self.prop = PropertyWidget()
        self.tabs = TabWidget(factory)

        vlayout = QVBoxLayout(self)
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.drawing)
        hlayout.addWidget(self.prop)
        hlayout.setStretch(0, 1)

        vlayout.addLayout(hlayout)
        vlayout.addWidget(self.tabs)
        vlayout.setStretch(0, 1)
        vlayout.setContentsMargins(11, 11, 11, 0)

        self.prop.onValueChanged.connect(self.drawing.update)
