from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QAbstractButton)
from PyQt5.Qt import pyqtSlot
from .DrawWidget import DrawWidget
from .PropertyWidget import PropertyWidget
from .TabWidget import TabWidget


class AppWidget(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        self.drawing = DrawWidget()
        self.prop = PropertyWidget()
        self.tabs = TabWidget()

        vlayout = QVBoxLayout(self)
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.drawing)
        hlayout.addWidget(self.prop)
        hlayout.setStretch(0, 1)

        vlayout.addLayout(hlayout)
        vlayout.addWidget(self.tabs)
        vlayout.setStretch(0, 1)
        self.setLayout(vlayout)

        vlayout.setContentsMargins(11, 11, 11, 0)

        self.prop.onValueChanged.connect(self.drawing.update)
        self.tabs.optimizerTab.group.buttonToggled.connect(self.toggledObjFunc)
        self.tabs.optimizerTab.sumRadio.toggle()

    @ pyqtSlot(QAbstractButton, bool)
    def toggledObjFunc(self, btn, checked):
        if btn == self.tabs.optimizerTab.sumRadio:
            self.drawing.obj_func_type = "sum"
        else:
            self.drawing.obj_func_type = "mean"
