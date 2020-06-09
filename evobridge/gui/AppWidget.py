from PyQt5.QtWidgets import (QWidget, QHBoxLayout)
from .DrawWidget import DrawWidget
from .PropertyWidget import PropertyWidget


class AppWidget(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        self.drawing = DrawWidget()
        self.prop = PropertyWidget()

        layout = QHBoxLayout(self)
        layout.addWidget(self.drawing)
        layout.addWidget(self.prop)
        layout.setStretch(0, 1)

        self.setLayout(layout)

        layout.setContentsMargins(11, 11, 11, 0)

        self.prop.onValueChanged.connect(self.drawing.update)
