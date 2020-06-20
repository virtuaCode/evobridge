from PyQt5.QtWidgets import QTabWidget, QLabel, QGroupBox, QWidget, QFormLayout, QHBoxLayout, QRadioButton, QButtonGroup, QVBoxLayout


class TabWidget(QTabWidget):

    def __init__(self):
        QTabWidget.__init__(self)
        self.setMinimumHeight(100)
        self.optimizerTab = OptimizerTab()
        self.addTab(self.optimizerTab, "Optimizer Settings")


class OptimizerTab(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        self.sumRadio = QRadioButton("Sum of absolute forces")
        self.meanRadio = QRadioButton("Mean of absolute forces")

        group = QButtonGroup(self)
        group.addButton(self.sumRadio)
        group.addButton(self.meanRadio)
        group.setExclusive(True)
        self.group = group

        btns = QVBoxLayout()
        btns.addWidget(self.meanRadio)
        btns.addWidget(self.sumRadio)

        box = QGroupBox("Objective Function")
        box.setLayout(btns)

        self.layout = QHBoxLayout()
        self.layout.addWidget(box)
        self.layout.addStretch()

        self.setLayout(self.layout)
