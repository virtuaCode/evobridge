from PyQt5.QtWidgets import QTabWidget, QLabel, QGroupBox, QLineEdit, QWidget, QFormLayout, QHBoxLayout, QRadioButton, QButtonGroup, QVBoxLayout, QComboBox, QSlider, QAbstractButton
from PyQt5.Qt import Qt, QDoubleValidator, pyqtSlot
from ..lib.optimize import OptimizerFactory


class TabWidget(QTabWidget):

    def __init__(self, factory: OptimizerFactory):
        QTabWidget.__init__(self)
        self.setMinimumHeight(100)
        self.optimizerTab = OptimizerTab(factory)
        self.addTab(self.optimizerTab, "Hill Climber")


class OptimizerTab(QWidget):

    def __init__(self, factory: OptimizerFactory):
        QWidget.__init__(self)

        self.factory = factory

        self.sumRadio = QRadioButton("Sum of absolute forces")
        self.meanRadio = QRadioButton("Mean of absolute forces")
        self.rootMeanRadio = QRadioButton(
            "Root mean square of absolute forces")

        group = QButtonGroup(self)
        group.addButton(self.meanRadio)
        group.addButton(self.sumRadio)
        group.addButton(self.rootMeanRadio)
        group.setExclusive(True)
        self.group = group

        btns = QVBoxLayout()
        btns.addWidget(self.sumRadio)
        btns.addWidget(self.meanRadio)
        btns.addWidget(self.rootMeanRadio)

        objBox = QGroupBox("Objective Function")
        objBox.setLayout(btns)

        self.obfRadio = QRadioButton("One Bit Flip")
        self.pbfRadio = QRadioButton("Probability Bit Flip")

        groupMutate = QButtonGroup(self)
        groupMutate.addButton(self.obfRadio)
        groupMutate.addButton(self.pbfRadio)
        groupMutate.setExclusive(True)
        self.groupMutate = groupMutate

        mutateLayout = QVBoxLayout()
        mutateLayout.addWidget(self.obfRadio)
        mutateLayout.addWidget(self.pbfRadio)
        mutateBox = QGroupBox("Mutation")
        mutateBox.setLayout(mutateLayout)

        self.thresholdRate = QLineEdit(
            str(self.factory.getHillClimberAcceptOption("threshold.rate")))
        self.thresholdRate.setAlignment(Qt.AlignRight)
        self.thresholdRate.setValidator(QDoubleValidator(0.0, 1.0, 4))
        self.thresholdTemp = QLineEdit(
            str(self.factory.getHillClimberAcceptOption("threshold.temp")))
        self.thresholdTemp.setAlignment(Qt.AlignRight)
        self.thresholdTemp.setValidator(QDoubleValidator(0.0, 100000000.0, 4))

        thresLayout = QVBoxLayout()
        thresLayout.addWidget(QLabel("Rate:"))
        thresLayout.addWidget(self.thresholdRate)
        thresLayout.addWidget(QLabel("Temperature:"))
        thresLayout.addWidget(self.thresholdTemp)

        thresholdBox = QGroupBox("Threshold Accepting")
        thresholdBox.setLayout(thresLayout)

        self.layout = QHBoxLayout()
        self.layout.addWidget(objBox)
        self.layout.addWidget(mutateBox)
        self.layout.addWidget(thresholdBox)
        self.layout.addStretch()

        self.setLayout(self.layout)

        self.group.buttonToggled.connect(
            self.selectEvaluate)
        self.groupMutate.buttonToggled.connect(
            self.selectMutate)
        self.thresholdRate.textChanged.connect(
            self.setThresholdRate)
        self.thresholdTemp.textChanged.connect(
            self.setThresholdTemp)

        # apply config

        mutate_switcher = {
            "one-bit": self.obfRadio,
            "prob-bit": self.pbfRadio
        }

        mutate_switcher.get(
            self.factory.getHillClimberMutate(), self.obfRadio).toggle()

        evaluate_switcher = {
            "abs-sum": self.sumRadio,
            "mean": self.meanRadio,
            "root-mean-square": self.rootMeanRadio
        }

        evaluate_switcher.get(
            self.factory.getHillClimberEvaluate(), self.rootMeanRadio).toggle()

        self.thresholdTemp.setText(
            str(self.factory.getHillClimberAcceptOption("threshold.temp")))
        self.thresholdRate.setText(
            str(self.factory.getHillClimberAcceptOption("threshold.rate")))

    @pyqtSlot(QAbstractButton, bool)
    def selectEvaluate(self, btn, checked):
        if btn == self.sumRadio:
            self.factory.setHillClimberEvaluate("abs-sum")
        elif btn == self.meanRadio:
            self.factory.setHillClimberEvaluate("mean")
        else:
            self.factory.setHillClimberEvaluate("root-mean-square")

    @pyqtSlot(str)
    def setThresholdRate(self, value):
        if self.thresholdRate.hasAcceptableInput():
            self.factory.setHillClimberAcceptOption(
                "threshold.rate", float(value))

    @pyqtSlot(str)
    def setThresholdTemp(self, value):
        if self.thresholdTemp.hasAcceptableInput():
            self.factory.setHillClimberAcceptOption(
                "threshold.temp", float(value))

    @pyqtSlot(QAbstractButton, bool)
    def selectMutate(self, btn, bool):
        if btn == self.obfRadio:
            self.factory.setHillClimberMutate("one-bit")
        elif btn == self.pbfRadio:
            self.factory.setHillClimberMutate("prob-bit")
