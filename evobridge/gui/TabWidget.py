from PyQt5.QtWidgets import (QTabWidget, QLabel, QGroupBox, QLineEdit, QWidget, QFormLayout, QHBoxLayout, QDoubleSpinBox,
                             QRadioButton, QButtonGroup, QVBoxLayout, QComboBox, QSlider, QAbstractButton, QSpinBox)
from PyQt5.Qt import Qt, QDoubleValidator, pyqtSlot
from ..lib import optimize as o


class TabWidget(QTabWidget):

    def __init__(self, factory: o.OptimizerFactory):
        QTabWidget.__init__(self)

        self.factory = factory

        self.setMinimumHeight(100)
        self.optimizerTab = HillclimberTab(factory)
        self.geneticTab = GeneticTab(factory)
        self.addTab(self.optimizerTab, "Hill Climber")
        self.addTab(self.geneticTab, "SPEA2")
        self.currentChanged.connect(self.tabChanged)

    @pyqtSlot(int)
    def tabChanged(self, index: int):
        self.factory.config["optimizer"] = [o.HILL_CLIMBER, o.SPEA2][index]


class GeneticTab(QWidget):

    def __init__(self, factory: o.OptimizerFactory):
        QWidget.__init__(self)

        self.factory = factory

        groupPop = QGroupBox()
        groupPop.setMinimumWidth(200)

        formPop = QFormLayout()

        self.popField = QSpinBox()
        self.popField.setMinimum(1)
        self.popField.setMaximum(10000)
        self.popField.setSingleStep(10)
        self.popField.setValue(self.factory.getSPEA2Population())

        self.archiveField = QSpinBox()
        self.archiveField.setMinimum(1)
        self.archiveField.setMaximum(10000)
        self.archiveField.setSingleStep(1)
        self.archiveField.setValue(self.factory.getSPEA2Archive())

        formPop.addRow(QLabel("Population:"), self.popField)
        formPop.addRow(QLabel("Archive:"), self.archiveField)

        groupPop.setLayout(formPop)

        groupMutate = QGroupBox("Mutation")
        groupMutate.setMinimumWidth(200)
        formMutate = QFormLayout()

        self.stepField = QDoubleSpinBox()
        self.stepField.setMinimum(0)
        self.stepField.setMaximum(255)
        self.stepField.setSingleStep(0.25)
        self.stepField.setValue(
            self.factory.getSPEA2MutateOption("default.creep-step"))

        self.probField = QDoubleSpinBox()
        self.probField.setMinimum(0)
        self.probField.setMaximum(1)
        self.probField.setSingleStep(0.01)
        self.probField.setValue(
            self.factory.getSPEA2MutateOption("default.creep-prob"))

        formMutate.addRow(QLabel("Probability:"), self.probField)
        formMutate.addRow(QLabel("Step Size:"), self.stepField)

        groupMutate.setLayout(formMutate)

        groupCrossover = QGroupBox("Crossover (K-Point)")
        groupCrossover.setMinimumWidth(200)
        formCrossover = QFormLayout()

        self.probCField = QDoubleSpinBox()
        self.probCField.setMinimum(0)
        self.probCField.setMaximum(1)
        self.probCField.setSingleStep(0.01)
        self.probCField.setValue(
            self.factory.getSPEA2CrossoverOption("k-point.probability"))

        self.nodesField = QSpinBox()
        self.nodesField.setMinimum(0)
        self.nodesField.setMaximum(100)
        self.nodesField.setSingleStep(1)
        self.nodesField.setValue(
            self.factory.getSPEA2CrossoverOption("k-point.nodes"))

        self.membersField = QSpinBox()
        self.membersField.setMinimum(0)
        self.membersField.setMaximum(100)
        self.membersField.setSingleStep(1)
        self.membersField.setValue(
            self.factory.getSPEA2CrossoverOption("k-point.members"))

        self.materialsField = QSpinBox()
        self.materialsField.setMinimum(0)
        self.materialsField.setMaximum(100)
        self.materialsField.setSingleStep(1)
        self.materialsField.setValue(
            self.factory.getSPEA2CrossoverOption("k-point.materials"))

        formCrossover.addRow(QLabel("Probability:"), self.probCField)
        formCrossover.addRow(QLabel("Nodes:"), self.nodesField)
        formCrossover.addRow(QLabel("Members:"), self.membersField)
        formCrossover.addRow(QLabel("Materials:"), self.materialsField)

        groupCrossover.setLayout(formCrossover)

        groupSelection = QGroupBox("Selection (Tournament)")
        groupSelection.setMinimumWidth(200)
        formSelection = QFormLayout()

        self.opponentsField = QSpinBox()
        self.opponentsField.setMinimum(0)
        self.opponentsField.setMaximum(100)
        self.opponentsField.setSingleStep(1)
        self.opponentsField.setValue(
            self.factory.getSPEA2SelectOption("tournament.opponents"))

        formSelection.addRow(QLabel("Opponents:"), self.opponentsField)
        groupSelection.setLayout(formSelection)

        layout = QHBoxLayout()
        layout.addWidget(groupPop)
        layout.addWidget(groupMutate)
        layout.addWidget(groupCrossover)
        layout.addWidget(groupSelection)
        layout.addStretch()

        self.popField.valueChanged.connect(self.setPopulation)
        self.probField.valueChanged.connect(self.setProbMutate)
        self.probCField.valueChanged.connect(self.setProbCrossover)
        self.nodesField.valueChanged.connect(self.setNodesCrossover)
        self.membersField.valueChanged.connect(self.setMembersCrossover)
        self.materialsField.valueChanged.connect(self.setMaterialsCrossover)
        self.archiveField.valueChanged.connect(self.setArchive)
        self.opponentsField.valueChanged.connect(self.setOpponents)
        self.stepField.valueChanged.connect(self.setStepMutate)

        self.setLayout(layout)

    @pyqtSlot(int)
    def setOpponents(self, value):
        if self.opponentsField.hasAcceptableInput():
            self.factory.setSPEA2SelectOption(
                "tournament.opponents", int(value))

    @pyqtSlot(float)
    def setProbCrossover(self, value):
        if self.probCField.hasAcceptableInput():
            self.factory.setSPEA2CrossoverOption(
                "k-point.probability", float(value))

    @pyqtSlot(int)
    def setNodesCrossover(self, value):
        if self.nodesField.hasAcceptableInput():
            self.factory.setSPEA2CrossoverOption(
                "k-point.nodes", int(value))

    @pyqtSlot(int)
    def setMembersCrossover(self, value):
        if self.membersField.hasAcceptableInput():
            self.factory.setSPEA2CrossoverOption(
                "k-point.members", int(value))

    @pyqtSlot(int)
    def setMaterialsCrossover(self, value):
        if self.materialsField.hasAcceptableInput():
            self.factory.setSPEA2CrossoverOption(
                "k-point.materials", int(value))

    @pyqtSlot(float)
    def setProbMutate(self, value):
        if self.probField.hasAcceptableInput():
            self.factory.setSPEA2MutateOption(
                "default.creep-prob", float(value))

    @pyqtSlot(float)
    def setStepMutate(self, value):
        if self.stepField.hasAcceptableInput():
            self.factory.setSPEA2MutateOption(
                "default.creep-step", float(value))

    @pyqtSlot(int)
    def setPopulation(self, value):
        if self.popField.hasAcceptableInput():
            self.factory.setSPEA2Population(int(value))

    @pyqtSlot(int)
    def setArchive(self, value):
        if self.archiveField.hasAcceptableInput():
            self.factory.setSPEA2Archive(int(value))


class HillclimberTab(QWidget):

    def __init__(self, factory: o.OptimizerFactory):
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

        thresholdTab = ThresholdTab(factory)
        recordTab = RecordTab(factory)
        annealingTab = AnnealingTab(factory)

        acceptTabs = QTabWidget()
        acceptTabs.addTab(thresholdTab, "Threshold Accepting")
        acceptTabs.addTab(recordTab, "Record-to-Record")
        acceptTabs.addTab(annealingTab, "Simulated Annealing")
        acceptTabs.setMaximumWidth(500)
        acceptTabs.currentChanged.connect(self.setAccepting)

        layout = QHBoxLayout()
        layout.addWidget(objBox, 0)
        layout.addWidget(mutateBox, 0)
        layout.addWidget(acceptTabs, 1)
        layout.addStretch(0)

        # self.layout.addStretch()

        self.setLayout(layout)

        self.group.buttonToggled.connect(
            self.selectEvaluate)
        self.groupMutate.buttonToggled.connect(
            self.selectMutate)

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

    @pyqtSlot(QAbstractButton, bool)
    def selectEvaluate(self, btn, checked):
        if btn == self.sumRadio:
            self.factory.setHillClimberEvaluate("abs-sum")
        elif btn == self.meanRadio:
            self.factory.setHillClimberEvaluate("mean")
        else:
            self.factory.setHillClimberEvaluate("root-mean-square")

    @pyqtSlot(QAbstractButton, bool)
    def selectMutate(self, btn, bool):
        if btn == self.obfRadio:
            self.factory.setHillClimberMutate("one-bit")
        elif btn == self.pbfRadio:
            self.factory.setHillClimberMutate("prob-bit")

    @pyqtSlot(int)
    def setAccepting(self, index):
        if index == 0:
            self.factory.setHillClimberAccept("threshold")
        elif index == 1:
            self.factory.setHillClimberAccept("record-to-record")
        elif index == 2:
            self.factory.setHillClimberAccept("simulated-annealing")


class ThresholdTab(QWidget):
    def __init__(self, factory):
        QWidget.__init__(self)

        self.factory = factory

        self.damping = QLineEdit(
            str(self.factory.getHillClimberAcceptOption("threshold.damping")))
        self.damping.setAlignment(Qt.AlignRight)
        self.damping.setValidator(QDoubleValidator(0.0, 1.0, 4))
        self.temp = QLineEdit(
            str(self.factory.getHillClimberAcceptOption("threshold.temp")))
        self.temp.setAlignment(Qt.AlignRight)
        self.temp.setValidator(QDoubleValidator(0.0, 100000000.0, 4))

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Damping:"))
        layout.addWidget(self.damping)
        layout.addWidget(QLabel("Temperature:"))
        layout.addWidget(self.temp)

        self.setLayout(layout)

        self.damping.textChanged.connect(
            self.setDamping)
        self.temp.textChanged.connect(
            self.setTemp)

    @pyqtSlot(str)
    def setDamping(self, value):
        if self.damping.hasAcceptableInput():
            self.factory.setHillClimberAcceptOption(
                "threshold.damping", float(value))

    @pyqtSlot(str)
    def setTemp(self, value):
        if self.temp.hasAcceptableInput():
            self.factory.setHillClimberAcceptOption(
                "threshold.temp", float(value))


class RecordTab(QWidget):
    def __init__(self, factory):
        QWidget.__init__(self)

        self.factory = factory

        self.damping = QLineEdit(
            str(self.factory.getHillClimberAcceptOption("record-to-record.damping")))
        self.damping.setAlignment(Qt.AlignRight)
        self.damping.setValidator(QDoubleValidator(0.0, 1.0, 4))
        self.temp = QLineEdit(
            str(self.factory.getHillClimberAcceptOption("record-to-record.temp")))
        self.temp.setAlignment(Qt.AlignRight)
        self.temp.setValidator(QDoubleValidator(0.0, 100000000.0, 4))

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Damping:"))
        layout.addWidget(self.damping)
        layout.addWidget(QLabel("Temperature:"))
        layout.addWidget(self.temp)

        self.setLayout(layout)

        self.damping.textChanged.connect(
            self.setDamping)
        self.temp.textChanged.connect(
            self.setTemp)

    @pyqtSlot(str)
    def setDamping(self, value):
        if self.damping.hasAcceptableInput():
            self.factory.setHillClimberAcceptOption(
                "record-to-record.damping", float(value))

    @pyqtSlot(str)
    def setTemp(self, value):
        if self.damping.hasAcceptableInput():
            self.factory.setHillClimberAcceptOption(
                "record-to-record.temp", float(value))


class AnnealingTab(QWidget):
    def __init__(self, factory):
        QWidget.__init__(self)

        self.factory = factory

        self.damping = QLineEdit(
            str(self.factory.getHillClimberAcceptOption("simulated-annealing.damping")))
        self.damping.setAlignment(Qt.AlignRight)
        self.damping.setValidator(QDoubleValidator(0.0, 1.0, 4))
        self.temp = QLineEdit(
            str(self.factory.getHillClimberAcceptOption("simulated-annealing.temp")))
        self.temp.setAlignment(Qt.AlignRight)
        self.temp.setValidator(QDoubleValidator(0.0, 100000000.0, 4))

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Damping:"))
        layout.addWidget(self.damping)
        layout.addWidget(QLabel("Temperature:"))
        layout.addWidget(self.temp)

        self.setLayout(layout)

        self.damping.textChanged.connect(
            self.setDamping)
        self.temp.textChanged.connect(
            self.setTemp)

    @pyqtSlot(str)
    def setDamping(self, value):
        if self.damping.hasAcceptableInput():
            self.factory.setHillClimberAcceptOption(
                "simulated-annealing.damping", float(value))

    @pyqtSlot(str)
    def setTemp(self, value):
        if self.damping.hasAcceptableInput():
            self.factory.setHillClimberAcceptOption(
                "simulated-annealing.temp", float(value))
