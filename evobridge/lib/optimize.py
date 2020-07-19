from .functions import create_onebit_mutate, create_propbit_mutate, create_threshold_accept
from ..gui.Objects import ObjectiveFunction
from ..gui.State import State
from .lsearch import LocalSearchOptimizer


class OptimizerFactory():

    CONFIG_VERSION = 1

    def __init__(self):
        self.config = {
            "version": self.CONFIG_VERSION,
            "optimizer": "hill-climber",
            "hill-climber": {
                "mutate": "prob-bit",
                "mutate-variants": {
                    "one-bit": {},
                    "prob-bit": {}
                },
                "accept": "threshold",
                "accept-variants": {
                    "threshold": {
                        "temp": 100000.0,
                        "rate": 0.95
                    },
                    "great-deluge": {},
                    "simulated-annealing": {},
                },
                "evaluate": "root-mean-square",
                "evaluate-variants": {
                    "root-mean-square": {},
                    "abs-sum": {},
                    "mean": {}
                }
            }
        }

    def getHillClimberMutate(self):
        return self.config["hill-climber"]["mutate"]

    def setHillClimberMutate(self, value: str):
        assert value in self.config["hill-climber"]["mutate-variants"].keys()
        self.config["hill-climber"]["mutate"] = value

    def getHillClimberAccept(self):
        return self.config["hill-climber"]["accept"]

    def setHillClimberAccept(self, value: str):
        assert value in self.config["hill-climber"]["accept-variants"].keys()
        self.config["hill-climber"]["accept"] = value

    def getHillClimberEvaluate(self):
        return self.config["hill-climber"]["evaluate"]

    def setHillClimberEvaluate(self, value: str):
        assert value in self.config["hill-climber"]["evaluate-variants"].keys()
        self.config["hill-climber"]["evaluate"] = value

    def getHillClimberAcceptOption(self, key: str):
        path = key.split(".")

        assert len(path) > 1

        d = self.config["hill-climber"]["accept-variants"]

        for i in range(len(path)-1):
            d = d[path[i]]
            assert path[i+1] in d.keys()

        assert type(d[path[-1]]) != dict

        return d[path[-1]]

    def setHillClimberAcceptOption(self, key: str, value):
        path = key.split(".")

        assert len(path) > 1

        d = self.config["hill-climber"]["accept-variants"]

        for i in range(len(path)-1):
            d = d[path[i]]
            assert path[i+1] in d.keys()

        assert type(d[path[-1]]) != dict
        assert type(value) == type(d[path[-1]])

        d[path[-1]] = value

    def createOptimizer(self, state: State):

        if self.config["optimizer"] == "hill-climber":
            optimizer = self.config["hill-climber"]
            if optimizer["mutate"] == "one-bit":
                mutate = create_onebit_mutate()
            else:
                mutate = create_propbit_mutate()

            if optimizer["accept"] == "threshold":
                options = optimizer["accept-variants"]["threshold"]
                accept = create_threshold_accept(
                    options["temp"], options["rate"])
            elif optimizer["accept"] == "simulated-annealing":
                # TODO
                return None
            elif optimizer["accept"] == "great-deluge":
                # TODO
                return None

            if optimizer["evaluate"] == "abs-sum":
                objfn = ObjectiveFunction.Sum
            elif optimizer["evaluate"] == "mean":
                objfn = ObjectiveFunction.Mean
            elif optimizer["evaluate"] == "root-mean-square":
                objfn = ObjectiveFunction.RootMeanSquare

            return LocalSearchOptimizer(state, mutate=mutate, accept=accept, objFunc=objfn)

        else:
            # TODO Implement Genetic Algorithm
            return None
