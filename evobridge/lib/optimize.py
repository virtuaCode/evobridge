from .functions import (create_onebit_mutate, create_probbit_mutate, create_tournament_select, create_k_point_crossover,
                        create_threshold_accept, create_record_to_record_accept, create_simulated_annealing_accept, create_creep_mutate)
import random
import numpy as np
from itertools import chain
from ..gui.State import State
from .statics import solve
from ..gui.Objects import Material, ObjectiveFunction, Mutation, MATERIAL_COSTS, MAX_MEMBER_LENGTHS
from .genotype import toGenotype, fromGenotype
from .graycode import grayToStdbin, stdbinToGray
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QWidget
from enum import Enum
from .bridge import Bridge
from abc import abstractmethod, ABCMeta
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import traceback
from ..gui.PlotWidget import PlotWidget
from ..gui.TestWidget import TestWidget
import networkx as nx
import pprint

HILL_CLIMBER = "hill-climber"
SPEA2 = "spea2"
MUTATE = "mutate"
ACCEPT = "accept"
SELECT = "select"
CROSSOVER = "crossover"
EVALUTATE = "evaluate"
_VARIANTS = "-variants"
MUTATE_VARIANTS = MUTATE + _VARIANTS
ACCEPT_VARIANTS = ACCEPT + _VARIANTS
SELECT_VARIANTS = SELECT + _VARIANTS
CROSSOVER_VARIANTS = CROSSOVER + _VARIANTS
EVALUTATE_VARIANTS = EVALUTATE + _VARIANTS


class SolveError(Exception):
    pass


class OptimizerFactory():

    CONFIG_VERSION = 3

    def __init__(self):
        self.config = {
            "version": self.CONFIG_VERSION,
            "optimizer": HILL_CLIMBER,
            HILL_CLIMBER: {
                MUTATE: "prob-bit",
                "mutate-variants": {
                    "one-bit": {},
                    "prob-bit": {}
                },
                ACCEPT: "threshold",
                ACCEPT_VARIANTS: {
                    "threshold": {
                        "temp": 100000.0,
                        "damping": 0.95
                    },
                    "record-to-record": {
                        "temp": 100000.0,
                        "damping": 0.95
                    },
                    "simulated-annealing": {
                        "temp": 100000.0,
                        "damping": 0.95
                    },
                },
                EVALUTATE: "root-mean-square",
                EVALUTATE_VARIANTS: {
                    "root-mean-square": {},
                    "abs-sum": {},
                    "mean": {}
                }
            },
            SPEA2: {
                "population": 100,
                "archive": 10,
                MUTATE: "default",
                MUTATE_VARIANTS: {
                    "default": {
                        "creep-step": 10.0,
                        "creep-prob": 0.05
                    }
                },
                CROSSOVER: "k-point",
                CROSSOVER_VARIANTS: {
                    "k-point": {
                        "probability": 0.5,
                        "nodes": 3,
                        "materials": 5,
                        "members": 20
                    }
                },
                SELECT: "tournament",
                SELECT_VARIANTS: {
                    "tournament": {
                        "opponents": 5
                    }
                }
            }
        }

    def getSPEA2Mutate(self):
        return self.config[SPEA2][MUTATE]

    def getSPEA2Population(self):
        return self.config[SPEA2]["population"]

    def setSPEA2Population(self, pop):
        self.config[SPEA2]["population"] = pop

    def getSPEA2Archive(self):
        return self.config[SPEA2]["archive"]

    def setSPEA2Archive(self, archive):
        self.config[SPEA2]["archive"] = archive

    def getSPEA2MutateOption(self, key):
        return self._getOption(SPEA2, MUTATE_VARIANTS, key)

    def setSPEA2MutateOption(self, key, value):
        self._setOption(SPEA2, MUTATE_VARIANTS, key, value)

    def getSPEA2CrossoverOption(self, key):
        return self._getOption(SPEA2, CROSSOVER_VARIANTS, key)

    def setSPEA2CrossoverOption(self, key, value):
        self._setOption(SPEA2, CROSSOVER_VARIANTS, key, value)

    def getSPEA2SelectOption(self, key):
        return self._getOption(SPEA2, SELECT_VARIANTS, key)

    def setSPEA2SelectOption(self, key, value):
        self._setOption(SPEA2, SELECT_VARIANTS, key, value)

    def getSPEA2Crossover(self):
        return self.config[SPEA2][CROSSOVER]

    def getSPEA2Select(self):
        return self.config[SPEA2][SELECT]

    def getHillClimberMutate(self):
        return self.config[HILL_CLIMBER][MUTATE]

    def setHillClimberMutate(self, value: str):
        assert value in self.config[HILL_CLIMBER][MUTATE_VARIANTS].keys()
        self.config[HILL_CLIMBER][MUTATE] = value

    def getHillClimberAccept(self):
        return self.config[HILL_CLIMBER][ACCEPT]

    def setHillClimberAccept(self, value: str):
        assert value in self.config[HILL_CLIMBER][ACCEPT_VARIANTS].keys()
        self.config[HILL_CLIMBER][ACCEPT] = value

    def getHillClimberEvaluate(self):
        return self.config[HILL_CLIMBER][EVALUTATE]

    def setHillClimberEvaluate(self, value: str):
        assert value in self.config[HILL_CLIMBER][EVALUTATE_VARIANTS].keys()
        self.config[HILL_CLIMBER][EVALUTATE] = value

    def getHillClimberAcceptOption(self, key: str):
        return self._getOption(HILL_CLIMBER, ACCEPT_VARIANTS, key)

    def setHillClimberAcceptOption(self, key: str, value):
        self._setOption(HILL_CLIMBER, ACCEPT_VARIANTS, key, value)

    def _getOption(self, algorithm, variants, key):
        path = key.split(".")

        assert len(path) > 1

        d = self.config[algorithm][variants]

        for i in range(len(path)-1):
            d = d[path[i]]
            assert path[i+1] in d.keys()

        assert type(d[path[-1]]) != dict

        return d[path[-1]]

    def _setOption(self, algorithm, variants, key, value):
        path = key.split(".")

        assert len(path) > 1

        d = self.config[algorithm][variants]

        for i in range(len(path)-1):
            d = d[path[i]]
            assert path[i+1] in d.keys()

        assert type(d[path[-1]]) != dict
        assert type(value) == type(d[path[-1]])

        d[path[-1]] = value

    def createOptimizer(self, state: State):
        pprint.pprint(self.config)

        if self.config["optimizer"] == HILL_CLIMBER:
            optimizer = self.config[HILL_CLIMBER]
            if optimizer[MUTATE] == "one-bit":
                mutate = create_onebit_mutate()
            else:
                mutate = create_probbit_mutate()

            if optimizer[ACCEPT] == "threshold":
                options = optimizer[ACCEPT_VARIANTS]["threshold"]
                accept = create_threshold_accept(
                    options["temp"], options["damping"])

            elif optimizer[ACCEPT] == "simulated-annealing":
                options = optimizer[ACCEPT_VARIANTS]["simulated-annealing"]
                accept = create_simulated_annealing_accept(
                    options["temp"], options["damping"])

            elif optimizer[ACCEPT] == "record-to-record":
                options = optimizer[ACCEPT_VARIANTS]["record-to-record"]
                accept = create_record_to_record_accept(
                    options["temp"], options["damping"])

            if optimizer[EVALUTATE] == "abs-sum":
                objfn = ObjectiveFunction.Sum
            elif optimizer[EVALUTATE] == "mean":
                objfn = ObjectiveFunction.Mean
            elif optimizer[EVALUTATE] == "root-mean-square":
                objfn = ObjectiveFunction.RootMeanSquare

            return LocalSearchOptimizer(state, mutateMembers=mutate, accept=accept, objFunc=objfn)

        elif self.config["optimizer"] == SPEA2:
            optimizer = self.config[SPEA2]
            population = optimizer["population"]
            archive = optimizer["archive"]

            if optimizer[MUTATE] == "default":
                options = optimizer[MUTATE_VARIANTS]["default"]
                step = options["creep-step"]
                prob = options["creep-prob"]
                mutateNode = create_creep_mutate(prob, step)
                mutateMem = create_probbit_mutate()
                mutateMat = create_probbit_mutate()

            if optimizer[CROSSOVER] == "k-point":
                options = optimizer[CROSSOVER_VARIANTS]["k-point"]

                crossover_prob = options["probability"]
                crossover = create_k_point_crossover(
                    options["nodes"], options["members"], options["materials"])

            if optimizer[SELECT] == "tournament":
                options = optimizer[SELECT_VARIANTS]["tournament"]

                select = create_tournament_select(
                    population, options["opponents"])

            return GeneticOptimizer(state, mutateNodes=mutateNode, mutateMembers=mutateMem, mutateMaterials=mutateMat, crossover=crossover, crossover_prob=crossover_prob, select=select, pop=population, archive=archive)


class Optimizer(object, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, state: State, objFunc, node_weight=10, street_weight=8):
        self.bridge = Bridge(state, node_weight=node_weight,
                             street_weight=street_weight)
        self.objFunc = objFunc
        self.genotypes = [self.bridge.genotype]
        self.fitness = [self._evaluate(self.bridge.genotype)]
        self.forces = [self._solve(self.bridge.genotype)]
        self.fitness_graph = None
        self.threshold_graph = None

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _evaluate(self):
        pass

    def _solve(self, genotype=None):
        b = self.bridge
        if genotype:
            b.setGenotype(genotype)
        supports = b.supports
        nodes = b.nodes
        members = b.members
        materials = b.materials
        loads = b.loads

        try:
            x, det, MF, RF = solve(nodes, members, supports, loads)
        except BaseException as e:
            return (None, None, None, None)

        return (x, det, MF, RF)

    def _repair(self, genotype):
        self.bridge.setGenotype(genotype)
        nodes = self.bridge.nodes.shape[0]
        members = self.bridge.members.shape[0]
        supports = np.sum(self.bridge.supports[:, 1:3])

        x = (supports + members) - 2 * nodes

        nod, mem, mat = map(np.copy, genotype)

        while x < 0:
            ind = np.argwhere(mem == False).reshape(-1)
            if len(ind) == 0:
                break
            pos = np.random.choice(ind)
            mem[pos] = True
            x += 1

        while x > 0:
            ind = np.argwhere(mem == True).reshape(-1)
            if len(ind) == 0:
                break
            pos = np.random.choice(ind)
            mem[pos] = False
            x -= 1

        return (nod, mem, mat)

    def plot(self, window):
        w = PlotWidget(window, self.genotypes, self.fitness, self.forces, self.bridge,
                       fitness_graph=self.fitness_graph, threshold_graph=self.threshold_graph)
        w.setWindowTitle('Statistics')

        return w


class LocalSearchOptimizer(Optimizer):

    def __init__(self, state: State,
                 mutateMembers=None,
                 mutateNodes=None,
                 mutateMaterials=None,
                 accept=None,
                 debug=False,
                 objFunc=ObjectiveFunction.RootMeanSquare):
        super().__init__(state, objFunc)
        self.mutateMaterials = mutateMaterials or create_probbit_mutate()
        self.mutateMembers = mutateMembers or create_probbit_mutate()
        self.mutateNodes = mutateNodes or create_creep_mutate(0.05, 2)
        self.accept = accept or create_threshold_accept(
            10000, 0.99)

    def mutate(self, genotype):
        _nodes, _members, _materials = genotype
        return self.mutateNodes(_nodes), self.mutateMembers(_members), self.mutateMaterials(_materials)

    def run(self, progress=None, max_iter=1000):
        (genotype, F, T) = self._local_search(
            self._evaluate, self.accept, self.bridge.genotype, max_iter=max_iter, progress=progress)
        self.bridge.setGenotype(genotype)
        self.genotypes = [genotype]
        self.forces = [self._solve(genotype=genotype)]
        self.fitness = [self._evaluate(genotype)]
        self.fitness_graph = F
        self.threshold_graph = T

    def _local_search(self, eval, accept, species, max_iter=1000, progress=None):
        if progress:
            progress.setEnabled(True)
            progress.setMaximum(max_iter)
            progress.setMinimum(0)
            progress.setValue(0)
            QGuiApplication.processEvents()

        t = 0
        A = species
        F = [np.array(eval(A))]
        T = [accept(F[t], F[t], t)[1]]

        while t < max_iter:
            B = self._repair(self.mutate(A))
            Bf = np.array(list(eval(B)))
            t += 1

            if progress and t % 100:
                progress.setValue(t)
                QGuiApplication.processEvents()

            accepted, threshold = accept(F[t-1], Bf, t)

            T.append(threshold)

            if accepted:
                A = B
                F.append(Bf)
            else:
                F.append(F[-1])

        if progress:
            progress.setEnabled(False)
            progress.setValue(max_iter)
            QGuiApplication.processEvents()

        return (A, F, T)

    def _in_allowed_area(self, nodes):
        # TODO: implement line intersection check
        # return not np.any(((nodes[:, 0] < 20) & (nodes[:, 1] < 40)) | ((nodes[:, 0] > 235) & (nodes[:, 1] < 40)))
        return True

    def _evaluate(self, genotype):
        b = self.bridge
        b.setGenotype(genotype)
        supports = b.supports
        nodes = b.nodes
        members = b.members
        materials = b.materials
        loads = b.loads

        LARGE = 10e10

        # TODO:
        if not self._in_allowed_area(nodes):
            return LARGE, LARGE, LARGE

        try:
            (x, detA, MFmat, RFmat) = solve(
                nodes, members, supports, loads)
        except:
            traceback.print_exc()
            return LARGE, LARGE, LARGE

        tmat = np.transpose(nodes[members], axes=(0, 2, 1))
        submat = np.subtract.reduce(tmat, axis=2)
        member_length = np.hypot.reduce(
            submat, axis=1, dtype=float).reshape(-1, 1)
        length = np.sum(member_length)

        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        G.add_edges_from(members.reshape(-1, 2))

        components = nx.algorithms.components.number_connected_components(G)

        costs = np.sum(member_length * np.array(MATERIAL_COSTS)
                       [np.array(materials)])

        max_member_lengths = np.array([MAX_MEMBER_LENGTHS[
            int(m)] for m in materials]).reshape(-1, 1)

        node_distance = np.sum(cdist(nodes, nodes)**2)

        too_long = np.sum(
            member_length[member_length > max_member_lengths]**3)**(1/3)

        length_sum = np.sum(member_length)

        num_too_long = len(member_length[member_length > max_member_lengths])

        # if too_long > 0:
        #    return tuple([too_long * np.max(G.degree())**components]*3)

        if x == 0 and detA != 0:
            if self.objFunc == ObjectiveFunction.Sum:
                forces = np.sum(np.abs(MFmat))
            elif self.objFunc == ObjectiveFunction.Mean:
                forces = np.mean(np.abs(MFmat))
            else:
                #forces = np.sqrt(np.mean(MFmat**2))
                forces = abs(np.mean(MFmat))

            if too_long > 0:
                return tuple(map(lambda x: x * np.sqrt(too_long)*num_too_long, [forces, costs, length]))
            else:
                return forces, costs, length

        else:
            return tuple([LARGE*(abs(x)+1) + node_distance]*3)


class GeneticOptimizer(Optimizer):

    def __init__(self, state: State,
                 mutateMembers=None,
                 mutateNodes=None,
                 mutateMaterials=None,
                 crossover=None,
                 select=None,
                 crossover_prob=0.5,
                 pop=100,
                 archive=20,
                 progress_every=100,
                 debug=False,
                 objFunc=ObjectiveFunction.RootMeanSquare):
        super().__init__(state, objFunc)
        self.pop = pop
        self.archive = archive
        self.mutateMembers = mutateMembers or create_probbit_mutate()
        self.mutateMaterials = mutateMaterials or create_probbit_mutate()
        self.mutateNodes = mutateNodes or create_creep_mutate(0.05, 10)
        self.select = select or create_tournament_select(pop, 6)
        self.crossover = crossover or create_k_point_crossover(6, 30)
        self.crossover_prob = crossover_prob
        self.progress_every = progress_every

    def mutate(self, genotype):
        _nodes, _members, _materials = genotype
        return self.mutateNodes(_nodes), self.mutateMembers(_members), self.mutateMaterials(_materials)

    def run(self, progress=None, max_iter=1000):
        (R, Rf, Rfs) = self._spea2(max_iter=max_iter, progress=progress)
        self.genotypes = R
        self.forces = list(map(lambda g: self._solve(genotype=g), R))
        self.fitness = np.array(Rf)
        print(self.fitness)
        self.fitness_graph = Rfs
        self.threshold_graph = []

    def _spea2(self, max_iter=1000, progress=None):
        if progress:
            progress.setEnabled(True)
            progress.setMaximum(max_iter)
            progress.setMinimum(0)
            progress.setValue(0)
            QGuiApplication.processEvents()

        t = 0
        P = [self.bridge.randomGenotype() for i in range(self.pop)]
        R = []
        Rf = []
        Rf_mean = []

        def dominates(a, b):
            return (a < b).any() or (a == b).all()

        while t < max_iter:
            Pf = np.array([self._evaluate(p) for p in P])
            # Pf = np.random.randint(0, 10, (self.pop, 2))
            # Pf = np.hstack([np.repeat([[0]], self.pop, axis=0),
            #                np.arange(self.pop).reshape(-1, 1)])

            PR = P + R

            PRf = np.vstack([Pf, np.array(Rf).reshape(-1, 3)])
            PRF = np.zeros(PRf.shape[0], dtype=float)

            dom_mat = cdist(PRf, PRf, metric=dominates).astype(np.bool)

            dom = np.count_nonzero(dom_mat, axis=0)

            knn = NearestNeighbors(n_neighbors=round((
                self.pop + self.archive)**0.5) + 1, n_jobs=1)
            knn.fit(PRf)

            dists, points = knn.kneighbors(PRf, return_distance=True)

            for i in range(PRf.shape[0]):
                PRF[i] = np.sum(dom[~dom_mat[i, :]], initial=0)

            PRF = PRF + 1/(np.max(dists, axis=1)**2+2)

            R_, = np.where(dom_mat.all(axis=1))

            while len(R_) > self.archive:
                R_ = np.delete(R_, np.argmin(dists[:, -1][R_]))

            if len(R_) < self.archive:
                sort_F = np.sort(PRF[np.where(np.any(~dom_mat, axis=1))])[
                    :self.archive-len(R_)]
                idx = np.argwhere(np.isin(PRF, sort_F))[
                    :len(sort_F)].reshape(-1)

                R_ = np.append(R_, idx)

            R = [PR[i] for i in R_]
            Rf = [PRf[i] for i in R_]
            Rf_mean.append(sum(map(lambda t: t[0]+t[1], Rf))/self.archive)

            P_ = self.select(PRF[:len(Pf)])

            P__ = []

            for i in range(int(self.pop/2)):
                u = random.random()
                A1, A2 = P[P_[2*i]], P[P_[2*i+1]]

                if u <= self.crossover_prob:
                    B, C = self.crossover(A1, A2)
                else:
                    B, C = A1, A2

                B = self._repair(self.mutate(B))
                C = self._repair(self.mutate(C))

                P__.append(B)
                P__.append(C)

            t += 1

            P = P__

            if progress and t % self.progress_every:
                progress.setValue(t)
                print(Rf[0])
                QGuiApplication.processEvents()

        if progress:
            progress.setEnabled(False)
            progress.setValue(max_iter)
            QGuiApplication.processEvents()

        return (R, Rf, Rf_mean)

    def _in_allowed_area(self, nodes):
        # TODO: implement line intersection check
        # return not np.any(((nodes[:, 0] < 20) & (nodes[:, 1] < 40)) | ((nodes[:, 0] > 235) & (nodes[:, 1] < 40)))
        return True

    def _evaluate(self, genotype):
        LARGE = 10e10
        b = self.bridge
        b.setGenotype(genotype)
        supports = b.supports
        nodes = b.nodes
        members = b.members
        materials = b.materials
        loads = b.loads

        tmat = np.transpose(nodes[members], axes=(0, 2, 1))
        submat = np.subtract.reduce(tmat, axis=2)
        member_length = np.hypot.reduce(
            submat, axis=1, dtype=float).reshape(-1, 1)
        length = np.sum(member_length)

        # TODO:
        if not self._in_allowed_area(nodes):
            return LARGE, LARGE, LARGE

        try:
            (x, detA, MFmat, RFmat) = solve(
                nodes, members, supports, loads)
        except:
            traceback.print_exc()
            return LARGE, LARGE, LARGE

        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        G.add_edges_from(members.reshape(-1, 2))

        components = nx.algorithms.components.number_connected_components(G)

        costs = np.sum(member_length * np.array(MATERIAL_COSTS)
                       [np.array(materials)])

        max_member_lengths = np.array([MAX_MEMBER_LENGTHS[
            int(m)] for m in materials]).reshape(-1, 1)

        node_distance = np.sum(cdist(nodes, nodes)**2)

        too_long = np.sum(
            member_length[member_length > max_member_lengths]**3)**(1/3)

        length_sum = np.sum(member_length)

        num_too_long = len(member_length[member_length > max_member_lengths])

        # if too_long > 0:
        #    return tuple([too_long * np.max(G.degree())**components]*3)

        if x == 0 and detA != 0:
            if self.objFunc == ObjectiveFunction.Sum:
                forces = np.sum(np.abs(MFmat))
            elif self.objFunc == ObjectiveFunction.Mean:
                forces = np.mean(np.abs(MFmat))
            else:
                #forces = np.sqrt(np.mean(MFmat**2))
                forces = abs(np.mean(MFmat))

            if too_long > 0:
                return tuple(map(lambda x: x * np.sqrt(too_long)*num_too_long, [forces, costs, length]))
            else:
                return forces, costs, length

        else:
            return tuple([LARGE*(abs(x)+1) + node_distance]*3)
