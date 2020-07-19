import random
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ..gui.State import State, max_lengths
from .statics import solve
from ..gui.Objects import Material, ObjectiveFunction, Mutation
import matplotlib.colors as col
import matplotlib.cm as cm
from .genotype import toGenotype, fromGenotype
from .graycode import grayToStdbin, stdbinToGray
from PyQt5.QtGui import QGuiApplication
from enum import Enum
from .functions import create_onebit_mutate, create_threshold_accept
from .bridge import Bridge
from abc import abstractmethod, ABCMeta


class Optimizer(object, metaclass=ABCMeta):

    MAX_MEMBER_LENGTHS = max_lengths

    @abstractmethod
    def __init__(self, state: State, node_weight=10, street_weight=8):
        self.bridge = Bridge(state, node_weight=node_weight,
                             street_weight=street_weight)

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
            det, MF, RF = solve(nodes, members, supports, loads)
        except e:
            raise e

        if det == 0:
            raise ArithmeticError("Determinant is zero")

        if len(MF) == 0 or len(RF) == 0:
            raise ArithmeticError("Not solveable")

        return (det, MF, RF)

    def plot(self, figsize=None, title=None, ylim=None, xlim=None, show_loads=False, show_fitness_graph=False):
        try:
            det, MFmat, RFmat = self._solve()
        except Exception as e:
            raise ArithmeticError from e

        b = self.bridge
        genotype = b.genotype
        supports = b.supports
        nodes = b.nodes
        members = b.members
        materials = b.materials
        loads = b.loads

        fitness = self._evaluate(genotype)

        tmat = np.transpose(nodes[members], axes=(0, 2, 1))
        submat = np.subtract.reduce(tmat, axis=2)
        member_lengths = np.hypot.reduce(submat, axis=1, dtype=float)
        max_member_lengths = np.array([self.MAX_MEMBER_LENGTHS[
            int(m)] for m in materials]).reshape(-1, 1)

        rows = 2 + int(show_fitness_graph)
        fig, axs = plt.subplots(rows, 1, figsize=figsize)

        ax = axs[0]
        material_labels = ["Street", "Wood", "Steel"]
        colors = ["black", "peru", "brown"]
        cmap = ListedColormap(colors, name="materials")
        norm = col.Normalize(vmin=0, vmax=2)

        for i in range(len(members[:, 0])):
            ls = ":" if member_lengths[i] > max_member_lengths[i] else "-"
            c = cmap(norm(int(materials[i, 0])))
            l = material_labels[int(materials[i])]

            ax.plot(nodes[members[i, :], 0],
                    nodes[members[i, :], 1], linestyle=ls, color=c, label=l, lw=5)

        def legend_without_duplicate_labels(ax):
            handles, labels = ax.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(
                zip(handles, labels)) if l not in labels[:i]]
            ax.legend(*zip(*unique))

        mask = np.ones(nodes.shape[0], dtype=bool)
        mask[supports[:, 0].ravel()] = 0
        _nodes = nodes[mask].reshape(-1, 2)
        _supports = nodes[np.invert(mask)].reshape(-1, 2)

        ax.scatter(_nodes[:, 0], _nodes[:, 1], s=100,
                   zorder=3, color="w", edgecolors="black")
        ax.scatter(_supports[:, 0], _supports[:, 1], s=100, marker="s",
                   zorder=3, color="w", edgecolors="black")

        ax.set_aspect('equal', adjustable='box')

        if ylim:
            ax.ylim(ylim)

        if xlim:
            ax.xlim(xlim)

        plt.grid()

        ax = axs[1]

        cmap = plt.get_cmap('coolwarm')
        abs_val = np.abs(MFmat)
        max_abs = np.max(abs_val)
        norm = col.Normalize(vmin=-max_abs, vmax=max_abs)

        for i in range(len(members[:, 0])):
            force = MFmat[i, 0]
            ls = ":" if member_lengths[i] > max_member_lengths[i] else "-"

            ax.plot(nodes[members[i, :], 0],
                    nodes[members[i, :], 1], linestyle=ls, color=cmap(norm(force)), lw=5)
            _nodes = nodes[members[i, :]]
            center = (np.sum(_nodes, axis=0)/2.0).reshape(2, -1)

            if np.abs(force) < 1e-12:
                c = "k"
            elif force < 0:
                c = "blue"
            else:
                c = "red"

            ax.text(*center, "{:.1f}".format(force),
                    backgroundcolor="w", color=c, weight="medium")

        mask = np.ones(nodes.shape[0], dtype=bool)
        mask[supports[:, 0].ravel()] = 0
        _nodes = nodes[mask].reshape(-1, 2)
        _supports = nodes[np.invert(mask)].reshape(-1, 2)

        ax.scatter(_nodes[:, 0], _nodes[:, 1], s=100,
                   zorder=3, color="w", edgecolors="black")
        ax.scatter(_supports[:, 0], _supports[:, 1], s=100, marker="s",
                   zorder=3, color="w", edgecolors="black")
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

        if show_loads:
            for l in loads:
                x, y = nodes[l[0].astype(int)]
                dx, dy = l[1:]
                ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.1,
                         width=0.1, facecolor="w", edgecolor="black")

        ax.set_aspect('equal', adjustable='box')

        if ylim:
            ax.ylim(ylim)

        if xlim:
            ax.xlim(xlim)

        if title:
            fig.suptitle(title)
        else:
            fig.suptitle("Fitness: {}".format(fitness))

        if show_fitness_graph:
            ax = axs[2]
            # ax.set_yscale("log")
            ax.plot(range(len(self.fitness_graph)),
                    self.fitness_graph, label="Best")
            ax.plot(range(len(self.threshold_graph)),
                    self.threshold_graph, label="Accept")

            ax.grid()
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Fitness")

        fig.show()


class LocalSearchOptimizer(Optimizer):

    def __init__(self, state: State,
                 mutate=None,
                 accept=None,
                 debug=False,
                 objFunc=ObjectiveFunction.Sum):
        super().__init__(state)
        self.objFunc = objFunc
        self.mutate = mutate or create_onebit_mutate()
        self.accept = accept or create_threshold_accept(
            10000, 0.99)

    def run(self, progress=None, max_iter=1000):
        (genotype, F, T) = self._local_search(
            self._evaluate, self.mutate, self.accept, self.bridge.genotype, max_iter=max_iter, progress=progress)
        self.bridge.setGenotype(genotype)
        self.fitness_graph = F
        self.threshold_graph = T

    def _local_search(self, eval, mutate, accept, species, max_iter=1000, progress=None):
        if progress:
            progress.setEnabled(True)
            progress.setMaximum(max_iter)
            progress.setMinimum(0)
            progress.setValue(0)
            QGuiApplication.processEvents()

        t = 0
        A = species
        F = [eval(A)]
        T = [accept(F[t], F[t], t)[1]]

        while t < max_iter:
            B = tuple(map(mutate, A))
            Bf = eval(B)
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

        if not self._in_allowed_area(nodes):
            return 10e6

        tmat = np.transpose(nodes[members], axes=(0, 2, 1))
        submat = np.subtract.reduce(tmat, axis=2)
        member_length = np.hypot.reduce(
            submat, axis=1, dtype=float).reshape(-1, 1)

        try:
            (detA, MFmat, RFmat) = solve(
                nodes, members, supports, loads)
        except:
            return 10e6

        if detA == 0:
            return 10e6

        max_member_lengths = np.array([self.MAX_MEMBER_LENGTHS[
            int(m)] for m in materials]).reshape(-1, 1)

        too_long = member_length[member_length > max_member_lengths]

        if self.objFunc == ObjectiveFunction.Sum:
            value = np.sum(np.abs(MFmat))
        elif self.objFunc == ObjectiveFunction.Mean:
            value = np.mean(np.abs(MFmat))
        else:
            value = np.sqrt(np.mean(MFmat**2))

        return value + np.sum(np.power(too_long, 2), initial=0)
