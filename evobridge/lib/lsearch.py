import random
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from ..gui.State import State
from .statics import solve
from ..gui.Objects import Material
import matplotlib.colors as col
import matplotlib.cm as cm
from .graycode import grayToStdbin
from PyQt5.QtGui import QGuiApplication


class Optimizer():

    def run():
        raise NotImplementedError(
            "run not implemented for class " + cls.__name__)


class LocalSearchOptimizer(Optimizer):

    MAX_MEMBER_LENGTH = 40

    def __init__(self, state: State,
                 mutate=None,
                 accept=None,
                 node_weight=10,
                 street_weight=8,
                 debug=False,
                 onupdate=None):
        state = state.clone()
        self.objFunc = "sum"
        self.mutate = mutate or LocalSearchOptimizer.create_onebit_mutate()
        self.accept = accept or LocalSearchOptimizer.create_threshold_accept(
            10000, 0.99)
        self.supports = np.array([[i, int(node.h_support), int(node.v_support)] for i, node in enumerate(
            state.nodes) if node.v_support or node.h_support], dtype=int).reshape((-1, 3))
        fixed_nodes = [node for node in state.nodes if node in set(
            [node for member in state.members if member.material == Material.STREET for node in [member.a, member.b]])]
        genotype_nodes = [
            node for node in state.nodes if node not in fixed_nodes]
        self.nodes = [*fixed_nodes, *genotype_nodes]
        self.fixed_nodes_pos = np.array(
            [[node.x, 255-node.y] for node in fixed_nodes]).reshape((-1, 2))
        self.genotype = bytearray(chain.from_iterable(
            [[int(node.x), int(255-node.y)] for node in genotype_nodes]))
        self.members_idx = np.array(
            [[self.nodes.index(member.a), self.nodes.index(member.b)] for member in state.members])
        node_loads = np.hstack([np.arange(len(self.nodes)).reshape((-1, 1)), np.full((len(self.nodes), 2), [
                               0, -node_weight])]).reshape((-1, 3))
        street_loads = np.array([[self.nodes.index(node), 0, -street_weight * member.length() / 2]
                                 for member in state.members for node in [member.a, member.b]]).reshape((-1, 3))
        self.loads = np.vstack(
            [node_loads, street_loads])
        genotype = bytearray(chain.from_iterable(
            [[int(node.x), int(255-node.y)] for node in genotype_nodes]))

        if debug:
            print("Created {}:".format(self.__class__.__name__))
            print("Fixed Nodes:", self.fixed_nodes_pos)
            print("Genotype:", self.genotype)
            print("Members:", self.members_idx)
            print("Supports:", self.supports.shape, self.supports)
            print("Loads:", self.loads)

    @ staticmethod
    def create_threshold_accept(temp, damping, minimize=True):
        return lambda Af, Bf, t: Bf < Af or abs(Af - Bf) <= damping ** t * temp

    def run(self, progress=None, max_iter=1000, objFunc="sum"):
        self.objFunc = objFunc
        (genotype, F) = self._local_search(
            self._evaluate, self.mutate, self.accept, self.genotype, max_iter=max_iter, progress=progress)
        self.genotype = genotype
        self.fitness = F

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

        while t < max_iter:
            B = mutate(A)
            Bf = eval(B)
            t += 1

            if progress and t % 100:
                progress.setValue(t)
                QGuiApplication.processEvents()

            if accept(F[t-1], Bf, t):
                A = B
                F.append(Bf)
            else:
                F.append(F[-1])

        if progress:
            progress.setEnabled(False)
            progress.setValue(max_iter)
            QGuiApplication.processEvents()

        return (A, F)

    def _in_allowed_area(self, nodes):
        # TODO: implement line intersection check
        # return not np.any(((nodes[:, 0] < 20) & (nodes[:, 1] < 40)) | ((nodes[:, 0] > 235) & (nodes[:, 1] < 40)))
        return True

    def _evaluate(self, genotype):
        nodes = np.array(
            [*self.fixed_nodes_pos.flatten(), *grayToStdbin(genotype)]).reshape(-1, 2)
        supports = self.supports
        members = self.members_idx
        loads = self.loads

        if not self._in_allowed_area(nodes):
            return 10e10

        tmat = np.transpose(nodes[members], axes=(0, 2, 1))
        submat = np.subtract.reduce(tmat, axis=2)
        member_length = np.hypot.reduce(submat, axis=1, dtype=float)

        (detA, MFmat, RFmat) = solve(
            nodes, members, supports, loads)

        if detA == 0:
            return 10e10

        # if np.any(member_length > self.MAX_MEMBER_LENGTH):
        too_long = member_length[member_length > self.MAX_MEMBER_LENGTH]
        #    return -np.power(np.log(np.sum(too_long)+1)+1, 2)
        if self.objFunc == "sum":
            value = np.sum(np.abs(MFmat))
        else:
            value = np.mean(np.abs(MFmat))

        return value + np.power(np.sum(too_long, initial=0), 2)

    def _solve(self, genotype=None):
        genotype = grayToStdbin(genotype or self.genotype)
        genotype = np.array(
            genotype, dtype=int).reshape(-1, 2)
        nodes = np.vstack([self.fixed_nodes_pos, genotype])
        supports = self.supports
        members = self.members_idx
        loads = self.loads
        det, MF, RF = solve(nodes, members, supports, loads)

        if det == 0:
            raise ArithmeticError("Determinant is zero")

        if len(MF) == 0 or len(RF) == 0:
            raise ArithmeticError("Not solveable")

        return (det, MF, RF)

    def plot(self, figsize=None, title=None, ylim=None, xlim=None, show_loads=False):
        try:
            det, MFmat, RFmat = self._solve()
        except Exception as e:
            raise ArithmeticError from e

        fitness = self._evaluate(self.genotype)

        genotype = np.array(
            grayToStdbin(self.genotype), dtype=int).reshape(-1, 2)
        Nodes = np.vstack([self.fixed_nodes_pos, genotype])
        Members = self.members_idx
        Supports = self.supports
        Loads = self.loads

        tmat = np.transpose(Nodes[Members], axes=(0, 2, 1))
        submat = np.subtract.reduce(tmat, axis=2)
        member_lengths = np.hypot.reduce(submat, axis=1, dtype=float)

        plt.figure(figsize=figsize)
        plt.subplot(2, 1, 1)
        plt.grid()

        cmap = plt.get_cmap('coolwarm')
        abs_val = np.abs(MFmat)
        max_abs = np.max(abs_val)
        norm = col.Normalize(vmin=-max_abs, vmax=max_abs)

        for i in range(len(Members[:, 0])):
            force = MFmat[i, 0]
            plt.plot(Nodes[Members[i, :], 0],
                     Nodes[Members[i, :], 1], linestyle=(":" if member_lengths[i] > self.MAX_MEMBER_LENGTH else "-"), color=cmap(norm(force)), lw=5)
            nodes = Nodes[Members[i, :]]
            center = (np.sum(nodes, axis=0)/2.0).reshape(2, -1)

            if np.abs(force) < 1e-12:
                c = "k"
            elif force < 0:
                c = "blue"
            else:
                c = "red"

            plt.text(*center, "{:.1f}".format(force),
                     backgroundcolor="w", color=c, weight="medium")

        mask = np.ones(Nodes.shape[0], dtype=bool)
        mask[Supports[:, 0].ravel()] = 0
        nodes = Nodes[mask].reshape(-1, 2)
        supports = Nodes[np.invert(mask)].reshape(-1, 2)

        plt.scatter(nodes[:, 0], nodes[:, 1], s=100,
                    zorder=3, color="w", edgecolors="black")
        plt.scatter(supports[:, 0], supports[:, 1], s=100, marker="s",
                    zorder=3, color="w", edgecolors="black")
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))

        if show_loads:
            for l in Loads:
                x, y = Nodes[l[0].astype(int)]
                dx, dy = l[1:]
                plt.arrow(x, y, dx, dy, head_width=0.2, head_length=0.1,
                          width=0.1, facecolor="w", edgecolor="black")

        plt.gca().set_aspect('equal', adjustable='box')

        if ylim:
            plt.ylim(ylim)

        if xlim:
            plt.xlim(xlim)

        if title:
            plt.title(title)
        else:
            plt.title("Fitness: {}".format(fitness))

        ax = plt.subplot(2, 1, 2)
        ax.set_yscale("log")
        plt.plot(range(len(self.fitness)), self.fitness)
        plt.grid()
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")

        plt.show()

    @ staticmethod
    def create_onebit_mutate():
        def mutate(genotype):
            assert type(genotype) is bytearray
            child = bytearray(len(genotype))

            for i, b in enumerate(genotype):
                child[i] = b

            index = random.randint(0, 8*len(genotype)-1)
            pos = index % 8
            child[index//8] ^= 1 << pos

            return child
        return mutate

    @ staticmethod
    def create_propbit_mutate():
        def mutate(genotype):
            assert type(genotype) is bytearray

            mutrate = 0.125/len(genotype)
            child = bytearray(len(genotype))

            for i, b in enumerate(genotype):
                child[i] = b
                mask = 0
                for k in range(8):
                    mask = mask << 1
                    if random.random() < mutrate:
                        mask ^= 1

                child[i] ^= mask

            return child
        return mutate
