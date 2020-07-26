from PyQt5.QtWidgets import QWidget, QVBoxLayout, QToolBar, QAction, QActionGroup, QMainWindow
from PyQt5.QtGui import QIcon
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib.colors as col
import matplotlib.cm as cm
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
from sklearn.preprocessing import normalize
from ..gui.Objects import Material, ObjectiveFunction, Mutation, MATERIAL_COSTS, MAX_MEMBER_LENGTHS
from PyQt5.QtCore import (Qt, pyqtSlot)


class PlotWidget(QMainWindow):
    def __init__(self, window, genotypes, fitness, forces, bridge, fitness_graph=None, threshold_graph=None):
        super().__init__(window)

        self.genotypes = genotypes
        self.fitness = fitness
        self.forces = forces
        self.bridge = bridge
        self.threshold_graph = threshold_graph
        self.fitness_graph = fitness_graph
        self.cached_canvas = [None]*len(genotypes)
        self.current_fig = None
        self.current_toolbar = None
        self.current_canvas = None

        toolbar = QToolBar("Individuals")
        self.addToolBar(Qt.BottomToolBarArea, toolbar)

        self.actiongroup = QActionGroup(self)

        for i in range(len(self.genotypes)):
            action = QAction(QIcon(), str(i+1), self)
            action.setCheckable(True)
            self.actiongroup.addAction(action)

        self.actiongroup.triggered.connect(self.plot_genotype)
        self.boxlayout = QVBoxLayout()

        toolbar.addActions(self.actiongroup.actions())
        self.actiongroup.setExclusive(True)
        self.actiongroup.actions()[0].activate(QAction.Trigger)
        # self.plot(0)

    @pyqtSlot(QAction)
    def plot_genotype(self, action: QAction):
        idx = self.actiongroup.actions().index(action)
        self.plot(idx)

    def onpick(self, event):
        self.plot(event.ind[0])
        self.actiongroup.actions()[event.ind[0]].activate(QAction.Trigger)

    def plot(self, index: int):
        if self.current_toolbar is not None:
            plt.close(self.current_fig)
            self.removeToolBar(self.current_toolbar)
            self.current_toolbar = None

        statically_correct = True

        (x, det, MFmat, RFmat) = self.forces[index]

        if x is None or x != 0 or det == 0:
            statically_correct = False

        b = self.bridge
        b.setGenotype(self.genotypes[index])
        genotype = b.genotype
        supports = b.supports
        nodes = b.nodes
        members = b.members
        materials = b.materials
        loads = b.loads

        force, cost, length = self.fitness[index]

        tmat = np.transpose(nodes[members], axes=(0, 2, 1))
        submat = np.subtract.reduce(tmat, axis=2)
        member_lengths = np.hypot.reduce(submat, axis=1, dtype=float)
        max_member_lengths = np.array([MAX_MEMBER_LENGTHS[
            int(m)] for m in materials]).reshape(-1, 1)

        rows = int(self.fitness_graph is not None) * 100
        fig = plt.figure(figsize=plt.figaspect(
            0.5 if self.fitness_graph is not None else 0.33))
        # fig, axs = plt.subplots(rows, 1, figsize=figsize)

        self.current_fig = fig
        canvas = FigureCanvas(fig)

        ax = plt.subplot(121 + rows)
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

        plt.grid()

        if statically_correct:
            ax = plt.subplot(122 + rows)

            cmap = plt.get_cmap('coolwarm')
            abs_val = np.abs(MFmat)
            max_abs = np.max(abs_val)
            norm = col.Normalize(vmin=-max_abs, vmax=max_abs)

            for i in range(len(members[:, 0])):
                f = MFmat[i, 0]  # TODO out of bounds
                ls = ":" if member_lengths[i] > max_member_lengths[i] else "-"

                ax.plot(nodes[members[i, :], 0],
                        nodes[members[i, :], 1], linestyle=ls, color=cmap(norm(f)), lw=5)
                _nodes = nodes[members[i, :]]
                center = (np.sum(_nodes, axis=0)/2.0).reshape(2, -1)

                if np.abs(f) < 1e-12:
                    c = "k"
                elif f < 0:
                    c = "blue"
                else:
                    c = "red"

                ax.text(*center, "{:.1f}".format(f),
                        backgroundcolor="w", color=c, weight="medium")

            mask = np.ones(nodes.shape[0], dtype=bool)
            mask[supports[:, 0].ravel()] = 0
            _nodes = nodes[mask].reshape(-1, 2)
            _supports = nodes[np.invert(mask)].reshape(-1, 2)

            ax.scatter(_nodes[:, 0], _nodes[:, 1], s=100,
                       zorder=3, color="w", edgecolors="black")
            ax.scatter(_supports[:, 0], _supports[:, 1], s=100, marker="s",
                       zorder=3, color="w", edgecolors="black")
            # plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            ax2_divider = make_axes_locatable(ax)
            cax2 = ax2_divider.append_axes("right", size="7%", pad="2%")
            cb2 = fig.colorbar(cm.ScalarMappable(
                norm=norm, cmap=cmap), cax=cax2, orientation="vertical")
            # cax2.xaxis.set_ticks_position("right")

            ax.set_aspect('equal', adjustable='box')

            fig.suptitle(
                "Forces: {:0.5f}, Costs: {:0.5f}, Lenghts: {:0.5f}".format(force, cost, length))

        if self.fitness_graph:
            if len(self.fitness) > 1:
                ax = plt.subplot(124 + rows, projection='3d')
                X = self.fitness

                face_col = normalize(X, norm="max", axis=0)

                ax.scatter(X[:, 0], X[:, 1], zs=X[:, 2],
                           facecolors=face_col,  marker="o", picker=True)
                ax.set_xlabel("Forces")
                ax.set_ylabel("Costs")
                ax.set_zlabel("Lengths")

                ax = plt.subplot(123 + rows)
            else:
                ax = plt.subplot(112 + rows)

            # ax.set_yscale("log")
            color = 'tab:blue'
            ax.plot(range(len(self.fitness_graph)),
                    self.fitness_graph, label="Best Fitness", color=color)
            ax.set_ylabel("Fitness", color=color)
            ax.set_xlabel("Iterations")
            ax.tick_params(axis='y', labelcolor=color)
            ax.grid()

            if self.threshold_graph:
                ax = ax.twinx()  # instantiate a second axes that shares the same x-axis

                color = "tab:orange"
                # we already handled the x-label with ax1
                ax.set_ylabel('Threshold', color=color)
                ax.plot(range(len(self.threshold_graph)),
                        self.threshold_graph, label="Accept", color=color)
                ax.tick_params(axis='y', labelcolor=color)

        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        fig.canvas.mpl_connect('pick_event', self.onpick)

        self.current_fig = fig
        self.current_canvas = canvas

        self.current_toolbar = NavigationToolbar(canvas, self)
        self.current_toolbar.setObjectName("Figure Control")
        self.addToolBar(Qt.BottomToolBarArea,
                        self.current_toolbar)
        self.insertToolBarBreak(self.current_toolbar)
        self.setCentralWidget(canvas)
