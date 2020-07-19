import numpy as np
from ..gui.Objects import Material
from ..lib.genotype import toGenotype, fromGenotype


class Bridge():
    def __init__(self, state, node_weight=10, street_weight=8):
        state = state.clone()

        street_nodes = [node for node in state.nodes if node in set(
            [node for member in state.members if member.material == Material.STREET for node in [member.a, member.b]])]

        genotype_nodes = [
            node for node in state.nodes if node not in street_nodes]

        nodes = [*street_nodes, *genotype_nodes]

        assert len(nodes) > 0, "At least 1 Node is required"

        nodes_array = np.array([[int(node.x), int(255-node.y)]
                                for node in genotype_nodes])

        members_idx = np.array(
            [[*sorted([nodes.index(member.a), nodes.index(member.b)]), member.material.value] for member in state.members if not (member.a in street_nodes and member.b in street_nodes)])

        members_array = np.full((len(nodes)-1, len(genotype_nodes)), 0.0)

        members_array[np.nonzero(
            np.tri(*members_array.shape, k=-len(street_nodes)))] = np.nan

        for a, b, m in members_idx:
            members_array[a, b - len(street_nodes)] = m

        self.supports = np.array([[i, int(node.h_support), int(node.v_support)] for i, node in enumerate(
            state.nodes) if node.v_support or node.h_support], dtype=int).reshape((-1, 3))

        self.street_nodes_pos = np.array(
            [[node.x, 255-node.y] for node in street_nodes]).reshape((-1, 2))

        self.street_members_idx = np.array(
            [[*sorted([nodes.index(member.a), nodes.index(member.b)]), member.material.value] for member in state.members if member.a in street_nodes and member.b in street_nodes])

        # Genotype
        self.genotype = toGenotype(nodes_array, members_array)

        node_loads = np.hstack([np.arange(len(nodes)).reshape((-1, 1)), np.full((len(nodes), 2), [
                               0, -node_weight])]).reshape((-1, 3))

        street_loads = np.array([[nodes.index(node), 0, -street_weight * member.length() / 2]
                                 for member in state.members for node in [member.a, member.b] if member.material == Material.STREET]).reshape((-1, 3))

        self.loads = np.vstack([node_loads, street_loads])

        self.nodes, self.members, self.materials = self._unpackGenotype(
            self.genotype)

    def _unpackGenotype(self, genotype):
        _nodes, _members = fromGenotype(*genotype)
        nodes = np.vstack([self.street_nodes_pos, _nodes])
        members = np.argwhere(_members > 0)
        materials = np.vstack([self.street_members_idx[:, 2].reshape(-1, 1), np.array(
            [int(_members[x, y]) for x, y in members]).reshape(-1, 1)])
        members[:, 1] += len(self.street_nodes_pos)
        members = np.vstack(
            [self.street_members_idx[:, 0:2], members]).astype(int)

        return nodes, members, materials

    def setGenotype(self, genotype):
        self.genotype = genotype
        self.nodes, self.members, self.materials = self._unpackGenotype(
            genotype)

    def getGenotype(self):
        return self.genotype
