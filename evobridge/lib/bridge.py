import numpy as np
import random
from ..gui.Objects import Material, MAX_MEMBER_LENGTHS
from ..lib.genotype import toGenotype, fromGenotype, stdbinToGray


class Bridge():
    def __init__(self, state, node_weight=10, street_weight=20):
        state = state.clone()

        street_nodes = [node for node in state.nodes if (node in set(
            [node for member in state.members if member.material == Material.STREET for node in [member.a, member.b]]))]

        fixed_nodes = [node
                       for node in state.nodes if node not in street_nodes and (node.v_support or node.h_support)]

        genotype_nodes = [
            node for node in state.nodes if node not in street_nodes + fixed_nodes]

        nodes = [*street_nodes, *fixed_nodes, *genotype_nodes]

        assert len(nodes) > 0, "At least 1 Node is required"

        nodes_array = np.array([[node.x, 255-node.y]
                                for node in genotype_nodes])

        members_idx = np.array(
            [[*sorted([nodes.index(member.a), nodes.index(member.b)])] for member in state.members])

        members_array = np.full((len(nodes), len(nodes)), False)

        if len(members_idx) > 0:
            members_array[members_idx[:, 0], members_idx[:, 1]] = True

        materials_idx = np.array([[*sorted([nodes.index(member.a), nodes.index(member.b)])]
                                  for member in state.members if member.material == Material.STEEL])

        materials_array = np.full((len(nodes), len(nodes)), False)

        if len(materials_idx) > 0:
            materials_array[materials_idx[:, 0], materials_idx[:, 1]] = True

        self.supports = np.array([[i, int(node.h_support), int(node.v_support)] for i, node in enumerate(
            nodes) if node.v_support or node.h_support], dtype=int).reshape((-1, 3))

        self.fixed_nodes_pos = np.array(
            [[node.x, 255-node.y] for node in street_nodes + fixed_nodes]).reshape((-1, 2))

        self.street_members_idx = np.array(
            [[*sorted([nodes.index(member.a), nodes.index(member.b)]), member.material.value] for member in state.members if member.a in street_nodes and member.b in street_nodes])

        # Genotype
        self.genotype = toGenotype(nodes_array, members_array, materials_array)

        node_loads = np.hstack([np.arange(len(nodes)).reshape((-1, 1)), np.full((len(nodes), 2), [
            0, -node_weight])]).reshape((-1, 3))

        street_loads = np.array([[nodes.index(node), 0, -street_weight * member.length() / 2]
                                 for member in state.members for node in [member.a, member.b] if member.material == Material.STREET]).reshape((-1, 3))

        self.loads = np.vstack([node_loads, street_loads])

        self.nodes, self.members, self.materials = self._unpackGenotype(
            self.genotype)

    def _unpackGenotype(self, genotype):
        _nodes, _members, _materials = fromGenotype(*genotype)
        nodes = np.vstack([self.fixed_nodes_pos, _nodes])

        _members[self.street_members_idx[:, 0],
                 self.street_members_idx[:, 1]] = False
        members = np.argwhere(_members)

        tmat = np.transpose(nodes[members], axes=(0, 2, 1))
        submat = np.subtract.reduce(tmat, axis=2)
        member_length = np.hypot.reduce(
            submat, axis=1, dtype=float).reshape(-1, 1)

        materials = (_materials[_members]).astype(int).reshape(-1, 1)+1

        max_member_lengths = np.array([MAX_MEMBER_LENGTHS[
            int(m)] for m in materials]).reshape(-1, 1)

        allowed_members = np.argwhere(
            member_length <= max_member_lengths)[:, 0]

        materials = np.vstack([self.street_members_idx[:, 2].reshape(-1, 1).astype(
            int), materials[allowed_members]])

        members = np.vstack(
            [self.street_members_idx[:, 0:2], members[allowed_members]]).astype(int)

        return nodes, members, materials

    def setGenotype(self, genotype):
        self.genotype = genotype
        self.nodes, self.members, self.materials = self._unpackGenotype(
            self.genotype)

    def getGenotype(self):
        return self.genotype

    def randomGenotype(self):
        _nodes, _members, _materials = self.genotype

        # nodes = np.array(
        #    list(map(lambda x: round(random.uniform(0, 255)*4)/4, range(len(_nodes)))))
        members = np.random.random(len(_members)) >= 0.5
        materials = np.random.random(len(_materials)) >= 0.5

        return (_nodes, members, materials)
