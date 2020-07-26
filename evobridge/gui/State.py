from .Objects import Node, Rock, Member, Material
from PyQt5.QtGui import QPainter


class State():

    def __init__(self, nodes=[], rocks=[], members=[]):
        self.nodes = nodes
        self.rocks = rocks
        self.members = members

    def clone(self):
        new_nodes = [n.clone() for n in self.nodes]
        new_rocks = [r.clone() for r in self.rocks]
        new_members = [Member(new_nodes[self.nodes.index(m.a)], new_nodes[self.nodes.index(
            m.b)], material=m.material) for m in self.members]
        return State(new_nodes, new_rocks, new_members)

    @classmethod
    def loadState(cls, path):
        nodes = []
        rocks = []
        members = []
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith(Node.__name__):
                    nodes.append(Node.fromString(line))
                elif line.startswith(Rock.__name__):
                    rocks.append(Rock.fromString(line))
                elif line.startswith(Member.__name__):
                    members.append(Member.fromString(line, nodes))

        return cls(nodes, rocks, members)

    def saveState(self, path):
        with open(path, "w") as f:
            for rock in self.rocks:
                f.write(rock.toString() + "\n")
            for node in self.nodes:
                f.write(node.toString() + "\n")
            for member in self.members:
                f.write(member.toString(self.nodes) + "\n")

    def addNode(self, node: Node):
        self.nodes.append(node)

    def addRock(self, rock: Rock):
        self.rocks.append(rock)

    def removeRock(self, rock: Rock):
        assert isinstance(rock, Rock)
        self.rocks.remove(rock)

    def removeObject(self, obj):
        if isinstance(obj, Node):
            self.removeNode(obj)

        if isinstance(obj, Rock):
            self.removeRock(obj)

    def toggleMember(self, node_a: Node, node_b: Node, material):
        a = self.nodes.index(node_a)
        b = self.nodes.index(node_b)
        if a < 0:
            return
        if b < 0:
            return
        if a == b:
            return
        if a >= len(self.nodes):
            return
        if b >= len(self.nodes):
            return

        member = Member(node_a, node_b, material)

        if member in self.members:
            self.members.remove(member)
        else:
            self.members.append(member)

    def removeNode(self, node: Node):
        # TODO bug when deleting nodes
        assert isinstance(node, Node)
        self.members = [
            member for member in self.members if not member.connectedTo(node)]
        self.nodes.remove(node)

    def getClickedObject(self, x, y):
        for node in self.nodes:
            if node.inClickArea(x, y):
                return node
        for rock in self.rocks:
            if rock.inClickArea(x, y):
                return rock

        return None

    def draw(self, p: QPainter, scale: float):
        for rock in self.rocks:
            rock.draw(p, scale)

        for member in self.members:
            member.draw(p, scale)

        for node in self.nodes:
            node.draw(p, scale)
