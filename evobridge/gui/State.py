from .Objects import Node, Rock, Member
from PyQt5.QtGui import QPainter


class State():

    def __init__(self, nodes=[], rocks=[], members=[]):
        self.nodes = nodes
        self.rocks = rocks
        self.members = members

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

    def toggleMember(self, node_a: Node, node_b: Node):
        a = self.nodes.index(node_a)
        b = self.nodes.index(node_b)
        assert a >= 0
        assert b >= 0
        assert a != b
        assert a < len(self.nodes)
        assert b < len(self.nodes)

        member = Member(node_a, node_b)

        if member in self.members:
            self.members.remove(member)
        else:
            self.members.append(member)

    def removeNode(self, node: Node):
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

    def draw(self, p: QPainter):
        for rock in self.rocks:
            rock.draw(p)

        for member in self.members:
            member.draw(p)

        for node in self.nodes:
            node.draw(p)
