import math

from PyQt5.QtCore import QPointF, QRectF, pyqtSlot, pyqtSignal, Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QTransform
import random


class StateObject():
    x = 0
    y = 0
    selected = False

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def inClickArea(self, x, y):
        raise NotImplementedError(
            "Class %s doesn't implement inClickArea()" % (self.__class__.__name__))

    def draw(self, p: QPainter):
        raise NotImplementedError(
            "Class %s doesn't implement draw()" % (self.__class__.__name__))

    def info(self):
        raise NotImplementedError(
            "Class %s doesn't implement info()" % (self.__class__.__name__))

    def setX(self, x):
        self.x = x

    def setY(self, y):
        self.y = y


class Rock(StateObject):

    def __init__(self, x, y, w, h):
        super().__init__(x, y)
        self.w = w
        self.h = h

    @classmethod
    def fromString(cls, line: str):
        cname, *params = line.split(" ")
        assert cname == cls.__name__

        return cls(*map(int, params))

    def toString(self):
        params = map(int, [self.x, self.y, self.w, self.h])
        return " ".join(map(str, [self.__class__.__name__, *params]))

    def inClickArea(self, x, y):
        return x >= self.x and \
            x <= self.x + self.w and \
            y >= self.y and \
            y <= self.y + self.h

    def draw(self, p: QPainter):
        p.save()
        if self.selected:
            pen = QPen(QColor("red"))
            pen.setWidthF(2)
        else:
            pen = QPen(QColor("black"))
            pen.setWidthF(1)

        pen.setCosmetic(True)
        p.setPen(pen)
        brush = QBrush(QColor("Black"))
        transform = QTransform()
        transform.scale(1/p.scaling, 1/p.scaling)
        brush.setTransform(transform)
        brush.setStyle(Qt.BDiagPattern)

        p.setBrush(brush)
        p.drawRect(QRectF(self.x, self.y, self.w, self.h))
        p.restore()


class Node(StateObject):
    radius = 2
    support = False

    def __init__(self, x, y, support=0):
        super().__init__(x, y)
        self.support = bool(support)

    @classmethod
    def fromString(cls, line: str):
        cname, *params = line.split(" ")
        assert cname == cls.__name__

        return cls(*map(int, params))

    def toString(self):
        return " ".join(map(str, [self.__class__.__name__, int(self.x), int(self.y), int(self.support)]))

    def inClickArea(self, x, y):
        return math.hypot(self.x - x, self.y - y) <= self.radius

    def draw(self, p: QPainter):
        p.save()

        if self.selected:
            pen = QPen(QColor("red"))
            pen.setWidthF(2)
        else:
            pen = QPen(QColor("black"))
            pen.setWidthF(1)

        pen.setCosmetic(True)
        p.setPen(pen)
        brush = QBrush(QColor(255, 255, 255, 127))
        p.setBrush(brush)
        if self.support:
            p.drawRect(QRectF(self.x-self.radius, self.y -
                              self.radius, self.radius*2, self.radius*2))
        else:
            p.drawEllipse(QRectF(self.x-self.radius, self.y -
                                 self.radius, self.radius*2, self.radius*2))
        p.drawLine(QPointF(self.x - 0.5, self.y),
                   QPointF(self.x + 0.5, self.y))
        p.drawLine(QPointF(self.x, self.y - 0.5),
                   QPointF(self.x, self.y + 0.5))
        p.restore()


class Member():

    def __init__(self, a: Node, b: Node):
        self.a = a
        self.b = b

    @classmethod
    def fromString(cls, line: str, nodes):
        cname, *params = line.split(" ")
        assert cname == cls.__name__
        a, b = map(int, params)
        return cls(nodes[a], nodes[b])

    def toString(self, nodes):
        a = nodes.index(self.a)
        b = nodes.index(self.b)
        return " ".join(map(str, [self.__class__.__name__, int(a), int(b)]))

    def draw(self, p: QPainter):
        p.save()
        pen = QPen()
        pen.setWidthF(4)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.drawLine(self.a.x, self.a.y, self.b.x, self.b.y)
        p.restore()

    def connectedTo(self, node):
        return self.a == node or self.b == node

    def __eq__(self, other):
        return isinstance(other, type(self)) and \
            ((self.a == other.a and self.b == other.b) or
                (self.a == other.b and self.b == other.a))
