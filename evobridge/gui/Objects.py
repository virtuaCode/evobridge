import math

from PyQt5.QtCore import QPointF, QRectF, pyqtSlot, pyqtSignal, Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QTransform, QPainterPath
import random
from enum import Enum


class Material(Enum):
    STREET = 0
    WOOD = 1
    STEEL = 2


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

    def draw(self, p: QPainter, scale: float):
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

    def clone(self):
        return Rock(self.x, self.y, self.w, self.h)

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

    def draw(self, p: QPainter, scale: float):
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
        transform.scale(1/scale, 1/scale)
        brush.setTransform(transform)
        brush.setStyle(Qt.BDiagPattern)

        p.setBrush(brush)
        p.drawRect(QRectF(self.x, self.y, self.w, self.h))
        p.restore()


class Node(StateObject):
    radius = 2
    h_support = False
    v_support = False

    def __init__(self, x, y, h_support=0, v_support=0):
        super().__init__(x, y)
        self.h_support = bool(h_support)
        self.v_support = bool(v_support)

    def clone(self):
        return Node(self.x, self.y, h_support=self.h_support, v_support=self.v_support)

    @classmethod
    def fromString(cls, line: str):
        cname, *params = line.split(" ")
        assert cname == cls.__name__

        return cls(*map(int, params))

    def toString(self):
        return " ".join(map(str, [self.__class__.__name__, int(self.x), int(self.y), int(self.h_support), int(self.v_support)]))

    def inClickArea(self, x, y):
        return math.hypot(self.x - x, self.y - y) <= self.radius

    def draw(self, p: QPainter, scale: float):
        p.save()

        if self.selected:
            pen = QPen(QColor("red"))
            pen.setWidthF(2)
        else:
            pen = QPen(QColor("black"))
            pen.setWidthF(1)

        pen.setCosmetic(True)
        p.setPen(pen)
        brush = QBrush(QColor("black"))
        brush.setStyle(Qt.Dense7Pattern)
        brush.setTransform(QTransform().scale(1/scale, 1/scale))
        p.setBrush(brush)

        if self.v_support and self.h_support:
            p.drawRect(QRectF(self.x-self.radius, self.y -
                              self.radius, self.radius*2, self.radius*2))
        elif self.h_support:
            self.drawSupport(p, 3)
        elif self.v_support:
            self.drawSupport(p, 2)
        else:
            p.drawEllipse(QRectF(self.x-self.radius, self.y -
                                 self.radius, self.radius*2, self.radius*2))
        p.drawLine(QPointF(self.x - 0.5, self.y),
                   QPointF(self.x + 0.5, self.y))
        p.drawLine(QPointF(self.x, self.y - 0.5),
                   QPointF(self.x, self.y + 0.5))
        p.restore()

    def drawSupport(self, p: QPainter, dir=0):
        # 0 down, 1 left, 2 up, 3 right
        p.save()
        #p.setTransform(QTransform().translate(self.x, self.y).rotate(90*dir))
        # p.rotate(90*dir)
        p.translate(self.x, self.y)
        p.rotate(90*dir)
        path = QPainterPath()
        path.moveTo(self.radius, 0)
        path.lineTo(self.radius, -self.radius)
        path.lineTo(-self.radius, -self.radius)
        path.lineTo(-self.radius, 0)
        # path.cubicTo(self.x+self.radius, self.y+self.radius*, self.x -
        #             self.radius, self.y+self.radius*math.sqrt(2), self.x-self.radius, self.y)
        path.arcTo(-self.radius, -self.radius,
                   self.radius*2, self.radius*2, 180, 180)
        path.closeSubpath()
        p.drawPath(path)
        p.restore()


class Member():

    def __init__(self, a: Node, b: Node, material=Material.WOOD):
        self.a = a
        self.b = b
        self.material = material

    @classmethod
    def fromString(cls, line: str, nodes):
        cname, *params = line.split(" ")
        assert cname == cls.__name__
        a, b, m = map(int, params)
        return cls(nodes[a], nodes[b], Material(m))

    def toString(self, nodes):
        a = nodes.index(self.a)
        b = nodes.index(self.b)
        return " ".join(map(str, [self.__class__.__name__, int(a), int(b), self.material.value]))

    def draw(self, p: QPainter, scale: float):
        p.save()
        pen = QPen()

        if self.material == Material.STREET:
            pen.setColor(QColor("black"))
        elif self.material == Material.WOOD:
            pen.setColor(QColor("peru"))
        elif self.material == Material.STEEL:
            pen.setColor(QColor("brown"))

        pen.setWidthF(4)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.drawLine(self.a.x, self.a.y, self.b.x, self.b.y)
        p.restore()

    def connectedTo(self, node):
        return self.a == node or self.b == node

    def length(self):
        return math.hypot(self.a.x - self.b.x, self.a.y - self.b.y)

    def __eq__(self, other):
        return isinstance(other, type(self)) and \
            ((self.a == other.a and self.b == other.b) or
                (self.a == other.b and self.b == other.a))
