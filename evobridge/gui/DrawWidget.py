
from PyQt5.QtGui import (
    QColor, QKeyEvent, QMouseEvent, QPainter, QPen, QWheelEvent)
from PyQt5.QtWidgets import (QFileDialog, QFrame)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QPointF
from .State import State
from .Objects import Node, Rock, Material
import os
import math
from operator import sub


class DrawWidget(QFrame):
    onGridSizeChange = pyqtSignal(int)
    onCursorChange = pyqtSignal(float, float)
    onObjectChange = pyqtSignal(list)

    click_pos = None
    clicked_object = None
    move_objects = False
    control_pressed = False
    scroll_enabled = True
    zoom_enabled = True
    moved_rest = None
    mouse_moved = False
    left_clicked = False
    snap = 1
    min_move_distance = 10
    selected_material = Material.WOOD

    def __init__(self):
        QFrame.__init__(self)

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.setLineWidth(2)

        self.state = State()
        self.selectedObjects = []

        self.setMouseTracking(True)
        self.painter = QPainter()
        self.zoom = 0
        self.gridsize = 10
        self.every = int(
            max(1, self.gridsize * int(self.gridsize/self.scale())))
        self.x_offset = 0
        self.y_offset = 0

        self.startX = -1
        self.startY = -1

        self.onGridSizeChange.emit(self.every)

    @pyqtSlot()
    def addNewNode(self):
        x, y = self.center()
        self.state.addNode(Node(x, y))
        self.repaint()

    @pyqtSlot()
    def addNewRock(self):
        x, y = self.center()
        self.state.addRock(Rock(x, y, 40, 40))
        self.repaint()

    @pyqtSlot()
    def toggleWood(self):
        self.selected_material = Material.WOOD

    @pyqtSlot()
    def toggleSteel(self):
        self.selected_material = Material.STEEL

    @pyqtSlot()
    def toggleStreet(self):
        self.selected_material = Material.STREET

    def setState(self, state: State):
        self.state = state
        self.selectedObjects.clear()
        self.emitSelectedObjects()
        self.repaint()

    def center(self):
        width, height = self.size().width(), self.size().height()
        return self.posToWorldPos(width/2, height/2)

    def paintEvent(self, event):

        p = self.painter
        p.begin(self)

        p.translate(QPointF(self.x_offset, self.y_offset))
        scale = 2**self.zoom
        p.scale(scale, scale)

        self.drawGrid(p)
        self.state.draw(p, scale)

        p.end()

        super().paintEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if self.zoom_enabled:
            delta = event.angleDelta().y()
            curscale = 2**self.zoom

            mx_before = (event.x() - self.x_offset) / curscale
            my_before = (event.y() - self.y_offset) / curscale

            self.zoom += delta / 240
            scale = 2**self.zoom

            mx_after = (event.x() - self.x_offset) / scale
            my_after = (event.y() - self.y_offset) / scale

            self.x_offset -= (mx_before - mx_after) * scale
            self.y_offset -= (my_before - my_after) * scale

            self.every = int(max(1, self.gridsize * int(self.gridsize/scale)))
            self.onGridSizeChange.emit(self.every)
        self.repaint()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Delete:
            for o in self.selectedObjects:
                self.state.removeObject(o)
            self.selectedObjects.clear()

            self.repaint()

    def posToWorldPos(self, x, y):
        scale = 2**self.zoom
        return ((x - self.x_offset) / scale, (y - self.y_offset) / scale)

    def worldPosToPos(self, x, y):
        scale = 2**self.zoom
        return (x * scale + self.x_offset, y * scale + self.y_offset)

    def mousePressEvent(self, event: QMouseEvent):
        self.setFocus(Qt.ActiveWindowFocusReason)

        self.prevX = event.x()
        self.prevY = event.y()

        if event.buttons() & Qt.LeftButton:
            self.zoom_enabled = False
            self.left_clicked = True

            (mx, my) = self.posToWorldPos(event.x(), event.y())

            obj = self.state.getClickedObject(mx, my)
            self.clicked_object = obj

            if obj is not None:
                self.scroll_enabled = False
                if event.modifiers() & Qt.ControlModifier:
                    self.control_pressed = True
                    if obj in self.selectedObjects:
                        obj.selected = False
                        self.selectedObjects.remove(obj)
                    else:
                        obj.selected = True
                        self.selectedObjects.append(obj)
                else:
                    if obj not in self.selectedObjects:
                        for o in self.selectedObjects:
                            o.selected = False

                        self.selectedObjects.clear()
                        obj.selected = True
                        self.selectedObjects.append(obj)

                self.click_pos = (event.x(), event.y())

        elif event.buttons() & Qt.RightButton:
            (mx, my) = self.posToWorldPos(event.x(), event.y())

            obj = self.state.getClickedObject(mx, my)

            if isinstance(obj, Node):
                for o in self.selectedObjects:
                    if isinstance(o, Node):
                        self.state.toggleMember(obj, o, self.selected_material)

        self.emitSelectedObjects()
        self.repaint()

    def mouseReleaseEvent(self, event):
        if self.clicked_object is not None:
            if not self.move_objects:
                if self.control_pressed:
                    pass
                else:
                    for o in self.selectedObjects:
                        o.selected = False

                    self.selectedObjects.clear()
                    self.clicked_object.selected = True
                    self.selectedObjects.append(self.clicked_object)
        elif not self.mouse_moved and self.left_clicked:
            for o in self.selectedObjects:
                o.selected = False
                self.selectedObjects.clear()

        self.clicked_object = None
        self.move_objects = False
        self.control_pressed = False
        self.scroll_enabled = True
        self.zoom_enabled = True
        self.mouse_moved = False
        self.left_clicked = False

        self.repaint()

    def mouseMoveEvent(self, event: QMouseEvent):
        repaint = False
        mx, my = self.posToWorldPos(event.x(), event.y())

        self.onCursorChange.emit(mx, my)

        if self.scroll_enabled and (event.buttons() & Qt.MiddleButton):
            self.x_offset -= self.prevX - event.x()
            self.y_offset -= self.prevY - event.y()
            repaint = True

        if self.clicked_object is None:
            if self.scroll_enabled and (event.buttons() & Qt.LeftButton):
                self.mouse_moved = True
                self.x_offset -= self.prevX - event.x()
                self.y_offset -= self.prevY - event.y()
                repaint = True

        else:
            if not self.move_objects:
                x, y = self.click_pos
                distance = math.hypot(x-event.x(), y-event.y())

                if distance > self.min_move_distance:
                    self.move_objects = True
                    click_x, click_y = self.posToWorldPos(x, y)
                    self.moved_rest = (mx - click_x, my - click_y)
            else:
                rest_x, rest_y = self.moved_rest
                cur_pos = self.posToWorldPos(event.x(), event.y())
                prev_pos = self.posToWorldPos(self.prevX, self.prevY)
                nx, ny = tuple(map(sub, cur_pos, prev_pos))

                snap = self.snap

                move_x, move_y = (rest_x + nx) // snap, (rest_y + ny) // snap

                for o in self.selectedObjects:
                    o.x += move_x * snap
                    o.y += move_y * snap

                self.moved_rest = ((rest_x + nx) % snap, (rest_y + ny) % snap)

                self.emitSelectedObjects()
                repaint = True
            #self.scroll_view = True

        self.prevX = event.x()
        self.prevY = event.y()

        if repaint:
            self.repaint()

    def emitSelectedObjects(self):
        self.onObjectChange.emit(self.selectedObjects)

    def drawGrid(self, p: QPainter):
        p.save()

        pen = QPen()
        pen.setColor(QColor(184, 184, 184))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(1)
        pen.setCosmetic(True)
        p.setPen(pen)

        p.drawLine(255, 0, 255, 255)
        p.drawLine(0, 255, 255, 255)

        pen.setStyle(Qt.SolidLine)
        p.setPen(pen)

        for x in range(255):
            if x % self.every == 0:
                p.drawLine(x, 0, x, 255)
        for y in range(255):
            if y % self.every == 0:
                p.drawLine(0, y, 255, y)

        p.restore()

    def scale(self):
        return 2**self.zoom
