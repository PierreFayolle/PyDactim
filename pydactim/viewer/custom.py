from PySide6.QtWidgets import QLabel, QFrame, QScrollArea, QMenu, QMessageBox, QCheckBox, QComboBox
from PySide6.QtCore import (
    Qt, QEvent, QSize, QPoint, QPointF, QRectF, QRect, 
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QBrush, QPen, QPaintEvent, QCursor

from scipy.interpolate import interp2d
import nibabel as nib
import numpy as np

from pydactim.viewer.utils import create_lut
from pydactim.viewer.settings import *

class ThumbnailFrame(QFrame):
    def __init__(self, path, replace_data, create_overlay):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.path = path
        self.replace_data = replace_data
        self.create_overlay = create_overlay

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.installEventFilter(self)  # Install event filter to handle events
        self.setCursor(Qt.PointingHandCursor)  # Set the cursor to pointing hand by default

    def showContextMenu(self, pos):
        menu = QMenu(self)

        action1 = menu.addAction("Replace image")
        action2 = menu.addAction("Add as overlay")
        
        action = menu.exec_(self.mapToGlobal(pos))
        
        if action == action1:
            self.replace_data(self.path)
        elif action == action2:
            self.create_overlay(self.path)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                self.replace_data(self.path)

        return super().eventFilter(obj, event)
    
    def enterEvent(self, event):
        # Set the cursor to pointing hand when entering the frame
        self.setCursor(Qt.PointingHandCursor)
        event.accept()

    def leaveEvent(self, event):
        # Restore the cursor to the default arrow when leaving the frame
        self.setCursor(Qt.ArrowCursor)
        event.accept()

class CustomLabel(QLabel):
    def __init__(self, path, view_data, update_slice, update_aif, axis, size=(1000,1000)):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)  # Center the image in the label
        self.setMinimumSize(size[0], size[1]) # Set the minimum size for the QLabel
        self.setStyleSheet("background-color: #000000;")
        self.setStyleSheet(f"border: 1px solid {BORDER_COLOR};")
        self.installEventFilter(self)  # Install event filter to handle events
        self.setMouseTracking(True)
        
        self.path = path
        self.data = view_data.data
        self.slice_index = view_data.slice
        self.pixdim = view_data.pixdim
        self.z = view_data.z
        self.axis = axis
        self.state = "3D"

        self.drag_start = QPoint()  # Store the starting point of the drag
        self.dragging = False  # Flag to track whether dragging is in progress
        
        self.contrast_adjustment_start = False
        self.contrast_min = self.data.min()  # Minimum value for contrast windowing
        self.contrast_max = self.data.max()  # Maximum value for contrast windowing+
        self.window_center = int((self.contrast_min + self.contrast_max) / 2)  # Initial contrast center
        self.window_width = int(self.contrast_max - self.contrast_min)  # Initial contrast width

        self.zoom_factor = 1.0  # Initial zoom factor
        self.zooming = False  # Flag to track whether zooming is in progress
        
        self.panning = False
        self.pan_start = QPoint()
        self.pan_x = 0  # Initialize pan_x to 0
        self.pan_y = 0  # Initialize pan_y to 0

        self.mosaic = False

        self.mouse_current = 0
        self.mouse_threshold = 1

        self.overlay = False
        self.overlay_data = None
        self.overlay_opacity = 50

        self.roiable = True
        self.roiing = False
        self.roi_start = QPoint()
        self.roi_x = 500
        self.roi_y = 500
        self.roi_width = 30

        self.update_slice = update_slice
        self.update_aif = update_aif
        self.selected_widget = "info_widget"

    def set_overlay(self, data, pixdim, lut):
        self.overlay = True
        self.overlay_data = data
        self.overlay_pixdim = pixdim
        self.overlay_lut = create_lut(lut)
        self.update_image()

    def update_image(self):
        current_data = self.data
        current_data = np.clip(current_data, self.window_center - self.window_width / 2, self.window_center + self.window_width / 2)
        current_data = (current_data - (self.window_center - self.window_width / 2)) / (self.window_width + 1e-12) * 255

        current_data = current_data.astype('uint8')
        current_data = self.lut[current_data]
        current_data = np.ascontiguousarray(current_data)

        # Create a QImage from the slice data
        self.qimage = QImage(current_data, current_data.shape[1], current_data.shape[0], current_data.shape[1]*3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(self.qimage)

        pixel_width, pixel_height, _ = self.pixdim
        scaled_width = int(pixmap.width() * pixel_width)
        scaled_height = int(pixmap.height() * pixel_height)
        pixmap = pixmap.scaled(scaled_width, scaled_height)

        # # Resize the pixmap to fit the application window with some scaling factor (e.g., 2 times larger)
        self.scale_factor = 1 / self.zoom_factor
        pixmap = pixmap.scaled(self.size() * self.scale_factor, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        if self.state == "3D":
            if pixmap.width() > THREED_WIDTH or pixmap.height() > THREED_HEIGHT:
                if pixmap.width() > pixmap.height():
                    new_width = THREED_WIDTH
                    new_height = int(pixmap.height() * (THREED_WIDTH / pixmap.width()))
                else:
                    new_height = THREED_HEIGHT
                    new_width = int(pixmap.width() * (THREED_HEIGHT / pixmap.height()))

                pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            pixmap = pixmap.scaled(ONED_WIDTH, ONED_HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        if self.overlay:
            current_overlay = self.overlay_data
            current_overlay = np.clip(current_overlay, self.window_center - self.window_width / 2, self.window_center + self.window_width / 2)
            current_overlay = (current_overlay - (self.window_center - self.window_width / 2)) / (self.window_width + 1e-12) * 255

            current_overlay = current_overlay.astype('uint8')
            current_overlay = self.overlay_lut[current_overlay]
            current_overlay = np.ascontiguousarray(current_overlay)

            current_overlay = QImage(current_overlay.data, current_overlay.shape[1], current_overlay.shape[0], current_overlay.shape[1]*3, QImage.Format_RGB888)
            pixel_width, pixel_height, _ = self.overlay_pixdim
            scaled_width = int(current_overlay.width() * pixel_width)
            scaled_height = int(current_overlay.height() * pixel_height)
            current_overlay = current_overlay.scaled(scaled_width, scaled_height)
            current_overlay = current_overlay.scaled(pixmap.size(), Qt.KeepAspectRatio)
            
            painter = QPainter()
            painter.begin(pixmap)
            painter.setOpacity(self.overlay_opacity / 100)
            painter.drawImage(0, 0, current_overlay)
            painter.end()

        if "perf" in self.path and self.roiable and self.selected_widget == "perf_widget":
            self.painterInstance = QPainter(pixmap)
            self.penRectangle = QPen(Qt.red)
            self.penRectangle.setWidth(3)

            # draw rectangle on painter
            self.painterInstance.setPen(self.penRectangle)
            self.painterInstance.drawRect(self.roi_x, self.roi_y,self.roi_width, self.roi_width)
            self.painterInstance.end()
            
        self.pixmap = pixmap
        self.setPixmap(self.pixmap)
    
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.RightButton:
                # Start image dragging when right button is pressed
                self.drag_start = event.position()
                self.dragging = True
            elif event.button() == Qt.LeftButton:
                # Start image dragging when left button is pressed
                self.drag_start = event.position()
                label_width = self.width()
                x_pos = event.position().x()
                y_pos = event.position().y()
                # if 0 < x_pos < 0.1 * label_width or 0.9 * label_width < x_pos < label_width:
                #     self.zooming = True
                if x_pos > self.roi_x and x_pos < self.roi_x + self.roi_width:
                    self.roiing = True
                    self.roi_start = event.position()
                    self.roi_start_x = self.roi_x
                    self.roi_start_y = self.roi_y
                # else:
                #     self.panning = True
                #     self.pan_start = event.position()
                #     self.pan_start_x = self.pan_x
                #     self.pan_start_y = self.pan_y
            elif event.button() == Qt.MiddleButton:
                # Start contrast adjustment when middle button is pressed
                self.drag_start = event.position()
                self.contrast_adjustment_start = self.window_center, self.window_width

        # MOUSE RELEASE EVENT
        elif event.type() == QEvent.Type.MouseButtonRelease:
            self.mouse_current = 0
            if event.button() == Qt.RightButton:
                # End image dragging when right button is released
                self.dragging = False
            elif event.button() == Qt.LeftButton:
                if self.zooming:
                    self.zooming = False
                elif self.panning:
                    self.panning = False
                elif self.roiing:
                    self.roiing = False
            elif event.button() == Qt.MiddleButton:
                # End contrast adjustment when left button is released
                self.contrast_adjustment_start = None

        # MOUSE WHEEL SCROLL EVENT
        elif event.type() == QEvent.Type.Wheel:
            # Handle wheel event for scrolling
            num_degrees = event.angleDelta().y() / 8
            num_steps = num_degrees / 15
            new_index = self.slice_index + num_steps
            new_index = max(0, min(new_index, self.z - 1))
            if new_index != self.slice_index:
                self.update_slice(int(new_index), self.axis)
                self.update_image()
                self.update_aif()

        # MOUSE RIGHT CLICK FOR SLICES
        elif event.type() == QEvent.Type.MouseMove and self.dragging:
            # Handle image dragging when right button is pressed and mouse is moved
            delta = self.drag_start - event.position()
            num_steps = delta.y()
            new_index = self.slice_index + num_steps 
            new_index = max(0, min(new_index, self.z - 1))
            if new_index != self.slice_index:
                self.update_slice(int(new_index), self.axis)
                self.update_image()
                self.update_aif()
            self.drag_start = event.position()

            # delta = self.drag_start - event.position()
            # self.mouse_current += delta.y()
            # print(f"{self.mouse_current =}")
            # if self.mouse_current >= self.mouse_threshold:
            #     print(f"++++ {self.slice_index =}; {self.z =}")
            #     self.slice_index = min(self.slice_index + 1, self.z - 1)
            #     self.update_slice(self.slice_index, self.axis)
            #     print(f"New {self.slice_index =}")
            #     self.update_image()
            #     self.update_aif()
            #     self.mouse_current = 0
            # elif self.mouse_current <= -self.mouse_threshold:
            #     print(f"----{self.slice_index =}; {self.z =}")
            #     self.slice_index = max(0, self.slice_index - 1)
            #     print(f"New {self.slice_index =}")
            #     self.update_slice(self.slice_index, self.axis)
            #     self.update_image()
            #     self.update_aif()
            #     self.mouse_current = 0
            # self.drag_start = event.position()

        # MOUSE WHEEL CLICK FOR WINDOWING
        elif event.type() == QEvent.Type.MouseMove and self.contrast_adjustment_start:
            # Handle contrast adjustment when left button is pressed and mouse is moved
            delta = self.drag_start - event.position()
            num_steps_x = int(delta.x() * 1.1)
            num_steps_y = int(delta.y() * 1.1)

            # Calculate new window center and width based on the mouse movement
            self.window_width = int(self.window_width - num_steps_x) 
            if self.window_width < 1: self.window_width = 1
            self.window_center = int(self.window_center + num_steps_y)
            if self.window_center < 0: self.window_center = 0

            self.drag_start = event.position()
            self.update_image()

        # # MOUSE LEFT CLICK FOR ZOOMING
        elif event.type() == QEvent.Type.MouseMove and self.roiing and self.roiable:
            delta = event.position() - self.roi_start
            self.roi_x = self.roi_start_x - (-1*delta.x())
            self.roi_y = self.roi_start_y - (-1*delta.y())
            self.update_image()
            self.update_aif()

        # elif event.type() == QEvent.Type.MouseMove and self.zooming:
        #     # Handle zooming when left button is pressed and mouse is moved
        #     label_width = self.width()

        #     delta = self.drag_start - event.position()
        #     self.zoom_factor += delta.y() * -0.005

        #     # Calculate the new zoom factor based on the mouse movement
        #     if self.zoom_factor > 1.0: self.zoom_factor = 1.0
        #     if self.zoom_factor < 0.3: self.zoom_factor = 0.3

        #     self.drag_start = event.position()
        #     self.update_image()

        # # MOUSE LEFT CLICK FOR PANNING
        # elif event.type() == QEvent.Type.MouseMove and self.panning:
        #     # Handle panning when left button is pressed and mouse is moved
        #     delta = event.position() - self.pan_start
        #     self.pan_x = self.pan_start_x - (-1*delta.x())
        #     self.pan_y = self.pan_start_y - (-1*delta.y())
        #     self.update_image()

        # if event.type() == QEvent.Enter:
        #     x = event.pos().x()
        #     y = event.pos().y()
        #     print("Mouse is over the label", self.pixmap.pixel(x,y))
            
        # elif event.type() == QEvent.Leave:
        #     print("Mouse is not over the label", event.pos())

        return super().eventFilter(obj, event)

    def enterEvent(self, event):
        self.setStyleSheet(f"border: 1px solid {TEXT_COLOR};")

    # def mouseMoveEvent(self, event):
    #     # Get the mouse position relative to the top-left corner of the label
    #     global_mouse_pos = QCursor.pos()

    #     # Convert global mouse position to local coordinates of the label
    #     local_mouse_pos = self.mapFromGlobal(global_mouse_pos)
    #     x = local_mouse_pos.x()
    #     y = local_mouse_pos.y()
        # Do something with the mouse position
        # print("################################")
        # print(f"{self.axis}: ({local_mouse_pos.x()}, {local_mouse_pos.y()})")
        # print(QColor(self.qimage.pixel(x,y)).getRgb())
        

    def leaveEvent(self, event):
        self.setStyleSheet(f"border: 1px solid {LIGHT_BORDER_COLOR};")

class ScrollArea(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(SCROLL_WIDTH, WINDOW_HEIGHT)
        self.setMaximumSize(SCROLL_WIDTH, WINDOW_HEIGHT)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Hide horizontal scrollbar
        self.setWidgetResizable(True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            event.ignore()
        elif event.key() == Qt.Key_Left:
            event.ignore()
        else:
            return super().keyPressEvent(event)

class ComboBox(QComboBox):
    def __init__(self):
        super().__init__()

    def keyPressEvent(self, event):
        if event.key():
            event.ignore()
        else:
            super().keyPressEvent(event)

class AnimatedToggle(QCheckBox):
    def __init__(self, toggle_overlay, width=40):
        QCheckBox.__init__(self)

        self.setFixedSize(width, 22)
        self.setCursor(Qt.PointingHandCursor)

        self._bg_color = LIGHT_BORDER_COLOR
        self._circle_color = TEXT_COLOR
        self._active_color = VERY_LIGHT_BG_COLOR

        self.toggle_overlay = toggle_overlay
        self.setChecked(True)

        self.stateChanged.connect(self.toggle_overlay)

    def debug(self):
        print(f"Status: {self.isChecked()}")

    def hitButton(self, pos: QPoint):
        return self.contentsRect().contains(pos)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)

        height = 20

        rect = QRect(0,0,self.width(), height)

        if not self.isChecked():
            p.setBrush(QColor(self._bg_color))
            p.drawRoundedRect(0,0,rect.width(), height, height / 2, height / 2)

            p.setBrush(QColor(self._circle_color))
            p.drawEllipse(3,2,18,18)
        else:
            p.setBrush(QColor(self._active_color))
            p.drawRoundedRect(0,0,rect.width(), height, height / 2, height / 2)

            p.setBrush(QColor(self._circle_color))
            p.drawEllipse(self.width() - 20,1,18,18)

        p.end()