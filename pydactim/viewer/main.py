import sys
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication, QPushButton, QSlider, QComboBox, QToolButton, QLineEdit, QLabel, QMainWindow, QCheckBox, QScrollArea, QStackedLayout, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QAction, QGuiApplication, QIcon
from PySide6.QtCore import Qt, QEvent, QRect, QSize
import pyqtgraph as pg
import matplotlib.pyplot as plt
import glob
from io import StringIO

from pydactim.transformation import (n4_bias_field_correction,
    registration, apply_transformation, resample, histogram_matching, 
    apply_mask, substract, susan, normalize, crop, apply_crop, copy_affine, skull_stripping, 
    remove_small_object, tissue_classifier, add_tissue_class, prediction_glioma, uncertainty_prediction_glioma)
from pydactim.sorting import sort_dicom
from pydactim.conversion import convert_dicom_to_nifti
from pydactim.viewer.custom import CustomLabel, ThumbnailFrame, ScrollArea, ComboBox, AnimatedToggle
from pydactim.viewer.view_panel import ViewPanel
from pydactim.viewer.utils import create_thumbnail, get_darkModePalette, reset_layout
from pydactim.viewer.settings import *

class NiftiViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        print("INFO - Initializing Pydactim Viewer")
        self.setWindowTitle("PyDactim Viewer")

        # self.nifti_found = glob.glob('./**/*.nii.gz', recursive=True)
        self.init_tabs()
        # self.init_all()
        # self.open_volume()
        self.dynamic_windows = []

    def init_all(self):
        self.filename = self.nifti_found[0]

        print("INFO - Creating UI")
        self.init_ui()
        print("INFO - Loading images")
        self.load_sequences(self.nifti_found)

        self.lut = None
        self.overlay_lut = None

        self.overlay = False
        self.overlay_path = None

        sys.stdout = CustomOutputStream(self.log_label, self.log_scroll)

    def init_tabs(self):
        print("INFO - Creating menu tab")
        # Add a menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        image_menu = menubar.addMenu("Image")
        transform_menu = menubar.addMenu("Transforms")
        view_menu = menubar.addMenu("View")

        # Add actions to the File menu
        action = QAction("Open directory", self)
        action.triggered.connect(self.open_volume)
        file_menu.addAction(action)

        action = QAction("Add file", self)
        action.triggered.connect(self.add_file)
        file_menu.addAction(action)

        action = QAction("Screenshot", self)
        action.triggered.connect(self.screenshot)
        file_menu.addAction(action)

        action = QAction("Sort DICOM directory", self)
        action.triggered.connect(lambda x: self.transforms("Sort DICOM"))
        edit_menu.addAction(action)

        action = QAction("Convert DICOM to NIfTI", self)
        action.triggered.connect(lambda x: self.transforms("Convert DICOM"))
        edit_menu.addAction(action)

        # Add actions to the Tool menu
        action = QAction("Histogram whole image", self)
        action.triggered.connect(self.whole_histogram)
        image_menu.addAction(action)

        action = QAction("Histogram current image", self)
        action.triggered.connect(self.add_current_histogram)
        image_menu.addAction(action)

        # action = QAction("Reset image", self)
        # action.triggered.connect(self.reset_image)
        # image_menu.addAction(action)

        action = QAction("Reset windowing", self)
        action.triggered.connect(self.reset_contrast)
        image_menu.addAction(action)

        # Add actions to the View menu
        action = QAction("Axial view", self)
        action.triggered.connect(lambda x: self.toggle_view("Axial"))
        view_menu.addAction(action)

        action = QAction("Coronal view", self)
        action.triggered.connect(lambda x: self.toggle_view("Coronal"))
        view_menu.addAction(action)

        action = QAction("Sagittal view", self)
        action.triggered.connect(lambda x: self.toggle_view("Sagittal"))
        view_menu.addAction(action)

        action = QAction("Mosaic view", self)
        action.triggered.connect(lambda x: self.toggle_view("Mosaic"))
        view_menu.addAction(action)

        # if not self.ensure_axial:
        action = QAction("3D view", self)
        action.triggered.connect(lambda x: self.toggle_view("3D"))
        view_menu.addAction(action)

        action = QAction("Auto Crop", self)
        action.triggered.connect(lambda x: self.transforms("Auto crop"))
        transform_menu.addAction(action)

        action = QAction("Apply Crop", self)
        action.triggered.connect(lambda x: self.transforms("Apply crop"))
        transform_menu.addAction(action)

        action = QAction("Apply mask", self)
        action.triggered.connect(lambda x: self.transforms("Apply mask"))
        transform_menu.addAction(action)

        action = QAction("Resample voxels", self)
        action.triggered.connect(lambda x: self.transforms("Resample voxels"))
        transform_menu.addAction(action)

        action = QAction("Dilate", self)
        action.triggered.connect(lambda x: self.transforms("Dilate"))
        transform_menu.addAction(action)

        action = QAction("Registration", self)
        action.triggered.connect(lambda x: self.transforms("Registration"))
        transform_menu.addAction(action)

        action = QAction("Susan", self)
        action.triggered.connect(lambda x: self.transforms("Susan"))
        transform_menu.addAction(action)

        action = QAction("Skull stripping", self)
        action.triggered.connect(lambda x: self.transforms("Skull stripping"))
        transform_menu.addAction(action)

    def init_ui(self):
        print("INFO - Creating left panel")
        # Sequences layout
        self.scroll_area = ScrollArea()

        print("INFO - Creating right panel")

        # ICON GUI
        self.selected_widget = "info_widget"
        self.icon_widget = QWidget()
        self.icon_layout = QHBoxLayout()

        self.icon_info_button = QToolButton()
        self.icon_info_button.setIcon(QIcon("info.png"))
        self.icon_info_button.setText("Image details")
        self.icon_info_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.icon_info_button.setIconSize(QSize(50,50))
        self.icon_info_button.setCursor(Qt.PointingHandCursor)
        self.icon_info_button.clicked.connect(self.show_info)

        self.icon_perf_button = QToolButton()
        self.icon_perf_button.setIcon(QIcon("perf.png"))
        self.icon_perf_button.setText("Perfusion Maps")
        self.icon_perf_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.icon_perf_button.setIconSize(QSize(50,50))
        self.icon_perf_button.setCursor(Qt.PointingHandCursor)
        self.icon_perf_button.clicked.connect(self.show_perf)

        self.icon_layout.addWidget(self.icon_info_button)
        self.icon_layout.addWidget(self.icon_perf_button)
        self.icon_layout.addStretch()
        self.icon_widget.setLayout(self.icon_layout)

        # INFO GUI
        self.info_widget = QWidget()
        self.info_layout = QVBoxLayout()

        self.text_label = QLabel(alignment=Qt.AlignmentFlag.AlignTop)
        self.text_label.setStyleSheet(f"background-color: {LIGHT_BG_COLOR}")
        self.text_label.setStyleSheet("border: 1px solid white")
        self.text_label.setStyleSheet("padding: 0px 15px")
        self.text_label.setMinimumSize(INFO_WIDTH, 0)

        self.plot_aif = pg.PlotWidget()
        self.plot_aif.getPlotItem().hideAxis('left')

        self.lut_widget = QWidget()
        self.lut_layout = QVBoxLayout()
        self.lut_label = QLabel("Look-up-table")
        self.lut_button = ComboBox()
        self.lut_button.addItems(['Grayscale', 'Red', 'Green', 'Blue', 'Rainbow', 'Spectral', 'Viridis', 'Qualitative'])
        self.lut_button.currentTextChanged.connect(self.change_lut)
        self.lut_layout.addWidget(self.lut_label)
        self.lut_layout.addWidget(self.lut_button)
        self.lut_widget.setLayout(self.lut_layout)

        self.log_title = QLabel("<h2><b>Log:</b></h2>")
        self.log_label = QLabel()
        self.log_label.setAlignment(Qt.AlignTop)
        self.log_scroll = QScrollArea()
        self.log_scroll.setStyleSheet(f"background-color: black; color: {TEXT_COLOR};")
        self.log_scroll.setWidget(self.log_label)
        self.log_scroll.setFixedHeight(300)
        self.log_scroll.setWidgetResizable(True)
        self.log_layout = QVBoxLayout()
        self.log_layout.addStretch()
        self.log_layout.addWidget(self.log_title)
        self.log_layout.addWidget(self.log_scroll)
        self.log_widget = QWidget()
        self.log_widget.setLayout(self.log_layout)

        self.info_layout.addWidget(self.text_label)
        self.info_layout.addWidget(self.lut_widget)
        self.info_layout.addStretch()
        self.info_layout.addWidget(self.log_widget)
        self.info_widget.setLayout(self.info_layout)

        # PERF GUI
        self.perf_widget = QWidget()
        self.perf_layout = QVBoxLayout()
        self.plot_aif = pg.PlotWidget()
        self.plot_aif.getPlotItem().hideAxis('left')
        self.perf_layout.addWidget(self.plot_aif)
        self.perf_widget.setLayout(self.perf_layout)

        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(self.icon_widget)
        self.right_layout.addWidget(self.info_widget)
        self.right_layout.addWidget(self.perf_widget)
        self.right_layout.addStretch()
        self.perf_widget.hide()
 
        self.right_widget = QWidget()
        self.right_widget.setLayout(self.right_layout)

        print("INFO - Creating middle panel")
        # View label that contains the image label(s)
        self.view_widget = ViewPanel(self.filename, self.update_text, self.plot_aif)
        self.view_widget.update_text()

        print("INFO - Creating main layout")
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.scroll_area)
        self.main_layout.addWidget(self.view_widget)
        self.main_layout.addWidget(self.right_widget)

        central_widget = QWidget(self)
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        print("INFO - Adding style sheet")
        # self.update_image()
        # self.create_histogram()
        self.setStyleSheet(
            """
            QWidget {
                background-color: $BG_COLOR;
            }
          
            QScrollArea {
                background-color: $BG_COLOR;
                border: 0px solid $BG_COLOR;
            }
            QScrollBar:vertical {
                border: 0px solid $BG_COLOR;
                background: $LIGHT_BG_COLOR;
                width:10px;    
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {         
        
                min-height: 0px;
                border: 0px solid red;
                border-radius: 4px;
                background-color: $BORDER_COLOR;
            }
            QScrollBar::add-line:vertical {       
                height: 0px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }
            QScrollBar::sub-line:vertical {
                height: 0 px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }
            QToolButton::hover{
                background-color: #000000;
            }
            """.replace('$BG_COLOR', BG_COLOR)\
            .replace('$LIGHT_BG_COLOR', LIGHT_BG_COLOR)\
            .replace('$BORDER_COLOR', BORDER_COLOR)
        )

    def show_info(self):
        self.perf_widget.hide()
        self.info_widget.show()
        self.selected_widget = "info_widget"
        self.view_widget.axial_label.selected_widget = self.selected_widget
        self.view_widget.axial_label.update_image()

    def show_perf(self):
        self.info_widget.hide()
        self.perf_widget.show()
        self.selected_widget = "perf_widget"
        self.view_widget.axial_label.selected_widget = self.selected_widget
        self.view_widget.axial_label.update_image()

    def open_volume(self):
        self.dirname = QFileDialog.getExistingDirectory(self, "Select a directory", "C:\\", QFileDialog.ShowDirsOnly)
        # self.dirname = "D:/Studies/GLIOBIOPSY/data/sub-018/"
        self.nifti_found = glob.glob(os.path.join(self.dirname, '**/*.nii.gz'), recursive=True)
        self.init_all()

    def add_file(self):
        new_path = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "${HOME}",
            "NIfTI Compressed Files (*.nii.gz);; NIfTI Files (*.nii)",
        )[0]
        if new_path not in self.nifti_found:
            self.load_sequence(new_path)
            self.nifti_found.append(new_path)
            self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")

    def load_sequences(self, paths):
        self.thumbnail_widget = QWidget()
        self.thumbnail_layout = QVBoxLayout()
        self.thumbnail_layout.addStretch()

        self.thumbnail_title = QLabel()
        self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(paths)} images loaded:</h2>")
        self.thumbnail_layout.addWidget(self.thumbnail_title)
        for path in paths:
            self.load_sequence(path)

        self.thumbnail_widget.setLayout(self.thumbnail_layout)
        self.scroll_area.setWidget(self.thumbnail_widget)
        print("INFO - UI fully created")

    def load_sequence(self, path):
            print(f"INFO - Found:\t{path}")
            frame = ThumbnailFrame(path, self.replace_data, self.create_overlay)
            label_image = QLabel()

            thumbnail, shape, pixdim = create_thumbnail(path)
            thumbnail = (thumbnail - thumbnail.min()) / (thumbnail.max() - thumbnail.min()) * 255
            thumbnail = thumbnail.astype('uint8')
            image = QImage(thumbnail.data, thumbnail.shape[1], thumbnail.shape[0], thumbnail.shape[1], QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(image)
            pixel_width, pixel_height, _ = pixdim
            scaled_width = int(pixmap.width() * pixel_width)
            scaled_height = int(pixmap.height() * pixel_height)
            pixmap = pixmap.scaled(scaled_width, scaled_height)
            pixmap = pixmap.scaledToWidth(THUMBNAIL_WIDTH, Qt.SmoothTransformation)
            label_image.setPixmap(pixmap)

            label_info = QLabel()
            path = os.path.basename(path).split(".nii")[0]
            if len(path) > 32: path = path[:32] + "..."
            pixdim = [round(pixdim[0], 2), round(pixdim[1], 2), round(pixdim[2], 2)]
            label_info.setText(f"{path}<br>Shape: {shape}<br>Pixdim: {pixdim}<hr style='background-color: {LIGHT_BORDER_COLOR}'>")

            label_info.setStyleSheet("border: 0px solid #000000;")
            label_image.setStyleSheet("border: 0px solid #000000;")

            temp_layout = QVBoxLayout(frame)
            temp_layout.addWidget(label_image)
            temp_layout.addWidget(label_info)

            self.thumbnail_layout.addWidget(frame)

    def replace_data(self, path):
        self.view_widget.init(path)
        self.view_widget.update_text()
        self.remove_overlay()

    def create_overlay(self, path):
        if self.overlay: self.remove_overlay()
        self.overlay = True
        self.overlay_path = path

        self.overlay_label = QLabel("<h2><b>Overlay:</b></h2>")

        self.overlay_toggle = QWidget()
        self.overlay_toggle_layout = QHBoxLayout()
        self.overlay_toggle_label = QLabel("Toggle overlay visibility")
        self.overlay_toggle_button = AnimatedToggle(self.toggle_overlay_visibility)
        self.overlay_toggle_layout.addWidget(self.overlay_toggle_label)
        self.overlay_toggle_layout.addWidget(self.overlay_toggle_button)
        self.overlay_toggle.setLayout(self.overlay_toggle_layout)
        
        self.overlay_slider = QWidget()
        self.overlay_slider_layout = QHBoxLayout()
        self.overlay_slider_label = QLabel("Overlay opacity")
        self.overlay_slider_button = QSlider(Qt.Orientation.Horizontal, self)
        self.overlay_slider_button.setGeometry(50,50, 200, 50)
        self.overlay_slider_button.setMinimum(0)
        self.overlay_slider_button.setMaximum(100)
        self.overlay_slider_button.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.overlay_slider_button.setValue(50)
        self.overlay_slider_button.setTickInterval(1)
        self.overlay_slider_button.valueChanged.connect(self.change_overlay_opacity) 
        self.overlay_slider_value = QLabel("50%")
        self.overlay_slider_layout.addWidget(self.overlay_slider_label)
        self.overlay_slider_layout.addWidget(self.overlay_slider_button)
        self.overlay_slider_layout.addWidget(self.overlay_slider_value)
        self.overlay_slider.setLayout(self.overlay_slider_layout)

        self.overlay_lut = QWidget()
        self.overlay_lut_layout = QVBoxLayout()
        self.overlay_lut_label = QLabel("Look-up-table")
        self.overlay_lut_button = ComboBox()
        self.overlay_lut_button.addItems(['Grayscale', 'Red', 'Green', 'Blue', 'Rainbow', 'Spectral', 'Viridis', 'Qualitative'])
        self.overlay_lut_button.currentTextChanged.connect(self.change_overlay_lut)
        self.overlay_lut_layout.addWidget(self.overlay_lut_label)
        self.overlay_lut_layout.addWidget(self.overlay_lut_button)
        self.overlay_lut.setLayout(self.overlay_lut_layout)

        self.overlay_remove = QPushButton("Remove overlay")
        self.overlay_remove.setFixedSize(150, 60)
        self.overlay_remove.setDown(True)
        self.overlay_remove.clicked.connect(self.remove_overlay)

       # Insert the overlay label and button at the top of the layout
        self.info_layout.insertWidget(2, self.overlay_label)
        self.info_layout.insertWidget(3, self.overlay_toggle)
        self.info_layout.insertWidget(4, self.overlay_slider)
        self.info_layout.insertWidget(5, self.overlay_lut)
        self.info_layout.insertWidget(6, self.overlay_remove, alignment=Qt.AlignCenter)
        self.info_layout.addStretch()

        self.view_widget.add_overlay(self.overlay_path)

    def change_lut(self, lut):
        self.view_widget.change_lut(lut)

    def change_overlay_lut(self, lut):
        self.view_widget.change_overlay_lut(lut)

    def toggle_overlay_visibility(self):
        self.view_widget.layout.itemAt(0).widget().overlay = not self.view_widget.layout.itemAt(0).widget().overlay
        self.view_widget.layout.itemAt(0).widget().update_image()
        if self.view_widget.state == "3D":
            self.view_widget.layout.itemAt(1).widget().overlay = not self.view_widget.layout.itemAt(1).widget().overlay
            self.view_widget.layout.itemAt(2).widget().overlay = not self.view_widget.layout.itemAt(2).widget().overlay
            self.view_widget.layout.itemAt(1).widget().update_image()
            self.view_widget.layout.itemAt(2).widget().update_image()

    def change_overlay_opacity(self):
        self.overlay_slider_value.setText(f"{self.sender().value()}%")
        for i in reversed(range(self.view_widget.layout.count())):
            label = self.view_widget.layout.itemAt(i).widget()
            label.overlay_opacity = self.sender().value()
            label.update_image()

    def remove_overlay(self):
        if self.overlay:
            for i in reversed(range(self.view_widget.layout.count())):
                label = self.view_widget.layout.itemAt(i).widget()
                label.overlay = False
                label.overlay_data = None
                label.update_image()

            self.info_layout.removeWidget(self.overlay_label)
            self.info_layout.removeWidget(self.overlay_toggle)
            self.info_layout.removeWidget(self.overlay_slider)
            self.info_layout.removeWidget(self.overlay_lut)
            self.info_layout.removeWidget(self.overlay_remove)
            self.overlay_label.setParent(None)
            self.overlay_toggle.setParent(None)
            self.overlay_slider.setParent(None)
            self.overlay_lut.setParent(None)
            self.overlay_remove.setParent(None)

        self.overlay = False
        self.overlay_path = None

    def screenshot(self):
        screen = QApplication.primaryScreen()
        screenshot = screen.grabWindow(self.image_label.winId())
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png)")

        if file_path:
            # Take the screenshot of the label and save it to the chosen path
            screenshot.save(file_path, "png")
            print("Screenshot saved to:", file_path)
        
    def whole_histogram(self):
        # Flatten the 3D array to a 1D array
        image_flat = self.rotated_data.flatten()
        image_flat = np.where(image_flat == 0, np.NaN, image_flat)

        # Plot the histogram
        plt.hist(image_flat, bins=256, range=(1, np.amax(self.rotated_data)), alpha=0.75)
        plt.xlabel('Intensity')
        plt.xlim(left=1)
        plt.ylabel('Frequency')
        plt.title('Histogram of the whole image')
        plt.show()

    def toggle_view(self, state):
        self.view_widget.state = state
        reset_layout(self.view_widget.layout)
        self.view_widget.update_layout()
        self.view_widget.update_text()

    def toggle_mosaic(self):
        self.reset_layout(self.view_widget.layout)
        self.create_image_label(self.filename, "mosaic")

    def add_current_histogram(self):
        self.toggle_current_histo = not self.toggle_current_histo
        if not self.toggle_current_histo:
            self.info_layout.removeWidget(self.graphWidget_2)
            self.graphWidget_2.setParent(None)
        else:
            if self.info_layout.count() <= 1:
                self.update_histogram()
                self.info_layout.addWidget(self.graphWidget_2, 1,0) 

    def update_aif(self, aif):
        self.plot_aif.clear()
        self.plot_aif.plot(aif)

    def create_histogram(self):
        image_flat = self.data.flatten()
        counts, bins = np.histogram(image_flat,bins=256)
        counts = counts[1:]

        self.graphWidget_1 = pg.PlotWidget()
        self.graphWidget_1.plot(np.arange(0, len(counts), 1), counts, pen=pen)
        self.graphWidget_2 = pg.PlotWidget()

        self.info_layout.addWidget(self.graphWidget_1, 1,0)

    def update_histogram(self):
        self.graphWidget_2.clear()
        image_flat = self.rotated_data[:, :, self.slice_index].flatten()
        counts, bins = np.histogram(image_flat,bins=256)
        counts = counts[1:]

        self.graphWidget_2.plot(np.arange(0, len(counts), 1), counts, pen=pen) 

    def update_text(self):
        title = "<h2><b>Information:</b></h2>"
        info = f"{os.path.basename(self.view_widget.path)}<br>"
        info += f"Image dimensions: {self.view_widget.shape}<br>Pixel dimension: {self.view_widget.pixdim}<br>"
        if self.view_widget.state == "3D" :
            info += f"X: {self.view_widget.s_slice}<br>"
            info += f"Y: {self.view_widget.c_slice}<br>"
            info += f"Z: {self.view_widget.a_slice}<br>"
        else:
            # if hasattr(self.view_widget, "layout") and self.view_widget.layout.count() > 0:
                image_label = self.view_widget.get_state_label()
                if len(self.view_widget.shape) == 4:
                    phase_index = image_label.slice_index // self.view_widget.shape[2]
                    slice_index = image_label.slice_index % self.view_widget.shape[2]
                    if self.view_widget.state == "Mosaic":
                        info += f"Phase: {image_label.slice_index+1}<br><br>"
                    else:
                        info += f"Phase: {phase_index+1}<br>Slice: {slice_index+1}<br><br>"
                else: 
                    info += f"Slice: {image_label.slice_index+1}<br>"
                    info += f"Window width: {int(image_label.window_width)}<br>Window center: {int(image_label.window_center)}"
        self.text_label.setText(f"{title}{info}")

    def reset_contrast(self):
        for i in reversed(range(self.view_widget.layout.count())):
            widget = self.view_widget.layout.itemAt(i).widget()
            widget.window_center = (widget.contrast_min + widget.contrast_max) / 2  # Reinitial contrast center
            widget.window_width = widget.contrast_max - widget.contrast_min  # Reinitial contrast width
            widget.update_image()

    def transforms(self, transform):
        new_window = TransformsGUI(self.nifti_found, self.view_widget.path, transform, self.run_transform)
        new_window.show()
        self.dynamic_windows.append(new_window)

    def run_transform(self, *args):
        print(args)
        if args[0] == "Sort DICOM":
            sort_dicom(args[1], args[2], anonymize=False)
        elif args[0] == "Convert DICOM":
            convert_dicom_to_nifti(args[1], args[2])
        elif args[0] == "Auto crop":
            new_path, idx = crop(args[1], force=args[2])
            if new_path not in self.nifti_found:
                self.load_sequence(new_path)
                self.nifti_found.append(new_path)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
        elif args[0] == "Apply crop":
            idx = [int(val) for val in args[2].split(",")]
            print(idx)
            new_path = apply_crop(args[1], crop=idx, force=args[3])
            if new_path not in self.nifti_found:
                self.load_sequence(new_path)
                self.nifti_found.append(new_path)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
        elif args[0] == "Resample voxels":
            new_path = resample(args[1], float(args[2].replace(",", ".")))[0]
            if new_path not in self.nifti_found:
                self.load_sequence(new_path)
                self.nifti_found.append(new_path)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
        elif args[0] == "Dilate":
            new_path = remove_small_object(args[1], int(args[2]), force=args[3])
            if new_path not in self.nifti_found:
                self.load_sequence(new_path)
                self.nifti_found.append(new_path)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
        elif args[0] == "Susan":
            new_path = susan(args[1], offset=int(args[2]), force=args[3])
            if new_path not in self.nifti_found:
                self.load_sequence(new_path)
                self.nifti_found.append(new_path)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
        elif args[0] == "Registration":
            if args[3] is True:
                new_path, _ = registration(args[1], args[2], matrix=args[3], force=args[4])
            else:
                new_path = registration(args[1], args[2], matrix=args[3], force=args[4])
            if new_path not in self.nifti_found:
                self.load_sequence(new_path)
                self.nifti_found.append(new_path)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
        elif args[0] == "Apply mask":
            new_path = apply_mask(args[1], args[2], force=args[3])
            if new_path not in self.nifti_found:
                self.load_sequence(new_path)
                self.nifti_found.append(new_path)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
        elif args[0] == "Skull stripping":
            new_path2 = None
            if args[3] is True:
                new_path, new_path2 = skull_stripping(args[1], model_path=args[2], mask=args[3], force=args[4])
            else:
                new_path = skull_stripping(args[1], model_path=args[2], mask=args[3], force=args[4])
            if new_path not in self.nifti_found:
                self.load_sequence(new_path)
                self.nifti_found.append(new_path)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
            if new_path2 is not None and new_path2 not in self.nifti_found:
                self.load_sequence(new_path2)
                self.nifti_found.append(new_path2)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")

    def closeEvent(self, event):
        for window in self.dynamic_windows:
            window.close()
        event.accept()

    def center(self):
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()

        qr.moveCenter(cp)
        self.move(qr.topLeft())

class TransformsGUI(QWidget):
    def __init__(self, path, selected_path, transform, run_transform):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.run_transform = run_transform

        if transform == "Sort DICOM":
            title = QLabel("Path to the DICOM directory")
            input_dir = QPushButton("Browse")
            input_dir.clicked.connect(self.get_input_dir)
            self.input_dir = QLineEdit(self)
            layout.addWidget(title)
            layout.addWidget(input_dir)
            layout.addWidget(self.input_dir)

            title = QLabel("Path to the DICOM sorted directory")
            output_dir = QPushButton("Browse")
            output_dir.clicked.connect(self.get_output_dir)
            self.output_dir = QLineEdit(self)
            layout.addWidget(title)
            layout.addWidget(output_dir)
            layout.addWidget(self.output_dir)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, self.input_dir.text(), self.output_dir.text()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        elif transform == "Convert DICOM":
            title = QLabel("Path to the DICOM directory")
            input_dir = QPushButton("Browse")
            input_dir.clicked.connect(self.get_input_dir)
            self.input_dir = QLineEdit(self)
            layout.addWidget(title)
            layout.addWidget(input_dir)
            layout.addWidget(self.input_dir)

            title = QLabel("Path to the converted NIfTI file(s)")
            output_dir = QPushButton("Browse")
            output_dir.clicked.connect(self.get_output_dir)
            self.output_dir = QLineEdit(self)
            layout.addWidget(title)
            layout.addWidget(output_dir)
            layout.addWidget(self.output_dir)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, self.input_dir.text(), self.output_dir.text()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        elif transform == "Auto crop":
            title = QLabel("Path for auto crop")
            combo_box = QComboBox(self)
            combo_box.addItems(path)
            combo_box.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box)

            force = QCheckBox("Force")
            layout.addWidget(force)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), force.isChecked()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        elif transform == "Apply crop":
            title = QLabel("Path for applied crop")
            combo_box = QComboBox(self)
            combo_box.addItems(path)
            combo_box.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box)

            title = QLabel("Indices")
            idx = QLineEdit(self)
            idx.setPlaceholderText("x1, x2, y1, y2, z1, z2")
            layout.addWidget(title)
            layout.addWidget(idx)

            force = QCheckBox("Force")
            layout.addWidget(force)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), idx.text(), force.isChecked()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        elif transform == "Resample voxels":
            title = QLabel("Path for resample")
            combo_box = QComboBox(self)
            combo_box.addItems(path)
            combo_box.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box)

            title = QLabel("Voxel dimension (mm)")
            voxel = QLineEdit(self)
            voxel.setText("1")
            layout.addWidget(title)
            layout.addWidget(voxel)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), voxel.text()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        elif transform == "Dilate":
            title = QLabel("Path for dilate")
            combo_box = QComboBox(self)
            combo_box.addItems(path)
            combo_box.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box)

            title = QLabel("Minimum size of objects to be removed")
            voxel = QLineEdit(self)
            layout.addWidget(title)
            layout.addWidget(voxel)

            force = QCheckBox("Force")
            layout.addWidget(force)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), voxel.text(), force.isChecked()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        elif transform == "Susan":
            title = QLabel("Path for susan")
            combo_box = QComboBox(self)
            combo_box.addItems(path)
            combo_box.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box)

            title = QLabel("Offset")
            offset = QLineEdit(self)
            offset.setText("3")
            layout.addWidget(title)
            layout.addWidget(offset)

            force = QCheckBox("Force")
            layout.addWidget(force)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), offset.text(), force.isChecked()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        elif transform == "Skull stripping":
            title = QLabel("Path for skull strip")
            combo_box = QComboBox(self)
            combo_box.addItems(path)
            combo_box.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box)

            title = QLabel("Model path")
            dialog = QPushButton("Browse")
            dialog.clicked.connect(self.get_model_dir)
            self.model = QLineEdit(self)
            
            layout.addWidget(title)
            layout.addWidget(self.model)
            layout.addWidget(dialog)

            mask = QCheckBox("Mask")
            layout.addWidget(mask)

            force = QCheckBox("Force")
            layout.addWidget(force)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), self.model.text(), mask.isChecked(), force.isChecked()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        elif transform == "Registration":
            title = QLabel("Path for fixed image")
            combo_box = QComboBox(self)
            combo_box.addItems(path)
            combo_box.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box)

            title = QLabel("Path for moving image")
            combo_box_2 = QComboBox(self)
            combo_box_2.addItems(path)
            combo_box_2.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box_2)

            mat = QCheckBox("Matrix")
            layout.addWidget(mat)

            force = QCheckBox("Force")
            layout.addWidget(force)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), combo_box_2.currentText(), mat.isChecked(), force.isChecked()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        elif transform == "Apply mask":
            title = QLabel("Path for the image to be masked")
            combo_box = QComboBox(self)
            combo_box.addItems(path)
            combo_box.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box)

            title = QLabel("Path for the mask")
            combo_box_2 = QComboBox(self)
            combo_box_2.addItems(path)
            combo_box_2.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box_2)

            force = QCheckBox("Force")
            layout.addWidget(force)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), combo_box_2.currentText(), force.isChecked()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.setWindowTitle("PyDactim Transformation")
        self.show_centered()

    def get_model_dir(self):
        dialog = QFileDialog()
        dir = dialog.getExistingDirectory(None, "Select a directory", "C:\\", QFileDialog.ShowDirsOnly)
        self.model.setText(dir)

    def get_input_dir(self):
        dialog = QFileDialog()
        dir = dialog.getExistingDirectory(None, "Select a directory", "C:\\", QFileDialog.ShowDirsOnly)
        self.input_dir.setText(dir)

    def get_output_dir(self):
        dialog = QFileDialog()
        dir = dialog.getExistingDirectory(None, "Select a directory", "C:\\", QFileDialog.ShowDirsOnly)
        self.output_dir.setText(dir)

    def launch_transform(self, *args):
        self.run_transform(*args)
        self.close()
    
    def show_centered(self):
        app = QGuiApplication.instance()
        screen_geometry = app.primaryScreen().availableGeometry()
        center = screen_geometry.center()
        self.move(center.x() - self.width() / 2, center.y() - self.height() / 2)
        self.show()

class CustomOutputStream(StringIO):
    def __init__(self, label, scroll):
        super().__init__()
        self.label = label
        self.scroll = scroll

    def write(self, txt):
        super().write(txt)
        self.label.setText(self.getvalue())
        self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        )

def ViewerApp():
    app = QApplication(sys.argv + ['-platform', 'windows:darkmode=2'])
    app.setStyle('Fusion')
    app.setPalette(get_darkModePalette(app))

    # default_nifti_file = "path/to/your/default/nifti_file.nii.gz"  # Change this to the default NIfTI file path

    # if not os.path.isfile(default_nifti_file):
    #     default_nifti_file, _ = QFileDialog.getOpenFileName(None, "Open NIfTI file", "", "NIfTI Files (*.nii.gz)")

    # if not os.path.isfile(default_nifti_file):
    #     print("Error: No valid NIfTI file selected.")
    #     sys.exit(1)

    viewer = NiftiViewer()
    viewer.show()
    viewer.center()
    sys.exit(app.exec())

if __name__ == "__main__":
    ViewerApp()

