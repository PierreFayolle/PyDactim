import sys
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QFrame, QHeaderView, QApplication, QPushButton, QSlider, QDialog, QComboBox, QToolButton, QLineEdit, QLabel, QMainWindow, QCheckBox, QScrollArea, QStackedLayout, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QWidget, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QAction, QGuiApplication, QIcon
from PySide6.QtCore import Qt, QEvent, QRect, QSize, QSettings
import pyqtgraph as pg
import matplotlib.pyplot as plt
import glob
from io import StringIO
import imageio
import tempfile

from pydactim.transformation import (n4_bias_field_correction,
    registration, apply_transformation, resample, histogram_matching, 
    apply_mask, substract, susan, normalize, crop, apply_crop, copy_affine, skull_stripping, 
    remove_small_object, tissue_classifier, add_tissue_class, prediction_glioma, prediction_multiple_sclerosis, uncertainty_prediction_glioma)

from pydactim.sorting import sort_dicom
from pydactim.conversion import convert_dicom_to_nifti
from .custom import ThumbnailFrame, ScrollArea, ComboBox, AnimatedToggle
from .view_panel import ViewPanel
from .utils import create_thumbnail, get_darkModePalette, reset_layout
from .settings import *

class AiModelPathDialog(QDialog):
    def __init__(self, settings):
        super().__init__()

        self.settings = settings

        self.setWindowTitle("Select AI Model Directories")
        self.layout = QVBoxLayout()

        self.skull_stripping_label = QLabel("Skull Stripping Model Directory:")
        self.layout.addWidget(self.skull_stripping_label)
        self.skull_stripping_path = QLabel(settings.value("ai_model_paths/skull_stripping", ""))
        self.layout.addWidget(self.skull_stripping_path)
        self.skull_stripping_button = QPushButton("Select Skull Stripping Model Directory")
        self.skull_stripping_button.clicked.connect(self.select_skull_stripping_path)
        self.layout.addWidget(self.skull_stripping_button)

        self.add_separator()

        self.glioma_segmentation_label = QLabel("Glioma Segmentation Model directory (with weights and landmarks):")
        self.layout.addWidget(self.glioma_segmentation_label)
        self.glioma_segmentation_path = QLabel(settings.value("ai_model_paths/glioma", ""))
        self.layout.addWidget(self.glioma_segmentation_path)
        self.glioma_segmentation_button = QPushButton("Select Glioma Segmentation Model Directory")
        self.glioma_segmentation_button.clicked.connect(self.select_glioma_segmentation_path)
        self.layout.addWidget(self.glioma_segmentation_button)

        self.add_separator()

        self.ms_segmentation_label = QLabel("MS Segmentation Model directory (with weights and landmarks):")
        self.layout.addWidget(self.ms_segmentation_label)
        self.ms_segmentation_path = QLabel(settings.value("ai_model_paths/ms", ""))
        self.layout.addWidget(self.ms_segmentation_path)
        self.ms_segmentation_button = QPushButton("Select MS Segmentation Model Directory")
        self.ms_segmentation_button.clicked.connect(self.select_ms_segmentation_path)
        self.layout.addWidget(self.ms_segmentation_button)

        self.add_separator()

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_paths)
        self.layout.addWidget(self.save_button)

        self.setLayout(self.layout)

    def add_separator(self):
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(separator)

    def select_skull_stripping_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Skull Stripping Model Directory")
        if path: self.skull_stripping_path.setText(path)

    def select_glioma_segmentation_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Glioma Segmentation Model Directory")
        if path: self.glioma_segmentation_path.setText(path)

    def select_ms_segmentation_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select MS Segmentation Model Directory")
        if path: self.ms_segmentation_path.setText(path)

    def save_paths(self):
        self.settings.setValue("ai_model_paths/skull_stripping", self.skull_stripping_path.text())
        self.settings.setValue("ai_model_paths/glioma", self.glioma_segmentation_path.text())
        self.settings.setValue("ai_model_paths/ms", self.ms_segmentation_path.text())
        # Save other AI model paths similarly
        self.accept()

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
        # print("INFO - Loading images")
        # self.load_sequences(self.nifti_found)

        self.lut = None
        self.overlay_lut = None

        self.overlay = False
        self.overlay_path = None

        # sys.stdout = CustomOutputStream(self.log_label, self.log_scroll)

    def init_tabs(self):
        print("INFO - Creating menu tab")
        # Add a menu bar
        menubar = self.menuBar()
        self.file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        image_menu = menubar.addMenu("Image")
        transform_menu = menubar.addMenu("Scripts")
        view_menu = menubar.addMenu("View")

        # Add actions to the File menu
        action = QAction("Open directory", self)
        action.triggered.connect(lambda x: self.open_volume(None))
        self.file_menu.addAction(action)

        action = QAction("Add file", self)
        action.triggered.connect(lambda x: self.add_file(None))
        self.file_menu.addAction(action)

        self.recent_submenu = self.file_menu.addMenu("Recent data")
        settings = self.init_config()
        recent_paths = [settings.value(f"recent_paths/recent_path_{i}", "") for i in range(1, 11)]

        for path in recent_paths:
            if os.path.exists(path):
                action = QAction(path, self)
                if os.path.isdir(path): action.triggered.connect(lambda checked, p=path: self.open_volume(p))
                else: action.triggered.connect(lambda checked, p=path: self.add_file(p))
                self.recent_submenu.addAction(action)

        action = QAction("Settings", self)
        action.triggered.connect(self.settings)
        self.file_menu.addAction(action)

        action = QAction("Screenshot", self)
        action.triggered.connect(self.screenshot)
        self.file_menu.addAction(action)

        action = QAction("Gif", self)
        action.triggered.connect(self.gifshot)
        self.file_menu.addAction(action)

        action = QAction("Close", self)
        action.triggered.connect(lambda x: sys.exit(1))
        self.file_menu.addAction(action)

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

        action = QAction("3D view", self)
        action.triggered.connect(lambda x: self.toggle_view("3D"))
        view_menu.addAction(action)

        # AI
        ai_submenu = transform_menu.addMenu("AI segmentation")

        action = QAction("Brain", self)
        action.triggered.connect(lambda x: self.transforms("Skull stripping"))
        ai_submenu.addAction(action)

        ai_action_1 = QAction("Glioma", self)
        ai_action_1.triggered.connect(lambda x: self.transforms("Glioma"))
        ai_submenu.addAction(ai_action_1)

        ai_action_3 = QAction("Meningioma", self)
        ai_action_3.triggered.connect(lambda x: self.transforms("Meningioma"))
        ai_submenu.addAction(ai_action_3)

        ai_action_2 = QAction("Multiple Sclerosis", self)
        ai_action_2.triggered.connect(lambda x: self.transforms("Multiple Sclerosis"))
        ai_submenu.addAction(ai_action_2)

        # Crop
        crop_submenu = transform_menu.addMenu("Crop")

        action = QAction("Auto crop", self)
        action.triggered.connect(lambda x: self.transforms("Auto crop"))
        crop_submenu.addAction(action)

        action = QAction("Apply crop", self)
        action.triggered.connect(lambda x: self.transforms("Apply crop"))
        crop_submenu.addAction(action)

        filter_submenu = transform_menu.addMenu("Filter")

        action = QAction("Remove islands", self)
        action.triggered.connect(lambda x: self.transforms("Remove islands"))
        filter_submenu.addAction(action)

        action = QAction("Susan", self)
        action.triggered.connect(lambda x: self.transforms("Susan"))
        filter_submenu.addAction(action)

        registration_submenu = transform_menu.addMenu("Registration")

        action = QAction("Registration", self)
        action.triggered.connect(lambda x: self.transforms("Registration"))
        registration_submenu.addAction(action)

        action = QAction("Apply registration", self)
        action.triggered.connect(lambda x: self.transforms("Apply registration"))
        registration_submenu.addAction(action)

        other_submenu = transform_menu.addMenu("Others")

        action = QAction("Apply mask", self)
        action.triggered.connect(lambda x: self.transforms("Apply mask"))
        other_submenu.addAction(action)

        action = QAction("Bias field correction", self)
        action.triggered.connect(lambda x: self.transforms("Bias field correction"))
        other_submenu.addAction(action)

        action = QAction("Resample voxels", self)
        action.triggered.connect(lambda x: self.transforms("Resample voxels"))
        other_submenu.addAction(action)

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
        self.lut_button.addItems(['Grayscale', 'Red', 'Green', 'Blue', 'Rainbow', 'Spectral', 'Flow', 'Viridis', 'Qualitative', 'Random'])
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

        self.roi_widget = QWidget()
        self.roi_layout = QHBoxLayout()

        self.roi_x_widget = QWidget()
        self.roi_x_layout = QHBoxLayout()
        self.roi_x_title = QLabel("ROI x")
        self.roi_x = QLineEdit()
        self.roi_x.setFixedWidth(50)
        self.roi_x_layout.addWidget(self.roi_x_title)
        self.roi_x_layout.addWidget(self.roi_x)
        self.roi_x_layout.setAlignment(Qt.AlignLeft)
        self.roi_x_widget.setLayout(self.roi_x_layout)

        self.roi_y_widget = QWidget()
        self.roi_y_layout = QHBoxLayout()
        self.roi_y_title = QLabel("ROI y")
        self.roi_y = QLineEdit()
        self.roi_y.setFixedWidth(50)
        self.roi_y_layout.addWidget(self.roi_y_title)
        self.roi_y_layout.addWidget(self.roi_y)
        self.roi_y_layout.setAlignment(Qt.AlignLeft)
        self.roi_y_widget.setLayout(self.roi_y_layout)

        self.roi_z_widget = QWidget()
        self.roi_z_layout = QHBoxLayout()
        self.roi_z_title = QLabel("ROI z")
        self.roi_z = QLineEdit()
        self.roi_z.setFixedWidth(50)
        self.roi_z_layout.addWidget(self.roi_z_title)
        self.roi_z_layout.addWidget(self.roi_z)
        self.roi_z_layout.setAlignment(Qt.AlignLeft)
        self.roi_z_widget.setLayout(self.roi_z_layout)

        self.roi_table = QTableWidget()
        roi_header = ["Mean", "Std", "Min", "Max", "mm3"]
        self.roi_table.setColumnCount(len(roi_header))
        self.roi_table.setHorizontalHeaderLabels(roi_header)
        self.roi_table.setRowCount(1)
        self.roi_table.verticalHeader().setVisible(False)
        _ = [self.roi_table.setColumnWidth(i,70) for i in range(len(roi_header))]
        self.roi_table.horizontalHeader().setStretchLastSection(True)


        self.roi_layout.addWidget(self.roi_x_widget)
        self.roi_layout.addWidget(self.roi_y_widget)
        self.roi_layout.addWidget(self.roi_z_widget)
        self.roi_widget.setLayout(self.roi_layout)

        self.perf_layout.addWidget(self.plot_aif)
        self.perf_layout.addWidget(self.roi_widget)
        self.perf_layout.addWidget(self.roi_table)
        self.perf_widget.setLayout(self.perf_layout)

        # RIGHT LAYOUT
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
        self.view_widget = ViewPanel(self.filename, self.update_text, self.plot_aif, self.update_roi)
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

    def init_config(self):
        temp_dir = tempfile.gettempdir()
        config_path = os.path.join(temp_dir, 'pydactim_view_config.ini')
        settings = QSettings(config_path, QSettings.IniFormat)

        for i in range(1, 11):
            if not settings.contains(f"recent_paths/recent_path_{i}"):
                settings.setValue(f"recent_paths/recent_path_{i}", "")

        if not settings.contains("ai_model_paths/skull_stripping"):
            settings.setValue("ai_model_paths/skull_stripping", "")

        if not settings.contains("ai_model_paths/glioma"):
            settings.setValue("ai_model_paths/glioma", "")

        if not settings.contains("ai_model_paths/ms"):
            settings.setValue("ai_model_paths/ms", "")

        return settings

    def settings(self):
        self.open_ai_model_path_dialog()

    def add_recent_path(self, path):
        settings = self.init_config()
        recent_paths = [settings.value(f"recent_paths/recent_path_{i}", "") for i in range(1, 11)]
        
        # Ensure the path is not already in the list
        if path in recent_paths:
            recent_paths.remove(path)
        
        recent_paths.insert(0, path)
        
        # Limit to 10 recent paths
        if len(recent_paths) > 10:
            recent_paths = recent_paths[:10]
        
        for i in range(1, 11):
            settings.setValue(f"recent_paths/recent_path_{i}", recent_paths[i-1] if i-1 < len(recent_paths) else "")

        self.recent_submenu.clear()
        for path in recent_paths:
            if os.path.exists(path):
                action = QAction(path, self)
                if os.path.isdir(path): action.triggered.connect(lambda checked, p=path: self.open_volume(p))
                else: action.triggered.connect(lambda checked, p=path: self.add_file(p))
                self.recent_submenu.addAction(action)

    def open_ai_model_path_dialog(self):
        settings = self.init_config()
        dialog = AiModelPathDialog(settings)
        dialog.exec_()

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

    def open_volume(self, path=None):
        if path is None: self.dirname = QFileDialog.getExistingDirectory(self, "Select a directory", "C:\\", QFileDialog.ShowDirsOnly)
        else: self.dirname = path
        # self.dirname = "D:/Studies/GLIOBIOPSY/data/sub-018/"
        self.nifti_found = glob.glob(os.path.join(self.dirname, '**/*.nii*'), recursive=True)
        if not hasattr(self, "thumbnail_title"): 
            self.init_all()
            self.load_sequences(self.nifti_found)
        else: self.load_sequences(self.nifti_found)
        self.add_recent_path(self.dirname)

    def add_file(self, path=None):
        if path is None:
            new_path = QFileDialog.getOpenFileName(
                self,
                "Open File",
                "${HOME}",
                "NIfTI Compressed Files (*.nii.gz);; NIfTI Files (*.nii)",
            )[0]
        else: new_path = path

        if not hasattr(self, "nifti_found"):
            self.nifti_found = []
        if new_path not in self.nifti_found:
            self.nifti_found.append(new_path)
            if not hasattr(self, "thumbnail_title"): 
                self.init_all()
                self.load_sequences(self.nifti_found)
            else: self.load_sequence(new_path)
            self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
        self.add_recent_path(new_path)
        
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
        self.overlay_lut_button.addItems(['Grayscale', 'Red', 'Green', 'Blue', 'Rainbow', 'Spectral', 'Flow', 'Viridis', 'Qualitative', 'Random'])
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
        self.view_widget.layout.itemAt(0).widget().overlay_toggle = not self.view_widget.layout.itemAt(0).widget().overlay_toggle
        self.view_widget.layout.itemAt(0).widget().update_image()
        if self.view_widget.state == "3D":
            self.view_widget.layout.itemAt(1).widget().overlay_toggle = not self.view_widget.layout.itemAt(1).widget().overlay_toggle
            self.view_widget.layout.itemAt(2).widget().overlay_toggle = not self.view_widget.layout.itemAt(2).widget().overlay_toggle
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
                label.overlay_toggle = False
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
        dir = QFileDialog.getExistingDirectory(self, "Select a directory", "C:\\", QFileDialog.ShowDirsOnly)

        if self.view_widget.state != "3D":
            if self.view_widget.state == "Axial":
                screenshot = screen.grabWindow(self.view_widget.axial_label.winId())
                name = f"{os.path.basename(self.view_widget.path).replace(".nii", "").replace(".gz", "")}_axial.png"
            elif self.view_widget.state == "Coronal":
                screenshot = screen.grabWindow(self.view_widget.coronal_label.winId())
                name = f"{os.path.basename(self.view_widget.path).replace(".nii", "").replace(".gz", "")}_coronal.png"
            elif self.view_widget.state == "Sagittal":
                screenshot = screen.grabWindow(self.view_widget.sagittal_label.winId())
                name = f"{os.path.basename(self.view_widget.path).replace(".nii", "").replace(".gz", "")}_sagittal.png"

            screenshot.save(os.path.join(dir, name), "png")
            print(f"INFO - Successfully screenshot at: {os.path.join(dir, name)}")
        else:
            screenshot = screen.grabWindow(self.view_widget.axial_label.winId())
            name = f"{os.path.basename(self.view_widget.path).replace(".nii", "").replace(".gz", "")}_axial.png"
            screenshot.save(os.path.join(dir, name), "png")

            screenshot = screen.grabWindow(self.view_widget.coronal_label.winId())
            name = f"{os.path.basename(self.view_widget.path).replace(".nii", "").replace(".gz", "")}_coronal.png"
            screenshot.save(os.path.join(dir, name), "png")

            screenshot = screen.grabWindow(self.view_widget.sagittal_label.winId())
            name = f"{os.path.basename(self.view_widget.path).replace(".nii", "").replace(".gz", "")}_sagittal.png"
            screenshot.save(os.path.join(dir, name), "png")
        
    def gifshot(self):
        gif_array = []
        current_slice = self.view_widget.a_slice
        self.view_widget.axial_label.data = self.view_widget.adata[..., self.view_widget.a_slice]
        for slice in range(self.view_widget.shape[2]):
            print(slice)
            self.view_widget.update_slice(slice, "Axial")
            self.view_widget.axial_label.update_image()

            pixmap = self.view_widget.axial_label.pixmap

            # Convert the pixmap to a QImage
            image = pixmap.toImage()

            # Convert the QImage to a NumPy array
            width = image.width()
            height = image.height()
            buffer = image.constBits()
            dtype = np.uint8 if image.format() == QImage.Format.Format_RGB32 else np.uint32
            array = np.array(buffer).reshape(height, width, 4).astype(dtype)

            array = array[:, :, [2, 1, 0, 3]]
            gif_array.append(array)  # Extract RGB channels

        print("INFO - Making the gif file...")
        imageio.mimsave(f'{self.dirname}/animation.gif', gif_array, duration=0.375/(len(gif_array)/10))
        self.view_widget.update_slice(current_slice, "Axial")
        self.view_widget.axial_label.update_image()
 
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
        self.roi_z.setText(f"{self.view_widget.a_slice}")

    def update_roi(self, x, y, z, mean, std, min, max, mm3):
        self.roi_x.setText(str(int(x)))
        self.roi_y.setText(str(int(y)))
        self.roi_z.setText(str(int(z)))
        self.roi_table.setItem(0, 0, QTableWidgetItem(str(round(mean,3))))
        self.roi_table.setItem(0, 1, QTableWidgetItem(str(round(std,3))))
        self.roi_table.setItem(0, 2, QTableWidgetItem(str(round(min,3))))
        self.roi_table.setItem(0, 3, QTableWidgetItem(str(round(max,3))))
        self.roi_table.setItem(0, 4, QTableWidgetItem(str(round(mm3,3))))

    def reset_contrast(self):
        for i in reversed(range(self.view_widget.layout.count())):
            widget = self.view_widget.layout.itemAt(i).widget()
            widget.window_center = (widget.contrast_min + widget.contrast_max) / 2  # Reinitial contrast center
            widget.window_width = widget.contrast_max - widget.contrast_min  # Reinitial contrast width
            widget.update_image()

    def transforms(self, transform):
        new_window = TransformsGUI(self.nifti_found, self.view_widget.path, transform, self.run_transform, self.init_config)
        new_window.show()
        self.dynamic_windows.append(new_window)

    def run_transform(self, *args):
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
        elif args[0] == "Remove islands":
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
        elif args[0] == "Apply transformation":
            new_path = apply_transformation(args[1], args[2], args[3], force=args[3])
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
        elif args[0] == "Bias field correction":
            new_path2 = None
            if args[3] is True:
                new_path, new_path2 = n4_bias_field_correction(args[1], mask=args[2], force=args[4])
            else:
                new_path = n4_bias_field_correction(args[1], force=args[2])
            if new_path not in self.nifti_found:
                self.load_sequence(new_path)
                self.nifti_found.append(new_path)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
            if new_path2 is not None and new_path2 not in self.nifti_found:
                self.load_sequence(new_path2)
                self.nifti_found.append(new_path2)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
        elif args[0] == "Glioma":
            new_path = prediction_glioma(args[1], model_path=args[2], landmark_path=args[3], force=args[4])
            if new_path not in self.nifti_found:
                self.load_sequence(new_path)
                self.nifti_found.append(new_path)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
        elif args[0] == "Multiple Sclerosis":
            new_path, new_path2 = prediction_multiple_sclerosis(args[1], model_path=args[2], landmark_path=args[3], force=args[4])
            print(new_path)
            if new_path not in self.nifti_found:
                self.load_sequence(new_path)
                self.nifti_found.append(new_path)
                self.thumbnail_title.setText(f"<h2 style='margin-left: 10px'>{len(self.nifti_found)} images loaded:</h2>")
            if new_path2 not in self.nifti_found:
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
    def __init__(self, path, selected_path, transform, run_transform, conf):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.setWindowTitle("PyDactim Transformation")
        self.show_centered()

        settings = conf()

        self.path = path
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
            title = QLabel("Path for the image to auto crop")
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
            title = QLabel("Path for for the image to apply crop")
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
            title = QLabel("Path for image the to resample")
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

        elif transform == "Remove islands":
            title = QLabel("Path for the image in which removing islands")
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
            title = QLabel("Path for the image to susan")
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
            title = QLabel("Path for the image to skull strip")
            combo_box = QComboBox(self)
            combo_box.addItems(path)
            combo_box.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box)

            title = QLabel("Model path")
            dialog = QPushButton("Browse")
            dialog.clicked.connect(self.get_extra_dir)
            self.extra = QLineEdit(self)
            self.extra.setText(settings.value("ai_model_paths/skull_stripping", ""))
            
            layout.addWidget(title)
            layout.addWidget(self.extra)
            layout.addWidget(dialog)

            mask = QCheckBox("Mask")
            layout.addWidget(mask)

            force = QCheckBox("Force")
            layout.addWidget(force)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), self.extra.text(), mask.isChecked(), force.isChecked()))
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

        elif transform == "Apply transformation":
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

            title = QLabel("Matrix path")
            dialog = QPushButton("Browse")
            dialog.clicked.connect(self.get_extra_file)
            self.extra = QLineEdit(self)
            
            layout.addWidget(title)
            layout.addWidget(self.extra)
            layout.addWidget(dialog)

            force = QCheckBox("Force")
            layout.addWidget(force)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), combo_box_2.currentText(), self.extra.text(), force.isChecked()))
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

        elif transform == "Bias field correction":
            title = QLabel("Path for the image to correct")
            combo_box = QComboBox(self)
            combo_box.addItems(path)
            combo_box.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box)

            mask = QCheckBox("Mask")
            layout.addWidget(mask)

            force = QCheckBox("Force")
            layout.addWidget(force)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), mask.isChecked(), force.isChecked()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        elif transform == "Glioma":
            title = QLabel("Path for the input T2-FLAIR")
            combo_box = QComboBox(self)
            combo_box.addItems(path)
            combo_box.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box)

            title = QLabel("Model path (including landmarks)")
            dialog = QPushButton("Browse")
        
            dialog.clicked.connect(self.get_ai_dir)
            self.model = QLineEdit(self)
            self.extra = QLineEdit(self)

            path = settings.value("ai_model_paths/glioma", "")
            if path is not None: self.set_ai_dir(path)
            else: "C:\\"

            layout.addWidget(title)
            layout.addWidget(dialog)
            layout.addWidget(self.model)
            layout.addWidget(self.extra)

            force = QCheckBox("Force")
            layout.addWidget(force)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), self.model.text(), self.extra.text(), force.isChecked()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

        elif transform == "Multiple Sclerosis":
            title = QLabel("Path for the input T2-FLAIR")
            combo_box = QComboBox(self)
            combo_box.addItems(path)
            combo_box.setCurrentText(selected_path)
            layout.addWidget(title)
            layout.addWidget(combo_box)

            title = QLabel("Model path (including landmarks)")
            dialog = QPushButton("Browse")
        
            dialog.clicked.connect(self.get_ai_dir)
            self.model = QLineEdit(self)
            self.extra = QLineEdit(self)

            path = settings.value("ai_model_paths/ms", "")
            if path is not None and len(path) > 0: self.set_ai_dir(path)
            else: "C:\\"

            layout.addWidget(title)
            layout.addWidget(dialog)
            layout.addWidget(self.model)
            layout.addWidget(self.extra)

            force = QCheckBox("Force")
            layout.addWidget(force)

            button = QPushButton("Run Function", self)
            button.clicked.connect(lambda: self.launch_transform(transform, combo_box.currentText(), self.model.text(), self.extra.text(), force.isChecked()))
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignHCenter)

    def set_ai_dir(self, dir):
        for file in os.listdir(dir):
            if file.endswith(".pth"): self.model.setText(os.path.join(dir, file))
            elif file.endswith(".npy"): self.extra.setText(os.path.join(dir, file))

    def get_ai_dir(self):
        dialog = QFileDialog()
        dir = dialog.getExistingDirectory(None, "Select a directory", "C:\\", QFileDialog.ShowDirsOnly)
        self.set_ai_dir(dir)

    def get_extra_dir(self):
        dialog = QFileDialog()
        dir = dialog.getExistingDirectory(None, "Select a directory", "C:\\", QFileDialog.ShowDirsOnly)
        self.extra.setText(dir)

    def get_extra_file(self):
        file = QFileDialog.getOpenFileName(
            self,
            "Open File",
            os.path.dirname(self.path[0]),
            "Matrix files (*.tfm);;",
        )[0]
        self.extra.setText(file)

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

