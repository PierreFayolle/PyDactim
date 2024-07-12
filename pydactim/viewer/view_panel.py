from PySide6.QtWidgets import QMessageBox, QGridLayout, QWidget, QLabel
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtCore import Qt, QEvent
import pyqtgraph as pg

import nibabel as nib
import numpy as np

from custom import CustomLabel
from utils import is_dsc_param, create_lut, reset_layout, to_sagittal, to_coronal, to_axial, create_4D_grid, to_3D
from settings import *
from scipy import ndimage

class ViewData():
    def __init__(self, data, slice, pixdim, z):
        self.data = data
        self.slice = slice
        self.pixdim = pixdim
        self.z = z

class ViewPanel(QWidget):
    def __init__(self, path, update_text, plot_aif, update_roi, state="3D"):
        super().__init__()

        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("background-color: #000000")

        self.state = state

        if is_dsc_param(path):
            self.lut = "Rainbow"
        else:
            self.lut = "Grayscale"
        
        self.update_text = update_text
        self.plot_aif = plot_aif
        self.update_roi = update_roi
        self.init(path)

    def init(self, path):
        reset_layout(self.layout)

        self.path = path
        self.load_data(self.path)

        if is_dsc_param(self.path):
            self.lut = "Rainbow"
        else:
            self.lut = "Grayscale"

        if not self.ensure3D:
            self.sagittal_label = CustomLabel(self.path, self.s, self.update_slice, self.update_label_aif, self.update_label_roi, "Sagittal", size=(500,500))
            self.sagittal_label.lut = create_lut(self.lut)
            self.coronal_label = CustomLabel(self.path, self.c, self.update_slice, self.update_label_aif, self.update_label_roi, "Coronal", size=(500,500))
            self.coronal_label.lut = create_lut(self.lut)
            self.axial_label = CustomLabel(self.path, self.a, self.update_slice, self.update_label_aif, self.update_label_roi, "Axial", size=(500,500))
            self.axial_label.lut = create_lut(self.lut)

        if self.ensure3D:
            self.axial_label = CustomLabel(self.path, self.a, self.update_slice, self.update_label_aif, self.update_label_roi, "Axial", size=(1000,1000))
            self.axial_label.lut = create_lut(self.lut)
        
        self.update_layout()

    def load_data(self, path):
        self.img = nib.load(path)
        self.data = self.img.get_fdata().astype(np.float32)
        self.shape = self.img.shape
        self.pixdim = self.img.header["pixdim"][1:4]

        self.mosaic = False
        self.mosaic_data = None
        self.max = np.max(self.data)
        # self.plot_aif.setRange(yRange=[0, self.max])
        # self.data = ndimage.uniform_filter(self.data, size=3)
        # self.data = ndimage.gaussian_filter(self.data, sigma=3)


        self.ensure3D = False
        if len(self.shape) == 4:
            self.ensure3D = True
            self.state = "Axial"
            self.offset = self.shape[2]
            self.a_slice = self.shape[2] // 2
            self.adata, self.apixdim, self.az = to_3D(self.data, self.pixdim)  
            self.a = ViewData(self.adata[..., self.a_slice], self.a_slice, self.apixdim, self.az)
        else:
            self.offset = 1

            self.sdata, self.s_slice, self.spixdim, self.sz = to_sagittal(self.data, self.pixdim, path)
            self.s = ViewData(self.sdata[..., self.s_slice], self.s_slice, self.spixdim, self.sz)

            self.cdata, self.c_slice, self.cpixdim, self.cz = to_coronal(self.data, self.pixdim, path)
            self.c = ViewData(self.cdata[..., self.c_slice], self.c_slice, self.cpixdim, self.cz)

            self.adata, self.a_slice, self.apixdim, self.az = to_axial(self.data, self.pixdim, path)
            self.a = ViewData(self.adata[..., self.a_slice], self.a_slice, self.apixdim, self.az)

    def get_state_label(self):
        if self.state == "Sagittal": return self.sagittal_label
        elif self.state == "Coronal": return self.coronal_label
        elif self.state == "Axial": return self.axial_label

    def update_labels(self):
        if not self.ensure3D:
            self.sagittal_label.data = self.sdata[..., self.s_slice]
            self.sagittal_label.pixdim = self.spixdim

            self.coronal_label.data = self.cdata[..., self.c_slice]
            self.coronal_label.pixdim = self.cpixdim

        self.axial_label.data = self.adata[..., self.a_slice]
        self.axial_label.pixdim = self.apixdim

    def update_slice(self, idx, axis):
        if axis == "Sagittal":
            self.s_slice = idx
            self.sagittal_label.slice_index = self.s_slice
            self.sagittal_label.data = self.sdata[..., self.s_slice]
            if self.sagittal_label.overlay: self.sagittal_label.overlay_data = self.osdata[..., self.s_slice]
        elif axis == "Coronal":
            self.c_slice = idx
            self.coronal_label.slice_index = self.c_slice
            self.coronal_label.data = self.cdata[..., self.c_slice]
            if self.coronal_label.overlay: self.coronal_label.overlay_data = self.ocdata[..., self.c_slice]
        elif axis == "Axial":
            self.a_slice = idx
            self.axial_label.slice_index = self.a_slice
            if self.mosaic is True: self.axial_label.data = self.mosaic_data[..., self.a_slice]
            else: self.axial_label.data = self.adata[..., self.a_slice]
            if self.axial_label.overlay: self.axial_label.overlay_data = self.oadata[..., self.a_slice]
        self.update_text()
    
    def update_layout(self):
        if self.state == "3D":
            self.layout.addWidget(self.sagittal_label, 0, 1)
            self.layout.addWidget(self.coronal_label, 0, 0)
            self.layout.addWidget(self.axial_label, 1, 0)
            for i in reversed(range( self.layout.count())):
                label = self.layout.itemAt(i).widget()
                label.setMinimumSize(THREED_WIDTH, THREED_HEIGHT)
                label.state = self.state
                label.update_image()
        else:
            if self.state == "Sagittal":
                self.layout.addWidget(self.sagittal_label, 0,0)
            elif self.state == "Coronal":
                self.layout.addWidget(self.coronal_label, 0,0)
            elif self.state == "Axial":
                if len(self.shape) == 4:
                    if self.mosaic is True:
                        if self.mosaic_data is None: self.mosaic_data = create_4D_grid(self.data)
                        self.axial_label.mosaic = True
                        self.axial_label.data = self.mosaic_data[...,0]
                        self.a_slice = 0
                        self.axial_label.slice_index = self.a_slice
                        self.axial_label.z = self.data.shape[3]
                        self.offset = 1
                    else:
                        self.axial_label.data = self.adata[...,0]
                        self.axial_label.mosaic = False
                        self.a_slice = self.adata.shape[2] // 2
                        self.axial_label.slice_index = self.a_slice
                        self.axial_label.z = self.adata.shape[2]
                        self.offset = self.data.shape[2]

                self.layout.addWidget(self.axial_label, 0,0)

            label = self.layout.itemAt(0).widget()
            label.setMinimumSize(ONED_WIDTH, ONED_HEIGHT)
            label.state = self.state
            label.update_image()

    def update_label_aif(self):
        self.update_text()
        if ("perf." in self.path or "t2star." in self.path or "t1w." in self.path) and len(self.shape) == 4:
            x = min((self.axial_label.roi_x + (self.axial_label.roi_width // 2)) * self.shape[1] / ONED_WIDTH, self.shape[1]-1)
            y = min((self.axial_label.roi_y + (self.axial_label.roi_width // 2)) * self.shape[0] / ONED_WIDTH, self.shape[0]-1)
            z =  self.a_slice % self.shape[2]
            temp = np.rot90(self.data)
            aif = temp[int(y), int(x), z, :]

            idx = np.where(np.max(self.data[:,:,z,:], axis=-1) > 1000)
            mean = np.mean(self.data[idx[0], idx[1], 15, :], axis=0)

            self.plot_aif.clear()
            self.plot_aif.plot(aif, pen=pg.mkPen(color=(255, 0, 0)))
            self.plot_aif.plot(mean, pen=pg.mkPen(color=(255, 255, 255)))
            self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(100, 100, 255)))
            self.v_line.setPos(self.a_slice // self.shape[2])
            self.plot_aif.addItem(self.v_line)
        else:
            aif = None
            mean = None

    def update_label_roi(self):
        self.update_text()
        if (is_dsc_param(self.path) or is_dsc_param(self.axial_label.overlay_path)) and len(self.shape) == 3:
            self.plot_aif.clear()
            histo, bins = np.histogram(self.axial_label.roi, bins=49)
            # print(histo)
            self.plot_aif.plot(histo, pen=pg.mkPen(color=(255, 0, 0)))
            self.update_roi(
                self.axial_label.roi_xx, 
                self.axial_label.roi_yy, 
                self.axial_label.slice_index,
                self.axial_label.roi_mean,
                self.axial_label.roi_std,
                self.axial_label.roi_min,
                self.axial_label.roi_max,
                self.axial_label.roi_mm3,
            )
            # self.plot_aif.plot(mean, pen=pg.mkPen(color=(255, 255, 255)))
        
        # self.plot_aif(aif)

    def change_lut(self, lut):
        self.lut = lut
        color_lut = create_lut(lut)
        for i in reversed(range(self.layout.count())):
            label = self.layout.itemAt(i).widget()
            label.lut = color_lut
            label.update_image()

    def change_overlay_lut(self, lut):
        self.overlay_lut = lut
        color_lut = create_lut(lut)
        for i in reversed(range(self.layout.count())):
            label = self.layout.itemAt(i).widget()
            label.overlay_lut = color_lut
            label.update_image()

    def add_overlay(self, path):
        print("INFO - Loading overlay")
        self.opath = path
        self.oimg = nib.load(self.opath)
        self.odata = self.oimg.get_fdata()
        self.oshape = self.oimg.shape
        self.opixdim = self.oimg.header["pixdim"][1:4]

        self.osdata, _, self.ospixdim, _ = to_sagittal(self.odata, self.opixdim, self.opath)
        self.ocdata, _, self.ocpixdim, _ = to_coronal(self.odata, self.opixdim, self.opath)
        self.oadata, _, self.oapixdim, _ = to_axial(self.odata, self.opixdim, self.opath)

        # If not in the same space, alert user
        if self.oshape != self.shape:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Warning")
            dlg.setIcon(QMessageBox.Icon.Warning)
            dlg.setText("The overlay data is not in the\nsame space than the previous data")
            button = dlg.exec()

        if is_dsc_param(self.opath):
            lut = "Rainbow"
        else:
            lut = "Grayscale"
        self.sagittal_label.set_overlay(path, self.osdata[..., self.s_slice], self.ospixdim, lut)
        self.coronal_label.set_overlay(path, self.ocdata[..., self.c_slice], self.ocpixdim, lut)
        self.axial_label.set_overlay(path, self.oadata[..., self.a_slice], self.oapixdim, lut)
        print("INFO - Overlay successfully loaded")

    def keyPressEvent(self, event):
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key_S and not self.ensure3D:
                self.state = "Sagittal"
                reset_layout(self.layout)
                self.update_layout()
                self.update_text()
            elif event.key() == Qt.Key_C and not self.ensure3D:
                self.state = "Coronal"
                reset_layout(self.layout)
                self.update_text()
                self.update_layout()
            elif event.key() == Qt.Key_A:
                self.state = "Axial"
                self.mosaic = False
                reset_layout(self.layout)
                self.update_layout()
                self.update_text()
            elif event.key() == Qt.Key_F and not self.ensure3D:
                self.state = "3D"
                reset_layout(self.layout)
                self.update_layout()
                self.update_text()
            elif event.key() == Qt.Key_M:
                self.mosaic = True
                reset_layout(self.layout)
                self.update_layout()
                self.update_text()
            elif self.ensure3D:
                if event.key() == Qt.Key_Right:
                    new_index = self.a_slice + self.offset
                    if self.mosaic:
                        if new_index > self.data.shape[3] - 1: new_index -= self.offset
                    else:
                        if new_index > self.adata.shape[2] - 1: new_index -= self.offset
                    self.update_slice(new_index, "Axial")
                    self.v_line.setPos(self.a_slice // self.shape[2])
                    self.axial_label.update_image()
                elif event.key() == Qt.Key_Left:
                    new_index = max(self.a_slice - self.offset, self.a_slice % self.offset)
                    self.update_slice(new_index, "Axial")
                    self.v_line.setPos(self.a_slice // self.shape[2])
                    self.axial_label.update_image()

        return super().keyPressEvent(event)