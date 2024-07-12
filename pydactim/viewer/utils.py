from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
import nibabel as nib
from numba import njit
import numpy as np

def is_dsc_param(path):
        if "CBV" in path.upper() or "CBF" in path.upper() or "MTT" in path.upper() or "TTP" in path.upper():
            return True
        return False

@njit
def to_sagittal(data, pixdim, path):
    sdata = np.transpose(data, (1,2,0))
    spixdim = [pixdim[1], pixdim[2], pixdim[0]]
    sdata = np.rot90(sdata) if "anat" in path else np.rot90(sdata[:,:,::-1])
    sdata = np.fliplr(sdata)
    s_slice = sdata.shape[2] // 2
    return sdata, s_slice, spixdim, sdata.shape[2]

@njit
def to_coronal(data, pixdim, path):
    cdata = np.transpose(data, (0,2,1))
    cpixdim = [pixdim[0], pixdim[2], pixdim[1]]
    cdata = np.rot90(cdata[:,:,::-1])
    if "anat" in path:
        cdata = np.fliplr(cdata)
    elif "dwi" in path and "flirt" in path:
        cdata = np.fliplr(cdata)
    elif "perf" in path and "flirt" in path:
        cdata = np.fliplr(cdata)
    c_slice = cdata.shape[2] // 2
    return cdata, c_slice, cpixdim, cdata.shape[2]

@njit
def to_axial(data, pixdim, path):
    adata = np.rot90(data)
    apixdim = list(pixdim)
    if "anat" in path and not "flirt" in path:
        adata = np.fliplr(adata) 
    elif "dwi" in path and "flirt" in path:
        adata = np.fliplr(adata)
    elif "perf" in path and "flirt" in path:
        adata = np.fliplr(adata)
    a_slice = adata.shape[2] // 2
    return adata, a_slice, apixdim, adata.shape[2]

def reset_layout(layout):
    # Clear the layout
    for i in reversed(range(layout.count())):
        widget = layout.itemAt(i).widget()
        if widget is not None:
            layout.removeWidget(widget)
            widget.setParent(None)

def create_lut(lut):
    if lut == "Grayscale":
        color_lut = np.zeros((256, 3), dtype=np.uint8)
        color_lut[:, 0] = np.arange(256)
        color_lut[:, 1] = np.arange(256)
        color_lut[:, 2] = np.arange(256)
    elif lut == "Red":
        color_lut = np.zeros((256, 3), dtype=np.uint8)
        color_lut[:, 0] = np.arange(256)
        color_lut[:, 1] = 0
        color_lut[:, 2] = 0
    elif lut == "Green":
        color_lut = np.zeros((256, 3), dtype=np.uint8)
        color_lut[:, 0] = 0
        color_lut[:, 1] = np.arange(256)
        color_lut[:, 2] = 0
    elif lut == "Blue":
        color_lut = np.zeros((256, 3), dtype=np.uint8)
        color_lut[:, 0] = 0
        color_lut[:, 1] = 0
        color_lut[:, 2] = np.arange(256)
    elif lut == "Rainbow":
        colormap = plt.get_cmap("jet", 256)
        color_lut = (colormap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    elif lut == "Spectral":
        colormap = plt.get_cmap("nipy_spectral", 256)
        color_lut = (colormap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    elif lut == "Flow":
        colormap = plt.get_cmap("coolwarm", 256)
        color_lut = (colormap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    elif lut == "Viridis":
        colormap = plt.get_cmap("viridis", 256)
        color_lut = (colormap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    elif lut == "Qualitative":
        colormap = plt.get_cmap("Paired", 256)
        color_lut = (colormap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    elif lut == "Random":
        colormap = plt.get_cmap("prism", 256)
        color_lut = (colormap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    return color_lut

def create_thumbnail(path):
    img = nib.load(path)
    data = img.get_fdata()

    def thumbnail(data):
        if len(data.shape) == 4:
            data = data.transpose(0, 1, 3, 2).reshape(
                data.shape[0], 
                data.shape[1], 
                data.shape[2]*data.shape[3]
            )
            
        data = np.rot90(data)
        if "anat" in path:
            data = np.fliplr(data)
        elif "dwi" in path and "flirt" in path:
            data = np.fliplr(data)
        elif "perf" in path and "flirt" in path:
            data = np.fliplr(data)

        slice_index = data.shape[2] // 2
        data = data[..., slice_index]
        return data
        
    data = thumbnail(data)
    return data, list(img.shape), list(img.header["pixdim"][1:4])

def to_3D(array, pixdim):
    array = np.rot90(array)
    array = np.transpose(array, (0, 1, 3, 2))
    array = np.reshape(array,
        (array.shape[0], 
        array.shape[1], 
        array.shape[2]*array.shape[3])
    )
    pixdim = [pixdim[1], pixdim[0], pixdim[2]]
    return array, pixdim, array.shape[2]

@njit
def create_3D_grid(array):
    num_slices = array.shape[2]
    num_rows = int(np.ceil(np.sqrt(num_slices)))  # Number of rows in the grid
    num_cols = int(np.ceil(num_slices / num_rows))  # Number of columns in the grid

    # Create a blank canvas to store the slices
    canvas_height = array.shape[0] * num_rows
    canvas_width = array.shape[1] * num_cols
    canvas = np.zeros((canvas_height, canvas_width), dtype=array.dtype)

    # Fill the canvas with the slices
    for i in range(num_slices):
        row = i // num_cols
        col = i % num_cols
        canvas[row * array.shape[0]:(row + 1) * array.shape[0], col * array.shape[1]:(col + 1) * array.shape[1]] = array[:, :, i]
    return canvas

@njit
def create_4D_grid(array):
    array = np.rot90(array)
    num_slices = array.shape[2]
    num_rows = int(np.ceil(np.sqrt(num_slices)))  # Number of rows in the grid
    num_cols = int(np.ceil(num_slices / num_rows))  # Number of columns in the grid

    # Create a blank canvas to store the slices
    canvas_height = array.shape[0] * num_rows
    canvas_width = array.shape[1] * num_cols
    canvas = np.zeros((canvas_height, canvas_width, array.shape[3]), dtype=array.dtype)

    # Fill the canvas with the slices
    for i in range(array.shape[3]):
        temp = create_3D_grid(array[:,:,:,i])
        canvas[:,:,i] = temp
    return canvas

def get_darkModePalette(app=None):
    darkPalette = app.palette()
    darkPalette.setColor(QPalette.Window, QColor(53, 53, 53))
    darkPalette.setColor(QPalette.WindowText, Qt.white)
    darkPalette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    darkPalette.setColor(QPalette.Base, QColor(42, 42, 42))
    darkPalette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
    darkPalette.setColor(QPalette.ToolTipBase, Qt.white)
    darkPalette.setColor(QPalette.ToolTipText, Qt.white)
    darkPalette.setColor(QPalette.Text, Qt.white)
    darkPalette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    darkPalette.setColor(QPalette.Dark, QColor(35, 35, 35))
    darkPalette.setColor(QPalette.Shadow, QColor(20, 20, 20))
    darkPalette.setColor(QPalette.Button, QColor(53, 53, 53))
    darkPalette.setColor(QPalette.ButtonText, Qt.white)
    darkPalette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    darkPalette.setColor(QPalette.BrightText, Qt.red)
    darkPalette.setColor(QPalette.Link, QColor(42, 130, 218))
    darkPalette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    darkPalette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    darkPalette.setColor(QPalette.HighlightedText, Qt.white)
    darkPalette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127),)
    return darkPalette