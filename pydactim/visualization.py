import torchio as tio
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from .utils import get_name_of_path

def plot_histo(input_path):
    # Load your MRI image data into a numpy array
    mri_image = nib.load(input_path).get_fdata()

    # Flatten the 3D array to a 1D array
    image_flat = mri_image.flatten()

    # Plot the histogram
    plt.hist(image_flat, bins=256, alpha=0.75)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of MRI Image')
    plt.show()

def a_plot_histo(intput_data):
    # Flatten the 3D array to a 1D array
    image_flat = intput_data.flatten()

    # Plot the histogram
    plt.hist(image_flat, bins=256, range=(1, np.amax(intput_data)), alpha=0.75)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of MRI Image')
    plt.show()

def old_plot(*images):
    subject = tio.Subject(temp=tio.ScalarImage(images[0]))
    if len(images) > 1:
        for img in images:
            title = get_name_of_path(os.path.basename(img).replace(".nii.gz", ""))
            subject.add_image(tio.ScalarImage(img), title)
    else:
        title = get_name_of_path(os.path.basename(images[0]).replace(".nii.gz", ""))
        subject.add_image(tio.ScalarImage(images[0]), title)

    subject.remove_image("temp")
    subject.plot()

def plot(input_data, pixdim=None, slice=None, save=None):
    """ Plot image from a nifti file path or from an array.
        If using this function with an array, make sure to specify the pixdim value.

        Parameters
        ----------
        input_data : str/array
            Nifti file path or array that will be plotted

        pixdim : list, optional
            List of size 3 of the dimension of the voxels

        slice : int, optional
            The axial slice that will be displayed 
            
        save : str, optional
            The path to save the plot if defined

        """
    print(f"INFO - Plotting image")
    if type(input_data) is str:
        img = nib.load(input_data)
        data = img.get_fdata()
        pixdim = [round(dim, 2) for dim in img.header["pixdim"][1:4].astype(float)]
        title = f"{os.path.basename(input_data)} (shape={data.shape}, pixdim={pixdim})"
    else:
        data = input_data
        if pixdim is None:
            pixdim = [1,1,1] # Default pixdim
            print("WARNING - 'Pixdim' parameter is not defined, hence the pixel dimensions will be set to default (can deform the image)")
        if not isinstance(pixdim, list) or len(pixdim) != 3: raise ValueError(f"ERROR - The pixdim must be an instance of list with 3 elements, verify this:\n\t{pixdim :}")
        title = f"shape={data.shape}, pixdim={pixdim}"

    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    indices = np.array(data.shape) // 2
    
    i, j, k = indices
    if slice is not None: k = slice
    sag = np.rot90(np.fliplr(data[i, :, :]), -1)
    cor = np.rot90(np.fliplr(data[:, j, :]), -1)
    tra = np.rot90(np.fliplr(data[:, :, k]), -1)

    sag_aspect = pixdim[2] / pixdim[1]
    ax1.imshow(sag, aspect=sag_aspect, cmap="gray")
    ax1.set_title(f'Sag (slice={str(i)})', y=0.9)
    ax1.axis('off')

    cor_aspect = pixdim[2] / pixdim[0]
    ax2.imshow(cor, aspect=cor_aspect, cmap="gray")
    ax2.set_title(f'Cor (slice={str(j)})', y=0.9)
    ax2.axis('off')

    tra_aspect = pixdim[1] / pixdim[0]
    ax3.imshow(tra, aspect=tra_aspect, cmap="gray")
    ax3.set_title(f'Tra (slice={str(k)})', y=0.9)
    ax3.axis('off')

    fig.text(0.5, 0.95, title, horizontalalignment="center")
    if save is not None: plt.savefig(save)
    plt.tight_layout()
    plt.show()

class plot3D():
    def __init__(self, input_path):
        self.volume = nib.load(input_path)
        self.volume_data = self.check_orientation(self.volume)[::-1,:,::-1]
        self.multi_slice_viewer()

    def check_orientation(self, volume):
        x, y, z = nib.aff2axcodes(volume.affine)
        volume_data = volume.get_fdata()
        if x != 'R':
            volume_data = nib.orientations.flip_axis(volume_data, axis=0)
        if y != 'P':
            volume_data = nib.orientations.flip_axis(volume_data, axis=1)
        if z != 'S':
            volume_data = nib.orientations.flip_axis(volume_data, axis=2)
        return volume_data
        
    def multi_slice_viewer(self):
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=[18,10])

        ax1.volume = np.transpose(self.volume_data, (0,2,1))
        ax1.index = ax1.volume.shape[0] // 2
        ax1.imshow(ax1.volume[ax1.index], cmap="gray", animated=True)

        ax2.volume = np.transpose(ax1.volume, (2,1,0))
        ax2.index = ax2.volume.shape[0] // 2
        ax2.imshow(ax2.volume[ax2.index], cmap="gray", animated=True)

        ax3.volume = np.transpose(ax1.volume, (1,2,0))
        ax3.index = ax3.volume.shape[0] // 2
        ax3.imshow(ax3.volume[ax3.index], cmap="gray", animated=True)

        fig.canvas.mpl_connect('scroll_event', self.process_scroll)
        fig.canvas.mpl_connect('button_press_event', self.process_key)

        ax1.set_axis_off()
        ax2.set_axis_off()
        ax3.set_axis_off()

        plt.tight_layout()
        plt.show()

    def process_scroll(self, event):
        fig = event.canvas.figure
        ax = event.inaxes
        if ax == None: return

        if event.button == 'down':
            self.next_slice(ax)
            print("down")
        elif event.button == 'up':
            self.previous_slice(ax)
            print("up")
        fig.canvas.draw()

    def process_key(self, event):
        fig = event.canvas.figure
        ax = event.inaxes
        if ax == None: return
        
        x = int(event.xdata)
        y = int(event.ydata)

        axs = plt.gcf().get_axes()
        axs.remove(ax)
        self.clicked_slice(axs, x, y)

        fig.canvas.draw()

    def previous_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]
        ax.images[0].set_array((volume[ax.index]))

    def next_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array((volume[ax.index]))

    def clicked_slice(self, axs, x, y):
        axs[0].images[0].set_array((axs[0].volume[x]))
        axs[1].images[0].set_array((axs[1].volume[y]))
