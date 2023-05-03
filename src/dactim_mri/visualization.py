import torchio as tio
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

from dactim_mri.utils import get_name_of_path

def plot_histo(input_path):
    # Load your MRI image data into a numpy array
    mri_image = nib.load(input_path).get_fdata()

    # Flatten the 3D array to a 1D array
    image_flat = mri_image.flatten()

    # Plot the histogram
    plt.hist(image_flat, bins=256, range=(1, np.amax(mri_image)), alpha=0.75)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of MRI Image')
    plt.show()

def plot(*images):
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

if __name__ == '__main__':
    print(get_name_of_path(r"D:\Results\GLIOBIOPSY\derivative\sub-003\ses-01\anat\sub-003_ses-01_FLAIR_brain_1mm.nii.gz"))
    # plot(r"D:\Results\GLIOBIOPSY\derivative\sub-003\ses-01\anat\sub-003_ses-01_FLAIR_brain_1mm.nii.gz")