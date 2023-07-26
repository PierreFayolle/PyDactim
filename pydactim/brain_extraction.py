"""
https://github.com/MIC-DKFZ/HD-BET

Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A, Schlemmer HP, Heiland S, Wick W, Bendszus M, Maier-Hein KH, Kickingereder P. 
Automated brain extraction of multi-sequence MRI using artificial neural networks. 
Hum Brain Mapp. 2019; 1–13. 
https://doi.org/10.1002/hbm.24750

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from skimage.morphology import label
from abc import abstractmethod
import os


""" UTILS """
def get_params_fname(model_path, fold):
    return os.path.join(model_path, "%d.model" % fold)

def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def postprocess_prediction(seg):
    # basically look for connected components and choose the largest one, delete everything else
    print("running postprocessing... ")
    mask = seg != 0
    lbls = label(mask, connectivity=mask.ndim)
    lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
    largest_region = np.argmax(lbls_sizes[1:]) + 1
    seg[lbls != largest_region] = 0
    return seg

class SetNetworkToVal(object):
    def __init__(self, use_dropout_sampling=False, norm_use_average=True):
        self.norm_use_average = norm_use_average
        self.use_dropout_sampling = use_dropout_sampling

    def __call__(self, module):
        if isinstance(module, nn.Dropout3d) or isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout):
            module.train(self.use_dropout_sampling)
        elif isinstance(module, nn.InstanceNorm3d) or isinstance(module, nn.InstanceNorm2d) or \
                isinstance(module, nn.InstanceNorm1d) \
                or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or \
                isinstance(module, nn.BatchNorm1d):
            module.train(not self.norm_use_average)

""" CONFIG """
class BaseConfig(object):
    def __init__(self):
        pass

    @abstractmethod
    def get_split(self, fold, random_state=12345):
        pass

    @abstractmethod
    def get_network(self, mode="train"):
        pass

    @abstractmethod
    def get_basic_generators(self, fold):
        pass

    @abstractmethod
    def get_data_generators(self, fold):
        pass

    def preprocess(self, data):
        return data

    def __repr__(self):
        res = ""
        for v in vars(self):
            if not v.startswith("__") and not v.startswith("_") and v != 'dataset':
                res += (v + ": " + str(self.__getattribute__(v)) + "\n")
        return res

class HD_BET_Config(BaseConfig):
    def __init__(self):
        super(HD_BET_Config, self).__init__()

        self.EXPERIMENT_NAME = self.__class__.__name__ # just a generic experiment name

        # network parameters
        self.net_base_num_layers = 21
        self.BATCH_SIZE = 2
        self.net_do_DS = True
        self.net_dropout_p = 0.0
        self.net_use_inst_norm = True
        self.net_conv_use_bias = True
        self.net_norm_use_affine = True
        self.net_leaky_relu_slope = 1e-1

        # hyperparameters
        self.INPUT_PATCH_SIZE = (128, 128, 128)
        self.num_classes = 2
        self.selected_data_channels = range(1)

        # data augmentation
        self.da_mirror_axes = (2, 3, 4)

        # validation
        self.val_use_DO = False
        self.val_use_train_mode = False # for dropout sampling
        self.val_num_repeats = 1 # only useful if dropout sampling
        self.val_batch_size = 1 # only useful if dropout sampling
        self.val_save_npz = True
        self.val_do_mirroring = True # test time data augmentation via mirroring
        self.val_write_images = True
        self.net_input_must_be_divisible_by = 16  # we could make a network class that has this as a property
        self.val_min_size = self.INPUT_PATCH_SIZE
        self.val_fn = None

        # CAREFUL! THIS IS A HACK TO MAKE PYTORCH 0.3 STATE DICTS COMPATIBLE WITH PYTORCH 0.4 (setting keep_runnings_
        # stats=True but not using them in validation. keep_runnings_stats was True before 0.3 but unused and defaults
        # to false in 0.4)
        self.val_use_moving_averages = False

    def get_network(self, train=True, pretrained_weights=None):
        net = Network(self.num_classes, len(self.selected_data_channels), self.net_base_num_layers,
                               self.net_dropout_p, softmax_helper, self.net_leaky_relu_slope, self.net_conv_use_bias,
                               self.net_norm_use_affine, True, self.net_do_DS)

        if pretrained_weights is not None:
            net.load_state_dict(
                torch.load(pretrained_weights, map_location=lambda storage, loc: storage))

        if train:
            net.train(True)
        else:
            net.train(False)
            net.apply(SetNetworkToVal(self.val_use_DO, self.val_use_moving_averages))
            net.do_ds = False

        optimizer = None
        self.lr_scheduler = None
        return net, optimizer

    def get_data_generators(self, fold):
        pass

    def get_split(self, fold, random_state=12345):
        pass

    def get_basic_generators(self, fold):
        pass

    def on_epoch_end(self, epoch):
        pass

    def preprocess(self, data):
        data = np.copy(data)
        for c in range(data.shape[0]):
            data[c] -= data[c].mean()
            data[c] /= data[c].std()
        return data

""" NETWORDK ARCHITECTURE"""
class EncodingModule(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, dropout_p=0.3, leakiness=1e-2, conv_bias=True,
                 inst_norm_affine=True, lrelu_inplace=True):
        nn.Module.__init__(self)
        self.dropout_p = dropout_p
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.bn_1 = nn.InstanceNorm3d(in_channels, affine=self.inst_norm_affine, track_running_stats=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, filter_size, 1, (filter_size - 1) // 2, bias=self.conv_bias)
        self.dropout = nn.Dropout3d(dropout_p)
        self.bn_2 = nn.InstanceNorm3d(in_channels, affine=self.inst_norm_affine, track_running_stats=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, filter_size, 1, (filter_size - 1) // 2, bias=self.conv_bias)

    def forward(self, x):
        skip = x
        x = F.leaky_relu(self.bn_1(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.conv1(x)
        if self.dropout_p is not None and self.dropout_p > 0:
            x = self.dropout(x)
        x = F.leaky_relu(self.bn_2(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.conv2(x)
        x = x + skip
        return x

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=True):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

class LocalizationModule(nn.Module):
    def __init__(self, in_channels, out_channels, leakiness=1e-2, conv_bias=True, inst_norm_affine=True,
                 lrelu_inplace=True):
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.conv1 = nn.Conv3d(in_channels, in_channels, 3, 1, 1, bias=self.conv_bias)
        self.bn_1 = nn.InstanceNorm3d(in_channels, affine=self.inst_norm_affine, track_running_stats=True)
        self.conv2 = nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=self.conv_bias)
        self.bn_2 = nn.InstanceNorm3d(out_channels, affine=self.inst_norm_affine, track_running_stats=True)

    def forward(self, x):
        x = F.leaky_relu(self.bn_1(self.conv1(x)), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = F.leaky_relu(self.bn_2(self.conv2(x)), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        return x

class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels, leakiness=1e-2, conv_bias=True, inst_norm_affine=True,
                 lrelu_inplace=True):
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.upsample = Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.upsample_conv = nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=self.conv_bias)
        self.bn = nn.InstanceNorm3d(out_channels, affine=self.inst_norm_affine, track_running_stats=True)

    def forward(self, x):
        x = F.leaky_relu(self.bn(self.upsample_conv(self.upsample(x))), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        return x

class DownsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels, leakiness=1e-2, conv_bias=True, inst_norm_affine=True,
                 lrelu_inplace=True):
        nn.Module.__init__(self)
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.bn = nn.InstanceNorm3d(in_channels, affine=self.inst_norm_affine, track_running_stats=True)
        self.downsample = nn.Conv3d(in_channels, out_channels, 3, 2, 1, bias=self.conv_bias)

    def forward(self, x):
        x = F.leaky_relu(self.bn(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        b = self.downsample(x)
        return x, b

class Network(nn.Module):
    def __init__(self, num_classes=4, num_input_channels=4, base_filters=16, dropout_p=0.3,
                 final_nonlin=softmax_helper, leakiness=1e-2, conv_bias=True, inst_norm_affine=True,
                 lrelu_inplace=True, do_ds=True):
        super(Network, self).__init__()

        self.do_ds = do_ds
        self.lrelu_inplace = lrelu_inplace
        self.inst_norm_affine = inst_norm_affine
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.final_nonlin = final_nonlin
        self.init_conv = nn.Conv3d(num_input_channels, base_filters, 3, 1, 1, bias=self.conv_bias)

        self.context1 = EncodingModule(base_filters, base_filters, 3, dropout_p, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.down1 = DownsamplingModule(base_filters, base_filters * 2, leakiness=1e-2, conv_bias=True,
                                        inst_norm_affine=True, lrelu_inplace=True)

        self.context2 = EncodingModule(2 * base_filters, 2 * base_filters, 3, dropout_p, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.down2 = DownsamplingModule(2 * base_filters, base_filters * 4, leakiness=1e-2, conv_bias=True,
                                        inst_norm_affine=True, lrelu_inplace=True)

        self.context3 = EncodingModule(4 * base_filters, 4 * base_filters, 3, dropout_p, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.down3 = DownsamplingModule(4 * base_filters, base_filters * 8, leakiness=1e-2, conv_bias=True,
                                        inst_norm_affine=True, lrelu_inplace=True)

        self.context4 = EncodingModule(8 * base_filters, 8 * base_filters, 3, dropout_p, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.down4 = DownsamplingModule(8 * base_filters, base_filters * 16, leakiness=1e-2, conv_bias=True,
                                        inst_norm_affine=True, lrelu_inplace=True)

        self.context5 = EncodingModule(16 * base_filters, 16 * base_filters, 3, dropout_p, leakiness=1e-2,
                                       conv_bias=True, inst_norm_affine=True, lrelu_inplace=True)

        self.bn_after_context5 = nn.InstanceNorm3d(16 * base_filters, affine=self.inst_norm_affine, track_running_stats=True)
        self.up1 = UpsamplingModule(16 * base_filters, 8 * base_filters, leakiness=1e-2, conv_bias=True,
                                    inst_norm_affine=True, lrelu_inplace=True)

        self.loc1 = LocalizationModule(16 * base_filters, 8 * base_filters, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.up2 = UpsamplingModule(8 * base_filters, 4 * base_filters, leakiness=1e-2, conv_bias=True,
                                    inst_norm_affine=True, lrelu_inplace=True)

        self.loc2 = LocalizationModule(8 * base_filters, 4 * base_filters, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.loc2_seg = nn.Conv3d(4 * base_filters, num_classes, 1, 1, 0, bias=False)
        self.up3 = UpsamplingModule(4 * base_filters, 2 * base_filters, leakiness=1e-2, conv_bias=True,
                                    inst_norm_affine=True, lrelu_inplace=True)

        self.loc3 = LocalizationModule(4 * base_filters, 2 * base_filters, leakiness=1e-2, conv_bias=True,
                                       inst_norm_affine=True, lrelu_inplace=True)
        self.loc3_seg = nn.Conv3d(2 * base_filters, num_classes, 1, 1, 0, bias=False)
        self.up4 = UpsamplingModule(2 * base_filters, 1 * base_filters, leakiness=1e-2, conv_bias=True,
                                    inst_norm_affine=True, lrelu_inplace=True)

        self.end_conv_1 = nn.Conv3d(2 * base_filters, 2 * base_filters, 3, 1, 1, bias=self.conv_bias)
        self.end_conv_1_bn = nn.InstanceNorm3d(2 * base_filters, affine=self.inst_norm_affine, track_running_stats=True)
        self.end_conv_2 = nn.Conv3d(2 * base_filters, 2 * base_filters, 3, 1, 1, bias=self.conv_bias)
        self.end_conv_2_bn = nn.InstanceNorm3d(2 * base_filters, affine=self.inst_norm_affine, track_running_stats=True)
        self.seg_layer = nn.Conv3d(2 * base_filters, num_classes, 1, 1, 0, bias=False)

    def forward(self, x):
        seg_outputs = []

        x = self.init_conv(x)
        x = self.context1(x)

        skip1, x = self.down1(x)
        x = self.context2(x)

        skip2, x = self.down2(x)
        x = self.context3(x)

        skip3, x = self.down3(x)
        x = self.context4(x)

        skip4, x = self.down4(x)
        x = self.context5(x)

        x = F.leaky_relu(self.bn_after_context5(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace)
        x = self.up1(x)

        x = torch.cat((skip4, x), dim=1)
        x = self.loc1(x)
        x = self.up2(x)

        x = torch.cat((skip3, x), dim=1)
        x = self.loc2(x)
        loc2_seg = self.final_nonlin(self.loc2_seg(x))
        seg_outputs.append(loc2_seg)
        x = self.up3(x)

        x = torch.cat((skip2, x), dim=1)
        x = self.loc3(x)
        loc3_seg = self.final_nonlin(self.loc3_seg(x))
        seg_outputs.append(loc3_seg)
        x = self.up4(x)

        x = torch.cat((skip1, x), dim=1)
        x = F.leaky_relu(self.end_conv_1_bn(self.end_conv_1(x)), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        x = F.leaky_relu(self.end_conv_2_bn(self.end_conv_2(x)), negative_slope=self.leakiness,
                         inplace=self.lrelu_inplace)
        x = self.final_nonlin(self.seg_layer(x))
        seg_outputs.append(x)

        if self.do_ds:
            return seg_outputs[::-1]
        else:
            return seg_outputs[-1]

"""DATA LOADING"""
def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge', cval=0, anti_aliasing=False)

def preprocess_image(itk_image, is_seg=False, spacing_target=(1, 0.5, 0.5)):
    spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
    image = sitk.GetArrayFromImage(itk_image).astype(float)

    if len(image.shape) == 4:
        image = np.squeeze(image, axis=0)

    if not is_seg:
        if np.any([[i != j] for i, j in zip(spacing, spacing_target)]):
            image = resize_image(image, spacing, spacing_target).astype(np.float32)

        image -= image.mean()
        image /= image.std()
    else:
        new_shape = (int(np.round(spacing[0] / spacing_target[0] * float(image.shape[0]))),
                     int(np.round(spacing[1] / spacing_target[1] * float(image.shape[1]))),
                     int(np.round(spacing[2] / spacing_target[2] * float(image.shape[2]))))
        image = resize_segmentation(image, new_shape, 1)
    return image

def load_and_preprocess(mri_file):
    images = {}
    # t1
    images["T1"] = mri_file

    properties_dict = {
        "spacing": images["T1"].GetSpacing(),
        "direction": images["T1"].GetDirection(),
        "size": images["T1"].GetSize(),
        "origin": images["T1"].GetOrigin()
    }

    direction = list(properties_dict["direction"])
    if len(direction) == 16: 
        direction = direction[:11]
        direction.pop(3)
        direction.pop(6)
        properties_dict["direction"] = tuple(direction)

    for k in images.keys():
        images[k] = preprocess_image(images[k], is_seg=False, spacing_target=(1.5, 1.5, 1.5))

    properties_dict['size_before_cropping'] = images["T1"].shape
    imgs = []
    for seq in ['T1']:
        imgs.append(images[seq][None])
    all_data = np.vstack(imgs)
    return all_data, properties_dict

def segmentation_nifti(segmentation, dct, order=1):
    '''
    segmentation must have the same spacing as the original nifti (for now). segmentation may have been cropped out
    of the original image

    dct:
    size_before_cropping
    brain_bbox
    size -> this is the original size of the dataset, if the image was not resampled, this is the same as size_before_cropping
    spacing
    origin
    direction

    :param segmentation:
    :param dct:
    :return:
    '''
    old_size = dct.get('size_before_cropping')
    bbox = dct.get('brain_bbox')
    if bbox is not None:
        seg_old_size = np.zeros(old_size)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + segmentation.shape[c], old_size[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
                     bbox[1][0]:bbox[1][1],
                     bbox[2][0]:bbox[2][1]] = segmentation
    else:
        seg_old_size = segmentation
    if np.any(np.array(seg_old_size) != np.array(dct['size'])[[2, 1, 0]]):
        seg_old_spacing = resize_segmentation(seg_old_size, np.array(dct['size'])[[2, 1, 0]], order=order)
    else:
        seg_old_spacing = seg_old_size
    seg_resized_itk = sitk.GetImageFromArray(seg_old_spacing.astype(np.int32))
    seg_resized_itk.SetSpacing(np.array(dct['spacing'])[[0, 1, 2]])
    seg_resized_itk.SetOrigin(dct['origin'])
    seg_resized_itk.SetDirection(dct['direction'])
    return seg_resized_itk

def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    '''
    Taken from batchgenerators (https://github.com/MIC-DKFZ/batchgenerators) to prevent dependency

    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation, new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            reshaped_multihot = resize((segmentation == c).astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

""" PREDICT CASE """
def pad_patient_3D(patient, shape_must_be_divisible_by=16, min_size=None):
    if not (isinstance(shape_must_be_divisible_by, list) or isinstance(shape_must_be_divisible_by, tuple)):
        shape_must_be_divisible_by = [shape_must_be_divisible_by] * 3
    shp = patient.shape
    new_shp = [shp[0] + shape_must_be_divisible_by[0] - shp[0] % shape_must_be_divisible_by[0],
               shp[1] + shape_must_be_divisible_by[1] - shp[1] % shape_must_be_divisible_by[1],
               shp[2] + shape_must_be_divisible_by[2] - shp[2] % shape_must_be_divisible_by[2]]
    for i in range(len(shp)):
        if shp[i] % shape_must_be_divisible_by[i] == 0:
            new_shp[i] -= shape_must_be_divisible_by[i]
    if min_size is not None:
        new_shp = np.max(np.vstack((np.array(new_shp), np.array(min_size))), 0)
    return reshape_by_padding_upper_coords(patient, new_shp, 0), shp

def reshape_by_padding_upper_coords(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2,len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0,0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    if len(shape) == 2:
        res[0:0+int(shape[0]), 0:0+int(shape[1])] = image
    elif len(shape) == 3:
        res[0:0+int(shape[0]), 0:0+int(shape[1]), 0:0+int(shape[2])] = image
    return res

def predict_case_3D_net(net, patient_data, do_mirroring, num_repeats, BATCH_SIZE=None,
                           new_shape_must_be_divisible_by=16, min_size=None, main_device=0, mirror_axes=(2, 3, 4)):
    with torch.no_grad():
        pad_res = []
        for i in range(patient_data.shape[0]):
            t, old_shape = pad_patient_3D(patient_data[i], new_shape_must_be_divisible_by, min_size)
            pad_res.append(t[None])

        patient_data = np.vstack(pad_res)

        new_shp = patient_data.shape

        data = np.zeros(tuple([1] + list(new_shp)), dtype=np.float32)

        data[0] = patient_data

        if BATCH_SIZE is not None:
            data = np.vstack([data] * BATCH_SIZE)

        a = torch.rand(data.shape).float()

        if main_device == 'cpu':
            pass
        else:
            a = a.cuda(main_device)

        if do_mirroring:
            x = 8
        else:
            x = 1
        all_preds = []
        for i in range(num_repeats):
            for m in range(x):
                data_for_net = np.array(data)
                do_stuff = False
                if m == 0:
                    do_stuff = True
                    pass
                if m == 1 and (4 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, :, :, ::-1]
                if m == 2 and (3 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, :, ::-1, :]
                if m == 3 and (4 in mirror_axes) and (3 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, :, ::-1, ::-1]
                if m == 4 and (2 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, ::-1, :, :]
                if m == 5 and (2 in mirror_axes) and (4 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, ::-1, :, ::-1]
                if m == 6 and (2 in mirror_axes) and (3 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, ::-1, ::-1, :]
                if m == 7 and (2 in mirror_axes) and (3 in mirror_axes) and (4 in mirror_axes):
                    do_stuff = True
                    data_for_net = data_for_net[:, :, ::-1, ::-1, ::-1]

                if do_stuff:
                    _ = a.data.copy_(torch.from_numpy(np.copy(data_for_net)))
                    p = net(a)  # np.copy is necessary because ::-1 creates just a view i think
                    p = p.data.cpu().numpy()

                    if m == 0:
                        pass
                    if m == 1 and (4 in mirror_axes):
                        p = p[:, :, :, :, ::-1]
                    if m == 2 and (3 in mirror_axes):
                        p = p[:, :, :, ::-1, :]
                    if m == 3 and (4 in mirror_axes) and (3 in mirror_axes):
                        p = p[:, :, :, ::-1, ::-1]
                    if m == 4 and (2 in mirror_axes):
                        p = p[:, :, ::-1, :, :]
                    if m == 5 and (2 in mirror_axes) and (4 in mirror_axes):
                        p = p[:, :, ::-1, :, ::-1]
                    if m == 6 and (2 in mirror_axes) and (3 in mirror_axes):
                        p = p[:, :, ::-1, ::-1, :]
                    if m == 7 and (2 in mirror_axes) and (3 in mirror_axes) and (4 in mirror_axes):
                        p = p[:, :, ::-1, ::-1, ::-1]
                    all_preds.append(p)

        stacked = np.vstack(all_preds)[:, :, :old_shape[0], :old_shape[1], :old_shape[2]]
        predicted_segmentation = stacked.mean(0).argmax(0)
        uncertainty = stacked.var(0)
        bayesian_predictions = stacked
        softmax_pred = stacked.mean(0)
    return predicted_segmentation, bayesian_predictions, softmax_pred, uncertainty

""" RUN """
def apply_bet(img, bet):
    img_itk = sitk.GetArrayFromImage(img)
    img_bet = sitk.GetArrayFromImage(bet)
    img_itk[img_bet == 0] = 0
    return img_itk

def run_hd_bet(input_data, model_path, mode="accurate", device=0,
               postprocess=False, do_tta=True, keep_mask=True, overwrite=True):
    """

    :param input_data: str or list/tuple of str
    :param mode: fast or accurate
    :param config_file: config.py
    :param device: either int (for device id) or 'cpu'
    :param postprocess: whether to do postprocessing or not. Postprocessing here consists of simply discarding all
    but the largest predicted connected component. Default False
    :param do_tta: whether to do test time data augmentation by mirroring along all axes. Default: True. If you use
    CPU you may want to turn that off to speed things up
    :return:
    """

    list_of_param_files = []

    if mode == 'fast':
        params_file = get_params_fname(model_path, 0)

        list_of_param_files.append(params_file)
    elif mode == 'accurate':
        for i in range(5):
            params_file = get_params_fname(model_path, i)
            list_of_param_files.append(params_file)
    else:
        raise ValueError("Unknown value for mode: %s. Expected: fast or accurate" % mode)

    assert all([os.path.isfile(i) for i in list_of_param_files]), "Could not find parameter files"

    cf = HD_BET_Config()

    net, _ = cf.get_network(cf.val_use_train_mode, None)
    if device == "cpu":
        net = net.cpu()
    else:
        net.cuda(device)

    params = []
    for p in list_of_param_files:
        params.append(torch.load(p, map_location=lambda storage, loc: storage))


        try:
            data, data_dict = load_and_preprocess(input_data)
        except RuntimeError:
            print("\nERROR\nCould not read file", input_data, "\n")
            continue
        except AssertionError as e:
            print(e)
            continue


        softmax_preds = []

        for i, p in enumerate(params):
            net.load_state_dict(p)
            net.eval()
            net.apply(SetNetworkToVal(False, False))
            _, _, softmax_pred, _ = predict_case_3D_net(net, data, do_tta, cf.val_num_repeats,
                                                        cf.val_batch_size, cf.net_input_must_be_divisible_by,
                                                        cf.val_min_size, device, cf.da_mirror_axes)
            softmax_preds.append(softmax_pred[None])

        seg = np.argmax(np.vstack(softmax_preds).mean(0), 0)

        if postprocess:
            seg = postprocess_prediction(seg)

        seg = segmentation_nifti(seg, data_dict)
        final = apply_bet(input_data, seg)
        return final, sitk.GetArrayFromImage(seg)

if __name__ == "__main__":
    path = r"F:\Dev\taima-2022\data\nifti\AX_T1_MPRAGE.nii.gz"
    run_hd_bet(sitk.ReadImage(path), model_path=r"C:\Users\467355\Documents\HD-BET-master\HD_BET\hd-bet_params")