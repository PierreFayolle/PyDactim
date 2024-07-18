"""Mix between the MPUnet published in JMI and Unet++ declared with Pytorch"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3d(in_channels: int, out_channels: int):
    return nn.Conv3d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     padding=1)


class MPUnetPP2CBND(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int):
        super().__init__()
        self.dropout_p = 0.3  # The default slope of leaky rely in keras is 0.3, default in pytorch is 0.01
        # Encoder
        #   First level
        self.conv000 = conv3d(n_inputs, 6)
        self.conv001 = conv3d(6, 6)
        self.drop000 = nn.Dropout3d(self.dropout_p)
        self.bn000 = nn.BatchNorm3d(6)
        self.bn001 = nn.BatchNorm3d(6)
        #   Second level
        self.conv100 = conv3d(6, 7)
        self.conv101 = conv3d(7, 7)
        self.drop100 = nn.Dropout3d(self.dropout_p)
        self.bn100 = nn.BatchNorm3d(7)
        self.bn101 = nn.BatchNorm3d(7)
        #   Third level
        self.conv200 = conv3d(7, 8)
        self.conv201 = conv3d(8, 8)
        self.drop200 = nn.Dropout3d(self.dropout_p)
        self.bn200 = nn.BatchNorm3d(8)
        self.bn201 = nn.BatchNorm3d(8)

        # Bottleneck
        self.conv300 = conv3d(8, 9)
        self.conv301 = conv3d(9, 9)
        self.drop300 = nn.Dropout3d(self.dropout_p)
        self.bn300 = nn.BatchNorm3d(9)
        self.bn301 = nn.BatchNorm3d(9)

        # Middle of Unet++
        #   First level
        #       Second column
        self.conv002 = conv3d(13, 6)
        self.conv003 = conv3d(6, 6)
        self.conv_seg0 = nn.Conv3d(in_channels=6,
                                   out_channels=n_outputs,
                                   kernel_size=1,
                                   padding=0)
        self.drop002 = nn.Dropout3d(self.dropout_p)
        self.bn002 = nn.BatchNorm3d(6)
        self.bn003 = nn.BatchNorm3d(6)
        #       Third column
        self.conv004 = conv3d(19, 6)
        self.conv005 = conv3d(6, 6)
        self.conv_seg1 = nn.Conv3d(in_channels=6,
                                   out_channels=n_outputs,
                                   kernel_size=1,
                                   padding=0)
        self.drop004 = nn.Dropout3d(self.dropout_p)
        self.bn004 = nn.BatchNorm3d(6)
        self.bn005 = nn.BatchNorm3d(6)
        #   Second level
        #       Second column
        self.conv102 = conv3d(15, 7)
        self.conv103 = conv3d(7, 7)
        self.drop102 = nn.Dropout3d(self.dropout_p)
        self.bn102 = nn.BatchNorm3d(7)
        self.bn103 = nn.BatchNorm3d(7)

        # Decoder
        #   Third level
        self.conv202 = conv3d(17, 8)
        self.conv203 = conv3d(8, 8)
        self.drop202 = nn.Dropout3d(self.dropout_p)
        self.bn202 = nn.BatchNorm3d(8)
        self.bn203 = nn.BatchNorm3d(8)
        #   Second level
        self.conv104 = conv3d(22, 7)
        self.conv105 = conv3d(7, 7)
        self.drop104 = nn.Dropout3d(self.dropout_p)
        self.bn104 = nn.BatchNorm3d(7)
        self.bn105 = nn.BatchNorm3d(7)
        #   First level
        self.conv006 = conv3d(25, 6)
        self.conv007 = conv3d(6, n_outputs)
        self.drop006 = nn.Dropout3d(self.dropout_p)
        self.bn006 = nn.BatchNorm3d(6)

        # Leaky Relu slope parameter
        self.lr_slope = 0.3  # The default slope of leaky rely in keras is 0.3, default in pytorch is 0.01

    def _activ(self, input_layer):
        # Activation function
        return F.leaky_relu(input_layer,
                            negative_slope=self.lr_slope)

    def forward(self, image):
        # Encoder
        #   First level
        fms = self.drop000(self._activ(self.bn000(self.conv000(image))))
        fms = fms_00 = self._activ(self.bn001(self.conv001(fms)))
        #   Second level
        fms = self.drop100(self._activ(self.bn100(self.conv100(F.max_pool3d(fms, (2, 2, 2))))))
        fms = fms_10 = self._activ(self.bn101(self.conv101(fms)))
        #   Third level
        fms = self.drop200(self._activ(self.bn200(self.conv200(F.max_pool3d(fms, (2, 2, 2))))))
        fms = fms_20 = self._activ(self.bn201(self.conv201(fms)))

        # Bottleneck
        fms = self.drop300(self._activ(self.bn300(self.conv300(F.max_pool3d(fms, (2, 2, 2))))))
        fms = self._activ(self.bn301(self.conv301(fms)))

        # Middle of Unet++
        #   Second level
        #       Second column
        fms_11 = self.drop102(self._activ(self.bn102(self.conv102(torch.cat((fms_10, F.interpolate(fms_20,
                                                                                                   scale_factor=2,
                                                                                                   mode='nearest')
                                                                             ), dim=1)))))

        fms_11 = self._activ(self.bn103(self.conv103(fms_11)))

        #   First level
        #       Second column
        fms_01 = self.drop002(self._activ(self.bn002(self.conv002(torch.cat((fms_00,
                                                                             F.interpolate(fms_10,
                                                                                           scale_factor=2,
                                                                                           mode='nearest')
                                                                             ), dim=1)))))
        fms_01 = self._activ(self.bn003(self.conv003(fms_01)))
        seg_01 = self.conv_seg0(fms_01)
        #       Third column
        fms_02 = self.drop004(self._activ(self.bn004(self.conv004(torch.cat((fms_00,
                                                                             fms_01,
                                                                             F.interpolate(fms_11,
                                                                                           scale_factor=2,
                                                                                           mode='nearest')
                                                                             ), dim=1)))))
        fms_02 = self._activ(self.bn005(self.conv005(fms_02)))
        seg_02 = self.conv_seg1(fms_02)

        # Decoder
        #   Third level
        fms = self.drop202(self._activ(self.bn202(self.conv202(torch.cat((fms_20,
                                                                          F.interpolate(fms,
                                                                                        scale_factor=2,
                                                                                        mode='nearest')
                                                                          ),
                                                                         dim=1)))))
        fms = self._activ(self.bn203(self.conv203(fms)))
        #   Second level
        fms = self.drop104(self._activ(self.bn104(self.conv104(torch.cat((fms_10,
                                                                          fms_11,
                                                                          F.interpolate(fms,
                                                                                        scale_factor=2,
                                                                                        mode='nearest')
                                                                          ),
                                                                         dim=1)))))
        fms = self._activ(self.bn105(self.conv105(fms)))
        #   First level
        fms = self.drop006(self._activ(self.bn006(self.conv006(torch.cat((fms_00,
                                                                          fms_01,
                                                                          fms_02,
                                                                          F.interpolate(fms,
                                                                                        scale_factor=2,
                                                                                        mode='nearest')
                                                                          ),
                                                                         dim=1)))))
        fms = torch.sigmoid(self.conv007(fms) + seg_01 + seg_02)
        return fms


if __name__ ==  '__main__':
    import nibabel as nib

    weights_path = "210510231800/MPUnetPP2CBND_epoch_29_weights.pth"
    model = MPUnetPP2CBND(1, 1)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    patch = torch.rand((1, 1, 32, 32, 32))
    affine = np.eye(4, 4)
    with torch.no_grad():
        pred = model(patch).detach().numpy()[0, 0]
    nib.save(nib.Nifti1Image(pred, affine), f'test_pred.nii.gz')
