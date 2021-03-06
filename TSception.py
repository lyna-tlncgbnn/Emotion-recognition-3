import numpy as np
import paddle
import paddle.nn as nn
import h5py
import matplotlib_inline
from paddle.io import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import KFold
import paddle.nn.functional as F
import copy
from Tsception_data_process import  PrepareData

class TSception(nn.Layer):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2D(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2D(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool*0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
                                         int(self.pool*0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2D(num_T)
        self.BN_s = nn.BatchNorm2D(num_S)
        self.BN_fusion = nn.BatchNorm2D(num_S)
        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes),
        #    nn.Softmax()
        )

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
    #    out = torch.cat((out, y), dim=-1)
        out = paddle.concat([out,y],axis=-1)
        y = self.Tception3(x)
    #    out = torch.cat((out, y), dim=-1)
        out = paddle.concat([out,y],axis=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
    #    out_ = torch.cat((out_, z), dim=2)
        out_ = paddle.concat([out_,z],axis=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
    #    out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = paddle.squeeze(paddle.mean(out,axis=-1), axis= -1)
        out = self.fc(out)
        return out

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))


