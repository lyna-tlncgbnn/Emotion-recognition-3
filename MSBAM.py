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

class MSBAM(nn.Layer):
    def conv_block(self,in_chan,out_chan,kernel,step):
        conv3d = nn.Sequential(
            nn.Conv3D(in_channels=in_chan,out_channels=out_chan,kernel_size=kernel,stride=step),
            # in 3d-conv kernel_size = (depth,height,width) stride=(depth,height,width)
            nn.ELU(),
            nn.BatchNorm3D(num_features=out_chan)          
        )
        return conv3d 
    
    def __init__(self,num_classes):
        super(MSBAM,self).__init__()
        self.fe1_kernel = (128,9,5)
        self.fe1_step = (64,9,4)
        self.fe2_kernel = (64,9,5)
        self.fe2_step = (32,9,4)
        
        self.conv3d_1 = self.conv_block(in_chan = 1,out_chan = 1,kernel = self.fe1_kernel, step = self.fe1_step)
        self.conv3d_2 = self.conv_block(in_chan = 1,out_chan = 1,kernel = self.fe2_kernel, step = self.fe2_step)


        self.linear_1 = nn.Sequential(
                        nn.Linear(27,25),
                        nn.Dropout(0.7)
        )

        self.linear_2 = nn.Sequential(
                        nn.Linear(57,25),
                        nn.Dropout(0.7)
        )

        self.linear_3 = nn.Sequential(
                        nn.Linear(50,num_classes),
                        nn.Dropout(0.7)
                    #    nn.Softmax
        )
        #nn.Linear = [in_features, out_features, weight_attr=None, bias_attr=None, name=None]
        #Linear层只接受一个Tensor作为输入，形状为 [batch_size,∗,in_features] ，其中 ∗ 表示可以为任意个额外的维度

    def forward(self,input):
        fe1 = self.conv3d_1(input) # N-C-D-W-H 
        fe1_L = paddle.flatten(fe1[:,:,:,:,0],start_axis = 1 , stop_axis = -1)    #N-C-L                  
        fe1_R = paddle.flatten(fe1[:,:,:,:,1],start_axis = 1 , stop_axis = -1)            
        fe1_C = fe1_L-fe1_R       #N-C-L  
        fe1_concat = paddle.concat(x=[fe1_L,fe1_C,fe1_R],axis=-1)  #[start_axis:stop_axis]


        fe2 = self.conv3d_2(input) # N-C-D-W-H 
        fe2_L = paddle.flatten(fe2[:,:,:,:,0],start_axis = 1 , stop_axis = -1)    #N-C-L                  
        fe2_R = paddle.flatten(fe2[:,:,:,:,1],start_axis = 1 , stop_axis = -1)            
        fe2_C = fe2_L-fe2_R       #N-C-L  
        fe2_concat = paddle.concat(x=[fe2_L,fe2_C,fe2_R],axis=-1)  #[start_axis:stop_axis]

        af_lin_1 = self.linear_1(fe1_concat)
        af_lin_2 = self.linear_2(fe2_concat)

        f_concat = paddle.concat(x=[af_lin_1,af_lin_2],axis = -1)

        out = self.linear_3(f_concat)


        return out
 


