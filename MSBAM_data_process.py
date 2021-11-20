from scipy.io import loadmat
import os
import numpy as np
import h5py
import os.path as osp

class DataDel:
    def __init__(self,data_path = 'data/data_preprocessed_matlab',label = 'A'):
        self.num_class = 2
        self.dataset = 'DEAP'
        self.subject = 32
        self.data_path = data_path
        self.label_type = label
        self.original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                               'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
                               'CP2', 'P4', 'P8', 'PO4', 'O2','pad1','pad2','pad3','pad4','pad5','pad6','pad7','pad8','pad9'
                               ,'pad10','pad11','pad12','pad13','pad14','pad15','pad16','pad17','pad18','pad19','pad20','pad21','pad22','pad23'
                               ,'pad24','pad25','pad26','pad27','pad28','pad29','pad30','pad31','pad32','pad33','pad34','pad35','pad36'
                               ,'pad37','pad38','pad39','pad40','pad41','pad42','pad43','pad44','pad45','pad46','pad47','pad48','pad49']

        self.TS = ['pad1','pad2','pad3','Fp1','pad4','Fp2','pad5','pad6','pad7','pad8','pad9'
                    ,'pad10','AF3','pad11','AF4','pad12','pad13','pad14','F7','pad15','F3','pad16','Fz','pad17','F4',
                    'pad18','F8','pad19','FC5','pad20','FC1','pad21','FC2','pad22','FC6','pad23','T7'
                    ,'pad24','C3','pad25','Cz','pad26','C4','pad27','T8','pad28','CP5','pad29','CP1','pad30',
                    'CP2','pad31','CP6','pad32','P7','pad33','P3','pad34','Pz','pad35','P4','pad36','P8'
                     ,'pad37','pad38','pad39','PO3','pad40','PO4','pad41','pad42','pad43','pad44','pad45','pad46',
                     'O1','Oz','O2','pad47','pad48','pad49']

    def run(self,split=True, expand=True):
        """
        Parameters
        ----------
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN
        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """
        for sub in range(self.subject):
            data_, label_ = self.load_data_per_subject(sub)
            # select label type here
            label_ = self.label_selection(label_)

            data_ = np.pad(data_,((0,0),(0,49),(0,0)),'constant', constant_values=0)
            #填充到81通道 32+49 = 81
            data_ = self.basecorrect(data_)
            data_ = self.norm(data_)
            data_ = data_.reshape(40,81,7680)

            if split:
                data_, label_ = self.split(data=data_, label=label_)

            if expand:
                # expand one dimension for deep learning(CNNs)
                data_ = np.expand_dims(data_, axis=-3)

            data_ = data_.reshape(40,12,1,9,9,640)  

            print('Data and label prepared for sub{}!'.format(sub))
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_, label_, sub)


    def norm(self,train):
        """
        this function do standard normalization for EEG channel by channel
        :param train: training data
        :param test: testing data
        :return: normalized training and testing data
        """
        # data: sample x 89 x 60 x 128
        mean = 0
        std = 0
        for channel in range(train.shape[2]):
            mean = np.mean(train[:, :, channel, :])
            std = np.std(train[:, :, channel, :])
            train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
        return train

    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load

        Returns
        -------
        data: (40, 32, 8064) label: (40, 4)
        """
        sub += 1
        if (sub < 10):
            sub_code = str('s0' + str(sub) + '.mat')
        else:
            sub_code = str('s' + str(sub) + '.mat')

        subject_path = os.path.join(self.data_path, sub_code)
       #subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
        subject = loadmat(subject_path)
        label = subject['labels']
        data = subject['data'][:, 0:32, :]  # 
        #   data: 40 x 32 x 8064
        #   label: 40 x 4
        # reorder the EEG channel to build the local-global graphs
        
       

        print('data:' + str(data.shape) + ' label:' + str(label.shape))

        return data, label

    def basecorrect(self,data):
        baseline = 128
        data = data.reshape(40,81,63,128)

        bs1 = data[:,:,0,:]
        bs2 = data[:,:,1,:]
        bs3 = data[:,:,2,:]

        base = (bs1+bs2+bs3)/3
        base = base[:,:,np.newaxis,:]

        data = data - base
        data = data[:,:,3:,:]

        return data



    def reorder_channel(self, data, graph):
        """
        This function reorder the channel according to different graph designs
        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        if graph == 'TS':
            graph_idx = self.TS
        elif graph == 'O':
            graph_idx = self.original_order

        idx = []

        for chan in graph_idx:
            idx.append(self.original_order.index(chan))

        return data[:, idx, :]    #选择需要的通道

    
    def label_selection(self, label):
        """
        This function: 1. selects which dimension of labels to use
                       2. create binary label
        Parameters
        ----------
        label: (trial, 4)

        Returns
        -------
        label: (trial,)
        """
        if self.label_type == 'A':
            label = label[:, 1]
        elif self.label_type == 'V':
            label = label[:, 0]
        if self.num_class == 2:
            label = np.where(label <= 5, 0, label)
            label = np.where(label > 5, 1, label)
            print('Binary label generated!')
        return label

    def split(self, data, label, segment_length=5, overlap=0, sampling_rate=128):
        """
        This function split one trial's data into shorter segments
        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []

        number_segment = int((data_shape[2] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=1)
        label = np.stack([np.repeat(label[i], int(number_segment + 1)) for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        return data, label
    
    def save(self, data, label, sub):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID

        Returns
        -------
        None
        """
        save_path = os.getcwd()
        data_type = 'DATA_MSBAM_{}_{}'.format(self.dataset, self.label_type)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()   

    def getdata(self,path):
        dir = os.listdir(path)
        dir.sort()
        for dirt in dir:
            data = h5py.File(path+dirt,'r')
            if dirt == "sub0.hdf":
                value = data['data']
                label = data['label']
            else:
                value = np.concatenate((value,data['data']))
                label = np.concatenate((label,data['label']))
        value = np.swapaxes(value,-1,-3)
        return value,label
         