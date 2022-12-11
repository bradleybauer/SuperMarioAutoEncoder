import torchvision as tv
import torch as th
import torch.nn.functional as F

import random
import pickle

# TODO test with more or less SEQ_LEN
# number of images the network is given to predict the SEQ_LEN+1th image
SEQ_LEN = 5


class AddGaussianNoise:
    def __init__(self, p=.5, mean=0, std=1):
        self.p = p
        self.std = std
        self.mean = mean

    def forward(self, tensor):
        if random.random() < self.p:
            noise = th.randn(tensor.size())
            if self.std != 1:
                std = random.random() * self.std
                noise = noise * self.std
            if self.mean != 0:
                noise = noise + self.mean
            return tensor + noise
        return tensor
    

class MyDataset(th.utils.data.Dataset):
    def __init__(self,file=None,indices=[],dataset=None,name = 'root_set'):
        self.file = file

        assert((dataset is None) != (file is None))
        if dataset is None:
            f = open(self.file,'rb')
            self.dataset = pickle.load(f)
            f.close()
        else:
            self.dataset = dataset
        dataset = self.dataset

        self.mean = [0.2582, 0.3149, 0.4804]
        self.std = [0.1897, 0.1698, 0.2326]

        assert(name != 'root_set' or len(indices) == 0)

        if len(indices):
            self.indices = indices
        else:
            self.indices = list(range(len(self.dataset['images'])))
            random.shuffle(self.indices)

    def load_indices(self, indices):
        self.indices = indices

    def get_train_test_subsets(self):
        indices = self.indices.copy()
        train_size = int(len(indices)*.9)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        return MyDataset(indices=train_indices,dataset=self.dataset,name = 'train_set'), MyDataset(indices=test_indices,dataset=self.dataset,name = 'test_set')

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def state2vec(state):
        oh_world_stage = F.one_hot(th.as_tensor((state['w']-1) * 4 + (state['s']-1), dtype=th.int64),num_classes=8 * 4)
        oh_mario = F.one_hot(th.as_tensor(state['m'], dtype=th.int64),num_classes=3)
        x_pos = th.as_tensor([state['x']]) / 1000
        y_pos = th.as_tensor([state['y']]) / 200
        return th.cat([oh_world_stage,oh_mario,x_pos,y_pos])

    @staticmethod
    def action2vec(action):
        return F.one_hot(th.as_tensor(action, dtype=th.int64), num_classes=12)
    
    def get_at(self, index):
        png = self.dataset['images'][index]
        state = self.dataset['states'][index]
        action = self.dataset['actions'][index]

        x = tv.io.decode_png(th.as_tensor(png,dtype=th.uint8)) / 255.0
        state = self.state2vec(state)
        action = self.action2vec(action)

        return x, state, action
    
    def get_valid_offset(self, index: int):
        D = self.dataset['states']
        N = len(self)

        # a sequence of states is valid if no game reset's occur during the sequence
        # if the index is not the start of a valid sequence then move index forward until it is

        # there is probably always a valid seq that starts no more than 100 steps away from index
        for i in range(100):
            offset = (i + index) % N

            # assume the sequence starting at offset is valid
            valid = True

            # check SEQ_LEN+1 sequential states starting at index for signs that the sequence is not valid

            # if the world-stage values change
            world = D[offset]['w']
            stage = D[offset]['s']
            for j in range(1,SEQ_LEN+1):
                sub_offset = (offset + j) % N
                sub_world = D[sub_offset]['w']
                sub_stage = D[sub_offset]['s']
                if sub_world!=world or sub_stage!=stage:
                    valid = False
                    break

            if valid:
                # if time resets then most likely we've loaded a new level (which could be the same world & stage)
                t = D[offset]['t']
                for j in range(1,SEQ_LEN+1):
                    sub_offset = (offset + j) % N
                    sub_t = D[sub_offset]['t']
                    if sub_t < t:
                        valid = False
                        break
            
            # these checks should catch most of the invalid sequences
            # however I am not sure if it is a perfect filter

            if valid:
                return offset

        return 0

    def __getitem__(self,index):
        N = len(self)

        index = self.get_valid_offset(index)

        xs_, ss_, as_ = [], [], []

        for i in range(SEQ_LEN):
            x, s, a = self.get_at(min(index + i, N))
            xs_.append(x.unsqueeze_(0))
            ss_.append(s.unsqueeze_(0))
            as_.append(a.unsqueeze_(0))

        Y, Sy, _ = self.get_at(min(index + i + 1, N))

        X = th.cat(xs_)
        Sx = th.cat(ss_)
        A = th.cat(as_)

        return X, Sx, A, Y, Sy

    def unNormalize(self, x:th.Tensor):
        dev = x.device
        b,c,h,w = x.shape

        std = th.tensor(self.std).view(1,c,1).to(dev)
        mean = th.tensor(self.mean).view(1,c,1).to(dev)

        x = x.view(b,c,-1)
        x = x * std + mean
        return x.view(b,c,h,w)
