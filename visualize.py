import torch as th
import torch.nn.functional as F

from nes_py.wrappers import JoypadSpace
from nes_py.app.play_human import play_human
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT  # not actually all perms
env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

from AutoEncoder import AutoEncoder
from MyDataset import MyDataset
from MyLogger import get_logger
import cv2
import numpy as np


cuda = True
device = th.device("cuda:0" if cuda else "cpu")

ckpt = th.load('weights/autoencoder_ckpt.pth')
model = AutoEncoder().to(device)
model.load_state_dict(ckpt['model_state'])
model.eval()

# pygame.init()
# display = pygame.display.set_mode((512, 512))

def get_info(info):
    # del info['time'] # keep time so we can detect when mario died
    # when training the world model. we only train on transitions where the time variable is non-decreasing
    info['status'] = {'small':0, 'tall':1, 'fireball':2}[info['status']]
    return {'w':int(info['world']),
            's':int(info['stage']),
            'm':int(info['status']),
            't':int(info['time']),
            'x':int(info['x_pos']),
            'y':int(info['y_pos'])}

def get_tensor(state):
    image = state[35:215, 38:256-38].transpose(2, 0, 1) / 256 # *1 to copy
    return th.Tensor(image)

def callback(state, action, reward, done, info):
    with th.no_grad():
        tensor = get_tensor(state)

        state = get_info(info)
        state = MyDataset.state2vec(state)

        rec = model(tensor.unsqueeze(0).cuda(), state.cuda())
        rec = th.clamp(rec['reconstruction'],0,1).cpu()

        rec = F.interpolate(rec, scale_factor=512/180, mode='bilinear')
        rec = rec.numpy()[0]*255
        rec = rec.transpose(1, 2, 0).astype('uint8')[:,:,::-1]

        # input = tensor.numpy().transpose(1,2,0)[:,:,::-1]

        try:
            cv2.imshow("Color Image", rec)
            cv2.waitKey(1)

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            exit(0)


play_human(env, callback=callback)

