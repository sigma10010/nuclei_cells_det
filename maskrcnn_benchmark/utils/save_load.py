import torch
import torch.backends.cudnn as cudnn
import os
import numpy as np

def save_model(model, model_name, epoch):
    if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
    filename='./checkpoint/ckpt_%s_%05d.t7'%(model_name, epoch)
    state=model.state_dict()
    torch.save(state, filename)

def load_model(model, model_name, epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    filename='./checkpoint/ckpt_%s_%05d.t7'%(model_name, epoch)
    model = model.to(device)
    if device == 'cuda':
#         model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    state = torch.load(filename)
    model.load_state_dict(state) 
    return model


def save_losses(save_dir, phase, losses):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    filename = os.path.join(save_dir, '%s_losses.npy'% phase)
    np.save(filename, np.array(losses))

def load_losses(save_dir, phase):
    filename = os.path.join(save_dir, '%s_losses.npy'%phase)
    return np.load(filename, allow_pickle=True)