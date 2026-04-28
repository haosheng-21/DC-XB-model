import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import  collate_fn
from model_finetune import GNNTransformer_finetune
import json
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
# from utils import weightConstraint


def reward(data, batch_size):

    device = (torch.device('cuda:0') if torch.cuda.is_available()
          else torch.device('cpu'))
    # print(data)
    # print(data[0])
    predict_loader = DataLoader(data, batch_size = batch_size, shuffle = False, num_workers = 0, collate_fn = collate_fn)
    # print(predict_loader)
    model = GNNTransformer_finetune()
    state_dict = torch.load('/.../DC-XB-model/XBERT/checkpoint_topo/checkpoint.pth', map_location='cuda:0')
    model.load_state_dict(state_dict)
    model = model.to(device = device)
    res0 = []
    res1 = []
    res2 = []
    for j, (ele_vector, spg_tokens_id_A, spg_tokens_id_B, mac_properties, (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), target1, target2) in enumerate(predict_loader):
        model.eval()
        ele_vector = ele_vector.to(device = device)
        spg_tokens_id_B = spg_tokens_id_B.to(device = device)
        mac_properties = mac_properties.to(device = device)
        atom_fea = atom_fea.to(device = device)
        nbr_fea = nbr_fea.to(device = device)
        nbr_fea_idx = nbr_fea_idx.to(device = device)
        crystal_atom_idx = crystal_atom_idx
        output = model(ele_vector, spg_tokens_id_B, mac_properties, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        prediction = F.softmax(output, dim=1).cpu().detach().numpy()
        pred_label = np.argmax(prediction, axis = 1)
        pred_label2 = [float(i) for i in pred_label]
        res0.append(output.cpu().detach().numpy().tolist())
        res1.append(prediction.tolist())
        res2.append(pred_label2)
        

    return res0, res1, res2





    
    
