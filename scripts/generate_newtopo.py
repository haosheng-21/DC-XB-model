import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group
import chemparse
import numpy as np
from p_tqdm import p_map

import pdb

import os

from scripts.diffusers_patch.logprob import sample, diffusion_with_logprob
from torch_geometric.data import Data, Batch, DataLoader

from eval_utils import get_crystals_list, lattices_to_params_shape

from compute_metrics import Crystal

import sys
sys.path.append(".")

from XBERT.data import GNNTransformerData
from XBERT.predict import reward

import random

import json

device = (torch.device('cuda:0') if torch.cuda.is_available()
          else torch.device('cpu'))

class SampleDataset(Dataset):

    def __init__(self, dataset, total_num, seed = 9999):
        super().__init__()
        self.total_num = total_num
        self.dataset = dataset
        self.is_carbon = dataset == 'carbon'
        # self.seed = seed
        # np.random.seed(self.seed)
        self.indexes = np.random.choice(len(self.dataset), total_num)


    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):

        idx = self.indexes[index]
        data = self.dataset[idx]
        return data
        
        
        
def main(args):
    model_path = Path(args.model_path)
    model, loaders, cfg = load_model(
        model_path, load_data=True, testing=False)
    model = model.to(device = device)
    if args.load_model_decoder:
        print('loading finetuned model!!!!!!')
        model.decoder.load_state_dict(torch.load(args.load_model_decoder))
    
    train_loader, val_loader = loaders
    train_set = train_loader.dataset
    
    print('start generation')            
    test_set = SampleDataset(train_set, args.batch_size*10) # , seed = args.seed)
    test_loader = DataLoader(test_set, batch_size = args.batch_size)

    model.decoder.eval()
    outputs_all = []
    traj_all = []
    for idx, batch in enumerate(test_loader):
        batch = batch.to(device = device)
        outputs, traj = sample(model, batch)
        outputs_all.append(outputs)
        traj_all.append(traj)
        # if idx == 1:
        #     break
    
    # torch.save(outputs_all, "outputs_all.pth")
    # torch.save(traj_all, "traj_all.pth")
    # print(traj_all[0]['atom_types'].shape)
    # print(traj_all[0]['atom_types_ori'].shape)
    # print(traj_all[0]['atom_types'].device)
    # print(traj_all[0]['all_frac_coords'].device)
    # print(traj_all[0]['all_lattices'].device)
    # print(traj_all[0]['all_crys_fam'].device)
    # print(traj_all[0]['batch_split'].device)
    # print(traj_all[0]['all_logprobs_x_05'].device)
    # print(traj_all[0]['all_logprobs_crys_1'].device)
    # sys.exit()

    num_atoms, atom_types, frac_coords, lattices = [], [], [], []
    for i in range(len(outputs_all)):
        num_atoms.append(outputs_all[i]["num_atoms"].detach().cpu())
        atom_types.append(outputs_all[i]["atom_types"].detach().cpu())
        frac_coords.append(outputs_all[i]["frac_coords"].detach().cpu())
        lattices.append(outputs_all[i]["lattices"].detach().cpu())
    
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim = 0)
    frac_coords = torch.cat(frac_coords, dim=0)
    lattices = torch.cat(lattices, dim=0)
    
    lengths, angles = lattices_to_params_shape(lattices)
    
    crys_array_list = get_crystals_list(frac_coords,
                                        atom_types,
                                        lengths,
                                        angles,
                                        num_atoms)
                                        
    torch.save(crys_array_list, "crys_array_list_topo_3class_generated.pth")     
    
   
    # print(crys_array_list)   
    gen_crys = p_map(lambda x: Crystal(x), crys_array_list)
    # print(gen_crys)
    # sys.exit()
    # print(gen_crys[0].structure)
    # print(gen_crys[-1].structure)
    
    gen_crys_xbert = GNNTransformerData(gen_crys)
    
    res0, res1,  res2 = reward(gen_crys_xbert, args.batch_size)
    # print(res1)
    # print(torch.tensor(res1).shape)
    # print(torch.tensor(res1)[:,:,-1])
    # print(torch.tensor(res1)[:,:,-1].shape)
    # res0 = torch.tensor(res0).view(-1,2)
    # print(res0)
    # rewards = res0[:, 1] - res0[:, 0]
    # rewards = torch.tensor(res1)[:,:,-1]
    # print(rewards)
    # print(res1)
    ll = []
    for i in res2:
        ll+=i
    print(ll.count(0.0))
    print(ll.count(1.0))
    
    # print(loss_record)
    
    # with open('res2_new.json', 'w') as f:
    #     json.dump([ll, ll_tot], f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--seed', default=9999, type=int)
    parser.add_argument('--label', default='')
    parser.add_argument('--load_model_decoder', default = None)
    args = parser.parse_args()


    main(args)
