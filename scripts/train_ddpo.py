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
    os.makedirs("checkpoints_temp", exist_ok=True)
###----------load-model-and-data----------------------------------------------------------------------------------------------------
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, loaders, cfg = load_model(
        model_path, load_data=True, testing=False)
    model = model.to(device = device)
    if args.load_model_decoder:
        model.decoder.load_state_dict(torch.load(args.load_model_decoder))
    # print(model)
    # print(list(model.parameters())[1])
    # print(list(model.decoder.parameters())[0])

    # for name, param in model.decoder.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")
    # sys.exit()

    train_loader, val_loader = loaders
    train_set = train_loader.dataset
    # print(train_set)
###----------------------train-------------------------------------------------------------------------------------------------------    
    for epoch in range(30):
        test_set = SampleDataset(train_set, args.batch_size) # , seed = args.seed)
        test_loader = DataLoader(test_set, batch_size = args.batch_size)
        
        # print(len(test_set))
        # print(len(test_loader))
        
        ##----------------------------------------------------------------------------------------------------------------------------
        ##------------------------------------SAMPLING--------------------------------------------------------------------------------
        ##----------------------------------------------------------------------------------------------------------------------------
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
        # print(traj_all[0]['all_logprobs_x_05'])
        # print(traj_all[0]['all_logprobs_crys_1'].device)
        # sys.exit()
        ##------------------calculate rewards------------------------
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
        # torch.save(crys_array_list, "crys_array_list.pth")     
        
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
        rewards = 2*torch.tensor(res1)[:,:,1] - torch.tensor(res1)[:,:,0] - torch.tensor(res1)[:,:,2]
        # rewards = torch.tensor(res2)
        # rewards = torch.where((rewards == 0) | (rewards ==2), -1, rewards)
        # print(rewards)
        # print(res1)
        # print(res2)
        
        
        ##----------------shuffled-----------------------------------
        
        # print(traj_all[0])
        for i, dic in enumerate(traj_all):
            dic['atom_types_t'] =        dic['atom_types_ori'][:-1]
            dic['atom_types_t_1'] =      dic['atom_types_ori'][1:]
            dic['all_frac_coords_t'] =   dic['all_frac_coords'][:-1]
            dic['all_frac_coords_t_1'] = dic['all_frac_coords'][1:]
            dic['all_lattices_t'] =      dic['all_lattices'][:-1]
            dic['all_lattices_t_1'] =    dic['all_lattices'][1:]
            dic['all_crys_fam_t'] =      dic['all_crys_fam'][:-1]
            dic['all_crys_fam_t_1'] =    dic['all_crys_fam'][1:]
        
            dic.pop('atom_types')
            dic.pop('all_frac_coords')
            dic.pop('all_lattices')
            dic.pop('all_crys_fam')
        
            dic['rewards'] = rewards[i].to(device = device)
            
            dic['atom_types_t'] =        dic['atom_types_t'][:-1]
            dic['atom_types_t_1'] =      dic['atom_types_t_1'][:-1]
            dic['all_frac_coords_t'] =   dic['all_frac_coords_t'][:-1]
            dic['all_frac_coords_05'] =  dic['all_frac_coords_05'][:-1]
            dic['all_frac_coords_t_1'] = dic['all_frac_coords_t_1'][:-1]
            dic['all_lattices_t'] =      dic['all_lattices_t'][:-1]
            dic['all_lattices_t_1'] =    dic['all_lattices_t_1'][:-1]
            dic['all_crys_fam_t'] =      dic['all_crys_fam_t'][:-1]
            dic['all_crys_fam_t_1'] =    dic['all_crys_fam_t_1'][:-1]
            dic['all_logprobs_x_05'] =   dic['all_logprobs_x_05'][:,:-1]
            dic['all_logprobs_x_1'] =    dic['all_logprobs_x_1'][:,:-1]
            dic['all_logprobs_t_1'] =    dic['all_logprobs_t_1'][:,:-1]
            dic['all_logprobs_crys_1'] = dic['all_logprobs_crys_1'][:,:-1]
            
            
        # print(traj_all[-1].keys())
        # print(traj_all[-1]['rewards'])
        
        for i, dic in enumerate(traj_all):
        
            indices = list(range(dic['atom_types_t'].shape[0]))
            random.shuffle(indices)   
            # print(indices)
        
            dic['atom_types_t'] =        dic['atom_types_t'][indices]
            dic['atom_types_t_1'] =      dic['atom_types_t_1'][indices]
            dic['all_frac_coords_t'] =   dic['all_frac_coords_t'][indices]
            dic['all_frac_coords_05'] =  dic['all_frac_coords_05'][indices]
            dic['all_frac_coords_t_1'] = dic['all_frac_coords_t_1'][indices]
            dic['all_lattices_t'] =      dic['all_lattices_t'][indices]
            dic['all_lattices_t_1'] =    dic['all_lattices_t_1'][indices]
            dic['all_crys_fam_t'] =      dic['all_crys_fam_t'][indices]
            dic['all_crys_fam_t_1'] =    dic['all_crys_fam_t_1'][indices]
            dic['all_logprobs_x_05'] =   dic['all_logprobs_x_05'][:,indices]
            dic['all_logprobs_x_1'] =    dic['all_logprobs_x_1'][:,indices]
            dic['all_logprobs_t_1'] =    dic['all_logprobs_t_1'][:,indices]
            dic['all_logprobs_crys_1'] = dic['all_logprobs_crys_1'][:,indices]
            time = torch.tensor([999-i for i in range(999)])
            time = time[:-1]
            dic['time'] = time[indices]
            # dic['time'] = time
        
        # print(traj_all[0].keys())
        # print(traj_all[0]['time'])
        
        
        ##----------------------------------------------------------------------------------------------------------------------------
        ##------------------------------------TRAINING--------------------------------------------------------------------------------
        ##----------------------------------------------------------------------------------------------------------------------------
        
        model.decoder.train()
        optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=0.00014) #, weight_decay = 0.001)
        
        for in_epoch in range(1):
            for i, dic in enumerate(traj_all):
                # print(i)
                # print(dic.keys())
                for j in range(len(dic['time'])):
                    for k in range(1):
                        t = dic['time'][j]
                        times = torch.full((dic['batch_size'], ), t, device = model.device)
                        time_emb = model.time_embedding(times)
                                
                        alphas = model.beta_scheduler.alphas[t]
                        alphas_cumprod = model.beta_scheduler.alphas_cumprod[t]
                        sigmas = model.beta_scheduler.sigmas[t]
                        sigma_x = model.sigma_scheduler.sigmas[t]
                        sigma_norm = model.sigma_scheduler.sigmas_norm[t]
                        c0 = 1.0 / torch.sqrt(alphas)
                        c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
                        x_t = dic['all_frac_coords_t'][j]
                        l_t = dic['all_lattices_t'][j]
                        crys_fam_t = dic['all_crys_fam_t'][j]
                        t_t = dic['atom_types_t'][j]
                        # print(x_t.device)
                        # print(sys.exit())
                        # x_t = x_t.to(device = device)
                        # l_t = l_t.to(device = device)
                        # crys_fam_t.to(device = device)
                        # t_t = t_t.to(device = device)
                        
                        # print(time_emb.dtype)
                        # print(t_t.dtype)
                        # print(x_t.dtype)
                        # print(crys_fam_t.dtype)
                        # print(dic['num_atoms'].dtype)
                        # print(dic['batch_split'].dtype)
                        
                        pred_crys_fam, pred_x, pred_t = model.decoder(time_emb, t_t, x_t, crys_fam_t, dic['num_atoms'], dic['batch_split'])
                        
                        
                        x_t_minus_05, logprob_x_05 = diffusion_with_logprob(
                            model, 
                            t,
                            x_t, 
                            crys_fam_t, 
                            t_t, 
                            pred_crys_fam, 
                            pred_x, 
                            pred_t, 
                            dic['spacegroup'],
                            dic['batch_size'],
                            dic['batch_split'],
                            dic['anchor_index'], 
                            dic['ops'], 
                            dic['ops_inv'],
                            "coordinates05",
                            dic['all_frac_coords_05'][j],
                            dic['all_crys_fam_t'][j],
                            dic['atom_types_t'][j],
                            step_lr = 1e-5,
                            sigma_norm = sigma_norm,
                        )
                        t_t_minus_05 = t_t
                        crys_fam_t_minus_05 = crys_fam_t
                        ori_crys_fam = crys_fam_t_minus_05
                        
                        pred_crys_fam, pred_x, pred_t = model.decoder(time_emb, t_t_minus_05, dic['all_frac_coords_05'][j], crys_fam_t_minus_05, dic['num_atoms'], dic['batch_split'])
                        
                        ##计算下一步的结果与logprobs
                        x_t_minus_1, logprob_x_1 = diffusion_with_logprob(
                            model, 
                            t,
                            dic['all_frac_coords_05'][j], 
                            ori_crys_fam, 
                            t_t_minus_05, 
                            pred_crys_fam, 
                            pred_x, 
                            pred_t, 
                            dic['spacegroup'],
                            dic['batch_size'],
                            dic['batch_split'],
                            dic['anchor_index'], 
                            dic['ops'], 
                            dic['ops_inv'],
                            "coordinates1",
                            dic['all_frac_coords_t_1'][j],
                            dic['all_crys_fam_t_1'][j],
                            dic['atom_types_t_1'][j],
                            sigma_x = sigma_x,
                            sigma_norm = sigma_norm
                        )
                        
                        t_t_minus_1, logprob_t_1  = diffusion_with_logprob(
                            model, 
                            t,
                            dic['all_frac_coords_05'][j], 
                            ori_crys_fam, 
                            t_t_minus_05, 
                            pred_crys_fam, 
                            pred_x, 
                            pred_t, 
                            dic['spacegroup'],
                            dic['batch_size'],
                            dic['batch_split'],
                            dic['anchor_index'], 
                            dic['ops'], 
                            dic['ops_inv'],
                            "atom_type",
                            dic['all_frac_coords_t_1'][j],
                            dic['all_crys_fam_t_1'][j],
                            dic['atom_types_t_1'][j],
                            c0 = c0,
                            c1 = c1,
                            sigmas = sigmas
                        )
                        
                        crys_fam_t_minus_1, logprob_crys_fam_1= diffusion_with_logprob(
                            model, 
                            t,
                            dic['all_frac_coords_05'][j], 
                            ori_crys_fam, 
                            t_t_minus_05, 
                            pred_crys_fam, 
                            pred_x, 
                            pred_t, 
                            dic['spacegroup'],
                            dic['batch_size'],
                            dic['batch_split'],
                            dic['anchor_index'], 
                            dic['ops'], 
                            dic['ops_inv'],
                            "lattice",
                            dic['all_frac_coords_t_1'][j],
                            dic['all_crys_fam_t_1'][j],
                            dic['atom_types_t_1'][j],
                            c0 = c0,
                            c1 = c1,
                            sigmas = sigmas
                        )
                        
                        # print(j)
                        # print(logprob_x_1)
                                           
                        # print(dic['all_logprobs_x_05'][:,j])
                        # print(logprob_x_05)
                        # print(logprob_x_05.requires_grad)                    
                        
                        # print(dic['all_logprobs_x_1'][:,j])
                        # print(logprob_x_1)
                        # print(logprob_x_1.requires_grad)
                        
                        # print(dic['all_logprobs_t_1'][:,j])
                        # print(logprob_t_1)
                        # print(logprob_t_1.requires_grad)
                        
                        # print(dic['all_logprobs_crys_1'][:,j])
                        # print(logprob_crys_fam_1)
                        # print(logprob_crys_fam_1.requires_grad)
                        
                        # sys.exit()
                        num_div = dic['batch_record']*(3+3+102)+6
                        logprob = (logprob_x_05 + logprob_x_1 + logprob_t_1 + logprob_crys_fam_1)/num_div
                        sample_logprob = (dic['all_logprobs_x_05'][:,j] + dic['all_logprobs_x_1'][:,j] + dic['all_logprobs_t_1'][:,j] + dic['all_logprobs_crys_1'][:,j])/num_div
                        # print(f'------{in_epoch}----{i}--------')
                        # print(f'logprob: {logprob}')
                        # print(f'sample_logprob: {sample_logprob}')
                        
                        advantages = torch.clamp(
                                                    # dic["rewards"]-torch.mean(dic["rewards"]),
                                                    dic["rewards"],
                                                    -5,
                                                    5,
                                                )
                        # print(f'advantages: {advantages}')
                        ratio = torch.exp(logprob - sample_logprob)
                        # print(f'ratio: {ratio}')
                        
                        unclipped_loss = -advantages * ratio
                        # print(f'unclipped_loss: {unclipped_loss}')
                        # print(unclipped_loss.shape)
                        clipped_loss = -advantages * torch.clamp(
                                    ratio,
                                    1.0 - 1e-4,
                                    1.0 + 1e-4,
                                )
                        # print(clipped_loss.shape)
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if j%100 == 0:
                            print(f'epoch = {epoch}, in_epoch={in_epoch}, traj_dic={i}, sample={j}, sample_epoch={k}. loss = {loss}')
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    
        torch.save(
                   model.decoder.state_dict(),
                   f'checkpoints_temp/model_decoder_3class_epoch_{epoch}.pth'
                  )
    # sys.exit()               
    
    
###-------------------------------generation-and-test-------------------------------------------------------------------------------        
    print('start generation')            
    test_set = SampleDataset(train_set, 64*20)# , seed = args.seed)
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
    ##------------------calculate rewards------------------------
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
                                        
    torch.save(crys_array_list, "crys_array_list_topo_3class.pth")     
    
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
    # print(ll)
    print(ll.count(0.0))
    print(ll.count(1.0))
    # with open('res2_new.json', 'w') as f:
    #     json.dump(ll, f)




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
