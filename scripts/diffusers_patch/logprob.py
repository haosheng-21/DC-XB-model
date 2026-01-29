import torch
from tqdm import tqdm
from torch_scatter import scatter
from copy import deepcopy as dc
import json
import numpy as np
import sys

MAX_ATOMIC_NUM=100

def log_p_wrapped(x, sigma, N=10, T=1.0):

    zero_positions = torch.where(sigma == 0)
    new_tensor = torch.ones_like(sigma).to(x.device)
    new_tensor[zero_positions] = 0

    sigma[zero_positions] = 1

    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)/((2*torch.pi*sigma**2)**0.5)

    log_p_ = torch.log(p_)
    log_p_ = log_p_ * new_tensor

    return log_p_

    


def diffusion_with_logprob(
        self, 
        t,
        x_t, 
        crys_fam_t, 
        t_t, 
        pred_crys_fam, 
        pred_x, 
        pred_t, 
        spacegroup,
        batch_size,
        batch_split,
        anchor_index, 
        ops, 
        ops_inv,
        type,
        prev_x_t = None,
        prev_crys_fam_t = None,
        prev_t_t = None,
        **params
        ):
    if type == "coordinates05":
        ##产生随机数和一些常数
        rand_x = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
        rand_x_anchor = rand_x[anchor_index]
        rand_x_anchor = (ops_inv[anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x = (ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

        step_size = params["step_lr"] / (params["sigma_norm"] * (self.sigma_scheduler.sigma_begin) ** 2)
        std_x = torch.sqrt(2 * step_size)

        ##对decoder出的结果进行调整
        pred_x = pred_x * torch.sqrt(params["sigma_norm"])
        pred_x_proj = torch.einsum('bij, bj-> bi', ops_inv, pred_x)
        pred_x_anchor = scatter(pred_x_proj, anchor_index, dim=0, reduce = 'mean')[anchor_index]
        pred_x = (ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

        ##计算结果

        # x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
        # frac_coords_all = torch.cat([x_t_minus_05[anchor_index], torch.ones(ops.size(0),1).to(x_t_minus_05.device)], dim=-1).unsqueeze(-1) # N * 4 * 1
        # x_t_minus_05 = (ops @ frac_coords_all).squeeze(-1)[:,:3] % 1. # N * 3

        if prev_x_t == None:
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
            frac_coords_all = torch.cat([x_t_minus_05[anchor_index], torch.ones(ops.size(0),1).to(x_t_minus_05.device)], dim=-1).unsqueeze(-1) # N * 4 * 1
            x_t_minus_05 = (ops @ frac_coords_all).squeeze(-1)[:,:3] % 1. # N * 3
        else:
            x_t_minus_05 = prev_x_t


        ##计算logprob
        result, mean, std = x_t_minus_05, x_t - step_size * pred_x, std_x
        batch_record = torch.tensor([len(torch.unique(anchor_index[batch_split == i])) for i in range(batch_size)]).to(self.device)
        batch_record = torch.repeat_interleave(torch.arange(len(batch_record)).to(self.device), batch_record)

        unique_anchor_index = torch.unique(anchor_index)
        result = result[unique_anchor_index]
        mean = mean[unique_anchor_index]
        ops = ops[unique_anchor_index]

        mean = (ops[:, :3, :3] @ mean.unsqueeze(-1)).squeeze(-1) + ops[:,:3,3]

        std_ori = std*ops[:,:3,:3]
        std = torch.sqrt(torch.matmul(std_ori, std_ori.transpose(1, 2)).diagonal(dim1=-2, dim2=-1))


        logprob = log_p_wrapped(x = result-mean, sigma=std)
        # logprob = torch.nan_to_num(logprob, nan=0.0)
        logprob = torch.stack([logprob[batch_record == i].sum() for i in range(batch_size)])

        return x_t_minus_05, logprob
    
    elif type == "coordinates1":
        ##产生随机数和一些常数
        rand_x = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
        rand_x_anchor = rand_x[anchor_index]
        rand_x_anchor = (ops_inv[anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x = (ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

        adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
        step_size = (params["sigma_x"] ** 2 - adjacent_sigma_x ** 2)
        std_x = torch.sqrt((adjacent_sigma_x ** 2 * (params["sigma_x"] ** 2 - adjacent_sigma_x ** 2)) / (params["sigma_x"] ** 2))

        ##对decoder出的结果进行调整
        pred_x = pred_x * torch.sqrt(params["sigma_norm"])
        pred_x_proj = torch.einsum('bij, bj-> bi', ops_inv, pred_x)
        pred_x_anchor = scatter(pred_x_proj, anchor_index, dim=0, reduce = 'mean')[anchor_index]
        pred_x = (ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1) 

        ##计算结果
        if prev_x_t == None:
            x_t_minus_1 = x_t - step_size * pred_x + std_x * rand_x
            frac_coords_all = torch.cat([x_t_minus_1[anchor_index], torch.ones(ops.size(0),1).to(x_t_minus_1.device)], dim=-1).unsqueeze(-1) # N * 4 * 1
            x_t_minus_1 = (ops @ frac_coords_all).squeeze(-1)[:,:3] % 1. # N * 3
        else:
            x_t_minus_1 = prev_x_t

        ##计算logprob
        result, mean, std = x_t_minus_1, x_t - step_size * pred_x, std_x

        batch_record = torch.tensor([len(torch.unique(anchor_index[batch_split == i])) for i in range(batch_size)])
        batch_record = torch.repeat_interleave(torch.arange(len(batch_record)), batch_record)

        unique_anchor_index = torch.unique(anchor_index)
        result = result[unique_anchor_index]
        mean = mean[unique_anchor_index]
        ops = ops[unique_anchor_index]

        mean = (ops[:, :3, :3] @ mean.unsqueeze(-1)).squeeze(-1) + ops[:,:3,3]

        std_ori = std*ops[:,:3,:3]
        std = torch.sqrt(torch.matmul(std_ori, std_ori.transpose(1, 2)).diagonal(dim1=-2, dim2=-1))

        logprob = log_p_wrapped(x = result-mean, sigma=std)
        # logprob = torch.nan_to_num(logprob, nan=0.0)
        logprob = torch.stack([logprob[batch_record == i].sum() for i in range(batch_size)])

        return x_t_minus_1, logprob
    
    elif type == "atom_type":
        ##产生随机数和一些常数
        rand_t = torch.randn_like(t_t) if t > 1 else torch.zeros_like(t_t)
        rand_t = rand_t[anchor_index]
        
        ##对decoder出的结果进行调整
        pred_t = scatter(pred_t, anchor_index, dim=0, reduce = 'mean')[anchor_index]

        ##计算结果
        if prev_t_t == None:
            t_t_minus_1 = params["c0"] * (t_t - params["c1"] * pred_t) + params["sigmas"] * rand_t
            t_t_minus_1 = t_t_minus_1[anchor_index]
        else:
            t_t_minus_1 = prev_t_t

        ##计算logprob
        result, mean, std = t_t_minus_1, params["c0"] * (t_t - params["c1"] * pred_t), params["sigmas"]
        batch_record = torch.tensor([len(torch.unique(anchor_index[batch_split == i])) for i in range(batch_size)])
        batch_record = torch.repeat_interleave(torch.arange(len(batch_record)), batch_record)
        
        unique_anchor_index = torch.unique(anchor_index)

        result = result[unique_anchor_index]
        mean = mean[unique_anchor_index]
        std = std

        logprob = torch.log(torch.exp(-(result - mean) ** 2 / 2 / std ** 2)/(2*torch.pi*std**2)**0.5)
        logprob = torch.stack([logprob[batch_record == i].sum() for i in range(batch_size)])

        return t_t_minus_1, logprob
    
    elif type == "lattice":
        ##产生随机数和一些常数
        rand_crys_fam = torch.randn_like(crys_fam_t)
        rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(rand_crys_fam, spacegroup)

        ##计算结果
        if prev_crys_fam_t == None:
            crys_fam_t_minus_1 = params["c0"] * (crys_fam_t - params["c1"] * pred_crys_fam) + params["sigmas"] * rand_crys_fam
            crys_fam_t_minus_1 = self.crystal_family.proj_k_to_spacegroup(crys_fam_t_minus_1, spacegroup)
        else:
            crys_fam_t_minus_1 = prev_crys_fam_t

        ##计算 logprobs
        result, mean, std = crys_fam_t_minus_1, params["c0"] * (crys_fam_t - params["c1"] * pred_crys_fam), params["sigmas"]
        logprob = torch.log(torch.exp(-(result - mean) ** 2 / 2 / std ** 2)/(2*torch.pi*std**2)**0.5)

        logprob = self.crystal_family.proj_k_to_spacegroup(logprob, spacegroup)
        mask = logprob == -0.25 * np.log(3) * np.sqrt(2)
        logprob[mask] = 0.
        
        logprob = logprob.sum(dim=1)

        return crys_fam_t_minus_1, logprob




@torch.no_grad()
def sample(self, batch, diff_ratio = 1.0, step_lr = 1e-5):

    batch_size = batch.num_graphs

    x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
    x_T_all = torch.cat([x_T[batch.anchor_index], torch.ones(batch.ops.size(0),1).to(x_T.device)], dim=-1).unsqueeze(-1) # N * 4 * 1
    x_T = (batch.ops @ x_T_all).squeeze(-1)[:,:3] % 1. # N * 3

    t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)
    t_T = t_T[batch.anchor_index]

    crys_fam_T = torch.randn([batch_size, 6]).to(self.device)
    crys_fam_T = self.crystal_family.proj_k_to_spacegroup(crys_fam_T, batch.spacegroup)
    l_T = self.crystal_family.v2m(crys_fam_T)

    time_start = self.beta_scheduler.timesteps - 1
    
    traj = {time_start : {
        'num_atoms' : batch.num_atoms,
        'atom_types' : t_T,
        'frac_coords' : x_T % 1.,
        'lattices' : l_T,
        'crys_fam': crys_fam_T
    }}


    for t in tqdm(range(time_start, 0, -1)):

        times = torch.full((batch_size, ), t, device = self.device)
        time_emb = self.time_embedding(times)
        
        alphas = self.beta_scheduler.alphas[t]
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
        sigmas = self.beta_scheduler.sigmas[t]
        sigma_x = self.sigma_scheduler.sigmas[t]
        sigma_norm = self.sigma_scheduler.sigmas_norm[t]
        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

        x_t = traj[t]['frac_coords']
        l_t = traj[t]['lattices']
        crys_fam_t = traj[t]['crys_fam']
        t_t = traj[t]['atom_types']


        #----------Corrector----------------------------------------------------------------------------------------------
        ##decoder
        # print(time_emb.dtype)
        # print(t_t.dtype)
        # print(x_t.dtype)
        # print(crys_fam_t.dtype)
        # print(batch.num_atoms.dtype)
        # print(batch.batch.dtype)

        pred_crys_fam, pred_x, pred_t = self.decoder(time_emb, t_t, x_t, crys_fam_t, batch.num_atoms, batch.batch)

        ##计算下一步的结果与logprobs
        x_t_minus_05, logprob_x_05 = diffusion_with_logprob(
            self, 
            t,
            x_t, 
            crys_fam_t, 
            t_t, 
            pred_crys_fam, 
            pred_x, 
            pred_t, 
            batch.spacegroup,
            batch.batch_size,
            batch.batch,
            batch.anchor_index, 
            batch.ops, 
            batch.ops_inv,
            "coordinates05",
            step_lr = step_lr,
            sigma_norm = sigma_norm,
        )
        t_t_minus_05 = t_t
        crys_fam_t_minus_05 = crys_fam_t
        ori_crys_fam = crys_fam_t_minus_05

        #-----Predictor---------------------------------------------------------------------------------------------------
        ##decoder
        pred_crys_fam, pred_x, pred_t = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, crys_fam_t_minus_05, batch.num_atoms, batch.batch)
        
        ##计算下一步的结果与logprobs
        x_t_minus_1, logprob_x_1 = diffusion_with_logprob(
            self, 
            t,
            x_t_minus_05, 
            ori_crys_fam, 
            t_t_minus_05, 
            pred_crys_fam, 
            pred_x, 
            pred_t, 
            batch.spacegroup,
            batch.batch_size,
            batch.batch,
            batch.anchor_index, 
            batch.ops, 
            batch.ops_inv,
            "coordinates1",
            sigma_x = sigma_x,
            sigma_norm = sigma_norm
        )

        t_t_minus_1, logprob_t_1  = diffusion_with_logprob(
            self, 
            t,
            x_t_minus_05, 
            ori_crys_fam, 
            t_t_minus_05, 
            pred_crys_fam, 
            pred_x, 
            pred_t, 
            batch.spacegroup,
            batch.batch_size,
            batch.batch,
            batch.anchor_index, 
            batch.ops, 
            batch.ops_inv,
            "atom_type",
            c0 = c0,
            c1 = c1,
            sigmas = sigmas
        )

        crys_fam_t_minus_1, logprob_crys_fam_1= diffusion_with_logprob(
            self, 
            t,
            x_t_minus_05, 
            ori_crys_fam, 
            t_t_minus_05, 
            pred_crys_fam, 
            pred_x, 
            pred_t, 
            batch.spacegroup,
            batch.batch_size,
            batch.batch,
            batch.anchor_index, 
            batch.ops, 
            batch.ops_inv,
            "lattice",
            c0 = c0,
            c1 = c1,
            sigmas = sigmas
        )

        l_t_minus_1 = self.crystal_family.v2m(crys_fam_t_minus_1)

        traj[t - 1] = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_t_minus_1,
            'frac_coords' : x_t_minus_1 % 1.,
            'frac_coords_05' : x_t_minus_05 % 1.,
            'lattices' : l_t_minus_1,
            'crys_fam': crys_fam_t_minus_1,
            'logprobs_x_05': logprob_x_05,
            'logprobs_x_1': logprob_x_1,
            'logprobs_t_1': logprob_t_1,
            'logprobs_crys_1': logprob_crys_fam_1  
        }
        # print(crys_fam_t_minus_1.shape)
        # break
    
    batch_record = torch.tensor([len(torch.unique(batch.anchor_index[batch.batch == i])) for i in range(batch.batch_size)]).to(self.device)
    
    traj_stack = {
        'batch_size' : batch.batch_size,
        'num_atoms' : batch.num_atoms,
        'spacegroup' : batch.spacegroup,
        'batch_split' : batch.batch,
        'anchor_index' : batch.anchor_index,
        'ops' : batch.ops,
        'ops_inv' : batch.ops_inv,
        'batch_record': batch_record,
        'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
        'atom_types_ori' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]),
        'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
        'all_frac_coords_05' : torch.stack([traj[i]['frac_coords_05'] for i in range(time_start-1, -1, -1)]),
        'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)]),
        'all_crys_fam' : torch.stack([traj[i]['crys_fam'] for i in range(time_start, -1, -1)]),
        'all_logprobs_x_05': torch.stack([traj[i]['logprobs_x_05'] for i in range(time_start-1, -1, -1)], dim=1),
        'all_logprobs_x_1': torch.stack([traj[i]['logprobs_x_1'] for i in range(time_start-1, -1, -1)], dim=1),
        'all_logprobs_t_1': torch.stack([traj[i]['logprobs_t_1'] for i in range(time_start-1, -1, -1)], dim=1),
        'all_logprobs_crys_1': torch.stack([traj[i]['logprobs_crys_1'] for i in range(time_start-1, -1, -1)], dim=1),
 
    }
 
    res = dc(traj[0])
    res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1
    return res, traj_stack
