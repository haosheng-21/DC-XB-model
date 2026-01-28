#处理数据
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import pickle
import sys
sys.path.append("/.../DC+XB/XBERT")
from utils import ele_vec, get_atom_embedding, GaussianDistance, get_spg_tokens_A, get_spg_tokens_B, get_token_id

#---------------继承torch中的Dataset类，产生每一个材料的信息-----------------------------------------------------
class GNNTransformerData(Dataset):
    def __init__(self, gen_crys):
        self.data = gen_crys
        self.vocab_new = json.load(open('/.../DC+XB/XBERT/vocab_new.json', 'r'))
        self.vocab_newnew = json.load(open('/.../DC+XB/XBERT/vocab_newnew.json', 'r'))
        self.sg_symbol = json.load(open('/.../DC+XB/XBERT/sg_symbol.json', 'r'))
        # self.vocab_new = json.load(open('vocab_new.json', 'r'))
        # self.vocab_newnew = json.load(open('vocab_newnew.json', 'r'))
        # self.sg_symbol = json.load(open('sg_symbol.json', 'r'))


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):

        #读取信息
        
        crystal = self.data[idx].structure
        formula = crystal.formula
        sg = Structure.get_space_group_info(crystal)[-1]
        num_atoms = crystal.num_sites
        volume = crystal.volume
        num_electrons = sum(crystal.atomic_numbers)

        # formula = self.mat_dict['formula'][idxx]
        # sg = self.mat_dict['sg'][idxx]
        # num_atoms = self.mat_dict['num_atoms'][idxx]
        # volume = self.mat_dict['volume'][idxx]
        
        a = crystal.as_dict()['lattice']['a']
        b = crystal.as_dict()['lattice']['b']
        c = crystal.as_dict()['lattice']['c']
        alpha = crystal.as_dict()['lattice']['alpha']
        beta = crystal.as_dict()['lattice']['beta']
        gamma = crystal.as_dict()['lattice']['gamma']
        

        #生成向量
        ##元素
        ele_vector = ele_vec(formula)
        ##空间群A(新的)
        sg2 = self.sg_symbol[str(sg)]
        sg_tokens_A = get_spg_tokens_A(sg2)
        spg_tokens_id_A = get_token_id(sg_tokens_A, self.vocab_new)
        ##空间群B(gengxinde)
        sg_tokens_B = get_spg_tokens_B(sg2)
        spg_tokens_id_B = get_token_id(sg_tokens_B, self.vocab_newnew)
        ##一些宏观性质
        mac_properties = [num_atoms, volume, num_electrons]
        ##GNN的特征  ####已经check
        atom_fea = np.vstack([get_atom_embedding(crystal[i].as_dict()['species'][0]['element']) for i in range(len(crystal))])  
        all_nbrs = crystal.get_all_neighbors(8, include_index=True)
        all_nbrs = [sorted(nbr, key=lambda x: x[1]) for nbr in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < 12:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0]*(12 - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) + [8 + 1.]*(12 - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:12])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:12])))
        nbr_fea = np.array(nbr_fea)
        gdf = GaussianDistance(0, 8,0.2)
        nbr_fea = gdf.expand(nbr_fea)
        atom_fea = atom_fea.tolist()
        nbr_fea = nbr_fea.tolist()
        nbr_fea_idx = np.array(nbr_fea_idx, dtype = float).tolist()
        ##目标
        target1 = [a, b, c, alpha, beta, gamma]
        target2 = 0
        
        #一些列表转换为tensor
        # element_embedding = torch.tensor(element_embedding, dtype = torch.float32)
        # spg_tokens_id = torch.tensor(spg_tokens_id)
        # mac_property = torch.tensor(mac_property)
        # atom_fea = torch.tensor(atom_fea, dtype = torch.float32)
        # nbr_fea = torch.tensor(nbr_fea, dtype = torch.float32)
        # nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        # target1 = torch.tensor(target1)
        # target2 = torch.tensor(target2)
        return ele_vector, spg_tokens_id_A, spg_tokens_id_B, mac_properties, (atom_fea, nbr_fea, nbr_fea_idx), target1, target2, formula

#------------------------定义数据整理函数，将若干个材料的信息整理为一个batch--------------------------------------------------------
def collate_fn(dataset_list):
    batch_ele_vector, batch_spg_tokens_id_A, batch_spg_tokens_id_B, batch_mac_properties, batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, crystal_atom_idx, batch_target1, batch_target2 = [], [], [], [], [], [], [], [], [], []
    base_idx = 0
    for i, (ele_vector, spg_tokens_id_A, spg_tokens_id_B, mac_properties, (atom_fea, nbr_fea, nbr_fea_idx), target1, target2, _) in enumerate(dataset_list):
        n_i = len(atom_fea)
        batch_ele_vector.append(ele_vector)
        batch_spg_tokens_id_A.append(torch.tensor(spg_tokens_id_A))
        batch_spg_tokens_id_B.append(torch.tensor(spg_tokens_id_B))
        electron_numbers = [float(i) for i in list(bin(mac_properties[-1]))[2:]]
        electron_numbers2 = [0.]*(14-len(electron_numbers)) + electron_numbers
        batch_mac_properties.append(electron_numbers2)
        batch_atom_fea.append(torch.tensor(atom_fea, dtype = torch.float32))
        batch_nbr_fea.append(torch.tensor(nbr_fea, dtype = torch.float32))
        batch_nbr_fea_idx.append(torch.tensor(nbr_fea_idx, dtype = torch.int) + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        base_idx += n_i
        batch_target1.append(target1)
        batch_target2.append(target2)
    return   torch.tensor(batch_ele_vector, dtype = torch.float32), torch.cat(batch_spg_tokens_id_A, dim = 0), torch.cat(batch_spg_tokens_id_B, dim = 0), torch.tensor(batch_mac_properties, dtype = torch.float32),\
            (torch.cat(batch_atom_fea, dim = 0), torch.cat(batch_nbr_fea, dim = 0), torch.cat(batch_nbr_fea_idx, dim = 0), crystal_atom_idx), \
             torch.tensor(batch_target1, dtype = torch.float32), torch.tensor(batch_target2, dtype = torch.float32)


#-----------------------直接运行这个py文件会执行下面的命令(写代码时调试用的)----------------------------------------------------------
if __name__ == '__main__':
    data = GNNTransformerData('.')
    print(len(data))
    print(data[0][0])
    print(data[0][1])
    # print(data[52987][0].shape)
    # print(data[52987][1])
    # print(data[52987][1].shape)
    # print(data[52987][2])
    # print(data[52987][3][0])
    # print(data[52987][3][0].shape)
    # print(data[52987][3][1])
    # print(data[52987][3][1].shape)
    # print(data[52987][3][2])
    # print(data[52987][3][2].shape)
    # print(data[52987][4])
    # print(data[52987][5])
    # print(data[52987][6])
    print(collate_fn([data[0], data[1]])[0])
    print(collate_fn([data[0], data[1]])[1])
