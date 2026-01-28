#定义模型
import torch
import torch.nn as nn
from data import collate_fn
import json


class ConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_full = nn.Linear(169,128) #, dtype=torch.float64)
        self.bn1 = nn.BatchNorm1d(128) #, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn2 = nn.BatchNorm1d(64) #, dtype=torch.float64)
        self.softplus2 = nn.Softplus()
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_fea[nbr_fea_idx,:]
        A = atom_fea.unsqueeze(1).expand(N, M, 64)
        total_nbr_fea = torch.cat([A, atom_nbr_fea, nbr_fea], dim = 2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, 128)).view(N, M, 128)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter*nbr_core, dim = 1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_fea + nbr_sumed)
        return out


class GNNTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        #元素部分
        self.fc1 = nn.Linear(92, 256)
        self.softplus_fc1 = nn.Softplus()
        #空间群部分
        self.spg_embedding = nn.Embedding(280, 256)
        #宏观性质部分
        self.fc2 = nn.Linear(14, 256)
        self.softplus_fc2 = nn.Softplus()
        #cgcnn部分
        self.cgcnn_embedding = nn.Linear(92,64) #, dtype=torch.float64)
        self.convs = nn.ModuleList([ConvLayer() for _ in range(3)])
        self.conv_to_fc_softplus = nn.Softplus()
        self.conv_to_fc = nn.Linear(64, 256) #, dtype=torch.float64)
        #transformer部分
        encoder_layer = nn.TransformerEncoderLayer( d_model = 256,
                                                    nhead = 4,
                                                    dim_feedforward = 256,
                                                    dropout = 0.2,
                                                    activation = 'relu',
                                                    batch_first = True,
                                                    norm_first = True)
        
        norm = torch.nn.LayerNorm(normalized_shape=256, elementwise_affine=True)
        self.encoder = nn.TransformerEncoder(encoder_layer = encoder_layer,
                                             num_layers = 5,
                                             norm = norm)
        #计算输出
        # self.out_softmax = nn.Softmax(dim = -1)
        self.out = nn.Sequential (
                                        nn.Dropout(0.1),
                                        nn.Linear(8*256,6),
                                  )



    def forward(self, ele_vector, spg_tokens_id_B, mac_properties, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        #元素部分
        ele_vector = self.softplus_fc1(self.fc1(ele_vector))
        #空间群部分
        spg_tokens_id = self.spg_embedding(spg_tokens_id_B)
        #宏观性质部分
        mac_properties = self.softplus_fc2(self.fc2(mac_properties))
        #cgcnn部分
        atom_fea = self.cgcnn_embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        crys_fea = self.conv_to_fc(crys_fea)
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        #拼接
        spg_tokens_id = spg_tokens_id.view(-1, 256*5)
        total = torch.cat([spg_tokens_id, mac_properties, ele_vector, crys_fea], dim = 1)
        total = total.view(-1, 8, 256)
        #transformer部分
        memory = self.encoder(src = total)
        #计算输出
        # memory = self.out_softmax(memory)
        memory = memory.view(-1,8*256)
        # out = self.out(memory)
        return memory

    def pooling(self, atom_fea, crystal_atom_idx):
        summed_fea = [torch.mean(atom_fea[idx_map], dim = 0, keepdim = True) for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim = 0)



    
if __name__ == '__main__':
    data = json.load(open('data_sa_fin.json', 'r'))
    # print(data[0])

    model = GNNTransformer()
    result = model( collate_fn([data[0], data[1]])[0],\
                    collate_fn([data[0], data[1]])[1],\
                    collate_fn([data[0], data[1]])[2][0],\
                    collate_fn([data[0], data[1]])[2][1],\
                    collate_fn([data[0], data[1]])[2][2],\
                    collate_fn([data[0], data[1]])[2][3] )
    # print(model)
    print(result)


