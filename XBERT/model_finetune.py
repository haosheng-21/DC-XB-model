from model import GNNTransformer
import torch
import torch.nn as nn

class GNNTransformer_finetune(nn.Module):
    def __init__(self):
        super().__init__()
        net = GNNTransformer()
        state_dict = torch.load('/.../DC+XB/XBERT/checkpoint/best.pth', map_location='cuda:0')
        net.load_state_dict(state_dict)
        del net.out
        self.feature = net
        self.topo = nn.Sequential (
                                        nn.Dropout(0.1),
                                        nn.Linear(8*256,3)
                                          )

    def forward(self, ele_vector, spg_tokens_id_B, mac_properties, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        memory = self.feature(ele_vector, spg_tokens_id_B, mac_properties, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        out = self.topo(memory)
        return out
