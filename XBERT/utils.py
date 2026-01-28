#定义一些辅助函数
from pymatgen.core import Composition, Element
from pymatgen.core.structure import Structure
import json
import numpy as np

#------------------------------得到材料中每个元素的描述----------------------------------------------------------------------------
def get_element_embedding(x):
    comp = Composition(x)
    elements = ['Cr', 'K', 'La', 'Li', 'Pr', 'Ru', 'Hg', 'Pu', 'Ho', 'Tl', 'Fe', 'Y', 'Ni', 'Al', 'Pm', 'Ag', 'U', 'Sb', 'Zn', 'O', 'Rb', 'Nd', 'Ga', 'Ta', 'Na', 'Ac', 'Se', 'Ca', 'Ti', 'I', 'Co', 'H', 'As', 'Eu', 'Ce', 'Be', 'Cl', 'Sr', 'Pb', 'N', 'Kr', 'Tb', 'Pt', 'C', 'P', 'Hf', 'F', 'Tc', 'He', 'Sc', 'Sm', 'Ne', 'Si', 'Ge', 'Ir', 'Mo', 'Cu', 'Dy', 'Mg', 'W', 'Nb', 'Gd', 'Np', 'Er', 'Th', 'Cd', 'Xe', 'Tm', 'Pa', 'Ar', 'Ba', 'Bi', 'Os', 'Re', 'Lu', 'Mn', 'Au', 'In', 'Cs', 'S', 'Pd', 'Rh', 'Zr', 'B', 'Te', 'Sn', 'Br', 'V']
    ele_feature = json.load(open('/.../DC+XB/XBERT/ele_emb.json', 'r'))
    # ele_feature = json.load(open('ele_emb.json', 'r'))
    ele_embedding = []
    for i in elements:
        if comp.get_atomic_fraction(i) == 0:
            ele_embedding.append([0 for j in range(93)])
        else:
            ele_embedding.append([comp.get_atomic_fraction(i)] + ele_feature[i])
    return ele_embedding

#----------------------------得到材料中每个原子的描述---------------------------------------------------------------------------
def get_atom_embedding(x):
    ele_feature = json.load(open('/.../DC+XB/XBERT/ele_emb.json', 'r'))
    # ele_feature = json.load(open('ele_emb.json', 'r'))
    return ele_feature[x]

#----------------------------将边的信息编码为向量-------------------------------------------------------------------------------
class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2/self.var**2)

#--------------------------------将一个空间群用12个词表示，并将每个词用编码字典中的一个数表示---------------------------------
def get_spg_tokens_A(spg_symbol: str):
    spg_dict = json.load(open('/.../DC+XB/XBERT/spg_dict_new.json', 'r'))
    # spg_dict = json.load(open('spg_dict_new.json', 'r'))
    spg_tokens = list(spg_dict[spg_symbol].values())
    return spg_tokens

def get_spg_tokens_B(spg_symbol: str):
    spg_dict = json.load(open('/.../DC+XB/XBERT/spg_dict_newnew.json', 'r'))
    # spg_dict = json.load(open('spg_dict_newnew.json', 'r'))
    spg_tokens = list(spg_dict[spg_symbol].values())
    return spg_tokens

def get_token_id(tokens,vocab):
    tokens_id = []
    for token in tokens:
        # try:
        token = str(token)
        tokens_id.append(vocab[token])
        # except:
        #     print(tokens)
    return tokens_id

#-----------------------------------------------ele_vec-------------------------------
def ele_vec(x):
    ele_emb = json.load(open('/.../DC+XB/XBERT/ele_emb.json'))
    # ele_emb = json.load(open('ele_emb.json'))
    comp = Composition(x)
    s = 0
    for i in comp:
        s += comp.get_atomic_fraction(i) * np.array(ele_emb[str(i)])
    return list(s)



if __name__ == '__main__':
    # print(get_element_embedding('Na2 Cl'))
    # print(get_element_embedding('Na2 Cl')[0])
    # print(get_element_embedding('Na2 Cl')[24])
    print(ele_vec('Rb4 Zr2 O6'))
