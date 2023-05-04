from ast import Pass
import numpy as np
import random

class nasbench201_encoder:
    def __init__(self):
        pass

    def get_rand_code(self):
        return tuple(np.concatenate((np.random.randint(5,size=6),np.random.randint(4,size=1),np.random.randint(1,5,size=3))))
    
    def get_nrand_code(self, n):
        s = set()
        while len(s) < n:
            s.add(self.get_rand_code())
        return [list(c) for c in s]

    def get_rand_backbone_branch(self):
        pass

    def get_rand_backbone(self):
        pass
    
    def get_rand_branch(self):
        branch = np.random.randint(2, size=21)
        nones = np.count_nonzero(branch == 1)
        if nones>9:
            ones_index = np.where(branch==1)
            del_ones_index = random.sample(list(*ones_index),k=nones-9)
            for x in del_ones_index:
                branch[x] = 0
        return branch
    
    def get_rand_operations(self):
        return np.random.randint(3, size=5)
    
    def parse_code(self,code):
        newcode = code[:6]
        mapping = {'a':0,'b':2,'c':5,'d':1,'e':3,'f':4}
        backbone_choices = {0:np.array(['c','d','z']),1:np.array(['e','z','z']),2:np.array(['a','b','c']),3:np.array(['a','f','z'])}
        for n,i in enumerate(backbone_choices[code[6]]):
            if i not in mapping:
                break
            newcode[mapping[i]] = code[6+1+n]
        return newcode

class natsbenchSSS_encoder:
    def __init__(self):
        self.candidates = [8, 16, 24, 32, 40, 48, 56, 64]
        self.num_choices = 8
    def __len__(self):
        return 5
    def get_rand_code(self):
        return tuple(random.sample(range(0,self.num_choices),k=5))
    
    def get_nrand_code(self, n):
        s = set()
        while len(s) < n:
            s.add(self.get_rand_code())
        return [list(c) for c in s]

    def parse_code(self,code):
        if type(code)==np.int64 or type(code)==int:
            return int(code)
        if type(code)==np.ndarray and type(code[0])==np.float64:
            code = code.astype(int)
        if type(code)==list and type(code[0])==float:
            code = [int(x) for x in code]
        newcode=[]
        for c in code:
            newcode.append(self.candidates[c])
        index = 0
        for i,c in enumerate(newcode):
            index+=self.candidates.index(c)*pow(self.num_choices,len(self)-1-i)
        return index
    def encode_index(self,index):
        code = np.zeros(len(self))
        i=0
        while index > 0:
            code[len(self)-1-i] = index%self.num_choices
            index//=self.num_choices
            i+=1
        return code

def get_encoder(nasspace):
    if nasspace == 'nasbench201':
        return nasbench201_encoder()
    elif nasspace == 'natsbenchSSS':
        return natsbenchSSS_encoder()
    else:
        assert False, f"no such searchspace:{args.nasspace}"

