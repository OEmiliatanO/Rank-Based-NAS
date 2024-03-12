import numpy as np
import random
import itertools

class nasbench201_encoder:
    def __init__(self):
        self.search_space = self.construct_search_space()

    def __getitem__(self, n):
        return self.search_space[n]

    def __len__(self):
        return len(self.search_space)
    
    def construct_search_space(self):
        mapping = {'a':0,'b':2,'c':5,'d':1,'e':3,'f':4}
        backbone_choices = {0:('c','d','z'),1:('e','z','z'),2:('a','b','c'),3:('a','f','z')}

        def has_backbone(net):
            has = [True, True, True, True]
            for i, backbones in enumerate(backbone_choices.values()):
                for backbone in backbones:
                    if backbone not in mapping: break
                    has[i] &= (net[mapping[backbone]] != 0)
            return np.array(has).any()
                
        all_nets = list(itertools.product(range(5), repeat=6))
        all_nets = list(filter(has_backbone, all_nets))
        return all_nets

    def get_rand_code(self):
        return tuple(np.concatenate((np.random.randint(5,size=6),np.random.randint(4,size=1),np.random.randint(1,5,size=3))))
    
    def get_nrand_code(self, n):
        return random.sample(range(len(self.search_space)), n)

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
        code = self.search_space[code]
        """
        newcode = code[:6]
        mapping = {'a':0,'b':2,'c':5,'d':1,'e':3,'f':4}
        backbone_choices = {0:np.array(['c','d','z']),1:np.array(['e','z','z']),2:np.array(['a','b','c']),3:np.array(['a','f','z'])}
        for n,i in enumerate(backbone_choices[code[6]]):
            if i not in mapping:
                break
            newcode[mapping[i]] = code[6+1+n]
        return newcode
        """
        return code

class natsbenchSSS_encoder:
    def __init__(self):
        self.candidates = [8, 16, 24, 32, 40, 48, 56, 64]
        self.num_choices = 8

    def __getitem__(self, n):
        return n

    def __len__(self):
        return 32768

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
            index+=self.candidates.index(c)*pow(self.num_choices,5-1-i)
        return index

    def encode_index(self,index):
        code = np.zeros(len(self))
        i=0
        while index > 0:
            code[len(self)-1-i] = index%self.num_choices
            index//=self.num_choices
            i+=1
        return code

class nasbench101_encoder:
    def __init__(self):
        pass
    
    def get_rand_code(self):
        operations = self.get_rand_operations()
        backbone,branch = self.get_rand_backbone_branch()
        return tuple(backbone),tuple(branch),tuple(operations)
    
    def get_nrand_code(self, n):
        s = set()
        while len(s) < n:
            bb, br, op = self.get_rand_code()
            bb, br, op = tuple(bb), tuple(br), tuple(op)
            s.add((bb,br,op))
        s = list(s)
        for i in range(len(s)):
            s[i] = list(s[i])
            s[i][0], s[i][1], s[i][2] = list(s[i][0]), list(s[i][1]), list(s[i][2])
        return s

    def get_rand_backbone_branch(self):
        backbone = np.concatenate((np.ones(3),np.zeros(2)))
        random.shuffle(backbone)
        nones = np.count_nonzero(backbone == 1)
        if nones==0:
            del_zero_index = random.choice(range(5))
            backbone[del_zero_index] = 1
        #backbone_nodes = random.sample([0,1,2,3,4], random.randint(1,5))
        #backbone = [0,0,0,0,0]
        #for pos in backbone_nodes:
        #    backbone[pos] = 1
        
        branch = np.concatenate((np.ones(5),np.zeros(4)))
        branch = np.concatenate((branch,np.zeros(12)))
        random.shuffle(branch)
        #branch = np.array([random.randint(0,1) for i in range(21)])
        
        #check same edges
        matrix = np.zeros([7,7])
        previous = -1
        for x,v in enumerate(backbone):
            if v==0:
                continue
            if previous==-1:
                matrix[0][x+1] = 1
            else:
                matrix[previous+1][x+1] = 1
            previous = x
        matrix[previous+1][-1] = 1
        map_backbone = matrix[np.triu_indices_from(matrix,k=1)]
        combined_m = 2*map_backbone + branch

        #check whether the edges of c1 and c2 exceeds the upper bound 9. If so, randomly delete some edges.
        zeros = np.count_nonzero(combined_m == 0)
        if zeros<12:
            del_ones_index = random.sample(list(*np.where(combined_m==1)),k=12-zeros)
            for x in del_ones_index:
                branch[x] = 0
        
        return backbone,branch

    def get_rand_backbone(self):
        backbone = np.random.randint(2, size=5)
        nones = np.count_nonzero(backbone == 1)
        if nones==0:
            del_zero_index = random.choice(range(5))
            backbone[del_zero_index] = 1
        return backbone
    
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
    
    def parse_code(self,backbone,branch,operations):
        backbone, branch, operations = np.array(backbone), np.array(branch), np.array(operations)
        if len(backbone)!=5 or np.count_nonzero(backbone == 1)==0:
            print('backbone: ',backbone)
            print(f"[backbone == 1] = {backbone == 1}")
            print(f"np.count_nonzero(backbone == 1) shall not be 0 but {np.count_nonzero(backbone == 1)}")
            raise ValueError(f"Invalid backbone: {backbone}.")
        matrix = np.zeros([7,7])
        vector = np.zeros(7)
        #Connect branch
        counter = 0
        for i in range(6,0,-1):
            for j in range(1,i+1):
                matrix[6-i][6-i+j] = branch[counter]
                counter+=1
        #Connect backbone
        previous = -1
        for i,v in enumerate(backbone):
            if v==0:
                continue
            if previous==-1:
                matrix[0][i+1] = 1
            else:
                matrix[previous+1][i+1] = 1
            previous = i
        matrix[previous+1][-1] = 1
        #Operations mapping
        vector = ['input', '' ,'' ,'' ,'' ,'' , 'output']
        for i,x in enumerate(operations):
            if x==0:
                vector[i+1] = 'conv1x1-bn-relu'
            elif x==1:
                vector[i+1] = 'conv3x3-bn-relu'
            elif x==2:
                vector[i+1] = 'maxpool3x3'
        return matrix.astype('int32'),vector

def get_encoder(nasspace):
    if nasspace == 'nasbench201':
        return nasbench201_encoder()
    elif nasspace == 'natsbenchsss':
        return natsbenchSSS_encoder()
    elif nasspace == 'nasbench101':
        return nasbench101_encoder()
    else:
        assert False, f"no such searchspace:{nasspace}"

