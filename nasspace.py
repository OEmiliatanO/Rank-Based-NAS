from models import get_cell_based_tiny_net, get_search_spaces
from nas_201_api import NASBench201API as API
from nasbench import api as nasbench101api
from nas_101_api.model import Network
from nas_101_api.model_spec import ModelSpec
from nats_api import create
import itertools
import random
import numpy as np
from models.cell_searchs.genotypes import Structure
from copy import deepcopy
import pycls.models
from pycls.models.nas.nas import NetworkImageNet, NetworkCIFAR
from pycls.models.anynet import AnyNet
from pycls.models.nas.genotypes import GENOTYPES, Genotype
import json
import torch


class Nasbench201:
    def __init__(self, dataset, apiloc):
        self.dataset = dataset
        self.api = API(apiloc, verbose=False)
        self.operations = ['none',
                                            'skip_connect',
                                            'nor_conv_1x1', 
                                            'nor_conv_3x3', 
                                            'avg_pool_3x3' ]
        self.epochs = '12'
    
    def get_index_by_code(self,code):
        node_str = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*[self.operations[c] for c in code[:6]])
        index = self.api.query_index_by_arch(node_str)
        return index

    def get_arch_str_by_code(self,code):
        node_str = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*[self.operations[c] for c in code[:6]])
        return node_str

    def get_network(self, uid, args):
        if self.dataset == "cifar10":
            dataset_name = 'cifar10-valid'
        else:
            dataset_name = self.dataset
        config = self.api.get_net_config(uid, dataset_name)
        #config = self.api.get_net_config(uid, 'cifar10-valid')
        #config['num_classes'] = 1
        network = get_cell_based_tiny_net(config)
        return network
    def __iter__(self):
        for uid in range(len(self)):
            network = self.get_network(uid)
            yield uid, network
    def __getitem__(self, index):
        return index
    def __len__(self):
        return 15625
    def num_activations(self):
        network = self.get_network(0)
        return network.classifier.in_features
    #def get_12epoch_accuracy(self, uid, acc_type, trainval, traincifar10=False):
    #    archinfo = self.api.query_meta_info_by_index(uid)
    #    if (self.dataset == 'cifar10' or traincifar10) and trainval:
    #        #return archinfo.get_metrics('cifar10-valid', acc_type, iepoch=12)['accuracy']
    #        return archinfo.get_metrics('cifar10-valid', 'x-valid', iepoch=12)['accuracy']
    #    elif traincifar10:
    #        return archinfo.get_metrics('cifar10', acc_type, iepoch=12)['accuracy']
    #    else:
    #        return archinfo.get_metrics(self.dataset, 'ori-test', iepoch=12)['accuracy']
    def get_12epoch_accuracy(self, uid, acc_type, trainval, traincifar10=False):
        #archinfo = self.api.query_meta_info_by_index(uid)
        #if (self.dataset == 'cifar10' and trainval) or traincifar10:
        info = self.api.get_more_info(uid, 'cifar10-valid', iepoch=None, hp=self.epochs, is_random=True)
        #else:
        #    info = self.api.get_more_info(uid, self.dataset, iepoch=None, hp=self.epochs, is_random=True)
        return info['valid-accuracy']
    def get_final_accuracy(self, uid, acc_type, trainval):
        #archinfo = self.api.query_meta_info_by_index(uid)
        if self.dataset == 'cifar10' and trainval:
            info = self.api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar10-valid', 'x-valid')
            #info = self.api.query_by_index(uid, 'cifar10-valid', hp='200')
            #info = self.api.get_more_info(uid, 'cifar10-valid', iepoch=None, hp='200', is_random=True)
        else:
            info = self.api.query_meta_info_by_index(uid, hp='200').get_metrics(self.dataset, acc_type)
            #info = self.api.query_by_index(uid, self.dataset, hp='200')
            #info = self.api.get_more_info(uid, self.dataset, iepoch=None, hp='200', is_random=True)
        return info['accuracy']
        #return info['valid-accuracy']
        #if self.dataset == 'cifar10' and trainval:
        #    return archinfo.get_metrics('cifar10-valid', acc_type, iepoch=11)['accuracy']
        #else:
        #    #return archinfo.get_metrics(self.dataset, 'ori-test', iepoch=12)['accuracy']
        #    return archinfo.get_metrics(self.dataset, 'x-test', iepoch=11)['accuracy']
        ##dataset = self.dataset
        ##if self.dataset == 'cifar10' and trainval:
        ##    dataset = 'cifar10-valid'
        ##archinfo = self.api.get_more_info(uid, dataset, iepoch=None, use_12epochs_result=True, is_random=True)
        ##return archinfo['valid-accuracy']

    def get_accuracy(self, uid, acc_type, trainval=True):
        archinfo = self.api.query_meta_info_by_index(uid)
        if self.dataset == 'cifar10' and trainval:
            return archinfo.get_metrics('cifar10-valid', acc_type)['accuracy']
        else:
            return archinfo.get_metrics(self.dataset, acc_type)['accuracy']

    def get_accuracy_for_all_datasets(self, uid):
        archinfo = self.api.query_meta_info_by_index(uid,hp='200')

        c10 = archinfo.get_metrics('cifar10', 'ori-test')['accuracy']
        c10_val = archinfo.get_metrics('cifar10-valid', 'x-valid')['accuracy']

        c100 = archinfo.get_metrics('cifar100', 'x-test')['accuracy']
        c100_val = archinfo.get_metrics('cifar100', 'x-valid')['accuracy']

        imagenet = archinfo.get_metrics('ImageNet16-120', 'x-test')['accuracy']
        imagenet_val = archinfo.get_metrics('ImageNet16-120', 'x-valid')['accuracy']

        return c10, c10_val, c100, c100_val, imagenet, imagenet_val

    #def train_and_eval(self, arch, dataname, acc_type, trainval=True):
    #    unique_hash = self.__getitem__(arch)
    #    time = self.get_training_time(unique_hash)
    #    acc12 = self.get_12epoch_accuracy(unique_hash, acc_type, trainval)
    #    acc = self.get_final_accuracy(unique_hash, acc_type, trainval)
    #    return acc12, acc, time
    def train_and_eval(self, arch, dataname, acc_type, trainval=True, traincifar10=False):
        unique_hash = self.__getitem__(arch)
        time = self.get_training_time(unique_hash)
        acc12 = self.get_12epoch_accuracy(unique_hash, acc_type, trainval, traincifar10)
        acc = self.get_final_accuracy(unique_hash, acc_type, trainval)
        return acc12, acc, time
    def random_arch(self):
        return random.randint(0, len(self)-1)
    def get_training_time(self, unique_hash):
        #info = self.api.get_more_info(unique_hash, 'cifar10-valid' if self.dataset == 'cifar10' else self.dataset, iepoch=None, use_12epochs_result=True, is_random=True)


        #info = self.api.get_more_info(unique_hash, 'cifar10-valid', iepoch=None, use_12epochs_result=True, is_random=True)
        info = self.api.get_more_info(unique_hash, 'cifar10-valid', iepoch=None, hp='12', is_random=True)
        return info['train-all-time'] + info['valid-per-time']
        #if self.dataset == 'cifar10' and trainval:
        #    info = self.api.get_more_info(unique_hash, 'cifar10-valid', iepoch=None, hp=self.epochs, is_random=True)
        #else:
        #    info = self.api.get_more_info(unique_hash, self.dataset, iepoch=None, hp=self.epochs, is_random=True)

        ##info = self.api.get_more_info(unique_hash, 'cifar10-valid', iepoch=None, use_12epochs_result=True, is_random=True)
        #return info['train-all-time'] + info['valid-per-time']
    def mutate_arch(self, arch):
        op_names = get_search_spaces('cell', 'nas-bench-201')
        #config = self.api.get_net_config(arch, self.dataset)
        config = self.api.get_net_config(arch, 'cifar10-valid')
        parent_arch = Structure(self.api.str2lists(config['arch_str']))
        child_arch = deepcopy( parent_arch )
        node_id = random.randint(0, len(child_arch.nodes)-1)
        node_info = list( child_arch.nodes[node_id] )
        snode_id = random.randint(0, len(node_info)-1)
        xop = random.choice( op_names )
        while xop == node_info[snode_id][0]:
          xop = random.choice( op_names )
        node_info[snode_id] = (xop, node_info[snode_id][1])
        child_arch.nodes[node_id] = tuple( node_info )
        arch_index = self.api.query_index_by_arch( child_arch )
        return arch_index
    def query_index_by_arch(self, arch):
        return self.api.query_index_by_arch(arch)

class Nasbench101:
    def __init__(self, dataset, apiloc, args):
        self.dataset = dataset
        self.api = nasbench101api.NASBench(apiloc)
        self.args = args
    def get_accuracy(self, unique_hash, acc_type, trainval=True):
        spec = self.get_spec(unique_hash)
        _, stats = self.api.get_metrics_from_spec(spec)
        maxacc = 0
        for ep in stats:
            for statmap in stats[ep]:
                newacc = statmap['final_test_accuracy']
                if newacc > maxacc:
                    maxacc = newacc
        return maxacc*100
    def get_final_accuracy(self, uid, acc_type, trainval):
        spec = self.get_spec(uid)
        if trainval:
            return self.api.query(spec)["validation_accuracy"]*100
        return self.api.query(spec)["test_accuracy"]*100
    def get_acc_all(self, uid, args):
        spec = self.get_spec(uid)
        info = self.api.query(spec)
        return info["validation_accuracy"]*100, info["test_accuracy"]*100

    def get_training_time(self, unique_hash):
        spec = self.get_spec(unique_hash)
        _, stats = self.api.get_metrics_from_spec(spec)
        maxacc = -1.
        maxtime = 0.
        for ep in stats:
            for statmap in stats[ep]:
                newacc = statmap['final_test_accuracy']
                if newacc > maxacc:
                    maxacc = newacc
                    maxtime = statmap['final_training_time']
        return maxtime
    def get_network(self, unique_hash, args):
        spec = self.get_spec(unique_hash)
        network = Network(spec, self.args)
        return network
    def get_network_by_matrix(self, m, ops):
        spec = Modelspec(m, ops)
        return Network(spec, self.args)
    def get_spec(self, unique_hash):
        matrix = self.api.fixed_statistics[unique_hash]['module_adjacency']
        operations = self.api.fixed_statistics[unique_hash]['module_operations']
        spec = ModelSpec(matrix, operations)
        return spec
    def get_spec_by_arch(self, m, op):
        return ModelSpec(m, op)
    def query_index_by_arch(self, spec):
        return self.api._hash_spec(spec)
    def __iter__(self):
        for unique_hash in self.api.hash_iterator():
            network = self.get_network(unique_hash, self.args)
            yield unique_hash, network
    def __getitem__(self, index):
        return next(itertools.islice(self.api.hash_iterator(), index, None))
    def __len__(self):
        return len(self.api.hash_iterator())
    def num_activations(self):
        for unique_hash in self.api.hash_iterator():
            network = self.get_network(unique_hash)
            return network.classifier.in_features
    def train_and_eval(self, arch, dataname, acc_type, trainval=True, traincifar10=False):
        unique_hash = self.__getitem__(arch)
        time =12.* self.get_training_time(unique_hash)/108.
        acc = self.get_accuracy(unique_hash, acc_type, trainval)
        return acc, acc, time
    def random_arch(self):
        return random.randint(0, len(self)-1)
    def mutate_arch(self, arch):
        unique_hash = self.__getitem__(arch)
        matrix = self.api.fixed_statistics[unique_hash]['module_adjacency']
        operations = self.api.fixed_statistics[unique_hash]['module_operations']
        coords = [ (i, j) for i in range(matrix.shape[0]) for j in range(i+1, matrix.shape[1])]
        random.shuffle(coords)
        # loop through changes until we find change thats allowed
        for i, j in coords:
            # try the ops in a particular order
            for k in [m for m in np.unique(matrix) if m != matrix[i, j]]:
                newmatrix = matrix.copy()
                newmatrix[i, j] = k
                spec = ModelSpec(newmatrix, operations)
                try:
                    newhash = self.api._hash_spec(spec)
                    if newhash in self.api.fixed_statistics:
                        return [n for n, m in enumerate(self.api.fixed_statistics.keys()) if m == newhash][0]
                except:
                    pass


class NatsbenchSSS:
    def __init__(self, dataset, apiloc, args):
        self.dataset = dataset
        self.api = create(apiloc, 'sss', fast_mode=True, verbose=False)
        self.args=args

    def __len__(self):
        return 32768

    def __iter__(self):
        for uid in range(len(self)):
            network = self.get_network(uid, self.args)
            yield uid, network

    def __getitem__(self, index):
        return index

    def query_index_by_arch(self, arch):
        ind = self.api.query_index_by_arch(arch)
        if ind == -1:
            raise Exception(f"not found arch: {arch}. return -1.")
        return ind

    def get_network(self,index,args):
        index = self.api.query_index_by_arch(index)
        if args.dataset == "cifar10":
            dataname = "cifar10-valid"
        else:
            dataname = args.dataset
        config = self.api.get_net_config(index, dataname)
        #config['num_classes'] = 1
        network = get_cell_based_tiny_net(config)
        return network
    
    def get_acc(self, index, args,hp=90):
        index = self.api.query_index_by_arch(index)
        info = self.api.get_more_info(index, args.dataset, hp= hp)
        validation_accuracy, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset=args.dataset, hp=hp)
        return info['test-accuracy'],validation_accuracy

    def get_acc_all(self, index, args):
        info_cifar10 = self.api.get_more_info(index, 'cifar10', hp=90)
        info_cifar100 = self.api.get_more_info(index, 'cifar100', hp=90)
        info_imagenet = self.api.get_more_info(index, 'ImageNet16-120', hp=90)

        validation_accuracy_cifar10, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset='cifar10-valid', hp=90)
        validation_accuracy_cifar100, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset='cifar100', hp=90)
        validation_accuracy_imagenet, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset='ImageNet16-120', hp=90)        
        return validation_accuracy_cifar10,info_cifar10['test-accuracy'],validation_accuracy_cifar100,info_cifar100['test-accuracy'],validation_accuracy_imagenet,info_imagenet['test-accuracy']

    def get_training_time(self,index,args,hp):
        validation_accuracy, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset=args.dataset, hp=hp)
        return time_cost

    def get_final_accuracy(self, index, acc_type, trainval):
        if trainval:
            acc, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset=self.dataset, hp=90)
            return acc
        return self.api.get_more_info(index, self.dataset, hp=90)['test-accuracy']

class NatsbenchTSS:
    def __init__(self, dataset, apiloc, args):
        self.dataset = dataset
        self.api = create(apiloc, 'tss', fast_mode=True, verbose=False)
        self.args=args
        self.operations = ['none',
                                            'skip_connect',
                                            'nor_conv_1x1', 
                                            'nor_conv_3x3', 
                                            'avg_pool_3x3' ]

    def __len__(self):
        return 15625

    def __iter__(self):
        for uid in range(len(self)):
            network = self.get_network(uid, self.args)
            yield uid, network

    def __getitem__(self, index):
        return index

    def get_index_by_code(self,code):
        node_str = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*[self.operations[c] for c in code[:6]])
        index = self.api.query_index_by_arch(node_str)
        return index

    def get_arch_str_by_code(self,code):
        node_str = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*[self.operations[c] for c in code[:6]])
        return node_str

    def query_index_by_arch(self, arch):
        return self.api.query_index_by_arch(arch)

    def get_network(self,index,args):
        index = self.api.query_index_by_arch(index)
        if args.dataset == "cifar10":
            dataname = "cifar10-valid"
        else:
            dataname = args.dataset
        config = self.api.get_net_config(index, dataname)
        #config['num_classes'] = 1
        network = get_cell_based_tiny_net(config)
        return network
    
    def get_training_time(self,index,args,hp):
        validation_accuracy, latency, time_cost, current_total_time_cost = self.api.simulate_train_eval(index, dataset=args.dataset, hp=hp)
        return time_cost

    def get_final_accuracy(self, uid, acc_type, trainval):
        if self.dataset == 'cifar10' and trainval:
            info = self.api.query_meta_info_by_index(uid, hp='200').get_metrics('cifar10-valid', 'x-valid')
            #info = self.api.query_by_index(uid, 'cifar10-valid', hp='200')
            #info = self.api.get_more_info(uid, 'cifar10-valid', iepoch=None, hp='200', is_random=True)
        else:
            info = self.api.query_meta_info_by_index(uid, hp='200').get_metrics(self.dataset, acc_type)
            #info = self.api.query_by_index(uid, self.dataset, hp='200')
            #info = self.api.get_more_info(uid, self.dataset, iepoch=None, hp='200', is_random=True)
        return info['accuracy']

    def get_acc_all(self, index, args):
        index = self.api.query_index_by_arch(index)
        information = self.api.arch2infos_dict[index]['200']

        valid_info_cifar10 = information.get_metrics('cifar10-valid', 'x-valid')
        valid_acc_cifar10 = valid_info_cifar10['accuracy']
        test__info_cifar10 = information.get_metrics('cifar10', 'ori-test')
        test_acc_cifar10 = test__info_cifar10['accuracy']

        valid_info_cifar100 = information.get_metrics('cifar100', 'x-valid')
        test__info_cifar100 = information.get_metrics('cifar100', 'x-test')
        valid_acc_cifar100 = valid_info_cifar100['accuracy']
        test_acc_cifar100 = test__info_cifar100['accuracy']

        valid_info_imagenet = information.get_metrics('ImageNet16-120', 'x-valid')
        test__info_imagenet = information.get_metrics('ImageNet16-120', 'x-test')
        valid_acc_imagenet = valid_info_imagenet['accuracy']
        test_acc_imagenet = test__info_imagenet['accuracy']

        return valid_acc_cifar10,test_acc_cifar10,valid_acc_cifar100,test_acc_cifar100,valid_acc_imagenet,test_acc_imagenet

class ReturnFeatureLayer(torch.nn.Module):
    def __init__(self, mod):
        super(ReturnFeatureLayer, self).__init__()
        self.mod = mod
    def forward(self, x):
        return self.mod(x), x
                

def return_feature_layer(network, prefix=''):
    #for attr_str in dir(network):
    #    target_attr = getattr(network, attr_str)
    #    if isinstance(target_attr, torch.nn.Linear):
    #        setattr(network, attr_str, ReturnFeatureLayer(target_attr))
    for n, ch in list(network.named_children()):
        if isinstance(ch, torch.nn.Linear):
            setattr(network, n, ReturnFeatureLayer(ch))
        else:
            return_feature_layer(ch, prefix + '\t')
             

class NDS:
    def __init__(self, searchspace):
        self.searchspace = searchspace
        data = json.load(open(f'nds_data/{searchspace}.json', 'r'))
        try:
            data = data['top'] + data['mid']
        except Exception as e:
            pass
        self.data = data
    def __iter__(self):
        for unique_hash in range(len(self)):
            network = self.get_network(unique_hash)
            yield unique_hash, network
    def get_network_config(self, uid):
        return self.data[uid]['net']
    def get_network_optim_config(self, uid):
        return self.data[uid]['optim']
    def get_network(self, uid):
        netinfo = self.data[uid]
        config = netinfo['net']
        #print(config)
        if 'genotype' in config:
            #print('geno')
            gen = config['genotype']
            genotype = Genotype(normal=gen['normal'], normal_concat=gen['normal_concat'], reduce=gen['reduce'], reduce_concat=gen['reduce_concat'])
            if '_in' in self.searchspace:
                network = NetworkImageNet(config['width'], 1, config['depth'], config['aux'],  genotype)
            else:
                network = NetworkCIFAR(config['width'], 1, config['depth'], config['aux'],  genotype)
            network.drop_path_prob = 0.
            #print(config)
            #print('genotype')
            L = config['depth']
        else:
            if 'bot_muls' in config and 'bms' not in config:
                config['bms'] = config['bot_muls']
                del config['bot_muls']
            if 'num_gs' in config and 'gws' not in config:
                config['gws'] = config['num_gs']
                del config['num_gs']
            config['nc'] = 1
            config['se_r'] = None
            config['stem_w'] = 12
            L = sum(config['ds'])
            if 'ResN' in self.searchspace:
                config['stem_type'] = 'res_stem_in'
            else:
                config['stem_type'] = 'simple_stem_in'
            #"res_stem_cifar": ResStemCifar,
            #"res_stem_in": ResStemIN,
            #"simple_stem_in": SimpleStemIN,
            if config['block_type'] == 'double_plain_block':
                config['block_type'] = 'vanilla_block'
            network = AnyNet(**config)
        return_feature_layer(network)
        return network
    def __getitem__(self, index):
        return index
    def __len__(self):
        return len(self.data)
    def random_arch(self):
        return random.randint(0, len(self.data)-1)
    def get_final_accuracy(self, uid, acc_type, trainval):
        return 100.-self.data[uid]['test_ep_top1'][-1]


def get_search_space(args):
    if args.nasspace == 'nasbench201':
        #return Nasbench201(args.dataset, args.api_loc)
        return NatsbenchTSS(args.dataset, args.api_loc, args)
    elif args.nasspace == 'nasbench101':
        return Nasbench101(args.dataset, args.api_loc, args)
    elif args.nasspace == 'natsbenchsss':
        return NatsbenchSSS(args.dataset, args.api_loc, args)
    elif args.nasspace == 'natsbenchtss':
        return NatsbenchTSS(args.dataset, args.api_loc, args)
    elif args.nasspace == 'nds_resnet':
        return NDS('ResNet')
    elif args.nasspace == 'nds_amoeba':
        return NDS('Amoeba')
    elif args.nasspace == 'nds_amoeba_in':
        return NDS('Amoeba_in')
    elif args.nasspace == 'nds_darts_in':
        return NDS('DARTS_in')
    elif args.nasspace == 'nds_darts':
        return NDS('DARTS')
    elif args.nasspace == 'nds_darts_fix-w-d':
        return NDS('DARTS_fix-w-d')
    elif args.nasspace == 'nds_darts_lr-wd':
        return NDS('DARTS_lr-wd')
    elif args.nasspace == 'nds_enas':
        return NDS('ENAS')
    elif args.nasspace == 'nds_enas_in':
        return NDS('ENAS_in')
    elif args.nasspace == 'nds_enas_fix-w-d':
        return NDS('ENAS_fix-w-d')
    elif args.nasspace == 'nds_pnas':
        return NDS('PNAS')
    elif args.nasspace == 'nds_pnas_fix-w-d':
        return NDS('PNAS_fix-w-d')
    elif args.nasspace == 'nds_pnas_in':
        return NDS('PNAS_in')
    elif args.nasspace == 'nds_nasnet':
        return NDS('NASNet')
    elif args.nasspace == 'nds_nasnet_in':
        return NDS('NASNet_in')
    elif args.nasspace == 'nds_resnext-a':
        return NDS('ResNeXt-A')
    elif args.nasspace == 'nds_resnext-a_in':
        return NDS('ResNeXt-A_in')
    elif args.nasspace == 'nds_resnext-b':
        return NDS('ResNeXt-B')
    elif args.nasspace == 'nds_resnext-b_in':
        return NDS('ResNeXt-B_in')
    elif args.nasspace == 'nds_vanilla':
        return NDS('Vanilla')
    elif args.nasspace == 'nds_vanilla_lr-wd':
        return NDS('Vanilla_lr-wd')
    elif args.nasspace == 'nds_vanilla_lr-wd_in':
        return NDS('Vanilla_lr-wd_in')
    else:
        assert False, f"no such search space: {args.nasspace}"
