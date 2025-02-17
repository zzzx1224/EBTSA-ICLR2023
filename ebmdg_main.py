from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pdb
import os, shutil
import argparse
import time
from tensorboardX import SummaryWriter
from aug import *
import pdb
from ebm_dataset import *
import ebmdg_model
import sys
from utils import ReplayBuffer, ReservoirBuffer

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='learning rate')
parser.add_argument('--sparse', default=0, type=float, help='L1 panelty')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log_ebm', help='Log dir [default: log]')
parser.add_argument('--pretrain_dir', default='log1', help='Loading the trained backbone')
parser.add_argument('--dataset', default='PACS', help='datasets')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training [default: 32]')
parser.add_argument('--shuffle', type=int, default=0, help='Batch Size during training [default: 32]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--net', default='res18', help='res18 or res50')
parser.add_argument('--energy_type', default='sigmoid', help='sigmoid or square')
parser.add_argument('--test_domain', default='sketch', help='GPU to use [default: GPU 0]')
parser.add_argument('--train_domain', default='', help='GPU to use [default: GPU 0]')
parser.add_argument('--ite_train', default=True, type=bool, help='learning rate')
parser.add_argument('--max_ite', default=10000, type=int, help='max_ite')
parser.add_argument('--test_ite', default=50, type=int, help='learning rate')
parser.add_argument('--test_batch', default=40, type=int, help='learning rate')
parser.add_argument('--data_aug', default=1, type=int, help='whether sample')
parser.add_argument('--difflr', default=1, type=int, help='whether sample')
parser.add_argument('--norm', default='bn', help='bn or in')
parser.add_argument('--reslr', default=0.1, type=float, help='backbone learning rate')
parser.add_argument('--num_steps', default=20, type=int, help='Steps of gradient descent for ebm')
parser.add_argument('--step_lr', default=100.0, type=float, help='lr for langevin dynamic')
parser.add_argument('--isreplay', default=1, type=int, help='Use MCMC chains initialized from a replay buffer.')
parser.add_argument('--reservoir', default=0, type=int, help='Use a reservoir of past entires')
parser.add_argument('--buffer_size', default=500, type=int, help='size of replay buffer')
parser.add_argument('--l2_coeff', default=0, type=float, help='coefficient for l2 on energy')
parser.add_argument('--en_coeff', default=1, type=float, help='coefficient for energy on langevin dynamic')
parser.add_argument('--loss_coeff', default=1, type=float, help='coefficient for based energy loss')
parser.add_argument('--cla_coeff', default=1, type=float, help='coefficient for classification on langevin dynamic')
parser.add_argument('--kl_coeff', default=0.1, type=float, help='coefficient for classification on langevin dynamic')
parser.add_argument('--extra_sup', default=1, type=int, help='Use extra supervison for langevin dynamic.')
parser.add_argument('--spec_norm', default=1, type=int, help='Whether to use spectral normalization on weights')
parser.add_argument('--model', default='ebm', help='backbone or ebm or imgebm or jem')
parser.add_argument('--prenorm', default='no', help='tanh or sigmoid or no')
parser.add_argument('--zmethod', default='concat', help='concat or dotp or no')
parser.add_argument('--mctimes', default=10, type=int, help='sample number of MC')
parser.add_argument('--energy_level', default='low', help='low or high or mul')
parser.add_argument('--ispretrain', default=0, type=int, help='pretrained backbone.')
parser.add_argument('--clipgrad', default=1, type=int, help='clip the gradient during Langevin dynamic.')
parser.add_argument('--pebm', default=1, type=int, help='start with noisy negative samples.')
parser.add_argument('--sampz', default=0, type=int, help='sampling z from the distribution.')
parser.add_argument('--earlystop', default=1, type=int, help='early stopping for langevin dynamics.')
parser.add_argument('--transf', default=0, type=int, help='transformer encoder.')
parser.add_argument('--dztype', default='p', help='p or a')
parser.add_argument('--znet', default='z', help='z or zz')
parser.add_argument('--ctx_num', default=10, type=int, help='sample number of MC')
parser.add_argument('--ebmdrop', default=0, type=int, help='pretrained backbone.')
parser.add_argument('--noisyneg', default=0, type=int, help='noisy negative samples.')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
gpu_index = args.gpu
net_backbone = args.net
max_ite = args.max_ite
test_ite = args.test_ite
test_batch = args.test_batch
iteration_training = args.ite_train
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
test_domain = args.test_domain
train_domain = args.train_domain
difflr = args.difflr
res_lr = args.reslr
difflr = bool(difflr)
norm_method = args.norm
kl_coeff = args.kl_coeff
num_steps = args.num_steps
step_lr = args.step_lr
isreplay = bool(args.isreplay)
extra_sup = bool(args.extra_sup)
reservoir = bool(args.reservoir)
en_coeff = args.en_coeff
l2_coeff = args.l2_coeff
loss_coeff = args.loss_coeff
cla_coeff = args.cla_coeff
energy_type = args.energy_type
spec_norm = bool(args.spec_norm)
using_model = args.model
pretrain_dir = args.pretrain_dir
ispretrain = bool(args.ispretrain)
pebm = bool(args.pebm)
clipgrad = bool(args.clipgrad)
# pdb.set_trace()
mctimes = args.mctimes
buffer_size = args.buffer_size
prenorm = args.prenorm
ctx_num = args.ctx_num
data_aug = args.data_aug
data_aug = bool(data_aug)

LOG_DIR = os.path.join('logs', args.log_dir)
args.log_dir = LOG_DIR

name_file = sys.argv[0]
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)
os.mkdir(LOG_DIR + '/train_img')
os.mkdir(LOG_DIR + '/test_img')
os.mkdir(LOG_DIR + '/files')
os.system('cp %s %s' % (name_file, LOG_DIR))
os.system('cp %s %s' % ('*.py', os.path.join(LOG_DIR, 'files')))
os.system('cp -r %s %s' % ('models', os.path.join(LOG_DIR, 'files')))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
print(args)
LOG_FOUT.write(str(args)+'\n')

if args.net =='res18':
    feat_dim = 512
elif args.net == 'res50':
    feat_dim = 2048


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-2)
            if m.bias is not None:
                init.constant(m.bias, 0)

def kl_divergence(mu_q, sigma_q, mu_p, sigma_p):

        var_q = sigma_q**2 + 1e-6
        var_p = sigma_p**2 + 1e-6

        component1 = torch.log(var_p) - torch.log(var_q)
        component2 = var_q / var_p
        component3 = (mu_p - mu_q).pow(2)/ var_p

        KLD = 0.5 * torch.sum((component1 -1 +component2 +component3),1)
        # pdb.set_trace()
        return KLD

def log_string(out_str, print_out=True):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    if print_out:
        print(out_str)

st = ' '
log_string(st.join(sys.argv))

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# pdb.set_trace()
best_acc = 0  # best test accuracy
best_valid_acc = 0 # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


writer = SummaryWriter(log_dir=args.log_dir)

# Data
print('==> Preparing data..')

bird = False

decay_inter = [250, 450]

if args.dataset == 'PACS':
    NUM_CLASS = 7
    num_domain = 4
    batchs_per_epoch = 0
    # ctx_test = 2 * ctx_num
    ctx_test = ctx_num
    domains = ['art_painting', 'photo', 'cartoon', 'sketch']
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
        domains = train_domain.split(',')
    log_string('data augmentation is ' + str(data_aug))
    if data_aug:
        # log_string()
        transform_train = transforms.Compose([
            # transforms.RandomCrop(64, padding=4),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2),
            transforms.RandomHorizontalFlip(),
            ImageJitter(jitter_param),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    log_string('train_domain: ' + str(domains))
    log_string('test: ' + str(test_domain))
    
    all_dataset = PACS4ebm(test_domain)
    test_cont_data = rtPACS(test_domain, ctx_num)
    # all_dataset = PACS4ebm_samec(test_domain)

elif args.dataset == 'PACSp':
    NUM_CLASS = 7
    num_domain = 4
    batchs_per_epoch = 0
    # ctx_test = 2 * ctx_num
    ctx_test = ctx_num
    domains = ['art_painting', 'photo', 'cartoon', 'sketch']
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
        domains = train_domain.split(',')
    log_string('data augmentation is ' + str(data_aug))
    if data_aug:
        # log_string()
        transform_train = transforms.Compose([
            # transforms.RandomCrop(64, padding=4),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2),
            transforms.RandomHorizontalFlip(),
            ImageJitter(jitter_param),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    log_string('train_domain: ' + str(domains))
    log_string('test: ' + str(test_domain))
    
    all_dataset = PACS4ebmp(test_domain)
    test_cont_data = rtPACS(test_domain, ctx_num)
    # all_dataset = PACS4ebm_samec(test_domain)

elif args.dataset == 'office':
    NUM_CLASS = 65
    num_domain = 4
    batchs_per_epoch = 0
    ctx_test = ctx_num
    domains = ['art', 'clipart', 'product', 'real_World']
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
        domains = train_domain.split(',')
    log_string('data augmentation is ' + str(data_aug))
    if data_aug:
        # log_string()
        transform_train = transforms.Compose([
            # transforms.RandomCrop(64, padding=4),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2),
            transforms.RandomHorizontalFlip(),
            ImageJitter(jitter_param),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    if args.noisyneg:
        transform_neg = transforms.Compose([
            # transforms.RandomCrop(64, padding=4),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2),
            transforms.RandomHorizontalFlip(),
            ImageJitter(jitter_param),
            Addfilter(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform_neg = None

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    log_string('train_domain: ' + str(domains))
    log_string('test: ' + str(test_domain))
    
    all_dataset = Office4ebm(test_domain)
    test_cont_data = rtOF(test_domain, ctx_num)

elif args.dataset == 'mnist':
    NUM_CLASS = 10
    num_domain = 7
    batchs_per_epoch = 0   #20
    # pdb.set_trace()
    ctx_test = ctx_num
    domains = ['0', '15', '30', '45', '60', '75', '90']
    test_domain = test_domain.split(',')
    # assert test_domain in domains
    for tes_dom in test_domain:
        domains.remove(tes_dom)
    if train_domain:
        domains = train_domain.split(',')
    log_string('data augmentation is ' + str(data_aug))
    if data_aug:
        # log_string()
        transform_train = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(1, 1), interpolation=2),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2), ratio=(1.0, 1.0), interpolation=2),
            transforms.RandomHorizontalFlip(),
            # ImageJitter(jitter_param),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # ImageJitter(jitter_param),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    log_string('train_domain: ' + str(domains))
    log_string('test: ' + str(test_domain))
    
    all_dataset = MNIST4ebm(test_domain, len(domains))
    test_cont_data = rtMNIST(test_domain, ctx_num)

elif args.dataset == 'svhn':
    NUM_CLASS = 10
    num_domain = 4
    batchs_per_epoch = 0
    ctx_test = ctx_num
    test_domain = 'mnist'
    domains = ['svhn']
    log_string('data augmentation is ' + str(data_aug))
    if data_aug:
        # log_string()
        transform_train = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(1, 1), interpolation=2),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2), ratio=(1.0, 1.0), interpolation=2),
            transforms.RandomHorizontalFlip(),
            # ImageJitter(jitter_param),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # ImageJitter(jitter_param),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    log_string('train_domain: ' + train_domain)
    log_string('test: ' + str(test_domain))
    
    all_dataset = SVHN4ebm(test_domain)
    test_cont_data = rtSVHN(test_domain, ctx_num)

else:
    raise NotImplementedError

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

args.num_classes = NUM_CLASS
args.num_domains = num_domain
args.bird = bird

# Model
print('==> Building model..')

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


# backbone = ebmdg_model.net0(args.num_classes, net_backbone)
if args.dataset == 'mnist':
    backbone = ebmdg_model.net1(args.num_classes, 1)
elif args.dataset == 'svhn':
    backbone = ebmdg_model.net1(args.num_classes, 3)
else:
    backbone = ebmdg_model.net0(args.num_classes, net_backbone)
# backbone = ebmdg_model.net0_m(args.num_classes, net_backbone)

if using_model == 'ebm':
    ebms = ebmdg_model.ebmzs(net_backbone, num_domain-1, spec_norm, energy_type, args.zmethod, step_lr)
    zclaf = ebmdg_model.zzclsfier(feat_dim, NUM_CLASS)
elif using_model=='domainebm':
    ebms = ebmdg_model.domain_ebms(net_backbone, num_domain-1, spec_norm, energy_type, prenorm)
elif using_model == 'ebmz':
    ebms = ebmdg_model.ebmzs(net_backbone, num_domain-1, spec_norm, energy_type, args.zmethod, step_lr)
    if args.znet=='zz':
        zclaf = ebmdg_model.zzzclsfier(feat_dim, NUM_CLASS, args.ebmdrop)
    elif args.znet=='z':
        zclaf = ebmdg_model.zzclsfier(feat_dim, NUM_CLASS)

if args.transf:
    # encoder_layers = nn.TransformerEncoderLayer(512, 8, batch_first=True)
    encoder_layers = nn.TransformerEncoderLayer(512, 8)
    transformer_encoder = nn.TransformerEncoder(encoder_layers, 2).to(device)
    transformer_encoder.train()
    optimizerT = torch.optim.Adam(transformer_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# pdb.set_trace()
log_string(str(backbone.extra_repr))
log_string(str(zclaf.extra_repr))
log_string(str(ebms.extra_repr))

pc = get_parameter_number(ebms)
log_string('Total: %.4fM, Trainable: %.4fM' %(pc['Total']/float(1e6), pc['Trainable']/float(1e6)))

backbone = backbone.to(device)
ebms = ebms.to(device)
zclaf = zclaf.to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    backbone = nn.DataParallel(backbone)
    ebms = nn.DataParallel(ebms)
    zclaf = nn.DataParallel(zclaf)

# if isinstance(net,torch.nn.DataParallel):
#     net = net.module
if using_model == 'backbone':
    backbone.train()
    net = backbone
elif using_model == 'ebm':
    # pdb.set_trace()
    checkpoint = torch.load(os.path.join('logs', pretrain_dir, 'ckpt.t7'))
    backbone.load_state_dict(checkpoint['net'])
    zclaf.load_state_dict(checkpoint['znet'])
    best_valid_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    backbone.eval()
    zclaf.eval()
    net = ebms
else:
    if ispretrain:
        checkpoint = torch.load(os.path.join('logs', pretrain_dir, 'ckpt.t7'))
        backbone.load_state_dict(checkpoint['net'])
        zclaf.load_state_dict(checkpoint['znet'])
        best_valid_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        backbone.eval()
    else:
        backbone.train()

    ebms.train()
    zclaf.train()
    net = ebms
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# pdb.set_trace()

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_valid_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

WEIGHT_DECAY = args.weight_decay

if OPTIMIZER == 'momentum':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY, momentum=0.9)
elif OPTIMIZER == 'nesterov':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY, momentum=0.9, nesterov=True)
elif OPTIMIZER=='adam' and using_model=='backbone' and difflr and not ispretrain:
    optimizer = torch.optim.Adam([{'params': net.layer0.parameters(), 'lr':args.lr * res_lr},   # different lr
                                  {'params': net.layer1.parameters(), 'lr':args.lr * res_lr},
                                  {'params': net.layer2.parameters(), 'lr':args.lr * res_lr},
                                  {'params': net.layer3.parameters(), 'lr':args.lr * res_lr},
                                  {'params': net.layer4.parameters(), 'lr':args.lr * res_lr},
                                  {'params': net.classifiers.parameters()}], 
                                  lr=args.lr, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER=='adam' and using_model=='backbone' and not difflr:
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER=='adam' and using_model=='ebm':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER=='adam' and using_model=='ebmz' and ispretrain:
    optimizerE = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    optimizerC = torch.optim.Adam(zclaf.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER=='adam' and using_model=='ebmz' and not ispretrain:
    optimizerE = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    optimizerC = torch.optim.Adam([{'params': backbone.layer0.parameters(), 'lr':args.lr * res_lr},   # different lr
                                  {'params': backbone.layer1.parameters(), 'lr':args.lr * res_lr},
                                  {'params': backbone.layer2.parameters(), 'lr':args.lr * res_lr},
                                  {'params': backbone.layer3.parameters(), 'lr':args.lr * res_lr},
                                  {'params': backbone.layer4.parameters(), 'lr':args.lr * res_lr},
                                  {'params': zclaf.parameters()}], 
                                  lr=args.lr, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER=='adam' and using_model=='domainebm':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER=='adam' and using_model=='imgebm':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == 'rmsp':
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
else:
    raise NotImplementedError

# pdb.set_trace()

bases_list = [b for a, b in net.named_parameters() if a.endswith('bases')]
other_list = [b for a, b in net.named_parameters() if 'coef' not in a]

coef_list = [b for a, b in net.named_parameters() if 'coef' in a]
print([a for a, b in net.named_parameters() if 'coef' in a])
print([b.shape for a, b in net.named_parameters() if 'coef' in a])
log_string('Totally %d coefs.' %(len(coef_list)))

# global converge_count 
converge_count = 0

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def langevin_dynamics(domain_id, model, neg_z, num_steps, sample=False):
    z_noise = torch.randn_like(neg_z).detach()

    im_negs_samples = []
    # pdb.set_trace()

    for i in range(num_steps):
        z_noise.normal_()

        # if FLAGS.anneal:
        #     im_neg = im_neg + 0.001 * (num_steps - i - 1) / num_steps * z_noise
        # else:
        neg_z = neg_z + 0.001 * z_noise

        neg_z.requires_grad_(requires_grad=True)
        energy = model.forward(neg_z, domain_id)

        # if FLAGS.all_step:
        #     im_grad = torch.autograd.grad([energy.sum()], [im_neg], create_graph=True)[0]
        # else:
        # pdb.set_trace()
        z_grad = torch.autograd.grad([energy.sum()], [neg_z])[0]
        # pdb.set_trace()

        if i == num_steps - 1:
            # pdb.set_trace()
            neg_z_orig = neg_z
            neg_z = neg_z - step_lr * z_grad
            n = 128

            neg_z_with_grad = neg_z_orig[:n]

            if sample:
                pass
            else:
                energy = model.forward(neg_z_with_grad, domain_id)
                z_grad = torch.autograd.grad([energy.sum()], [neg_z_with_grad], create_graph=True)[0]

            neg_z_with_grad = neg_z_with_grad - step_lr * z_grad[:n]
            # pdb.set_trace()
            # neg_z_with_grad = torch.clamp(neg_z_with_grad, 0, 1)  # maybe need
        else:
            neg_z = neg_z - step_lr * z_grad

        neg_z = neg_z.detach()

        if sample:
            im_negs_samples.append(neg_z)

        # neg_z = torch.clamp(neg_z, 0, 1) # maybe need

    if sample:
        return neg_z, neg_z_with_grad, im_negs_samples, z_grad
    else:
        return neg_z, neg_z_with_grad, z_grad

def langevin_dynamics_a(domain_id, model, neg_z, qz, num_steps, clip, earlystop=False, sample=False):
    z_noise = torch.randn_like(neg_z).detach()

    im_negs_samples = []
    gradz_samples = []
    # pdb.set_trace()
    energy_m = 1
    step_lr = args.step_lr

    for i in range(num_steps):
        z_noise.normal_()

        # if FLAGS.anneal:
        #     im_neg = im_neg + 0.001 * (num_steps - i - 1) / num_steps * z_noise
        # else:
        neg_z = neg_z + 0.001 * z_noise

        neg_z.requires_grad_(requires_grad=True)
        energy = model.forward(neg_z, qz, domain_id)


        # if FLAGS.all_step:
        #     im_grad = torch.autograd.grad([energy.sum()], [im_neg], create_graph=True)[0]
        # else:
        # pdb.set_trace()
        z_grad = torch.autograd.grad([energy.sum()], [neg_z])[0]
        if clip:
            # if i==0:
                # print(z_grad.max(), z_grad.min())
            z_grad = torch.clamp(z_grad, -0.01, 0.01)
            # if i==0:
                # print(z_grad.max(), z_grad.min())

        ##### extra loss last iteration or all iterations  ########
        # if i == num_steps - 1:
        neg_z_orig = neg_z
        neg_z = neg_z - step_lr * z_grad
        n = 128

        neg_z_with_grad = neg_z_orig[:n]

        if sample:
            pass
        else:
            energy = model.forward(neg_z_with_grad, qz, domain_id)
            z_grad = torch.autograd.grad([energy.sum()], [neg_z_with_grad], create_graph=True)[0]

        neg_z_with_grad = neg_z_with_grad - step_lr * z_grad[:n]

        # neg_z_with_grad = torch.clamp(neg_z_with_grad, 0, 1)  # maybe need
        # else:
        #     neg_z = neg_z - step_lr * z_grad
        ############################################################

        # neg_z = neg_z.detach()

        # if sample:
        #     im_negs_samples.append(neg_z)

        # new_energy = model.forward(neg_z, qz, domain_id)

        if earlystop:
            # print('early stop')
            new_energy = model.forward(neg_z, qz, domain_id)

            mask = (new_energy >= energy).squeeze(-1)
            # pdb.set_trace()
            neg_z[mask] = neg_z_orig[mask]
            # new_energy = energy
            # print(i)
            # print(o_energy)

            # if mask.sum()==neg_z.size()[0]:
            #     log_string(str(i))
            #     neg_z = neg_z.detach()
            #     break

        # pdb.set_trace()
        gradz_samples.append(neg_z_with_grad)
        # neg_z_with_grad = torch.clamp(neg_z_with_grad, 0, 1)  # maybe need

        # pdb.set_trace()

        neg_z = neg_z.detach()

        if sample:
            im_negs_samples.append(neg_z)

        # neg_z = torch.clamp(neg_z, 0, 1) # maybe need

    if sample:
        return neg_z, neg_z_with_grad, im_negs_samples, z_grad
    else:
        # pdb.set_trace()
        return neg_z, gradz_samples, z_grad

def label_preseve_langevin_dynamics(domain_id, model, neg_z, qz, num_steps, clip, sample=False):
    z_noise = torch.randn_like(neg_z).detach()

    im_negs_samples = []
    gradz_samples = []
    # pdb.set_trace()
    energy_m = 1
    step_lr = args.step_lr
    print(model.models[domain_id].lr)
    if clip:
        print('clip')

    for i in range(num_steps):
        z_noise.normal_()

        # if FLAGS.anneal:
        #     im_neg = im_neg + 0.001 * (num_steps - i - 1) / num_steps * z_noise
        # else:
        neg_z = neg_z + 0.001 * z_noise

        neg_z.requires_grad_(requires_grad=True)
        energy = model.forward(neg_z, qz, domain_id)
        
        # pdb.set_trace()
        # if i==0:
        #     o_energy = torch.ones(energy.size()).cuda()

        # energy = (o_energy >= energy) * energy
        # o_energy = energy
        # print(i)
        # # print(o_energy)

        # if o_energy.sum()==0:
        #     log_string(str(i))
        #     neg_z = neg_z.detach()
        #     break

        # if i==0:
        #     energy0 = energy.mean()
        # elif energy.mean() <= 0.1*energy0:
        #     log_string(str(i))
        #     neg_z = neg_z.detach()
        #     break
        # elif i==num_steps:
            # log_string(str(i+1))

        # if i>=20 or (energy_m - energy.mean()) < 1e-4:
        #     step_lr = step_lr * 0.9

        # step_lr = max(step_lr, 10)

        # energy_m = energy.mean()
        # print(energy.mean())


        # if FLAGS.all_step:
        #     im_grad = torch.autograd.grad([energy.sum()], [im_neg], create_graph=True)[0]
        # else:
        # pdb.set_trace()
        z_grad = torch.autograd.grad([energy.sum()], [neg_z])[0]
        if clip:
            if i==0:
                print(z_grad.max(), z_grad.min())
            z_grad = torch.clamp(z_grad, -0.01, 0.01)
            if i==0:
                print(z_grad.max(), z_grad.min())

        neg_z_orig = neg_z.detach()
        neg_z = neg_z - step_lr * z_grad
        # n = 128

        # neg_z_with_grad = neg_z_orig[:n]
        new_energy = model.forward(neg_z, qz, domain_id)

        mask = (new_energy >= energy).squeeze(-1)
        # pdb.set_trace()
        neg_z[mask] = neg_z_orig[mask]
        # new_energy = energy
        # print(i)
        # print(o_energy)

        if mask.sum()==neg_z.size()[0]:
            log_string(str(i))
            neg_z = neg_z.detach()
            break

        # if sample:
        #     pass
        # else:
        #     energy = model.forward(neg_z_with_grad, qz, domain_id)
        #     z_grad = torch.autograd.grad([energy.sum()], [neg_z_with_grad], create_graph=True)[0]

        # neg_z_with_grad = neg_z_with_grad - step_lr * z_grad[:n]
        # pdb.set_trace()
        gradz_samples.append(neg_z)
        # neg_z_with_grad = torch.clamp(neg_z_with_grad, 0, 1)  # maybe need

        # pdb.set_trace()

        neg_z = neg_z.detach()

        if sample:
            im_negs_samples.append(neg_z)

        # neg_z = torch.clamp(neg_z, 0, 1) # maybe need

    if sample:
        return neg_z, neg_z_with_grad, im_negs_samples, z_grad
    else:
        # pdb.set_trace()
        return neg_z, gradz_samples, z_grad

if isreplay:
    replay_buffers = []
    for i in range(len(domains)):
        if reservoir:
            replay_buffers.append(ReservoirBuffer(buffer_size, args.num_classes))
        else:
            replay_buffers.append(ReplayBuffer(buffer_size, args.num_classes))

# global category_id
# category_id = 0

def train(epoch):
    log_string('\nEpoch: %d' % epoch)
    net.train()
    zclaf.train()
    if using_model=='ebmz' and not ispretrain:
        backbone.train()
    if args.transf:
        transformer_encoder.train()
    train_loss = 0
    extra_energy_for_ld = 0
    extra_class_for_ld = 0
    pcorrect = 0
    ptotal = 0
    ncorrect = 0
    ntotal = 0
    nncorrect = 0
    nntotal = 0

    for domain_id in range(3):
        # log_string('Domain ID:' + str(domain_id))
        # pdb.set_trace()
        if isreplay:
            replay_buffer = replay_buffers[domain_id]
        t0 = time.time()
        all_dataset.reset('train', domain_id, transform=transform_train)
        # all_dataset.reset('train', domain_id, category_id, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn)
        # pdb.set_trace()
        # print(time.time()-t0)
        for batch_idx, (inputs, targets, neg_inputs, neg_targets) in enumerate(trainloader):
            inputs, targets, neg_images, neg_labels = inputs.to(device), targets.to(device), neg_inputs.to(device), neg_targets.to(device)
            # print(time.time()-t0)
            # pdb.set_trace()

            if using_model == 'ebm':
                optimizer.zero_grad()

            elif using_model == 'ebmz':
                optimizerC.zero_grad()
                optimizerE.zero_grad()

            _, pos_z = backbone(inputs, domain_id)
            _, pos_zmu, pos_zsig = zclaf(pos_z, mctimes, domain_id)
            # _, pos_zmu, pos_zsig = zclaf(pos_z.detach(), mctimes, domain_id)
            # pdb.set_trace()
            _, neg_z = backbone(neg_images, domain_id)

            if isreplay and len(replay_buffer._storage[0]) >= BATCH_SIZE:
            # if isreplay and len(replay_buffer._storage[category_id]) >= BATCH_SIZE:
                # replay_batch, replay_labels = replay_buffer.sample(neg_z.size(0))
                replay_batch, replay_labels = replay_buffer.sample(neg_labels.cpu().numpy().tolist())
                # replay_batch = decompress_x_mod(replay_batch)
                replay_mask = (
                    np.random.uniform(
                        0,
                        1,
                        neg_z.size(0)) > 0.5) # ?? 
                # pdb.set_trace()
                neg_z[replay_mask] = torch.Tensor(replay_batch[replay_mask]).to(device)
                # neg_zmu[replay_mask] = torch.Tensor(replay_z[replay_mask]).to(device)
                # pdb.set_trace()
                neg_labels[replay_mask] = torch.LongTensor(replay_labels[replay_mask]).to(device)
            else:
                idxs = None

            _, neg_zmu, neg_zsig = zclaf(neg_z, mctimes, domain_id)

            pDz = []
            aDz = []
            nDz = []
            print(targets.unique())
            mask = torch.ones(NUM_CLASS).to(device)
            for cate in range(NUM_CLASS):
                if cate in targets.unique():
                    pDz.append(pos_z[targets==cate].mean(0,keepdim=True))
                    aDz.append(torch.cat([pos_z[targets==cate], neg_z[neg_labels==cate]], 0).mean(0,keepdim=True))
                    # Dz.append(pos_z[targets==cate].mean(0,keepdim=True))
                    # pdb.set_trace()
                else:
                    print('not class' + str(cate) + '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # pdb.set_trace()
                    aDz.append(pos_z.mean(0,keepdim=True))
                    pDz.append(pos_z.mean(0,keepdim=True))
                    mask[cate] = 0


            if args.dztype =='p':
                Dz = torch.cat(pDz, 0)
            elif args.dztype =='a':
                Dz = torch.cat(aDz, 0)

            _, pos_zD_mu, pos_zD_sig = zclaf(Dz, mctimes, domain_id)

            # KLD = (kl_divergence(pos_zmu, pos_zsig, pos_zD_mu[targets], pos_zD_sig[targets]) * mask[targets]).mean()
            # print(KLD)

            z_mu_samp = pos_zD_mu.unsqueeze(1).repeat(1, args.mctimes, 1)
            z_sigma_samp = pos_zD_sig.unsqueeze(1).repeat(1, args.mctimes, 1)
            # pdb.set_trace()
            eps_q = z_mu_samp.new(z_mu_samp.size()).normal_()
            qz = z_mu_samp + 1 * z_sigma_samp * eps_q  # 7*10*512
            zclassifier = qz.permute(2,1,0).contiguous().view(feat_dim, args.mctimes*NUM_CLASS)  #512*(10*7)
            # epos_z = pos_z.unsqueeze(1).repeat(1, args.mctimes, 1)
            # qz = torch.cat([epos_z, qz], -1)
            # qz = qz.view(epos_z.size()[0]*args.mctimes, 512*2)   
            # pdb.set_trace()
            pos_y = torch.mm(pos_z, zclassifier) # 128*512 512*10*7
            neg_y = torch.mm(neg_z, zclassifier)

            if using_model=='ebmz':
                pos_targets = targets.unsqueeze(1).repeat(1,args.mctimes)
                lossC_pos = criterion(pos_y.view(pos_y.size()[0], mctimes, NUM_CLASS).view(-1, NUM_CLASS), pos_targets.view(-1)) #criterion(ys, targets)
                KLD = (kl_divergence(pos_zmu, pos_zsig, pos_zD_mu[targets], pos_zD_sig[targets]) * mask[targets]).mean()

            # pdb.set_trace()

            # upd_neg_z, neg_z_with_grad, x_grad = langevin_dynamics(domain_id, net, neg_z, num_steps)
            neg_z = neg_z.detach()
            neg_zmu = neg_zmu.detach()
            if args.sampz:   #0322
                epos = neg_zmu.new(neg_zmu.size()).normal_()
                neg_zmu = neg_zmu + epos * neg_zsig.detach()

                epos_pz = neg_zmu.new(neg_zmu.size()).normal_()
                pebm_z = pos_zD_mu[neg_labels].detach() + epos_pz * pos_zD_sig[neg_labels].detach()
            else:
                pebm_z = pos_zD_mu[neg_labels].detach()

            if args.transf:
                neg_zmu = neg_zmu.unsqueeze(0)
                pzmu = pos_zD_mu.detach().unsqueeze(1).repeat(1, BATCH_SIZE, 1)
                neg_zmu = transformer_encoder(torch.cat([neg_zmu, pzmu], 0))
                neg_zmu = neg_zmu[0]
                # pdb.set_trace()

            if args.dataset!='PACSp':    #### 0504
                upd_neg_z, neg_z_with_grad, _ = langevin_dynamics_a(domain_id, net, neg_z, neg_zmu, num_steps, clipgrad, args.earlystop)
                if pebm:
                    pupd_neg_z, pneg_z_with_grad, _ = langevin_dynamics_a(domain_id, net, neg_z, pebm_z, num_steps, clipgrad, args.earlystop)
            else:
                upd_neg_z, neg_z_with_grad, _ = langevin_dynamics_a(domain_id, net, neg_z, neg_zmu, 1, clipgrad, args.earlystop)
                if pebm:
                    pupd_neg_z, pneg_z_with_grad, _ = langevin_dynamics_a(domain_id, net, neg_z, pebm_z, 1, clipgrad, args.earlystop)

            # pdb.set_trace()

            energy_pos = net.forward(pos_z.detach(), pos_zmu.detach(), domain_id)
            energy_neg = net.forward(upd_neg_z.clone(), neg_zmu.detach(), domain_id)
            raw_energy_neg = net.forward(neg_z.clone(), neg_zmu.detach(), domain_id)

            if isreplay and (upd_neg_z is not None):
                replay_buffer.add(upd_neg_z, neg_labels)

            loss = energy_pos.mean() - energy_neg.mean() + 1 #+ 1 - raw_energy_neg.mean()#
            if pebm:
                loss += energy_pos.mean() - net.forward(pupd_neg_z.clone(), pebm_z, domain_id).mean() + 1
            loss = loss + (torch.pow(energy_pos, 2).mean() + torch.pow(energy_neg, 2).mean()) * l2_coeff    
            loss = loss * loss_coeff
            # loss = torch.zeros(1).to(device)
            # print(time.time()-t0) 
            if extra_sup:
                extra_class = 0
                extra_energy = 0 #torch.zeros(1).to(device)
                extrakl = 0
                net.requires_grad_(False)
                zclaf.requires_grad_(False)
                backbone.requires_grad_(False)  ### 0328
                for grad_negz in neg_z_with_grad:    #### label preserving LD
                # extra_energy = net.forward(neg_z_with_grad, domain_id)
                    # if using_model=='ebm' or using_model == 'domainebm':
                    extra_neg_y = torch.mm(grad_negz, zclassifier)
                    # extra_neg_y = torch.mm(grad_negz, zclassifier.detach()) ## 0322
                    # pdb.set_trace()
                    extra_class += criterion(extra_neg_y.view(extra_neg_y.size()[0], mctimes, NUM_CLASS).view(-1, NUM_CLASS), neg_labels.unsqueeze(1).repeat(1,mctimes).view(-1))
                    extra_energy += net.forward(grad_negz, neg_zmu, domain_id).mean()

                    # _, extra_mu, extra_sig = zclaf(grad_negz, mctimes, domain_id)
                    # extrakl += (kl_divergence(extra_mu, extra_sig, neg_zmu, neg_zsig)).mean()
                    extrakl += torch.zeros(1).to(device)

                if pebm:
                    for pgrad_negz in pneg_z_with_grad:    #### label preserving LD
                        extra_neg_y = torch.mm(pgrad_negz, qz.detach().permute(2,1,0).contiguous().view(feat_dim, args.mctimes*NUM_CLASS))
                        # pdb.set_trace()
                        extra_class += criterion(extra_neg_y.view(extra_neg_y.size()[0], mctimes, NUM_CLASS).view(-1, NUM_CLASS), neg_labels.unsqueeze(1).repeat(1,mctimes).view(-1))
                        extra_energy += net.forward(pgrad_negz, pebm_z, domain_id).mean()   ## 0322

                        extrakl += torch.zeros(1).to(device)
                
                    # extra_class = criterion(backbone.forward(neg_images, domain_id, grad_negz)[0], neg_labels)
                        
                net.requires_grad_(True)
                zclaf.requires_grad_(True)
                backbone.requires_grad_(True) ### 0328
                extra_class = extra_class / len(neg_z_with_grad)
                extra_energy = extra_energy / len(neg_z_with_grad)
                extrakl = extrakl / len(neg_z_with_grad)
            else:
                extra_energy = torch.zeros(1).to(device)
                extra_class = torch.zeros(1).to(device)
                extrakl = torch.zeros(1).to(device)

            # pdb.set_trace()
            if using_model =='ebmz':
                loss = loss + en_coeff * extra_energy.mean() + cla_coeff * extra_class + kl_coeff * KLD + cla_coeff * lossC_pos + kl_coeff * extrakl

                train_loss = loss.item()
                extra_energy_for_ld = en_coeff * extra_energy.mean().item()
                extra_class_for_ld = cla_coeff * extra_class.item()
                posceloss = cla_coeff * lossC_pos.item()
                klloss = kl_coeff * KLD.item()
                # closs += ifcommon * common_loss.item()

                loss.backward()
                optimizerC.step()
                optimizerE.step()
                if args.transf:
                    optimizerT.step()
            elif using_model == 'ebm':
                loss = loss + en_coeff * extra_energy.mean() + cla_coeff * extra_class + kl_coeff * extrakl

                train_loss = loss.item()
                extra_energy_for_ld = en_coeff * extra_energy.mean().item()
                extra_class_for_ld = cla_coeff * extra_class.item()
                posceloss = 0
                klloss = kl_coeff * extrakl.item()
                # closs += ifcommon * common_loss.item()

                loss.backward()
                optimizer.step()
                if args.transf:
                    optimizerT.step()
            # pdb.set_trace()
            # print(time.time()-t0)

            # predicted = []
            if using_model == 'ebm' or using_model == 'domainebm' or using_model == 'ebmz':
                new_neg_y = torch.mm(upd_neg_z, qz.permute(2,1,0).contiguous().view(feat_dim, mctimes*NUM_CLASS))
                new_neg_y = new_neg_y.view(new_neg_y.size()[0], mctimes, NUM_CLASS).mean(1)

            # pdb.set_trace()
            _, pos_pred = pos_y.view(pos_y.size()[0], mctimes, NUM_CLASS).mean(1).max(1)
            _, neg_pred = neg_y.view(pos_y.size()[0], mctimes, NUM_CLASS).mean(1).max(1)
            _, new_neg_pred = new_neg_y.max(1)

            # pdb.set_trace()
            # print(time.time()-t0)
            pcorrect += pos_pred.eq(targets).sum().item()
            ptotal += targets.size(0)
            ncorrect += neg_pred.eq(neg_labels).sum().item()
            ntotal += neg_labels.size(0)
            nncorrect += new_neg_pred.eq(neg_labels).sum().item()
            # nntotal += targets.size(0)
            # print(time.time()-t0)


            # category_id += 1
            # if category_id == 7:
            #     category_id = 0
            if iteration_training and batch_idx>=batchs_per_epoch:
                break

        log_string('Domain: %d | Pos_energy: %.3f | Neg_energy: %.3f | Raw_neg_energy: %.3f | ExNeg_energy: %.3f | PosCEloss: %.3f | ExCEloss: %.3f | KLD:  %.3f | Loss: %.3f | Pos Acc: %.3f%% (%d/%d) | Neg Acc: %.3f%% (%d/%d) | New Neg Acc: %.3f%% (%d/%d)'  # print energy?
                % (domain_id, energy_pos.mean(), energy_neg.mean(), raw_energy_neg.mean(), extra_energy_for_ld, posceloss, extra_class_for_ld, klloss, train_loss/(batch_idx+1), 100.*pcorrect/ptotal, pcorrect, ptotal, 100.*ncorrect/ntotal, ncorrect, ntotal, 100.*nncorrect/ntotal, nncorrect, ntotal))

        writer.add_scalar('loss', train_loss/(batch_idx+1), epoch)
        writer.add_scalar('new_neg_acc', 100.*nncorrect/ntotal, epoch)
        log_string(str(time.time()-t0))

def validation(epoch):
    global best_valid_acc
    net.eval()
    zclaf.eval()
    if not ispretrain:
        backbone.eval()
    if args.transf:
        transformer_encoder.eval()
    test_loss = 0
    correct = 0
    raw_correct = 0
    new_correct = 0
    total = 0
    ac_correct = [0, 0, 0]
    # pdb.set_trace()
    # print(backbone.layer0[0].weight[0,0])
    
    # with torch.no_grad():
    if args.dataset=='office':
        test_cont_data.reset('val', 0, transform=transform_test)
    else:
        test_cont_data.reset('val', transform=transform_test)
    context_loader = torch.utils.data.DataLoader(test_cont_data, batch_size=(num_domain-1)*NUM_CLASS*ctx_test, shuffle=False, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn)
    for batch_idx, (inputs, targets) in enumerate(context_loader):
        context_img, context_label = inputs.to(device), targets.to(device)
        if using_model=='ebmz':
            optimizerC.zero_grad()
            optimizerE.zero_grad()
        elif using_model=='ebm':
            optimizer.zero_grad()
        if args.transf:
            optimizerT.zero_grad()

        _, neg_z = backbone(context_img, 0)
        neg_z = neg_z.view((num_domain-1),NUM_CLASS,ctx_test,neg_z.size()[-1])
        if args.dztype=='p':
            Dz = neg_z.mean(2).view((num_domain-1) * NUM_CLASS, neg_z.size()[-1])
            _, neg_zD_mu, _ = zclaf(Dz, 1, 0)  # 7*512
            neg_zD_mu = neg_zD_mu.view((num_domain-1), NUM_CLASS, neg_z.size()[-1])
        elif args.dztype=='a':
            Dz = neg_z.mean(2).mean(0).view(NUM_CLASS, neg_z.size()[-1])
            _, neg_zD_mu, _ = zclaf(Dz, 1, 0)  # 7*512
            neg_zD_mu = neg_zD_mu.view(NUM_CLASS, neg_z.size()[-1])

    for i in range(4):
        all_dataset.reset('val', i, transform=transform_test)
        # all_dataset.reset('val', i, 0, transform=transform_test)
        valloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=4)

        if using_model=='ebmz':
            optimizerC.zero_grad()
            optimizerE.zero_grad()
        elif using_model=='ebm':
            optimizer.zero_grad()

        if args.transf:
            optimizerT.zero_grad()

        for batch_idx, (inputs, targets, neg_images, neg_labels) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            y_spec = []
            y_news = []
            y_raws = []
            
            for domain_id in range(len(domains)):
                if i == domain_id:
                    continue
                else:
                    if using_model=='ebmz':
                        optimizerC.zero_grad()
                        optimizerE.zero_grad()
                    elif using_model=='ebm':
                        optimizer.zero_grad()

                    y_raw, z = backbone(inputs, domain_id)

                    _, pzmu, _ = zclaf(z, 1, domain_id)

                    if args.transf:
                        # pdb.set_trace()
                        pzmu = pzmu.unsqueeze(0)
                        nzmu = neg_zD_mu[domain_id].detach().unsqueeze(1).repeat(1, pzmu.size(1), 1)
                        pzmu = transformer_encoder(torch.cat([pzmu, nzmu], 0))
                        pzmu = pzmu[0]

                    if args.dztype=='p':
                        new_y = torch.mm(z, neg_zD_mu[domain_id].permute(1,0).contiguous().view(feat_dim, NUM_CLASS))
                        upd_neg_z, _, _ = langevin_dynamics_a(domain_id, net, z, pzmu, num_steps, clipgrad, args.earlystop)   ### 0328
                        y = torch.mm(upd_neg_z, neg_zD_mu[domain_id].permute(1,0).contiguous().view(feat_dim, NUM_CLASS))
                    elif args.dztype=='a':
                        new_y = torch.mm(z, neg_zD_mu.permute(1,0).contiguous().view(feat_dim, NUM_CLASS))
                        upd_neg_z, _, _ = langevin_dynamics_a(domain_id, net, z, pzmu, num_steps, clipgrad, args.earlystop)   ### 0328
                        y = torch.mm(upd_neg_z, neg_zD_mu.permute(1,0).contiguous().view(feat_dim, NUM_CLASS))

                    y_spec.append(torch.softmax(y, -1).view(y.size()[0], 1, NUM_CLASS))
                    y_news.append(torch.softmax(new_y, -1).view(y.size()[0], 1, NUM_CLASS))
                    y_raws.append(torch.softmax(y_raw, -1).view(y_raw.size()[0], 1, NUM_CLASS))
                    
            # pdb.set_trace()
            y_mean = torch.cat(y_spec, 1).mean(1)
            y_new_mean = torch.cat(y_news, 1).mean(1)
            y_raw_mean = torch.cat(y_raws, 1).mean(1)
            cls_loss = criterion(y_mean, targets)
            loss = cls_loss

            test_loss += loss.item()
            _, predicted = y_mean.max(1)
            _, raw_pred = y_raw_mean.max(1)
            _, new_pred = y_new_mean.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            raw_correct += raw_pred.eq(targets).sum().item()
            new_correct += new_pred.eq(targets).sum().item()

    log_string('VAL Loss: %.3f | Final Acc: %.3f%% (%d/%d) | New Acc: %.3f%% (%d/%d)'
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*new_correct/total, new_correct, total))

    writer.add_scalar('val_loss', test_loss/(batch_idx+1), epoch)
    writer.add_scalar('val_acc', 100.*correct/total, epoch)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_valid_acc:
        print('Saving..')
        log_string('The best validation Acc')
        if args.transf:
            state = {
                'net': net.state_dict(),
                'resnet': backbone.state_dict(),
                'znet': zclaf.state_dict(),
                'tran': transformer_encoder.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
        else:
            state = {
                'net': net.state_dict(),
                'resnet': backbone.state_dict(),
                'znet': zclaf.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(LOG_DIR, 'ckpt.t7'))
        best_valid_acc = acc
        return 0
    else:
        return 1

def test(epoch):
    global best_acc
    net.eval()
    zclaf.eval()
    if not ispretrain:
        backbone.eval()
    if args.transf:
        transformer_encoder.eval()
    test_loss = 0
    correct = 0
    total = 0
    new_correct = 0
    sele_correct = 0
    ac_correct = [0, 0, 0]
    raw_correct = [0, 0, 0]
    select_correct = [0, 0, 0]
    raw_energy = [0, 0, 0]
    new_energy = [0, 0, 0]
    select_energy = [0, 0, 0]
    num_preds = 1
    all_dataset.reset('test', 0, transform=transform_test)
    # all_dataset.reset('test', 0, 0, transform=transform_test)
    testloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=4)

    with torch.no_grad():
        # test_cont_data.reset('test', 0, transform=transform_test)
        if args.dataset=='office':
            test_cont_data.reset('val', 0, transform=transform_test)
        else:
            test_cont_data.reset('val', transform=transform_test)
        context_loader = torch.utils.data.DataLoader(test_cont_data, batch_size=(num_domain-1)*NUM_CLASS*ctx_test, shuffle=False, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn)
        for batch_idx, (inputs, targets) in enumerate(context_loader):
            context_img, context_label = inputs.to(device), targets.to(device)
            if using_model=='ebmz':
                optimizerC.zero_grad()
                optimizerE.zero_grad()
            elif using_model=='ebm':
                optimizer.zero_grad()

            if args.transf:
                optimizerT.zero_grad()

            _, neg_z = backbone(context_img, 0)
            # Dz = []
            neg_z = neg_z.view((num_domain-1),NUM_CLASS,ctx_test,neg_z.size()[-1])

            if args.dztype=='p':
                Dz = neg_z.mean(2).view((num_domain-1) * NUM_CLASS, neg_z.size()[-1])
                _, neg_zD_mu, _ = zclaf(Dz, 1, 0)  # 7*512
                neg_zD_mu = neg_zD_mu.view((num_domain-1), NUM_CLASS, neg_z.size()[-1])
            elif args.dztype=='a':
                Dz = neg_z.mean(2).mean(0).view(NUM_CLASS, neg_z.size()[-1])
                _, neg_zD_mu, _ = zclaf(Dz, 1, 0)  # 7*512
                neg_zD_mu = neg_zD_mu.view(NUM_CLASS, neg_z.size()[-1])

    for batch_idx, (inputs, targets, neg_images, neg_labels) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        y_spec = []
        y_news = []
        y_sele = []
        
        for domain_id in range(len(domains)):

            if using_model=='ebmz':
                optimizerC.zero_grad()
                optimizerE.zero_grad()
            elif using_model=='ebm':
                optimizer.zero_grad()

            if args.transf:
                optimizerT.zero_grad()

            _, z = backbone(inputs, domain_id)

            _, pzmu, _ = zclaf(z, 1, domain_id)

            if args.transf:
                # pdb.set_trace()
                pzmu = pzmu.unsqueeze(0)
                nzmu = neg_zD_mu[domain_id].detach().unsqueeze(1).repeat(1, pzmu.size(1), 1)
                pzmu = transformer_encoder(torch.cat([pzmu, nzmu], 0))
                pzmu = pzmu[0]

            if args.dztype=='p':
                new_y = torch.mm(z, neg_zD_mu[domain_id].permute(1,0).contiguous().view(feat_dim, NUM_CLASS))
                mask = ((- torch.softmax(new_y, -1) * torch.log(torch.softmax(new_y, -1))).sum(-1) < 0.4)
                # pdb.set_trace()
                upd_neg_z, _, _ = langevin_dynamics_a(domain_id, net, z, pzmu, num_steps, clipgrad, args.earlystop)   ### 0328
                y = torch.mm(upd_neg_z, neg_zD_mu[domain_id].permute(1,0).contiguous().view(feat_dim, NUM_CLASS))
                select_y = y + 0.1 - 0.1
                select_y[mask] = new_y[mask]
            elif args.dztype=='a':
                new_y = torch.mm(z, neg_zD_mu.permute(1,0).contiguous().view(feat_dim, NUM_CLASS))
                upd_neg_z, _, _ = langevin_dynamics_a(domain_id, net, z, pzmu, num_steps, clipgrad, args.earlystop)   ### 0328
                y = torch.mm(upd_neg_z, neg_zD_mu.permute(1,0).contiguous().view(feat_dim, NUM_CLASS))

            # pdb.set_trace()
            new_energy[domain_id] += (net.forward(upd_neg_z, pzmu, domain_id)).mean().item()
            raw_energy[domain_id] += (net.forward(z, pzmu, domain_id)).mean().item()
            # select_energy[domain_id] += 
            

            y_spec.append(torch.softmax(y, -1).view(y.size()[0], 1, NUM_CLASS))
            y_news.append(torch.softmax(new_y, -1).view(y.size()[0], 1, NUM_CLASS))
            y_sele.append(torch.softmax(select_y, -1).view(y.size()[0], 1, NUM_CLASS))
                    
        # pdb.set_trace()
        y_mean = torch.cat(y_spec, 1).mean(1)
        y_new_mean = torch.cat(y_news, 1).mean(1)
        y_sele_mean = torch.cat(y_sele, 1).mean(1)
        cls_loss = criterion(y_mean, targets)
        # pdb.set_trace()
        loss = cls_loss

        test_loss += loss.item()
        _, predicted = y_mean.max(1)
        _, new_pred = y_new_mean.max(1)
        _, sele_pred = y_sele_mean.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        new_correct += new_pred.eq(targets).sum().item()
        sele_correct += sele_pred.eq(targets).sum().item()
        for i in range(len(y_spec)):
            _, pred_sam = y_spec[i].squeeze().max(1)
            _, raw_pred_sam = y_news[i].squeeze().max(1)
            _, sele_pred_sam = y_sele[i].squeeze().max(1)
            ac_correct[i] += pred_sam.eq(targets).sum().item()
            raw_correct[i] += raw_pred_sam.eq(targets).sum().item()
            select_correct[i] += sele_pred_sam.eq(targets).sum().item()

        # for i in range(num_preds):
        #     _, ac_predicted = ys[:, i].max(1)
        #     ac_correct[i] += ac_predicted.eq(targets).sum().item()

    log_string('TEST Loss: %.3f | Acc: %.3f%% (%d/%d)  | Raw Acc: %.3f%% (%d/%d) | Select Acc: %.3f%% (%d/%d) | Acc: %.3f%%, %.3f%%, %.3f%%  | Raw Acc: %.3f%%, %.3f%%, %.3f%% | Select Acc: %.3f%%, %.3f%%, %.3f%% | Energy: %.3f, %.3f, %.3f | Raw energy: %.3f, %.3f, %.3f'
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100.*new_correct/total, new_correct, total, 100.*sele_correct/total, sele_correct, total, 100.*ac_correct[0]/total, 100.*ac_correct[1]/total, 100.*ac_correct[2]/total, 100.*raw_correct[0]/total, 100.*raw_correct[1]/total, 100.*raw_correct[2]/total, 100.*select_correct[0]/total, 100.*select_correct[1]/total, 100.*select_correct[2]/total, new_energy[0]/(batch_idx+1), new_energy[1]/(batch_idx+1), new_energy[2]/(batch_idx+1), raw_energy[0]/(batch_idx+1), raw_energy[1]/(batch_idx+1), raw_energy[2]/(batch_idx+1)))

    writer.add_scalar('test_loss', test_loss/(batch_idx+1), epoch)
    writer.add_scalar('test_acc', 100.*correct/total, epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        log_string('The best test Acc')
        if args.transf:
            state = {
                'net': net.state_dict(),
                'resnet': backbone.state_dict(),
                'znet': zclaf.state_dict(),
                'tran': transformer_encoder.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
        else:
            state = {
                'net': net.state_dict(),
                'resnet': backbone.state_dict(),
                'znet': zclaf.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
        # if not os.path.isdir('checkpoint'):
            # os.mkdir('checkpoint')
        torch.save(state, os.path.join(LOG_DIR, 'tckpt.t7'))
        best_acc = acc
        return 0
    else:
        return 1


decay_ite = [0.6*max_ite]


if not iteration_training:
    for epoch in range(start_epoch, start_epoch+decay_inter[-1]+50):
        if epoch in decay_inter:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
            log_string('In epoch %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
        train(epoch)
        if epoch % 5 == 0:
            _ = validation(epoch)
            _ = test(epoch)
else:
    for epoch in range(max_ite):   
        if epoch in decay_ite:
            for i in range(len(optimizerC.param_groups)):
                optimizerC.param_groups[i]['lr'] = optimizerC.param_groups[i]['lr']*0.1
            log_string('In iteration %d the LR is decay to %f' %(epoch, optimizerC.param_groups[0]['lr']))
        train(epoch)
        # pdb.set_trace()
        if epoch % test_ite == 0:
            if args.dataset!='office':
                _ = validation(epoch)
            _ = test(epoch)
