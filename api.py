<<<<<<< HEAD
import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
from torch.utils.data import DataLoader
from model import SCVfliter
from data_process import myDataset
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--nclass', default=3, type=int)
parser.add_argument('--split_rate', default=0.7, type=float)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--weight_decay', default=1e-6, type=float)
parser.add_argument('--epoches', default=15, type=int)
parser.add_argument('--dict_len', default=6, type=int)
parser.add_argument('--embed_size', default=20, type=int)
parser.add_argument('--num_hiddens', default=100, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--device', default='cuda', type=str)
args = parser.parse_args()

print('Load data...')
train_data = myDataset('train', args.nclass, args.split_rate)
test_data = myDataset('test', args.nclass, args.split_rate)
trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

net = SCVfliter(args.dict_len, args.embed_size, args.num_hiddens, args.num_layers, args.nclass).to(args.device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()

print('Start train...')
=======
import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
from torch.utils.data import DataLoader
from model import SCVfliter
from data_process import myDataset
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--nclass', default=3, type=int)
parser.add_argument('--split_rate', default=0.7, type=float)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--weight_decay', default=1e-6, type=float)
parser.add_argument('--epoches', default=15, type=int)
parser.add_argument('--dict_len', default=6, type=int)
parser.add_argument('--embed_size', default=20, type=int)
parser.add_argument('--num_hiddens', default=100, type=int)
parser.add_argument('--num_layers', default=2, type=int)
parser.add_argument('--device', default='cuda', type=str)
args = parser.parse_args()

print('Load data...')
train_data = myDataset('train', args.nclass, args.split_rate)
test_data = myDataset('test', args.nclass, args.split_rate)
trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

net = SCVfliter(args.dict_len, args.embed_size, args.num_hiddens, args.num_layers, args.nclass).to(args.device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()

print('Start train...')
>>>>>>> 5f5caeb (first commit)
train(net, trainloader, testloader, optimizer, criterion, args)