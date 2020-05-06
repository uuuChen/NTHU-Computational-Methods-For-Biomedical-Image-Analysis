# System
import argparse
import torch
import os
import torchvision.transforms as transforms
import torchvision.models as torch_models
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Custom
import util
from data_loader import CBIS_DDSM_Dataset, get_numpy_data2d_with_labels
import models

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root-dir", type=str, default='BIA_Assignment_2_Dataset-All/BIA_Assignment_2_Dataset/',
                    help="root path of data set")
parser.add_argument("--use-model", "-um", type=str, default='SVM',
                    help="use which model to train")
parser.add_argument("--load-model", "-lm", type=str, default='',
                    help="load model")
parser.add_argument("--save-model-path", "-smp", type=str, default='saves/',
                    help="save model directory path")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--resize-size', '-rs', type=int, default=256, metavar='S',
                    help='resize size (default: (256, 256))')
parser.add_argument('--epochs', '-ep', default=200, type=int, metavar='N',
                    help='number of total initial epochs to run (default: 200)')
parser.add_argument('--start-epoch', '-sep', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', '-bs', default=128, type=int,metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '-lr', default=0.1, type=float,
                    metavar='TLR', help='train learning rate')
parser.add_argument('--lr-drop-interval', '-lr-drop', default=50, type=int,
                    metavar='LRD', help='learning rate drop interval (default: 50)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
args = parser.parse_args()

# get data_loader
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.resize_size, args.resize_size)),
    # transforms.RandomCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: np.repeat(x, 3, axis=0)),  # copy 1 channel to generate 3 channels
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.resize_size, args.resize_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: np.repeat(x, 3, axis=0)),  # copy 1 channel to generate 3 channels
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_set = CBIS_DDSM_Dataset(args.root_dir, "train", method='crop', transform=train_transform)
test_set = CBIS_DDSM_Dataset(args.root_dir, "test", method='crop', transform=test_transform)
train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
train_data2d, train_labels = get_numpy_data2d_with_labels(train_loader)
test_data2d, test_labels = get_numpy_data2d_with_labels(test_loader)

# select device
use_cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')


def svm_process():
    util.topic_log('SVM')
    model = svm.SVC()
    model.fit(train_data2d, train_labels)
    predicts = model.predict(test_data2d)
    util.evaluate_log(predicts, test_labels)


def kmeans_process():
    util.topic_log('Kmeans')
    model = KMeans(n_clusters=2, random_state=0)
    model.fit(train_data2d)
    predicts = model.predict(test_data2d)
    util.evaluate_log(predicts, test_labels)


def random_foreset_process():
    util.topic_log('Random Forest')
    model = RandomForestClassifier()
    model.fit(train_data2d, train_labels)
    predicts = model.predict(test_data2d)
    util.evaluate_log(predicts, test_labels)


def alexnet_process():
    util.topic_log('Alexnet')
    pretrain_model = torch_models.alexnet(pretrained=True)
    model = models.FineTuneModel(pretrain_model, 'alexnet').to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    if args.load_model:
        util.load_model(model, optimizer, args, args.load_model)
    util.model_fit(model, optimizer, args, train_loader, test_loader, 'alexnet')


def resnet_process():
    util.topic_log('Resnet')
    pretrain_model = torch_models.resnet152(pretrained=True)
    model = models.FineTuneModel(pretrain_model, 'resnet').to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    if args.load_model:
        util.load_model(model, optimizer, args, args.load_model)
    util.model_fit(model, optimizer, args, train_loader, test_loader, 'resnet152')


def check_dir_exist():
    os.makedirs(f'{args.save_model_path}', exist_ok=True)


def main():
    check_dir_exist()
    # svm_process()
    # kmeans_process()
    # random_foreset_process()
    # alexnet_process()
    resnet_process()


if __name__ == '__main__':
    main()
