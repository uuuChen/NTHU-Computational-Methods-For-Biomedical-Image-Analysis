import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score


def topic_log(topic_str, length=40):
    num_of_dash = length - (len(topic_str) + 2)
    left_num_of_dash = num_of_dash // 2
    right_num_of_dash = num_of_dash - left_num_of_dash
    print('\n\n' + '-' * left_num_of_dash + " " + topic_str + " " + '-' * right_num_of_dash)


def evaluate_log(predicts, true_labels):
    accuracy = accuracy_score(true_labels, predicts)
    f1_score_ = f1_score(true_labels, predicts, 'macro')
    recall_score_ = recall_score(true_labels, predicts, 'macro')
    print(f'accuracy:     {accuracy * 100.: .2f} %')
    print(f'f1-score:     {f1_score_}')
    print(f'recall-score: {recall_score_}')


def model_fit(model, optimizer, args, train_loader, test_loader, model_name):
    best_top1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        top1 = train_epoch(model, optimizer, args, train_loader, epoch)
        if top1 > best_top1:
            save_model(model, optimizer, epoch, args.save_model_path + f'{model_name}_checkpoint.tar')
        test_epoch(model, args, test_loader)


def train_epoch(model, optimizer, args, train_loader, epoch):
    losses = AvgMeter()
    accuracies = AvgMeter()
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        top1_acc = accuracy(output, target, topk=(1,))[0]
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        accuracies.update(top1_acc, data.shape[0])
        losses.update(loss.item(), data.shape[0])
        if batch_idx % args.log_interval == 0:
            done = batch_idx * len(data)
            percentage = 100. * batch_idx / len(train_loader)
            print(f'Train Epoch: {epoch}\t[{done:5}/{len(train_loader.dataset)}({percentage:3.0f}%)]\t'
                  f'Loss: {losses.mean:.6f}\tAccuracy: {accuracies.mean:.3f}')
    return accuracies.mean


def test_epoch(model, args, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = list()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_model(model, optimizer, path, args):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    args.start_epoch = checkpoint['epoch']


class AvgMeter:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.mean = 0

    def update(self, value, count=1):
        self.sum += (value * count)
        self.count += count
        self.mean = self.sum / self.count


