'''
This script trains tumor/normal
'''
import torch
import torchvision
import PIL
import json
import time
import random
torch.set_printoptions(profile='full')
hyper_parameters = open('dataset/hyper_parameters_json', 'r')
hyper_parameters = json.load(hyper_parameters)
MAKE_NEW_LOG = 1
FILE_NAME = __file__+'_'+hyper_parameters['task']


def Write_log(_string):
    global MAKE_NEW_LOG
    if MAKE_NEW_LOG:
        MAKE_NEW_LOG = 0
        logFile = open('logs/' + FILE_NAME + '.log', 'w')
        logFile.writelines('Mission: ' + FILE_NAME + '\n' +
                           time.asctime(time.localtime())+' Starts!\n')
        logFile.close()
    logFile = open('logs/' + FILE_NAME + '.log', 'a')
    logFile.writelines(time.asctime(time.localtime()) + ' ' + _string + '\n')
    logFile.close()


file0 = hyper_parameters['file0']
file1 = hyper_parameters['file1']
for i in range(len(file0)):
    file0[i] = 'meta/' + file0[i]
for i in range(len(file1)):
    file1[i] = 'meta/' + file1[i]
if hyper_parameters["initialized"] == 0:
    Write_log(
        'Warning: hyper-parameters not initialized')
else:
    Write_log(
        json.dumps(hyper_parameters))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(hyper_parameters['GPU'])
if torch.cuda.is_available():
    Write_log(
        'using GPU '+str(torch.cuda.current_device()))
else:
    Write_log(
        '  '+'warning: using CPU')
data0 = []
data1 = []
for i in file0:  # mss=1 msi=0
    f = open(i, 'r')
    for line in f.readlines():
        data0.append(line[:-1])
    f.close()
for i in file1:
    f = open(i, 'r')
    for line in f.readlines():
        data1.append(line[:-1])
    f.close()
random.shuffle(data0)
random.shuffle(data1)
trainSet = []
validationSet = []
testSet = []
for i in range(len(data1)):  # mss=1 msi=0
    if 0 <= i % 200 < 170:
        trainSet.append((data1[i], 1))
    if 170 <= i % 200 < 195:
        validationSet.append((data1[i], 1))
    if 195 <= i % 200 < 200:
        testSet.append((data1[i], 1))
for i in range(len(data0)):  # mss=1 msi=0
    if 0 <= i % 200 < 170:
        trainSet.append((data0[i], 0))
    if 170 <= i % 200 < 195:
        validationSet.append((data0[i], 0))
    if 195 <= i % 200 < 200:
        testSet.append((data0[i], 0))
random.shuffle(trainSet)
random.shuffle(validationSet)
random.shuffle(testSet)
Write_log('train: {}, validation: {}, test: {}'.format(
    len(trainSet), len(validationSet), len(testSet)))


class NewDataset(torch.utils.data.Dataset):
    def __init__(self, data_set, transforms):
        self.data_set = data_set
        self.transforms = transforms

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        if index >= len(self.data_set):
            log.writelines(time.asctime(time.localtime()) +
                           '  '+"Error: __getitem__ out of range" + '\n')
        else:
            image_file, label = self.data_set[index]
            label_tensor = torch.Tensor([1])
            label_tensor[0] = label
            image = PIL.Image.open(image_file)
            image_tensor = self.transforms(image)
            image_tensor = image_tensor.type(torch.FloatTensor)
            return image_tensor, label_tensor


criterion = torch.nn.CrossEntropyLoss()
model = torchvision.models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)
net = model.to(device)
o = 0
for c in net.children():
    if o < 6:
        for p in c.parameters():
            p.requires_grad = False
    o += 1
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                             lr=hyper_parameters["LR"], weight_decay=hyper_parameters["L2"])
vali_acc = []


def abort():
    res = 0
    if len(vali_acc) > hyper_parameters["wait"]:
        res = 1
        for i in range(hyper_parameters['wait']):
            if vali_acc[-1] > vali_acc[-2 - i]:
                res = 0
    return res


def validation():
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, label in validation_dataloader:
            data, label = data.to(device), label.to(device)
            predicted = net(data)
            predicted = torch.nn.functional.softmax(predicted, dim=1)
            predicted = torch.max(predicted, 1)[1]
            label = label.squeeze()
            total += label.size(0)
            correct += (predicted == label).sum().item()
        Write_log("validation: acc is {:.4f}%".format(100 * correct / total))
        vali_acc.append(100 * correct / total)


def test():
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, label in test_dataloader:
            data, label = data.to(device), label.to(device)
            predicted = net(data)
            predicted = torch.nn.functional.softmax(predicted, dim=1)
            predicted = torch.max(predicted, 1)[1]
            label = label.squeeze()
            total += label.size(0)
            correct += (predicted == label).sum().item()
        Write_log("test: acc is {:.4f}%".format(100 * correct / total))


def train():
    Write_log("start training")
    iteration = 0
    validation()
    for epoch in range(hyper_parameters["EPOCH"]):
        for data, label in train_dataloader:
            iteration += 1
            if iteration % hyper_parameters["V-iteration"] == 0:
                validation()
            net.train()
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label.squeeze().long())
            loss.backward()
            optimizer.step()
        Write_log("epoch:{} loss:{:.6f}".format(epoch, loss))
        if abort():
            Write_log('training aborted')
            break
    Write_log("training finished")


train_dataset = NewDataset(trainSet, torchvision.transforms.Compose(
    [torchvision.transforms.RandomCrop(224, padding=5), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip(), torchvision.transforms.ToTensor()]))
train_dataloader = torch.utils.data.dataloader.DataLoader(
    train_dataset, batch_size=hyper_parameters['BS'], num_workers=hyper_parameters['NW'], shuffle=True, drop_last=True)
test_dataset = NewDataset(testSet, torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()]))
test_dataloader = torch.utils.data.dataloader.DataLoader(
    test_dataset, batch_size=hyper_parameters['BS'], num_workers=hyper_parameters['NW'], drop_last=True)
validation_dataset = NewDataset(validationSet, torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()]))
validation_dataloader = torch.utils.data.dataloader.DataLoader(
    validation_dataset, batch_size=hyper_parameters['BS'], num_workers=hyper_parameters['NW'], drop_last=True)
train()
test()
Write_log("start saving!")
torch.save(net, "models/"+FILE_NAME+'.model')
Write_log("saving finished!\nall done!")
