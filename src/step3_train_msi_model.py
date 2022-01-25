'''
Description:
    This script trains MSS-MSI tiles classifier
Author:
    Benson0704@outlook.com
Version:
    origin
WARNING:
    If model re-training is needed, dataset must be added manually as comments below
'''
import torch
import torchvision
import PIL
import json
import random
HYPER_PARAMETERS = open('../hyper_parameters.json', 'r')
HYPER_PARAMETERS = json.load(HYPER_PARAMETERS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(HYPER_PARAMETERS['GPU'])
'''
Now you can add your dataset as training data
data0 refers to those labeled as 'MSI'
data1 refers to those labeled as 'MSS'
those two python list should append the tile's path
    like: data0=['1.png','2.png']
later this scipt will automacally split them into train, test and validation sets
'''
data0 = []
data1 = []

random.shuffle(data0)
random.shuffle(data1)

train_set = []
validation_set = []
test_set = []
for i in range(len(data0)):
    if 0 <= i % 200 < 170:
        train_set.append((data0[i], 0))
    if 170 <= i % 200 < 195:
        validation_set.append((data0[i], 0))
    if 195 <= i % 200 < 200:
        test_set.append((data0[i], 0))

for i in range(len(data1)):
    if 0 <= i % 200 < 170:
        train_set.append((data1[i], 1))
    if 170 <= i % 200 < 195:
        validation_set.append((data1[i], 1))
    if 195 <= i % 200 < 200:
        test_set.append((data1[i], 1))

random.shuffle(train_set)
random.shuffle(validation_set)
random.shuffle(test_set)


class NewDataset(torch.utils.data.Dataset):
    def __init__(self, data_set, transforms):
        self.data_set = data_set
        self.transforms = transforms

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
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
iter = 0
for child in net.children():
    if iter < 6:
        for parameter in child.parameters():
            parameter.requires_grad = False
    iter += 1
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                             lr=HYPER_PARAMETERS["LR"], weight_decay=HYPER_PARAMETERS["L2"])
validation_accuracy = []


def abort():
    res = 0
    if len(validation_accuracy) > HYPER_PARAMETERS["wait"]:
        res = 1
        for i in range(HYPER_PARAMETERS['wait']):
            if validation_accuracy[-1] > validation_accuracy[-2 - i]:
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
        print("validation: acc is {:.4f}%".format(100 * correct / total))
        validation_accuracy.append(100 * correct / total)


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
        print("test: acc is {:.4f}%".format(100 * correct / total))


def train():
    print("start training")
    iteration = 0
    validation()
    for epoch in range(HYPER_PARAMETERS["EPOCH"]):
        for data, label in train_dataloader:
            iteration += 1
            if iteration % HYPER_PARAMETERS["V-iteration"] == 0:
                validation()
            net.train()
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label.squeeze().long())
            loss.backward()
            optimizer.step()
        print("epoch:{} loss:{:.6f}".format(epoch, loss))
        if abort():
            print('training aborted')
            break
    print("training finished")


if __name__ == '__main__':
    train_dataset = NewDataset(train_set, torchvision.transforms.Compose(
        [torchvision.transforms.RandomCrop(224, padding=5), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip(), torchvision.transforms.ToTensor()]))
    train_dataloader = torch.utils.data.dataloader.DataLoader(
        train_dataset, batch_size=HYPER_PARAMETERS['BS'], num_workers=HYPER_PARAMETERS['NW'], shuffle=True, drop_last=True)
    test_dataset = NewDataset(test_set, torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]))
    test_dataloader = torch.utils.data.dataloader.DataLoader(
        test_dataset, batch_size=HYPER_PARAMETERS['BS'], num_workers=HYPER_PARAMETERS['NW'], drop_last=True)
    validation_dataset = NewDataset(validation_set, torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]))
    validation_dataloader = torch.utils.data.dataloader.DataLoader(
        validation_dataset, batch_size=HYPER_PARAMETERS['BS'], num_workers=HYPER_PARAMETERS['NW'], drop_last=True)
    print('init done!')
    train()
    test()
    print("start saving!")
    torch.save(net, "../models/msi"+'.model')
    print("saving finished!\nall done!")
