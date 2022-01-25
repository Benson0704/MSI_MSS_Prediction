import torch
import torchvision
import PIL
import json
import time
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
testFilesMSI = data0
testFilesMSS = data1
Write_log('MSI: {}, MSS: {}'.format(len(data0), len(data1)))
L = []
F = []
P = {}
result = []
for i in range(len(testFilesMSS)):
    name = testFilesMSS[i].split('_&_')[0]
    F.append(testFilesMSS[i])
    if name in P:
        pass
    else:
        P[name] = 1
    testFilesMSS[i] = (testFilesMSS[i], 1)
    L.append(1)
for i in range(len(testFilesMSI)):
    name = testFilesMSI[i].split('_&_')[0]
    F.append(testFilesMSI[i])
    if name in P:
        pass
    else:
        P[name] = 0
    testFilesMSI[i] = (testFilesMSI[i], 0)
    L.append(0)

data = testFilesMSS+testFilesMSI


class NewDataset(torch.utils.data.Dataset):
    def __init__(self, data_set, transforms):
        self.data_set = data_set
        self.transforms = transforms

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        if index >= len(self.data_set):
            Write_log(time.asctime(time.localtime()) +
                      '  '+"Error: __getitem__ out of range" + '\n')
        else:
            image_file, label = self.data_set[index]
            label_tensor = torch.Tensor([1])
            label_tensor[0] = label
            image = PIL.Image.open(image_file)
            image_tensor = self.transforms(image)
            image_tensor = image_tensor.type(torch.FloatTensor)
            return image_tensor, label, index


dataset = NewDataset(data, torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()]))
dataloader = torch.utils.data.dataloader.DataLoader(
    dataset, batch_size=hyper_parameters['BS'], num_workers=hyper_parameters['NW'])
net = torch.load("models/train2MS.py_Low+Middle" + '.model')
Write_log('using model: '+"models/train2MS.py_Low+Middle" + '.model')
net.eval()
probability = []
label_list = []

with torch.no_grad():
    total = 0
    for image, label, index in dataloader:
        image = image.to(device)
        predicted = net(image)
        predicted = torch.nn.functional.softmax(predicted, dim=1)
        for i in range(len(index)):
            probability.append(float(predicted[i][1]))
            label_list.append(float(label[i]))
            total += 1
        predicted = torch.max(predicted, 1)[1]
        for i in range(len(index)):
            result.append(int(predicted[i]))
        if total % 2560 == 0:
            Write_log('{}/{} completed!'.format(total, len(data)))
print(len(data))
print(len(probability))
print(len(label_list))
auc = []
for i, pr in enumerate(probability):
    if pr >= 0.5:
        probability[i] = 1
    else:
        probability[i] = 0
auc = []
Pfinal = []
PLfinal = []
dict_data = {}
for i in range(len(F)):
    name = F[i].split('_&_')[0]
    if name in dict_data.keys():
        dict_data[name][1].append((F[i], L[i], result[i]))
    else:
        dict_data[name] = []
        dict_data[name].append(P[name])
        dict_data[name].append([])
        dict_data[name][1].append((F[i], L[i], result[i]))
Write_log('dict length ' + str(len(dict_data)))
d = []
for k in dict_data:
    real, li = dict_data[k]
    if len(li) <= 10:
        d.append(k)
for i in d:
    print('delete')
    del dict_data[i]
Write_log('edited dict length ' + str(len(dict_data)))
wrong = []
output = open('test.txt', 'w')
for k in dict_data:
    mss = 0
    msi = 0
    real, li = dict_data[k]
    for i in li:
        f, l, r = i
        if r == 1:
            mss += 1
        elif r == 0:
            msi += 1
        else:
            print('error')
            exit()
    Pfinal.append(mss/(mss+msi))
    PLfinal.append(real)
    output.writelines(str(k)+' '+str(mss/(mss+msi))+' '+str(real)+'\n')
output.close()
for i, _ in enumerate(Pfinal):
    print(Pfinal[i], PLfinal[i])
