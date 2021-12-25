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


file = hyper_parameters['testfile']
for i in range(len(file)):
    file[i] = 'meta/' + file[i]
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
data = []
'''
for i in file:  # mss=1 msi=0
    f = open(i, 'r')
    for line in f.readlines():
        data.append(line[:-1])
    f.close()
'''
tumorfile = open('tumor.txt', 'r')
for i in tumorfile.readlines():
    tmp = i.split(' ')
    if float(tmp[1]) < 0.5:
        data.append(tmp[0])
print(len(data))
for i, _ in enumerate(data):
    data[i] = (data[i], 0)


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
net = torch.load("models/" + FILE_NAME.replace('new', 'train') + '.model')
Write_log('using model: '+"models/" +
          FILE_NAME.replace('new', 'train') + '.model')
net.eval()
probability = []

with torch.no_grad():
    total = 0
    for image, label, index in dataloader:
        image = image.to(device)
        predicted = net(image)
        predicted = torch.nn.functional.softmax(predicted, dim=1)
        for i in range(len(index)):
            probability.append(float(predicted[i][1]))
            total += 1
        predicted = torch.max(predicted, 1)[1]
        if total % 2560 == 0:
            Write_log('{}/{} completed!'.format(total, len(data)))
output = []
f = open('MSS.txt', 'w')
for i, _ in enumerate(data):
    a, b = _
    output.append(a+' '+str(probability[i]))
output.sort()
for i in output:
    f.writelines(i+'\n')
f.close()
