import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json


def main(out, name):
    def img_preprocess(img_in):
        img = img_in.copy()
        img = img[:, :, ::-1]   				# 1
        img = np.ascontiguousarray(img)			# 2
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = transform(img)
        img = img.unsqueeze(0)					# 3
        return img
    # 定义获取梯度的函数

    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数

    def farward_hook(module, input, output):
        fmap_block.append(output)
    # 计算grad-cam并可视化

    def cam_show_img(img, feature_map, grads, out_dir, name):
        H, W, _ = img.shape
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
        grads = grads.reshape([grads.shape[0], -1])					# 5
        weights = np.mean(grads, axis=1)							# 6
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]							# 7
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cv2.resize(cam, (W, H))
        cv2.imwrite(os.path.join(out_dir,
                                 name.split('/')[-1]), img)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_img = 0.3 * heatmap + 0.7 * img

        path_cam_img = os.path.join(out_dir, "Edited-" + name.split('/')[-1])
        cv2.imwrite(path_cam_img, cam_img)
    path_img = name
    output_dir = '/home/wangbeining/data/OurData/cam/{}'.format(out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    # 只取标签名
    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()
    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)

    img_input = img_preprocess(img)

    # 加载 squeezenet1_1 预训练模型
    net = torch.load("models/" + 'train2MS.py_Low+Middle' + '.model')
    net.eval()
    # 注册hook
    net.layer4.register_forward_hook(farward_hook)  # 9
    net.layer4.register_backward_hook(backward_hook)

    # forward
    output = net(img_input.to(device))
    idx = np.argmax(output.cpu().data.numpy())
    # backward
    net.zero_grad()
    class_loss = output[0, idx]
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # 保存cam图片
    cam_show_img(img, fmap, grads_val, output_dir, name)


if __name__ == '__main__':
    '''
    '╡═╖╓╗п╧┘░й' : low
    '╓╨╖╓╗п╧┘░й' : middle
    '''
    '''
    hyper_parameters = open('dataset/hyper_parameters_json', 'r')
    f = open('cam/4_crop/middle.meta', 'r')
    for line in f.readlines():
        main('4_crop/middle', line.replace('\n', ''))
    f.close()
    f = open('cam/4_crop/other.meta', 'r')
    for line in f.readlines():
        main('4_crop/other', line.replace('\n', ''))
    f.close()

    f = open('cam/6_crop/low.meta', 'r')
    for line in f.readlines():
        main('6_crop/low', line.replace('\n', ''))
    f.close()
    
    f = open('cam/27_crop/middle.meta', 'r')
    for line in f.readlines():
        main('27_crop/middle', line.replace('\n', ''))
    f.close()
    f = open('cam/27_crop/other.meta', 'r')
    for line in f.readlines():
        main('27_crop/other', line.replace('\n', ''))
    f.close()
    
    f = open('cam/34_crop/low.meta', 'r')
    for line in f.readlines():
        main('34_crop/low', line.replace('\n', ''))
    f.close()
    
    f = open('cam/34_crop/other.meta', 'r')
    for line in f.readlines():
        main('34_crop/other', line.replace('\n', ''))
    f.close()
    
    f = open('cam/61_crop/low.meta', 'r')
    for line in f.readlines():
        main('61_crop/low', line.replace('\n', ''))
    f.close()
    '''
    f = open('cam/61_crop/other.meta', 'r')
    for line in f.readlines():
        main('61_crop/other', line.replace('\n', ''))
    f.close()
