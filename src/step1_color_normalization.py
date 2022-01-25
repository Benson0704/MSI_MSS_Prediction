'''
Description:
    This script resizes tiles to 224*224 and does the color normalization
Author:
    Benson0704@outlook.com
Version:
    origin
'''
import PIL
import numpy
import time
import staintools
import os


def get_file_paths(_targetDirectory, _list):
    path = _targetDirectory
    for i in os.listdir(path):
        if os.path.isfile(path + '/' + i):
            if i[-4:] == '.jpg':
                _list.append(path + '/' + i)
        else:
            if not os.path.exists((path + '/' + i).replace('ImageCrops Samples - zenodo', 'processed')):
                os.mkdir(
                    (path + '/' + i).replace('ImageCrops Samples - zenodo', 'processed'))
            get_file_paths(path+'/'+i, _list)


if __name__ == '__main__':
    if not os.path.exists('../dataset/processed'):
        os.mkdir('../dataset/processed')
    tile_list = []
    get_file_paths('../dataset/ImageCrops Samples - zenodo/', tile_list)
    # print(tile_list)
    print('Loading images done!')
    refImage = PIL.Image.open('../Ref.png')
    refImage = numpy.array(refImage)
    refImage.astype('uint8')
    normalizer = staintools.StainNormalizer('macenko')
    normalizer.fit(refImage)
    print('Initailizing normalizer done!')
    time_last = time.time()
    for iter, data in enumerate(tile_list):
        if iter % 100 == 0:
            print('{}/{} done, time used these 100 tiles:{:.3f}s'.format(iter,
                                                                         len(tile_list), time.time()-time_last))
            time_last = time.time()
        image = PIL.Image.open(data)
        image = image.resize((224, 224), PIL.Image.ANTIALIAS)
        image = numpy.array(image)
        image.astype('uint8')
        image = normalizer.transform(image)
        image = PIL.Image.fromarray(image)
        image.save(data.replace('ImageCrops Samples - zenodo', 'processed'))
    print('All done! step1 completed!')
