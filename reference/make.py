import os

l = []
'''
    '╡═╖╓╗п╧┘░й' : low
    '╓╨╖╓╗п╧┘░й' : middle
    '''


def Get_file_paths(_targetDirectory):
    path = _targetDirectory
    for i in os.listdir(path):
        if os.path.isfile(path + '/' + i):
            l.append(path + '/' + i)
        else:
            Get_file_paths(path+'/'+i)


Get_file_paths('dataset/new/61_crop/Other')
f = open('cam/61_crop/other.meta', 'w')
for i in l:
    f.writelines(i + '\n')
f.close()
l = []
Get_file_paths('dataset/new/61_crop/╡═╖╓╗п╧┘░й')
f = open('cam/61_crop/low.meta', 'w')
for i in l:
    f.writelines(i + '\n')
f.close()
