import  numpy as np
import os
import shutil
from core.data_loader import glob
import pandas as pd


samples_src=glob('sample_images/src','*/*',recursive=True)
samples_ref=glob('sample_images/ref','*/*',recursive=True)

#np.save('samples_src.npy',samples_src)
#np.save('samples_ref.npy',samples_ref)
src=np.load('samples_src.npy')
ref=np.load('samples_ref.npy')

ar1=pd.read_csv('dataset/dataindex_00.csv')

cls_by_folder = glob('dataset/', '*.csv', True)

arr = pd.read_csv('dataset/dataindex_00.csv', dtype='unicode')
arr = arr.loc[:, ['sex', 'path']]
for file_info in cls_by_folder:
    arrr = pd.read_csv(file_info, dtype='unicode')
    arrr = arrr.loc[:, ['sex', 'path']]
    arr=pd.concat([arr,arrr],axis=0)



arr.drop_duplicates()

print('aa')
'''
#pretrain 시 사용했던 0 class dataset
dslist=np.load('imagelists/train_0.npy')


#pretrain 시 사용했던 1 class dataset
dslist2=np.load('imagelists/train_1.npy')


#pretrain validation 시 사용했던 0 class dataset
dslist3=np.load('imagelists/val_0.npy')


#pretrain validation 시 사용했던 1 class dataset
dslist4=np.load('imagelists/val_1.npy')


for f in dslist:
    name = f.split('/')[-1][0:-2]
    if os.path.isfile('datasets/hair/female/'+name):
        shutil.copy('datasets/hair/female/'+name, 'datasets/hair_pretrain/female/' + name + 'pg')
    elif os.path.isfile('datasets/hair_val/female/'+name+ 'pg') :
        shutil.copy('datasets/hair_val/female/'+name+ 'pg', 'datasets/hair_pretrain/female/' + name + 'pg')
    else:
        print('there is no file')

for f in dslist2:
    name = f.split('/')[-1][0:-2]
    if os.path.isfile('datasets/hair/male/'+name):
        shutil.copy('datasets/hair/male/'+name, 'datasets/hair_pretrain/male/' + name + 'pg')
    elif os.path.isfile('datasets/hair_val/male/'+name+ 'pg') :
        shutil.copy('datasets/hair_val/male/'+name+ 'pg', 'datasets/hair_pretrain/male/' + name + 'pg')
    else:
        print('there is no file')


for f in dslist3:
    name = f.split('/')[-1][0:-2]
    if os.path.isfile('datasets/hair/female/'+name):
        shutil.copy('datasets/hair/female/'+name, 'datasets/hair_preval/female/' + name + 'pg')
    elif os.path.isfile('datasets/hair_val/female/'+name+ 'pg') :
        shutil.copy('datasets/hair_val/female/'+name+ 'pg', 'datasets/hair_preval/female/' + name + 'pg')
    else:
        print('there is no file')

for f in dslist4:
    name = f.split('/')[-1][0:-2]
    if os.path.isfile('datasets/hair/male/'+name):
        shutil.copy('datasets/hair/male/'+name, 'datasets/hair_preval/male/' + name + 'pg')
    elif os.path.isfile('datasets/hair_val/male/'+name+ 'pg') :
        shutil.copy('datasets/hair_val/male/'+name+ 'pg', 'datasets/hair_preval/male/' + name + 'pg')
    else:
        print('there is no file')
'''
