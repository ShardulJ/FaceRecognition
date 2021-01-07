import numpy as np
import pandas as pd
import glob
import os

main_path='/Users/sharduljanaskar/Documents/face_classification/trainset/'

df = pd.DataFrame(columns=['path','img_names','label'])

image_list = []

for root, subdirectories, files in os.walk(main_path):
    for file in files:
        if not file.startswith('.'):
            image_list.append(os.path.join(root, file))
        
    
df['path'] = image_list
df['img_names'] = [x.split('/')[8] for x in image_list]
df['label'] = [x.split('/')[7] for x in image_list]

df = df.sort_values(by=['path'])
df.to_csv('df_new.csv')

#print(len(labels))
print(df.head())

'''
def extract(dir):
    for
'''
