import pandas as pd
import numpy as np
import glob
import os

data_folder = "datasets/"
folders = ["Business","Entertainment","Politics","Sports","Tech"]

files = os.listdir('datasets/Business/')
folderslist = [f for f in os.listdir(data_folder) if not f.startswith('.')]

news = []
newstype = []
title = []
for folder in folders:
    folder_path = 'datasets/'+folder+'/'
    #list all files in a particular news category
    files = os.listdir(folder_path)
    for text_file in files:
        file_path = folder_path + "/" +text_file
        #read contents of a file
        with open(file_path, errors='replace') as f:
            titles = f.readline().strip()
            data = f.readlines()
        titles = ''.join(titles)
        data = ' '.join(data).replace("\n",'')
        #append the news article and it's category to two lists
        news.append(data)
        title.append(titles)
        newstype.append(folder)

datadict = {'title':title , 'news':news, 'type':newstype}
df = pd.DataFrame(datadict)
df.to_csv('CSVs/newsdatabase.csv')