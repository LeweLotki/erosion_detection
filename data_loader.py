from os import listdir, path
from pandas import read_csv
from cv2 import imread, cvtColor, resize, COLOR_RGB2GRAY

labels = read_csv('data_labels.csv')
resolution = 52

def load_data(folder):
    data = []
    for filename in listdir(folder):
        img = imread(path.join(folder,filename))
        img = cvtColor(img, COLOR_RGB2GRAY)
        img = resize(img, (resolution, resolution))
        idx = int(filename[0:-4])
        try:
            label = int(labels.loc[idx]['Code'])
            if img is not None:
                data.append([img,label])
                if label == 1:  data.append([img / 255,label])
        except:continue
    return data

df = load_data('data')
