import os
import numpy as np
from PIL import Image

def crop(img):
    width, height = img.size
    min_dim = min(width,height)
    left = (width-min_dim)//2
    right = left + min_dim
    top = (height-min_dim)//2
    bottom = top + min_dim
    img = img.crop((left, top, right, bottom))
    return img

def retrieve(scale_size, data, label, max_index):
    names = [] 
    input_path = os.getcwd() + data
    for root, dirs, files in os.walk(input_path,topdown = True):
        for name in files: 
            _, ending = os.path.splitext(name)
            if (ending == ".jpg" or ending == ".png"):
                names.append(name)
    all_img = []
    all_labels = []
    for i in range(len(names)):
        if (i >= max_index):
            break
        img = Image.open(os.path.join(input_path, names[i])).convert('RGB')
        #img = ImageOps.grayscale(img)
        img = crop(img)
        if (img.size[0] > 1000 or img.size[0] < 256):
            continue
        img.thumbnail((scale_size, scale_size))
        img = np.asarray(img)
        all_img.append(img)
        all_labels.append(label)
    return np.asarray(all_img, dtype=np.float16), np.asarray(all_labels, dtype=np.float16)
    
def store():
    scale_size = 256
    data_Yes1, label_Yes1 = retrieve(scale_size, "\\transverseYes\\", 1, 1900)
    print("yes1", data_Yes1.shape)
    
    data_No1, label_No1 = retrieve(scale_size, "\\transverseNo\\", 0, 1900)
    print("no1", data_No1.shape)
    
    '''
    data_Yes2, label_Yes2 = retrieve(scale_size, "\\transverseYes\\", 2, 1900) #1900
    print("yes2", data_Yes2.shape)
    data_No2, label_No2 = retrieve(scale_size, "\\transverseNo\\", 0, 1900//2) # 1900//2
    print("no2", data_No2.shape)
    data = np.concatenate((data_Yes1, data_No1, data_Yes2, data_No2))
    labels = np.concatenate((label_Yes1, label_No1, label_Yes2, label_No2))
    '''
    data = np.concatenate((data_Yes1, data_No1))
    labels = np.concatenate((label_Yes1, label_No1))
    
    dataSize = labels.shape[0]
    split_size = dataSize//5*4
    np.random.seed(0)
    train_indices = np.random.choice(dataSize,split_size, replace=False)
    all_indices = np.arange(dataSize)
    test_indices = np.delete(all_indices, train_indices)
    trainX = data[train_indices]
    testX = data[test_indices]
    trainY = labels[train_indices]
    testY = labels[test_indices]
    
    np.save("trainX.npy",trainX)
    np.save("testX.npy", testX)
    np.save("trainY.npy",trainY)
    np.save("testY.npy", testY)

if __name__ == "__main__":
    store()