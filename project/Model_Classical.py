from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from skimage.filters import prewitt_h, prewitt_v
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import csv

param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}

def Relu(img, threshold):
    return np.maximum(img, threshold)

def greyScale(img):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    return np.dot(img[:][...,:3], rgb_weights)
    
def centroid(img):
    img = img / np.sum(np.sum(img))
    # marginal distributions
    dx = np.sum(img, 1)
    dy = np.sum(img, 0)
    
    # expected values
    cx = np.sum(dx * np.arange(256))
    cy = np.sum(dy * np.arange(256))
    return int(cx), int(cy)


# extracting the avg brightness
def avg_brightness(img):
    w,h = img.shape
    avg_val = np.sum(np.sum(img, axis=0))/(w*h)
    return avg_val

# extracting the euclidean distance of the image centroid
def centroid_loc(img):
    w,h = img.shape
    cx, cy = centroid(img)
    dx = (w//2 - cx)**2
    dy = (h//2 - cy)**2
    euclidean_dist = np.sqrt(dx + dy)
    return euclidean_dist

# extracting the symmetry 
# compare the pixel match after a rotation
def symmetry(img):
    threshold = 10
    rotate_img = img.T
    diff_img = img - rotate_img
    abs_img = np.absolute(diff_img)
    count = 0
    for i in range(len(abs_img)):
        for j in range(len(abs_img[i])):
            if (abs_img[i][j] < threshold):
                count += 1
    return count/20

# extracting the pixel brightness around the centroid
def centroid_density(img):
    cx, cy = centroid(img)
    dist = 10
    total_val = 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            if (np.abs(cx - i) < 10 and np.abs(cy - j) < 10):
                total_val += img[i][j]
    avg_val = total_val / (dist*dist)
    return avg_val

# extract the avg brightness on the vertical edge plot
def vertical_val(img):
    edges_vertical = prewitt_v(img)
    return (avg_brightness(edges_vertical) +1)*100

# extract the avg brightness on the vertical edge plot
def horizontial_val(img):
    edges_horizontal = prewitt_h(img)
    return (avg_brightness(edges_horizontal)+1)*100

def reading_cvs(filename):
    X, Y = [],[]
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            f1 = float(row[0])
            f2 = float(row[1])
            f3 = float(row[2])
            f4 = float(row[3])
            f5 = float(row[4])
            f6 = float(row[5])
            Y.append(float(row[6]))
            X.append([f1, f2, f3, f4, f5, f6])
    return X, Y

def accuracy_calc(clf, X, Y):
    prediction = clf.predict(X)
    trainN = len(Y)
    numCorrect = 0
    for i in range(len(prediction)):
        if (prediction[i] == Y[i]):
            numCorrect += 1    
    accuracy = numCorrect/trainN
    return accuracy

def main():
    saving_feature = False
    if saving_feature:
        # loading in the data
        trainX = np.load("trainX.npy", allow_pickle=True)
        trainY = np.load("trainY.npy", allow_pickle=True)
        testX = np.load("testX.npy", allow_pickle=True)
        testY = np.load("testY.npy", allow_pickle=True)
        print("loading completed")
        
        
        trainX = trainX.astype(int)
        trainN = trainY.shape[0]
        
        # processing the data
        trainX = greyScale(trainX)
        trainX = Relu(trainX, 80)
        
        # extracting the features
        features = []
        for i in range(trainN):
            feature = []
            img = trainX[i]
            feature.append(avg_brightness(img))
            feature.append(centroid_loc(img))
            feature.append(symmetry(img))
            feature.append(centroid_density(img))
            feature.append(vertical_val(img))
            feature.append(horizontial_val(img))
            feature.append((trainY[i]))           
            
            features.append(feature)

        # saving the features to .csv
        with open('train_features.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(features)):
                writer.writerow(features[i])
        
        testX = testX.astype(int)
        testN = testY.shape[0]
        
        # processing the data
        testX = greyScale(testX)
        testX = Relu(testX, 80)
        
        # extracting the features
        features = []
        for i in range(testN):
            feature = []
            img = testX[i]
            feature.append(avg_brightness(img))
            feature.append(centroid_loc(img))
            feature.append(symmetry(img))
            feature.append(centroid_density(img))
            feature.append(vertical_val(img))
            feature.append(horizontial_val(img))
            feature.append((trainY[i]))           
            
            features.append(feature)

        # saving the features to .csv
        with open('test_features.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(features)):
                writer.writerow(features[i])
        print("feature selection completed")
        
        
    # reading the features from csv
    trainX, trainY = reading_cvs('train_features.csv')
    testX, testY = reading_cvs('test_features.csv')
    
    # using Support Vector Machine
    clf = svm.SVC()
    clf.fit(trainX, trainY)
    accuracy = accuracy_calc(clf, trainX, trainY)
    print("SVM train accuracy", accuracy)
    accuracy = accuracy_calc(clf, testX, testY)
    print("SVM test accuracy", accuracy)
    print()
    
    # using Random Forest
    clf = RandomForestClassifier(max_depth=3, random_state=0)
    clf.fit(trainX, trainY)
    accuracy = accuracy_calc(clf, trainX, trainY)
    print("RF train accuracy", accuracy)
    accuracy = accuracy_calc(clf, testX, testY)
    print("RF test accuracy", accuracy)
    print()
    
    # using Logistic Regression 
    clf = LogisticRegression(random_state=0)
    clf.fit(trainX, trainY)
    accuracy = accuracy_calc(clf, trainX, trainY)
    print("LR train accuracy", accuracy)
    accuracy = accuracy_calc(clf, testX, testY)
    print("LR test accuracy", accuracy)
    

    # PCA for plotting
    pca = PCA(n_components=2)
    pc = pca.fit_transform(testX)

    plt.title("PCA plot")
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    for i in range(len(testY)):

        if (testY[i] == 1.0):
            plt.scatter(pc[i][0], pc[i][1], marker = 'o', color = 'r')
        else:
            plt.scatter(pc[i][0], pc[i][1], marker = 'x', color = 'b')
if __name__ == "__main__":
    main()
