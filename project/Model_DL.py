import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def CNN():
    shape = [256, 256, 3]
    K = 1
    
    model = Sequential()
    model.add(layers.Conv2D(8, (11,11), strides=4,padding="VALID",input_shape=shape))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D(2, strides=2, padding="VALID"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(128, (5,5), strides=1,padding="VALID", kernel_regularizer= 'l2'))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D(2, strides=2, padding="VALID"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(256,(3,3),strides=1,padding='VALID', kernel_regularizer= 'l2'))
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(256,(3,3),strides=1,padding='VALID', kernel_regularizer= 'l2'))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D(2, strides=2, padding="VALID"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(128,(3,3),strides=1,padding='VALID', kernel_regularizer= 'l2'))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D(2, strides=2, padding="VALID"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(10))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(K, activation='sigmoid'))
    
    model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=['accuracy'])
    return model

'''
plotting the first layer filters
'''
def visualize(model):
    weights, bias = model.layers[0].get_weights()

    #normalize filter values between  0 and 1 for visualization
    f_min, f_max = weights.min(), weights.max()
    filters = (weights - f_min) / (f_max - f_min)  
    
    #plotting all the filters
    for i in range(filters.shape[3]):
        plt.figure()
        plt.imshow(filters[:,:,:, i])
        plt.title("Filter" + str(i+1))
    plt.show()
        
    
def plot(history):
    L_train = history.get("loss")
    Acc = history.get('accuracy')
    
    # ploting training loss
    iterations = np.arange(len(L_train))
    plt.plot(iterations,L_train)
    plt.title('Train Error')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')

    # ploting testing accuracy
    plt.figure()
    plt.plot(iterations,Acc)
    plt.title('Train Accuracy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
 
def result(model, testX, testY):
    # test preformance on testing dataset
    test_loss, test_acc = model.evaluate(testX, testY, batch_size= 300)
    print("The final accuracy on testing dataset is: " + str(test_acc))
    
def main():
    saving = False
    model = CNN()
    
    if (saving):    
        # loading the train dataset
        trainX = np.load("trainX.npy", allow_pickle=True)
        trainY = np.load("trainY.npy", allow_pickle=True)
    
        print("loading completed")
        
        # normalizing the data
        trainX = trainX/255.0

        # training the model on CNN
        history = model.fit(trainX, trainY, batch_size=300, steps_per_epoch = 10, epochs=80, verbose=1)
        plot(history.history)
        print("Training with CNN completed")
        
        # saving the model in h5
        model.save_weights('model_weights.h5')
        
    else:
        model.load_weights('model_weights.h5')
        
    # loading the test dataset
    testX = np.load("testX.npy", allow_pickle=True)
    testY = np.load("testY.npy", allow_pickle=True)
    testX = testX/255.0
    visualize(model)
    result(model, testX, testY)
    
if __name__ == "__main__":
    main()
