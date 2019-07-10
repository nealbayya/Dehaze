#Neal Bayya
#4 channel input version
from os import listdir
from PIL import Image
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import random
import pickle
import glob
import cv2 as cv

from keras import backend as K
from keras.models import *
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.optimizers import *
from keras.metrics import *
from keras.utils import *

class Generator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
    def __len__(self):
        return int(np.ceil(len(self.x_set)/float(self.batch_size)))
    def __getitem__(self, idx):
        bx = self.x_set[idx*self.batch_size:(idx+1)*self.batch_size]
        by = self.y_set[idx*self.batch_size:(idx+1)*self.batch_size]
        x = []
        for p in bx:
            xin = np.load(crops_path+p)['hazyin'] 
            xin[:,:,0:3] = xin[:,:,0:3] / 255
            x.append(xin)
        x = np.array(x)
        y = np.array(by)
        return x, y

def create_model(N = 2, M = 3):
    hazy = Input(shape=(128, 128, 4))
    x = hazy
    for i in range(M): #M controls number of times to downsample
        for j in range(N): #apply conv N times before downsample
            cj = Conv2D(32, (3,3), activation='relu', padding='same')(x)
            x = cj
        x = MaxPooling2D((2,2), padding='same')(x) #downsample
    x = Dropout(rate=0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(1)(x)
    return Model(inputs = hazy, outputs = x)

def create_model2():
    init = Input(shape=(128, 128, 4))
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(init)
    x = MaxPooling2D((2, 2), padding='same')(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(c2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(c3)
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(c4)
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(c5)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(c6)
    x = Dense(1)(x)
    return Model(init, x)

def create_model3():
    init = Input(shape=(128, 128, 4))
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(init)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    x = MaxPooling2D((2, 2), padding='same')(c3)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    x = MaxPooling2D((2, 2), padding='same')(c3)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    x = MaxPooling2D((2, 2), padding='same')(c3)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    x = MaxPooling2D((2, 2), padding='same')(c3)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    x = MaxPooling2D((2, 2), padding='same')(c3)
    x = Flatten()(x)
    x = Dense(1)(x)
    return Model(init, x)

def main():
    global path, image_path, crops_path 
    path = 'regression_data/HazeRD_data/'
    image_path = path + "simu/"
    crops_path = path + "crops4c/"
    ncrops = 5
    cropsize = 128

    random.seed(5)
    train_split = 2/3
    test_split = 1/6
    val_split = 1/6
    batch_size = 32
    MODEL_NAME = 'dmap_HAZERD_5cycle3'
 
    model_json_name = "models/" + MODEL_NAME + ".json"
    model_weights_name = "models/" + MODEL_NAME + ".h5"
    model_predictions_name = "models/" + MODEL_NAME + "pred.npz"
    
    #Read crops
    crop_names = []
    modelio = open(path + 'fourchannel_io.txt', 'r').readlines()
    x, y = [],[]
    error_crops = []
    for c_num, l in enumerate(modelio):
        labeling = l.split(' ')
        p = labeling[0]
        try:
            tmp = np.load(crops_path+p)['hazyin'] 
            tmp[:,:,0:3] = tmp[:,:,0:3] / 255
            x.append(p)
            crop_names.append(p)
            y.append(float(labeling[1]))
        except:
            error_crops.append(p)
    print("error parsing following crops: {}".format(error_crops))
    ncases = len(x)
    
    # ML
    indices = [i for i in range(ncases)]
    random.shuffle(indices)
    split1 = int(train_split*ncases)
    split2 = int((train_split+test_split)*ncases)
    X_train = np.array([x[idx] for idx in indices[:split1]])
    Y_train = np.array([y[idx] for idx in indices[:split1]])
    names_train = [crop_names[idx] for idx in indices[:split1]]
    X_test = np.array([x[idx] for idx in indices[split1:split2]])
    Y_test = np.array([y[idx] for idx in indices[split1:split2]])
    names_test = [crop_names[idx] for idx in indices[split1:split2]]
    X_val = np.array([x[idx] for idx in indices[split2:]])
    Y_val = np.array([y[idx] for idx in indices[split2:]])
    names_val = [crop_names[idx] for idx in indices[split2:]]

    print('X train shape: {}'.format(X_train.shape))
    print('Y train shape: {}'.format(Y_train.shape))
    
    print('X test shape: {}'.format(X_test.shape))
    print('Y test shape: {}'.format(Y_test.shape))
    
    print('X val shape: {}'.format(X_val.shape))
    print('Y val shape: {}'.format(Y_val.shape))
    ''' 
    #Training model
    model = create_model3()
    model.compile(optimizer='adam', loss='mae')
    trainGen = Generator(X_train, Y_train, batch_size)
    valGen = Generator(X_val, Y_val, batch_size)
    history = model.fit_generator(generator=trainGen,
                                           steps_per_epoch=len(Y_train)//batch_size,
                                           epochs=150,
                                           verbose=1,
                                           validation_data=valGen,
                                           validation_steps=len(Y_val)//batch_size,
                                           shuffle=True)
    print(history.history.keys())

    #plot loss curve
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(MODEL_NAME + "loss_plot.png")

    #save model
    model_json = model.to_json()
    with open(model_json_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_weights_name)
    print("Training complete. Saved model to disk")
    
    '''
    #Testing model
    json_file = open(model_json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights_name)
    loaded_model.compile(optimizer='adam', loss='mae')
    print("Loaded model from disk")
    testGen = Generator(X_test, Y_test, batch_size)
    pred_stats = loaded_model.evaluate_generator(testGen)
    print(pred_stats)
    pred = loaded_model.predict_generator(testGen)
    pred = [p[0] for p in pred]
    
    predfile = "pred" + MODEL_NAME + ".txt"
    predbuff = open(predfile, "w")
    predbuff.write("Crop Name\tPredicted\tY_Test\tDiff\n")
    for ntest, ptest, ytest in zip(names_test, pred, Y_test):
        predbuff.write(ntest + "\t" + "%0.2f"%ptest + "\t" + "%0.2f"%ytest +  "\t" + "%0.2f"%(ptest-ytest) + "\n")
        #predbuff.write("{}\t{}\t{}\t{}\n".format(ntest, ptest, ytest, ptest-ytest))
    predbuff.close()
    
            
if __name__ == '__main__':
    main()


