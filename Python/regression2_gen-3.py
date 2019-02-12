#Neal Bayya and George Tang
from os import listdir
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
        x0 = np.array([cv.imread(crops_path+p+'.png') for p in bx]) / 255.0
        y = np.array(by)
        return x0, y

def generate_crops(model_input, size=128, npatches=10):
    [rows, columns, nchannels] = model_input.shape
    crops = []
    for i in range(npatches):
        ul_row = random.randint(0,rows-size)
        ul_col = random.randint(0,columns-size)
        patch_input = model_input[ul_row:ul_row+size, ul_col: ul_col+size,:]
        crops.append(patch_input)
    return np.array(crops)

def crops_master(images, id2params, cropsize, ncrops):
    crop2betaf = open(path+'valcrops.txt', 'w')
    for img_num, i in enumerate(images):
        img_id = i[0:len(i)-6]
        params = id2params[img_id]
        img = cv.imread(image_path + i)
        img_crops = generate_crops(img, cropsize, ncrops)
        for c, img_crop in enumerate(img_crops):
            cn = i.split('.')[0] + "_" + str(c)
            crop2betaf.write(cn + " " + str(params[1]) + "\n")
            cv.imwrite(crops_path+cn+'.png', img_crop)
    crop2betaf.close()

def create_model():
    #encoder
    init = Input(shape=(128, 128, 3))
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

def main():
    global path, image_path, crops_path 
    #generate crops
    path = ''
    id2params = {}
    """
    for i, l in enumerate(open(path+'val2.txt', 'r')):
        if '?' not in l:
            continue
        split = l.split("?")
        img_id = split[0] + '?' + split[1]
        alpha = float(split[2])
        beta = float(split[3])
        id2params[img_id] = (alpha, beta)
    image_path = path + "HazyImages2/"
    images = sorted([i for i in listdir(image_path) if not i.startswith(('.', '_'))])
    print("number of images: {}".format(len(images)))
    """
    crops_path = path + "crops2/"
    ncrops = 10
    cropsize = 128
    #crops_master(images, id2params, cropsize, ncrops)
    #Read crops
    ncases = 16800#len(images)*ncrops
    print(ncases)
    crop_names = []
    crops2betaf = open(path + 'valcrops.txt', 'r')
    x, y = [], []
    for c_num, l in enumerate(crops2betaf):
        labeling = l.split(' ')
        x.append(labeling[0])
        crop_names.append(labeling[0])
        y.append(float(labeling[1]))
    # ML
    random.seed(5)
    train_split = 2/3
    test_split = 1/6
    val_split = 1/6
    batch_size = 32
    MODEL_NAME = 'regressor3'

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
    
    model_json_name = "models/" + MODEL_NAME + ".json"
    model_weights_name = "models/" + MODEL_NAME + ".h5"
    model_predictions_name = "models/" + MODEL_NAME + "pred.npz"
    """ 
    #Training model
    model = create_model()
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    #history = model.fit(x=X_train, y=Y_train, validation_data = (X_val, Y_val), epochs=100)
    trainGen = Generator(X_train, Y_train, batch_size)
    valGen = Generator(X_val, Y_val, batch_size)
    history = model.fit_generator(generator=trainGen,
                                           steps_per_epoch=len(Y_train)//batch_size,
                                           epochs=75,
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
    plt.savefig("loss_plot.png")

    #plot metric graph
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model mean absolute error')
    plt.ylabel('mean absolute error ')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("metric_plot.png")

    #save model
    model_json = model.to_json()
    with open(model_json_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_weights_name)
    print("Training complete. Saved model to disk")
    """
    '''
    #Testing model
    '''
    json_file = open(model_json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights_name)
    loaded_model.compile(optimizer='adam', loss='mae')
    print("Loaded model from disk")
    testGen = Generator(X_test, Y_test, 32)
    pred_stats = loaded_model.evaluate_generator(testGen)
    print(pred_stats)
    pred = loaded_model.predict_generator(testGen)
    pred = [p[0] for p in pred]
    with open("preds.pkl", "wb") as f:
        pickle.dump([names_test, X_test, Y_test, pred], f)
if __name__ == '__main__':
    main()
