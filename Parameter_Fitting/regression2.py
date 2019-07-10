#Neal Bayya
from os import listdir
import numpy as np
import numpy.matlib
from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
import random
import pickle
import glob

from keras import backend as K
from keras.models import *
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.optimizers import *
from keras.metrics import *
from keras.utils import *

def generate_crops(model_input, size=128, npatches=10):
    [rows, columns, nchannels] = model_input.shape
    crops = []
    for i in range(npatches):
        ul_row = random.randint(0,rows-size)
        ul_col = random.randint(0,columns-size)
        patch_input = model_input[ul_row:ul_row+size, ul_col: ul_col+size,:]
        crops.append(patch_input)
    return np.array(crops)

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
    x = Flatten()(c5)
    x = Dense(1)(x)
    return Model(init, x)

def main():
    #populate id2params
    path = 'regression_data/'
    id2params = {}
    for i, l in enumerate(open(path+'vals2.txt', 'r')):
        if '?' not in l:
            continue
        split = l.split("?")
        img_id = split[0] + '_' + split[1]
        alpha = float(split[2])
        beta = float(split[3])
        id2params[img_id] = (alpha, beta)
    
    #form crops and model io
    image_path = path + "HazyImages2/"
    crops_path = path + "crops2/"
    images = sorted([i for i in listdir(image_path) if not i.startswith(('.', '_'))])
    ncrops = 5
    cropsize = 128
    ncases = len(images)*ncrops
    model_input = np.zeros((ncases, cropsize, cropsize, 3))
    model_output = np.zeros(ncases)
    
    crop_names = []
    #if crops are already in crops path, do the following
    if len(listdir(crops_path)) > 0:
        gen_crops = sorted([i for i in listdir(crops_path) if not i.startswith(('.', '_'))])
        for c_num, c_name in enumerate(gen_crops):
            crop_block = imread(crops_path + c_name)
            model_input[c_num,:,:,:] = crop_block[:,:,0:3] 
            fst_underscr = c_name.index("_")           
            img_id  = c_name[0:c_name.index("_", fst_underscr+1)]
            model_output[c_num] = id2params[img_id][1]
            crop_names.append(img_id + "_" + str(c_num))
    else:
        for img_num, i in enumerate(images):
            img_id = i[0:len(i)-6]
            params = id2params[img_id]
            img = imread(image_path + i)
            img_crops = generate_crops(img, cropsize, ncrops)
            model_input[img_num*ncrops:img_num*ncrops+ncrops,:,:,:] = img_crops
            model_output[img_num*ncrops:img_num*ncrops+ncrops] = np.repeat(params[1], ncrops)
            for c, img_crop in enumerate(img_crops):
                cn = i.split('.')[0] + "_" + str(c)
                crop_names.append(cn)
                imsave(crops_path+cn+'.png', img_crop)

    # ML
    random.seed(7)
    train_split = 2/3
    test_split = 1/6
    val_split = 1/6

    MODEL_NAME = 'regressor_oc'

    indices = [i for i in range(ncases)]
    random.shuffle(indices)
    split1 = int(train_split*ncases)
    split2 = int((train_split+test_split)*ncases)
    X_train = np.array([model_input[idx,:,:,:] for idx in indices[:split1]])
    Y_train = np.array([model_output[idx] for idx in indices[:split1]])
    names_train = [crop_names[idx] for idx in indices[:split1]]
    X_test = np.array([model_input[idx,:,:,:] for idx in indices[split1:split2]])
    Y_test = np.array([model_output[idx] for idx in indices[split1:split2]])
    names_test = [crop_names[idx] for idx in indices[split1:split2]]
    X_val = np.array([model_input[idx,:,:,:] for idx in indices[split2:]])
    Y_val = np.array([model_output[idx] for idx in indices[split2:]])
    names_val = [crop_names[idx] for idx in indices[split2:]]

    model_json_name = "models/" + MODEL_NAME + ".json"
    model_weights_name = "models/" + MODEL_NAME + ".h5"
    model_predictions_name = "models/" + MODEL_NAME + "pred.npz"

    #Training model
    model = create_model()
    sgd = SGD(lr=0.01)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
    history = model.fit(x=X_train, y=Y_train, validation_data = (X_val, Y_val), epochs=100)
    print(history.history.keys())

    #plot loss curve
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("loss_plot_oc.png")

    #plot metric graph
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model mean absolute error')
    plt.ylabel('mean absolute error ')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("metric_plot_oc.png")

    #save model
    model_json = model.to_json()
    with open(model_json_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_weights_name)
    print("Training complete. Saved model to disk")

    #Testing model
    '''
    json_file = open(model_json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights_name)
    loaded_model.compile(optimizer='adam', loss='mean_squared_error')
    print("Loaded model from disk")
    pred_stats = loaded_model.evaluate(X_test, Y_test)
    print(pred_stats)
    pred = loaded_model.predict(X_test)
    pred = [p[0] for p in pred]
    with open("preds.pkl", "wb") as f:
        pickle.dump([names_test, X_test, Y_test, pred], f)

    '''
if __name__ == '__main__':
    main()
