#Neal Bayya
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
    def __init__(self, x_set, db_set, y_set, batch_size):
        self.x_set = x_set
        self.db_set = db_set
        self.y_set = y_set
        self.batch_size = batch_size
    def __len__(self):
        return int(np.ceil(len(self.x_set)/float(self.batch_size)))
    def __getitem__(self, idx):
        bx = self.x_set[idx*self.batch_size:(idx+1)*self.batch_size]
        bd = self.db_set[idx*self.batch_size:(idx+1)*self.batch_size]
        by = self.y_set[idx*self.batch_size:(idx+1)*self.batch_size]
        x0 = np.array([cv.imread(crops_path+p) for p in bx]) / 255.0
        d = np.array([[avgd] for avgd in bd])
        y = np.array(by)
        model_in = [x0, d]
        return model_in, y

def generate_crops(model_input, size=128, npatches=10):
    [rows, columns, nchannels] = model_input.shape
    crops = []
    for i in range(npatches):
        ul_row = random.randint(0,rows-size)
        ul_col = random.randint(0,columns-size)
        patch_input = model_input[ul_row:ul_row+size, ul_col: ul_col+size,:]
        crops.append(patch_input)
    return np.array(crops)

def crops_master(images, id2params, id2depth, cropsize, ncrops):
    modelio = open(path+'valcrops.txt', 'w')
    for img_num, i in enumerate(images):
        #Expected format of each image: SCENE_HAZE_LR
        scene = i[:i.index('_')]
        haze = i[i.index('_')+1 : i.rindex('_')]
        lr = i[i.rindex('_')+1 : i.index('.')]
        params = id2params[scene+'_'+haze]
        avgdepth = id2depth[scene+'_'+lr]
        img = cv.imread(image_path + i)
        img_crops = generate_crops(img, cropsize, ncrops)
        for c, img_crop in enumerate(img_crops):
            #Written format of each crop: SCENE_HAZE_LR_CROP
            cn = scene + "_" + haze + "_" + lr + "_" + str(c) + '.png'
            modelio.write(cn + ' ' + str(avgdepth) + ' ' + str(params[1]) + '\n' )#
            cv.imwrite(crops_path+cn, img_crop)
    modelio.close()

def extract_depth(scale = 1.0):
    depth_path  = "regression_data/depth/"
    scenes = [s for s in listdir(depth_path) if not s[0] in {'.', '_'}]
    img2depth = {}

    max_depth = 0
    maxd_img = ""
    min_depth = np.inf
    mind_img = ""
    for s in scenes:
        left_disp = cv.imread(depth_path+s+"/disp1.png").flatten() 
        left_disp = np.array([v/scale for v in left_disp if v != 0])

        right_disp = cv.imread(depth_path+s+"/disp5.png").flatten() 
        right_disp = np.array([v/scale for v in right_disp if v != 0])

        left_depth = 3740*0.016 / left_disp
        right_depth = 3740*0.016 / right_disp
        img2depth[s+'_0'] = np.mean(left_depth)
        img2depth[s+'_1'] = np.mean(right_depth)

        if np.amax(left_depth) > max_depth:
            max_depth = np.amax(left_depth)
            maxd_img = s+'_0'
        elif np.amax(right_depth) > max_depth:
            max_depth = np.amax(right_depth)
            maxd_img = s+'_1'

        if np.amin(left_depth) < min_depth:
            min_depth = np.amin(left_depth)
            mind_img = s+'_0'
        elif np.amin(right_depth) < min_depth:
            min_depth = np.amin(right_depth)
            mind_img = s+'_1'
    return img2depth

def create_model():
    init = Input(shape=(128, 128, 3))
    avg_depth = Input(shape=(1,))
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
    merged_data = concatenate([x, avg_depth])
    x = Dense(1)(merged_data)
    return Model(inputs = [init, avg_depth], outputs = x)

def create_model2(N=2,M=3):
    init = Input(shape=(128, 128, 3))
    avg_depth = Input(shape=(1,))
    x = init
    for i in range(M): #M controls number of times to downsample
        for j in range(N): #apply conv N times before downsample
            cj = Conv2D(32, (3,3), activation='relu', padding='same')(x)
            x = cj
        x = MaxPooling2D((2,2), padding='same')(x) #downsample
    x = Dropout(rate=0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(10, activation='relu')(x)
    merged_data = concatenate([x, avg_depth])
    x = Dense(1)(merged_data)
    return Model(inputs = [init, avg_depth], outputs = x)

def main():
    global path, image_path, crops_path 
    path = 'regression_data/'
    image_path = path + "HazyImages2/"
    crops_path = path + "crops2/"
    ncrops = 10
    cropsize = 128
    '''
    #generate crops
    id2params = {}
    for i, l in enumerate(open(path+'val2.txt', 'r')):
        if '?' not in l:
            continue
        split = l.split("?")
        img_id = split[0] + '_' + split[1]
        alpha = float(split[2])
        beta = float(split[3])
        id2params[img_id] = (alpha, beta)
    
    images = sorted([i for i in listdir(image_path) if not i.startswith(('.', '_'))])
    id2depth = extract_depth(3.0)
    crops_master(images, id2params, id2depth, cropsize, ncrops)
    '''
    
    #Read crops
    crop_names = []
    modelio = open(path + 'valcrops.txt', 'r').readlines()
    x, dbuffer, y = [],[],[]
    error_crops = []
    for c_num, l in enumerate(modelio):
        labeling = l.split(' ')
        p = labeling[0]
        try:
            tmp = np.array(cv.imread(crops_path+p)) / 255.0
            x.append(p)
            crop_names.append(p)
            dbuffer.append(float(labeling[1]))
            y.append(float(labeling[2]))
        except:
            error_crops.append(p)
    print("error parsing following crops: {}".format(error_crops))
    ncases = len(x)

    # ML
    random.seed(5)
    train_split = 2/3
    test_split = 1/6
    val_split = 1/6
    batch_size = 32
    MODEL_NAME = 'dbuffer2'
 
    model_json_name = "models/" + MODEL_NAME + ".json"
    model_weights_name = "models/" + MODEL_NAME + ".h5"
    model_predictions_name = "models/" + MODEL_NAME + "pred.npz"

    indices = [i for i in range(ncases)]
    random.shuffle(indices)
    split1 = int(train_split*ncases)
    split2 = int((train_split+test_split)*ncases)
    X_train = np.array([x[idx] for idx in indices[:split1]])
    D_train = np.array([dbuffer[idx] for idx in indices[:split1]])
    Y_train = np.array([y[idx] for idx in indices[:split1]])
    names_train = [crop_names[idx] for idx in indices[:split1]]
    X_test = np.array([x[idx] for idx in indices[split1:split2]])
    D_test = np.array([dbuffer[idx] for idx in indices[split1:split2]])
    Y_test = np.array([y[idx] for idx in indices[split1:split2]])
    names_test = [crop_names[idx] for idx in indices[split1:split2]]
    X_val = np.array([x[idx] for idx in indices[split2:]])
    D_val = np.array([dbuffer[idx] for idx in indices[split2:]])
    Y_val = np.array([y[idx] for idx in indices[split2:]])
    names_val = [crop_names[idx] for idx in indices[split2:]]

    print('X train shape: {}'.format(X_train.shape))
    print('D train shape: {}'.format(D_train.shape))
    print('Y train shape: {}'.format(Y_train.shape))
    
    print('X test shape: {}'.format(X_test.shape))
    print('D test shape: {}'.format(D_test.shape))
    print('Y test shape: {}'.format(Y_test.shape))
    
    print('X val shape: {}'.format(X_val.shape))
    print('D val shape: {}'.format(D_val.shape))
    print('Y val shape: {}'.format(Y_val.shape))
    
    #Training model
    '''
    model = create_model2()
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    trainGen = Generator(X_train, D_train, Y_train, batch_size)
    valGen = Generator(X_val, D_val, Y_val, batch_size)
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
    testGen = Generator(X_test, D_test, Y_test, batch_size)
    """pred_stats = loaded_model.evaluate_generator(testGen)
    print(pred_stats)
    pred = loaded_model.predict_generator(testGen)
    pred = [p[0] for p in pred]
    print("Predicted\tY_Test\tDiff")
    for ptest, ytest in zip(pred, Y_test):
        print("{}\t{}\t{}".format(ptest, ytest, ptest-ytest))
    """

                
                
if __name__ == '__main__':
    main()


