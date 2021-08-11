#!/usr/bin/env python
# coding: utf-8

# In[8]:


#!/usr/bin/env python
# coding: utf-8

import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import optparse, json, argparse, math
import ROOT
import time
import yaml
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import log_loss
from os import environ
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Reshape,Conv2D,MaxPooling2D,ConvLSTM2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from plotting.plotter import plotter
from root_numpy import root2array, tree2array
from sklearn.metrics import confusion_matrix
seed = 7
##small changes
np.random.seed(7)
rng = np.random.RandomState(31337)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',   action='store', required=True, help='Name of output varibles config')

args = parser.parse_args()
timestr=time.strftime("%Y%m%d-%H%M%S")

def plot_confusion_matrix(cm,
                          target_names,
                          output_name,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,#这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()
    #plt.savesig("test.png",dpi=350)
    plt.savefig("%s/confuse"%output_name+timestr+".pdf")

# 显示混淆矩阵
def plot_confuse(model, x_val, y_val, labels, output):
    predictions = model.predict_classes(x_val,batch_size=800)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False, target_names=labels, output_name=output, title='Confusion Matrix')

def GetYamlData(yaml_file):
    file = open(yaml_file, 'r')
    file_data = file.read()
    file.close()
    data = yaml.safe_load(file_data)
    return data

def LoadConfig():
    current_path = os.path.abspath(".")
    yaml_path_config = os.path.join(current_path, "%s"%args.config)
    data_config = GetYamlData(yaml_path_config)
    return data_config

def load_data(inputPath,variables,criteria,inputTree):
    # Load dataset to .csv format file
    my_cols_list=variables+['process', 'key', 'target']
    data = pd.DataFrame(columns=my_cols_list)
    keys=['bb','cc','gg','bkg1','bkg2']
    for key in keys :
        print(key)
        if 'bb' in key:
            sampleNames=['bb']          
            fileNames = ['output_2e2h_bb']
            target=0
        if 'cc' in key:
            sampleNames=['cc']
            fileNames = ['output_2e2h_cc']
            target=1
        if 'gg' in key:
            sampleNames=['gg']
            fileNames = ['output_2e2h_gg']
            target=2
        if 'bkg1' in key:
            sampleNames=['bkg1']
            fileNames = ['output_peakbackground']
            target=3
        if 'bkg2' in key:
            sampleNames=['bkg1']
            fileNames = ['output_contbackground']
            target=4
        #inputTree = '%s'%args.treename
        print(sampleNames)

        for process_index in range(len(fileNames)):
            fileName = fileNames[process_index]
            sampleName = sampleNames[process_index]

            try: tfile = ROOT.TFile(inputPath+"/"+fileName+".root")
            except :
                print(" file "+ inputPath+"/"+fileName+".root doesn't exits ")
                continue
            try: tree = tfile.Get(inputTree)
            except :
                print(inputTree + " deosn't exists in " + inputPath+"/"+fileName+".root")
                continue
            if tree is not None :
                print('criteria: ', criteria)
                #try: chunk_arr = tree2array(tree=tree, selection=criteria, start=0, stop=100) # Can use  start=first entry, stop = final entry desired
                try: chunk_arr = tree2array(tree=tree, selection=criteria) # Can use  start=first entry, stop = final entry desired
                except : continue
                else :
                    chunk_df = pd.DataFrame(chunk_arr, columns=variables)
                    chunk_df['process']=sampleName
                    chunk_df['key']=key
                    chunk_df['target']=target

                    data = data.append(chunk_df, ignore_index=True)

            tfile.Close()
        if len(data) == 0 : continue
        nbb = len(data.iloc[(data.target.values == 0) & (data.key.values==key) ])
        ncc = len(data.iloc[(data.target.values == 1) & (data.key.values==key) ])
        ngg = len(data.iloc[(data.target.values == 2) & (data.key.values==key) ])
        nbkg1 = len(data.iloc[(data.target.values == 3) & (data.key.values==key) ])
        nbkg2 = len(data.iloc[(data.target.values == 4) & (data.key.values==key) ])
        nOther00 = len(data.iloc[(data.target.values == 5) & (data.key.values==key) ])
        processfreq = data.groupby('key')
        samplefreq = data.groupby('process')

        if key == 'bb':
            print('Process bb frequency: ', len(processfreq.get_group('bb')))
        elif key == 'cc':
            print('Process cc frequency: ', len(processfreq.get_group('cc')))
        elif key == 'gg':
            print('Process gg frequency: ', len(processfreq.get_group('gg')))
        elif key == 'bkg1':
            print('Process bkg1 frequency: ', len(processfreq.get_group('bkg1')))
        elif key == 'bkg2':
            print('Process bkg2 frequency: ', len(processfreq.get_group('bkg2')))

        #print("TotalWeights = %f" % (data.iloc[(data.key.values==key)]["sampleWeight"].sum()))
        #nNW = len(data.iloc[(data["sampleWeight"].values < 0) & (data.key.values==key) ])
        #print(key, "events with -ve weights", nNW)

    print('<load_data> data columns: ', (data.columns.values.tolist()))
    n = len(data)
    nbb = len(data.iloc[data.target.values == 0])
    ncc = len(data.iloc[data.target.values == 1])
    ngg = len(data.iloc[data.target.values == 2])
    nbkg1 = len(data.iloc[data.target.values == 3])
    nbkg2 = len(data.iloc[data.target.values == 4])
    nOther = len(data.iloc[data.target.values == 5])
    print("Total length of nbb = %i, ncc = %i, ngg = %i, nbkg1 = %i, nbkg2 = %i" % (nbb, ncc, ngg, nbkg1, nbkg2))
    return data

def MakePlots(y_train, y_test, y_test_pred, y_train_pred, Wt_train, Wt_test, output):
    from sklearn.metrics import roc_curve, auc
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    figMVA, axesMVA = plt.subplots(2, 3, figsize=(30, 20))

    for i in range(5):
        print(i)
        if i==0:
            ax=axes[0,0]
            axMVA=axesMVA[0,0]
        if i==1:
            ax=axes[0,1]
            axMVA=axesMVA[0,1]
        if i==2:
            ax=axes[0,2]
            axMVA=axesMVA[0,2]
        if i==3:
            ax=axes[1,0]
            axMVA=axesMVA[1,0]
        if i==4:
            ax=axes[1,1]
            axMVA=axesMVA[1,1]
        nodename=["bb","cc","gg","bkg1","bkg2"]
    
        axMVA.hist([y_test_pred[:, i][(y_test[:, 0]==1)],
                    y_test_pred[:, i][(y_test[:, 1]==1)],
                    y_test_pred[:, i][(y_test[:, 2]==1)],
                    y_test_pred[:, i][(y_test[:, 3]==1)],
                    y_test_pred[:, i][(y_test[:, 4]==1)]],bins=np.linspace(0, 1, 21),label=["bb","cc","gg","bkg1","bkg2"],
                   weights=[Wt_test[(y_test[:, 0]==1)]/np.sum(Wt_test[(y_test[:, 0]==1)]),
                            Wt_test[(y_test[:, 1]==1)]/np.sum(Wt_test[(y_test[:, 1]==1)]),
                            Wt_test[(y_test[:, 2]==1)]/np.sum(Wt_test[(y_test[:, 2]==1)]),
                            Wt_test[(y_test[:, 3]==1)]/np.sum(Wt_test[(y_test[:, 3]==1)]),
                            Wt_test[(y_test[:, 4]==1)]/np.sum(Wt_test[(y_test[:, 4]==1)])],
                    histtype='step',linewidth=4,color=['red','green','blue','yellow','black'])

        axMVA.hist([y_train_pred[:, i][(y_train[:, 0]==1)],
                    y_train_pred[:, i][(y_train[:, 1]==1)],
                    y_train_pred[:, i][(y_train[:, 2]==1)],
                    y_train_pred[:, i][(y_train[:, 3]==1)],
                    y_train_pred[:, i][(y_train[:, 4]==1)]],bins=np.linspace(0, 1, 21),label=["bb_train","cc_train","gg_train","bkg1_train","bkg2_train"],
                   weights=[Wt_train[(y_train[:, 0]==1)]/np.sum(Wt_train[(y_train[:, 0]==1)]),
                            Wt_train[(y_train[:, 1]==1)]/np.sum(Wt_train[(y_train[:, 1]==1)]),
                            Wt_train[(y_train[:, 2]==1)]/np.sum(Wt_train[(y_train[:, 2]==1)]),
                            Wt_train[(y_train[:, 3]==1)]/np.sum(Wt_train[(y_train[:, 3]==1)]),
                            Wt_train[(y_train[:, 4]==1)]/np.sum(Wt_train[(y_train[:, 4]==1)])],
                    histtype='stepfilled',alpha=0.2,linewidth=1,color=['red','green','blue','yellow','black'])

        axMVA.set_title('MVA: Node '+str(nodename[i]),fontsize=20)
        axMVA.legend(loc="upper right",fontsize=10)
        #axMVA.set_ylim([0, 1])
        axMVA.set_xlim([0, 1])
    
        
        fpr, tpr, th = roc_curve(y_test[:, i], y_test_pred[:, i],sample_weight=Wt_test)
        fpr_tr, tpr_tr, th_tr = roc_curve(y_train[:, i], y_train_pred[:, i],sample_weight=Wt_train)
        
    
        roc_auc = auc(fpr, tpr)
        roc_auc_tr = auc(fpr_tr, tpr_tr)
    
        ax.plot(tpr, 1-fpr, label='ROC curve test (area = %0.2f)' % roc_auc,linewidth=4)
        ax.plot(tpr_tr, 1-fpr_tr, label='ROC curve train (area = %0.2f)' % roc_auc_tr,linewidth=4)
        #plt.plot([0, 1], [0, 1], 'k--')
        #ax.set_xlim([0.8, 1.0])
        #ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Signal efficiency',fontsize=20)
        ax.set_ylabel('Background rejection',fontsize=20)
        ax.set_title('ROC: Node '+str(i+1),fontsize=20)
        #ax.set_yscale("log", nonposy='clip')
        ax.legend(loc="lower left",fontsize=10)

    #axes[1,2].set_axis_off()
    axesMVA[1,2].set_axis_off()
    print(type(output), output)
    fig.savefig("%s/ROC"%output+timestr+".pdf")
    figMVA.savefig("%s/output"%output+timestr+".pdf")


# In[2]:


def load_trained_model(weights_path, num_variables, optimizer,nClasses):
    model = baseline_model(num_variables, optimizer,nClasses)
    model.load_weights(weights_path)
    return model

def normalise(x_train, x_test):
    mu = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_normalised = (x_train - mu) / std
    x_test_normalised = (x_test - mu) / std
    return x_train_normalised, x_test_normalised

def baseline_model(num_variables,optimizer,nClasses):
    model = Sequential()
    model.add(Dense(64,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    for index in range(0):
        model.add(Dense(64,activation='tanh'))
        model.add(Dropout(0.1))
    for index in range(0):
        model.add(Dense(32,activation='tanh'))
        model.add(Dropout(0.1))
    for index in range(0):
        model.add(Dense(16,activation='relu'))
        model.add(Dropout(0.1))
    model.add(Dense(nClasses, activation='softmax'))
    if optimizer=='Adam':
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.01),metrics=['acc'])
    if optimizer=='Nadam':
        model.compile(loss='categorical_crossentropy',optimizer=Nadam(lr=0.01),metrics=['acc'])
    return model

def check_dir(dir):
    if not os.path.exists(dir):
        print('mkdir: ', dir)
        os.makedirs(dir)

# Ratio always > 1. mu use in natural log multiplied into ratio. Keep mu above 1 to avoid class weights going negative.
def create_class_weight(labels_dict,mu=0.9):
    total = np.sum(list(labels_dict.values())) # total number of examples in all datasets
    keys = list(labels_dict.keys()) # labels
    class_weight = dict()
    print('total: ', total)

    for key in keys:
        # logarithm smooths the weights for very imbalanced classes.
        score = math.log(mu*total/float(labels_dict[key])) # natlog(parameter * total number of examples / number of examples for given label)
        #score = float(total/labels_dict[key])
        print('score = ', score)
        if score > 0.:
            class_weight[key] = score
        else :
            class_weight[key] = 1.
    return class_weight

# Convert the nan to zero in dataframe


def main():

    print('Using Keras version: ', tf.keras.__version__)
    
    ConfigData = LoadConfig()
    
    do_model_fit = 1
    
    number_of_classes = 5
    
    output_directory = ConfigData['output']
    
    check_dir(output_directory)
    
    selection_criteria1 = 'Entry$%2==0'
    selection_criteria2 = 'Entry$%2!=0'
    
    #Before split
    variable_list1 = ConfigData['variable'].split()
    
    #After split
    variable_list = ConfigData['variable'].split()
    
    print('DEBUG: ', variable_list)
    # Create list of headers for dataset .csv
    column_headers = []
    column_headers1 = []
    for key in variable_list:
        column_headers.append(key)
    for key in variable_list1:
        column_headers1.append(key)

    column_headers.append('weight')
    column_headers1.append('weight')
    print('DEBUG: ', column_headers1)
        
    inputs_file_path = ConfigData['input']
   
    treename = ConfigData['tree'] 
    
    # Load ttree into .csv including all variables listed in column_headers
    print('<train-DNN> Input file path: ', inputs_file_path)
    #outputdataframe_name = '%s/output_dataframe_%s.csv' %(output_directory,selection)
    outputdataframe_nametr = '%s/output_dataframe_tr.csv' %(output_directory)
    outputdataframe_namete = '%s/output_dataframe_te.csv' %(output_directory)
    
    print('<train-DNN> Creating new data .csv @: %s . . . . ' % (inputs_file_path))
    datatr = load_data(inputs_file_path, column_headers1, selection_criteria1, treename)
    datate = load_data(inputs_file_path, column_headers1, selection_criteria2, treename)
    
    print(datatr.head())
    
    datatr.fillna(0, inplace = True)
    datate.fillna(0, inplace = True)
    
    # In[4]:
    
    #Extra constant
    
    # Make instance of plotter tool
    Plotter = plotter()
    
    # Create statistically independant lists train/test data (used to train/evaluate the network)
    #traindataset, valdataset = train_test_split(data, test_size=0.2)
    traindataset =datatr.copy()
    valdataset=datate.copy()
    #valdataset.to_csv('valid_dataset.csv', index=False)
    training_columns = variable_list1
    #del training_columns[-4]
    print('<train-DNN> Training features: ', training_columns)
    
    # Select data from columns under the remaining column headers in traindataset
    X_train = traindataset[training_columns].astype('float32')
    print("Training numbers:"+str(len(traindataset)))
    print("Training numbers:"+str(len(traindataset[training_columns])))
    print("Training numbers:"+str(len(X_train)))
    Y_train = traindataset.target.astype(int)
    X_test = valdataset[training_columns].astype('float32')
    Y_test = valdataset.target.astype(int)
    
    num_variables = len(training_columns)

    ####################
    trainweights = traindataset.loc[:,'weight'] #Norm weight x xsec wt x (cpscaleweight)
    trainweights = np.array(trainweights)
    
    testweights = valdataset.loc[:,'weight']
    testweights = np.array(testweights)
    
    train_weights = traindataset['weight'].values
    test_weights = valdataset['weight'].values
    ###################
    inweightsROC = traindataset.loc[:,'weight'] #Norm weight
    trainweightsROC = np.array(trainweights)
    
    testweightsROC = valdataset.loc[:,'weight']
    testweightsROC = np.array(testweights)
    
    train_weightsROC = traindataset['weight'].values
    test_weightsROC = valdataset['weight'].values
    ###################

    # Fit label encoder to Y_train
    newencoder = LabelEncoder()
    newencoder.fit(Y_train)
    # Transform to encoded array
    encoded_Y = newencoder.transform(Y_train)
    encoded_Y_test = newencoder.transform(Y_test)
    # Transform to one hot encoded arrays
    # Y_train = np_utils.to_categorical(encoded_Y)
    # Y_test = np_utils.to_categorical(encoded_Y_test)
    Y_train = to_categorical(encoded_Y)
    Y_test = to_categorical(encoded_Y_test)
    #optimizer = 'Nadam'
    optimizer = 'Adam'#'Nadam'
    if do_model_fit == 1:
        histories = []
        labels = []
        # Define model and early stopping                                                                                                         
        early_stopping_monitor = EarlyStopping(patience=20,monitor='val_loss',verbose=1)
        model3 = baseline_model(num_variables,optimizer,number_of_classes)
        #model3 = newCNN_model(num_variables,optimizer,number_of_classes,1000,0.40)
        history3 = model3.fit(X_train,Y_train,validation_data=(X_test, Y_test, testweights), epochs=200, batch_size=800, verbose=1, shuffle=True, sample_weight=trainweights)
        #history3 = model3.fit(X_train,Y_train,validation_data=(X_test, Y_test, testweights), epochs=5, batch_size=800, verbose=1, shuffle=True, sample_weight=trainweights)
        histories.append(history3)
        labels.append(optimizer)
    
        # Make plot of loss function evolution                                                                                               
        Plotter.plot_training_progress_acc(histories, labels)
        acc_progress_filename = 'DNN_loss'+timestr+'.pdf'
        Plotter.save_plots(dir=output_directory, filename=acc_progress_filename)
        # Which model do you want the rest of the plots for?                                                                                         
        model = model3
    else:
        # Which model do you want to load?                  
        model_name = os.path.join(output_directory,'model.h5')
        print('<train-DNN> Loaded Model: %s' % (model_name))
        model = load_trained_model(model_name,num_variables,optimizer,number_of_classes)
    # Node probabilities for training sample events
    result_probs = model.predict(np.array(X_train))
    result_classes = model.predict_classes(np.array(X_train))
    
    # Node probabilities for testing sample events     
    result_probs_test = model.predict(np.array(X_test))
    result_classes_test = model.predict_classes(np.array(X_test))
    # Store model in file                                        
    model_output_name = os.path.join(output_directory,'model.h5')
    model.save(model_output_name)
    weights_output_name = os.path.join(output_directory,'model_weights.h5')
    model.save_weights(weights_output_name)
    model_json = model.to_json()
    model_json_name = os.path.join(output_directory,'model_serialised.json')
    with open(model_json_name,'w') as json_file:
        json_file.write(model_json)
    model.summary()
    model_schematic_name = os.path.join(output_directory,'model_schematic.pdf')
    #plot_model(model, to_file=model_schematic_name, show_shapes=True, show_layer_names=True)
   
    labels = ['bb', 'cc', 'gg', 'bkg1', 'bkg2']
 
    # Initialise output directory.     
    Plotter.output_directory = output_directory
    
    MakePlots(y_train=Y_train, y_test=Y_test, y_test_pred=result_probs_test, y_train_pred=result_probs, Wt_train=train_weightsROC, Wt_test=test_weightsROC, output=output_directory)
    #MakePlots(y_train=Y_train, y_test=Y_test, y_test_pred=result_probs_test, y_train_pred=result_probs, Wt_train=train_weightsROC, Wt_test=test_weightsROC)
    plot_confuse(model, X_test, Y_test, labels, output_directory)


# In[ ]:

if __name__ == '__main__':
    main()	
