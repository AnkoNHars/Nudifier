from __future__ import division
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import *
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
#import lightgbm as lgb
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import os
from PIL import Image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences, make_sampling_table, skipgrams
sns.set()

# fonctions utiles
def add_conv(y, filters, kernel_size = (3,3), conv_strides = (2,2), pool_strides = False, batch_norm = False, dropout = False, input_shape = None, activation = 'lrelu', gnoise = False):
    y = Conv2D(int(filters), kernel_size, strides=conv_strides, padding='same')(y)
    if batch_norm:  y = BatchNormalization()(y)
    if gnoise: y = GaussianNoise(gnoise)(y)
    if activation != None and activation != 'lrelu': y = Activation(activation)(y)
    elif activation == 'lrelu': y = LeakyReLU(alpha=0.2)(y)
    if dropout: y = Dropout(dropout)(y)
    if pool_strides: y = MaxPooling2D(strides = pool_strides)(y)
    return y


def filter_gen(n, reverse = False, base = 32):
    flt = base
    filters = []
    for x in range(n):
        filters.append(flt)
        flt *= 2
    if reverse: filters.reverse()
    return filters

def add_res_block(y, filters, kernel_size = (3,3), strides = (1,1),**kwargs):
    shortcut = y
    y = add_conv(y, filters, kernel_size = kernel_size, conv_strides = strides, pool_strides = False, dropout = False, **kwargs)
    y = add_conv(y, filters, kernel_size = kernel_size, conv_strides = strides, pool_strides = False, dropout = False, activation = None, **kwargs)
    y = Add()([shortcut, y])
    y = Activation('relu')(y)
    return y


def add_deconv(y, filters, kernel_size = (2,2), strides = (2,2), batch_norm = False, activation = 'lrelu', dropout = False, gnoise = False):
    y = Conv2DTranspose(filters,kernel_size = kernel_size, strides = strides)(y)
    if batch_norm: y = BatchNormalization()(y)
    if gnoise: y = GaussianNoise(gnoise)(y)
    if activation != None and activation != 'lrelu': y = Activation(activation)(y)
    elif activation == 'lrelu': y = LeakyReLU(alpha=0.2)(y)
    if dropout: y = Dropout(dropout)(y)
    return y 

def add_deconv1d(y, filters, prev_filters, kernel_size = 3, conv_strides = 2, batch_norm = True,
                 dropout = False, activation = 'lrelu'):
    
    y = Reshape((-1,1,prev_filters))(y)
    y = Conv2DTranspose(filters = filters, kernel_size = (kernel_size,1), strides = 
                        (conv_strides, 1))(y)
    if batch_norm: y = BatchNormalization()(y)
    if activation != None and activation != 'lrelu': y = Activation(activation)(y)
    elif activation == 'lrelu': y = LeakyReLU(alpha=0.2)(y)
    if dropout: y = Dropout(dropout)(y)
    y = Reshape((-1, filters))(y)
    return y 

def add_res_block1d(y, filters, **kwargs):
    shortcut = y
    y = add_conv1d(y, filters, pool_strides= False, **kwargs)
    y = add_conv1d(y, filters, pool_strides = False, activation = None, **kwargs)
    y = Add()([shortcut, y])
    y = Activation('relu')(y)
    return y

def add_conv1d(y, filters, kernel_size = 3, conv_strides = 1, pool_strides = 2, batch_norm = True, dropout = False, input_shape = None, activation = 'lrelu'):
    
    """Add convolution1D + batch norm + act + dropout to model"""
    y = Conv1D(int(filters), kernel_size, strides=conv_strides, padding='same')(y)
    if batch_norm:  y = BatchNormalization()(y)
    if activation != None and activation != 'lrelu': y = Activation(activation)(y)
    elif activation == 'lrelu': y = LeakyReLU(alpha=0.2)(y)
    if dropout: y = Dropout(dropout)(y)
    if pool_strides: y = MaxPooling1D(strides = pool_strides)(y)
    return y


def add_dense(y, units, batch_norm = True, dropout = 0.4, act = 'lrelu'):
    """ Add dense layer + bath norm + act + dropout to model"""
    y = Dense(units)(y)
    if batch_norm:  y = BatchNormalization()(y)
    if act!= None and act != 'lrelu': y = Activation(act)(y)
    else: y = LeakyReLU(alpha=0.2)(y)
    if dropout > 0: y = Dropout(dropout)(y)
    return y

def get_config(params={}):
    d_params = {'default_act':'lrelu',
                   'hidden_layers':4,
                   'batch_norm':False,
                   'dropout':0.4,
                   'input_shape':None,
                   'output_shape':None,
                   'optimizer':'adam',
                   'loss_func':'mse',
                   'metrics':[],
                   'print':True
                   }
    for p in params.keys():
        d_params[p] = params[p]
    return d_params

def get_fully_connected(params={}):
    d_params = {'out_func':'softmax',
                'batch_norm':True,
                'hidden_layers':4,
                'units': [512,256,128,64],
                'loss_func':'binary_crossentropy',
                'metrics': ['accuracy']
               }
    for p in params.keys():
        d_params[p] = params[p]
        
    params = get_config(d_params)
    inp = Input(shape=params['input_shape'])
    y = inp
    for l in range(params['hidden_layers']):
        y = add_dense(y, params['units'][l], params['batch_norm'], params['dropout'],
                      params['default_act'])
    out = add_dense(y, params['output_shape'], False, False, params['out_func'])
    model = Model(inputs = inp, outputs = out)
    model.compile(loss = params['loss_func'],optimizer = params['optimizer'],
                  metrics = params['metrics'])
    if params['print'] : print(model.summary())
    return model

def get_convolutional_2d(params={}):
    d_params = {'out_func': 'softmax',
                'hidden_layers':4,
                'filters':[32,64,264,512],
                'res_blocks':0,
                'metrics':['accuracy'],
                'loss_func':'categorical_crossentropy',
                'kernel_size':(3,3),
                'conv_strides':(2,2),
                'pool_strides': False,
                'dropout':False,
                'res_filters':64
               }
    for p in params.keys():
        d_params[p] = params[p]
        
    params = get_config(d_params)
    
    inp = Input(shape=params['input_shape'])
    y = inp
    for x in range(params['hidden_layers']):
        y = add_conv(y, params['filters'][x], params['kernel_size'], params['conv_strides'],params['pool_strides'], params['batch_norm'],
                     params['dropout'], None, params['default_act'])
    for x in range(params['res_blocks']):
        y = add_res_block(y, params['res_filters'], params['kernel_size'])
    y = Flatten()(y)
    out = add_dense(y, params['output_shape'], False, False, params['out_func'])
    model = Model(inputs = inp, outputs = out)
    model.compile(loss = params['loss_func'],optimizer = params['optimizer'],
                  metrics = params['metrics'])
    if params['print'] : print(model.summary())
    return model

def get_siamois_1d(params={}, return_encoder = False):
    d_params = {'hidden_convs':1,
                'hidden_denses':2,
                'units':[256,128],
                'filters':[64],
                'metrics':['accuracy'],
                'out_func':'softmax',
                'loss_func':'categorical_crossentropy',
                'conv_strides':2,
                'kernels':[2,4,6],
                'conv_dropout':False,
                'dense_dropout':0.2,
                'pool_strides':False,
                'embedding':True,
                'embedding_vectors':8
               }
    for p in params.keys():
        d_params[p] = params[p]
        
    params = get_config(d_params)
    
    liste_out = []
    
    inp = Input(shape=params['input_shape'])
    #emb = Embedding(1, params['embedding_vectors'],input_length=params['input_shape'])(inp)
    shape = list(params['input_shape']) + [1]
    emb = Reshape(shape)(inp)
    for i,ker in enumerate(params['kernels']):
        y = emb
        for x in range(params['hidden_convs']):
            y = add_conv1d(y, params['filters'][x], params['kernels'][i], params['conv_strides'], params['pool_strides'],
                         params['batch_norm'], params['dropout'],None,params['default_act'])
            
        y = Flatten()(y)
        liste_out.append(y)
    
    y = concatenate(liste_out)
    #encoded = y
    for x in range(params['hidden_denses']):
        if x == 1: encoded = y
        y = add_dense(y, params['units'][x], params['batch_norm'], params['dense_dropout'],params['default_act'])
    
    out = add_dense(y, params['output_shape'], False, False, params['out_func'])
    model = Model(inputs = inp, outputs = out)
    model.compile(loss = params['loss_func'],optimizer = params['optimizer'],
                  metrics = params['metrics'])
    if params['print'] : print(model.summary())
    if not return_encoder:
        return model
    else:
        encoder= Model(inputs = inp, outputs = encoded)
        return model, encoder
    
    
              
def fully_connected_grid_search(input_shape, output_shape, params = {}):
    o_optimizers = ['rmsprop','adam']
    o_epochs = [5,10]
    o_batches = [16,32]
    o_layers = [4,6]
    o_units = [[512,256,128,64,32,16,8],[64,64,64,64,64,64,64,64]]
    o_batch_norm = [True,False]
    o_dropout = [False,0.2,0.8]
    o_act = ['lrelu','relu','elu']
    param_grid = dict(optimizer = o_optimizers, epochs = o_epochs, batch_size = o_batches,
                     layers = o_layers, units = o_units, act = o_act, batch_norm = o_batch_norm)

    def _flc(optimizer = 'adam', layers=4, act = 'lrelu',
             batch_norm = True, units = [512,256,128,64,32,16,8],dropout=0.4):
        d_params = {'input_shape':input_shape,
              'output_shape':output_shape,
              'optimizer':optimizer,
              'hidden_layers':layers,
              'default_act':act,
              'batch_norm':batch_norm,
              'units':units,
              'dropout':dropout
             }
        for p in params.keys(): d_params[p] = params[p]
        return get_fully_connected(d_params)
    model = KerasClassifier(build_fn = _flc, verbose = 0)
    grid = GridSearchCV(estimator = model, param_grid = param_grid)
    return grid

def lgbm_grid_search(X_train, y_train, X_test, y_test,params={}):
    d_params = {'objectives': ['binary'],
              'num_leaves':[26,27,28],
              'max_bin':[150,175],
              'learning_rate':[0.1,0.09,0.08],
              'boosting':['gbdt'],
              'num_iterations':[100,125],
              #'max_depth': [-1, 25,100,250],
              #'feature_fraction':[0.5,0.66,1.0],
              'reg_alpha': [0.0,1.0],
              'reg_lambda':[0.0,1.0],
              #'min_gain_to_split':[0.0, 0.25,0.5,1],
              #'min_data_in_leaf':[10,20,30],
              #'min_child_weight':[0.001,0.01,0.1,1],
              #'scoring' :['accuracy']
             }
    for p in params.keys():
        d_params[p] = params[p]
    lgbm = lgb.LGBMClassifier()
    grid = GridSearchCV(estimator=lgbm, param_grid = d_params)
    grid.fit(X_train, y_train, verbose=1)
    print(grid.best_score_)
    print(grid.best_params_)
    lgbm = lgb.LGBMClassifier(**grid.best_params_)
    lgbm.fit(X_train, y_train, verbose=1)
    print('{0:.2f} %'.format(lgbm.score(X_test, y_test) * 100))
    return grid
              
def rfc_grid_search(X_train, y_train, X_test, y_test,params={}):
    d_params = { 'n_estimators': [150],
                'max_features': [None],
                'max_depth' : [None],
                'criterion' :['entropy'],
                'min_samples_split':[2],
                'min_samples_leaf':[6]
                }
    for p in params.keys():
        d_params[p] = params[p]
    RFC= RandomForestClassifier()
    grid = GridSearchCV(estimator=RFC, param_grid = d_params)
    grid.fit(X_train, y_train)
    print(grid.best_score_)
    print(grid.best_params_)
    RFC = RandomForestClassifier(**grid.best_params_)
    RFC.fit(X_train, y_train)
    print('{0:.2f} %'.format(RFC.score(X_test, y_test) * 100))
    return grid
    

    
   
                       
        

def getColByType(df): 
    """returns a dict of col sorted by type """
    g = df.columns.to_series().groupby(df.dtypes).groups
    x = {k.name: v for k, v in g.items()}
    return x

def tell_me_all_about(Serie):
    """ prints a top unique elmts + describe method of the serie"""
    print(sorted(Serie.unique()))
    print(Serie.describe())
    
def plot_confusion_matrix(y_pred, y_true, classes,
                          normalize=False,
                          title='Confusion matrix',cmap=plt.cm.Blues):
    from sklearn.metrics import confusion_matrix
    import itertools
    
    matseq = confusion_matrix(y_pred, y_true)
    row_sums = matseq.sum(axis=1, dtype=float)
    cm = 100 * matseq / row_sums[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:.2f} %'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    
def plot_roc(model, X_test, y_test, figsize=(6,5)):
    """ 
    plot ROC AUC using sklearn functions 
    
    
    """
    y_score = model.predict_proba(X_test)
    y_true = np.array(pd.get_dummies(y_test))
    fpr = []
    tpr = []
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_true[:,0], y_score[:,0])
    roc_auc= auc(fpr, tpr)
    plt.figure(figsize=figsize)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.03, 1.0])
    plt.ylim([-0.03, 1.05])
    plt.plot([0,1e-6,1], [0,1,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbe ROC | 50/50')
    plt.legend(loc="lower right")
    plt.show()

    
def plotFeatImpo(mod, columns, figsize=(20, 5)) : 
    """Plot Features Importance, given the columns of the database in format Index or List"""
    importance = mod.feature_importances_
    m = max(importance)
    lab = tuple(columns)
    #sns.set()
    plt.figure(figsize=figsize)
    plt.plot(importance, "-o", markersize=12)
    plt.xticks(np.arange(len(lab)), (lab), rotation=90, fontsize=12)
    plt.yticks(np.arange(0, m+0.02, 0.02))
    #plt.savefig('AttributsImportants.png')
    plt.show()

def get_data_balance(data):
    known_classes = {}
    for d in data:
        if str(d) in known_classes.keys():
            known_classes[str(d)] += 1
        else: 
            known_classes[str(d)] = 1
    total = sum(known_classes.values())
    dic = {list(known_classes.keys())[x]: str(round(list(known_classes.values())[x] / total *100, 2))+'%' for x in range(len(list(known_classes.keys())))}
    return dic

def balance_data(X,y):
    print(get_data_balance(y))
    known_classes = {}
    for d in y:
        if str(d) in known_classes.keys():
            known_classes[str(d)] += 1
        else: 
            known_classes[str(d)] = 1
    toRemove = {known_classes.keys()[x]: known_classes.values()[x] - min(list(known_classes.values())) for x in range(len(list(known_classes.keys())))}
    print(toRemove)
    new_X = []
    new_y = []
    for x in range(X.shape[0]):
        if toRemove[str(y[x])] > 0: toRemove[str(y[x])] -= 1
        else: 
            new_X.append(X[x])
            new_y.append(y[x])
    
    new_X = np.asarray(new_X)
    new_y = np.asarray(new_y)
    print(get_data_balance(new_y))
    print(new_y.shape)
    return new_X, new_y

def getLastFromDir(path):
    i = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            try:
                i.append(int(file.split('.')[0]))
            except: pass
    i.sort()
    try:
        last = i[-1]
    except: last = 0
    return last

class ImageIterator():
    def __init__(self, path, batch_size=32, shuffle = False, ext = '.png'):
        self.batch_size = batch_size
        self.path = path
        self.shuffle = shuffle
        self._n = 0
        
def show_array(array):
    return Image.fromarray(np.round(array).astype('uint8'), 'RGB')

def show_encoded_array(array):
    array = array * 127.5 + 127.5
    return show_array(array)

def tokenize_data(X,vocab_size, max_words, tokenizer = None, split = ' ', savePath = None, pad_to_max_length = False):
    if tokenizer == None:
        tokenizer = Tokenizer(num_words=vocab_size, split=split, )
        tokenizer.fit_on_texts(X)
    elif type(tokenizer) == str:
        tokenizer = load_tokenizer(tokenizer)
    X = tokenizer.texts_to_sequences(X)
    if savePath!= None:
        with open(savePath, 'wb') as f:
            pickle.dump(tokenizer, f)
    if pad_to_max_length: X.append(np.zeros(max_words))
    X = pad_sequences(X, maxlen = max_words, padding='post')
    if pad_to_max_length: X.remove(np.zeros(max_words))
    return X, tokenizer
class SimilarityCallback:
    def __init__(self, vocab_map, valid_model, vocab_size):
        self.vocab_map = vocab_map
        self.vocab_map[0] = 'UNK'
        self.valid_model = valid_model
        self.vocab_size = int(vocab_size/10)
    def run_sim(self):
        valid_examples = np.random.choice(self.vocab_size, 5, replace=False)
        for i in range(5):
            valid_word = self.vocab_map[valid_examples[i]]
            top_k = 5  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = self.vocab_map[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    def _get_sim(self,valid_word_idx):
        sim = np.zeros((self.vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(self.vocab_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = self.valid_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

def vectorize_from_dic(seqs, vec_dic):
    sfr = []
    for seq in seqs:
        ns = []
        for tk in seq:
            ns.append(vec_dic[tk])
        sfr.append(ns)
    return np.asarray(sfr)
def get_sim_model(vocab_size, X, vocab_map, vector_dim=64, save_path = 'sim_model.ckpt'):
    emb = Embedding(input_dim = vocab_size, output_dim = vector_dim, input_length=1)
    word_input = Input(shape=(1,))
    context_input = Input(shape=(1,))
    word = emb(word_input)
    context = emb(context_input)
    vectorizer = Model(inputs = word_input, outputs = word)
    similarity = dot([word, context], axes=2, normalize=True)
    print(similarity.shape)
    sim_model = Model(inputs=[word_input, context_input], outputs=similarity)
    merged = dot([word, context], axes=0, normalize=False)
    merged = Flatten()(merged)
    output = Dense(1, activation = 'sigmoid',)(merged)
    model = Model(inputs = [word_input, context_input], outputs=output)
    print(model.summary())
    model.compile(loss='binary_crossentropy',optimizer='adam')
    print('Trainning embedding...')
    sim_cb = SimilarityCallback(vocab_map, sim_model, vocab_size)
    for e in range(25):
        sampling_table = make_sampling_table(vocab_size)    
        couples, labels = skipgrams(X.flatten(),vocab_size,window_size = 3, sampling_table = sampling_table)
        word_target, word_context = zip(*couples)
        word_target = np.array(word_target, dtype="int32")
        word_context = np.array(word_context, dtype="int32")
        
        for x in range(3):
            model.fit([word_target, word_context],labels, epochs = 1, verbose = 1, batch_size = 2048,
              shuffle = True, validation_split = 0.1)
        if e%10 == 0:
            sim_cb.run_sim()
            sim_model.save(save_path)
            vectorizer.save('word2{}vec.ckpt'.format(vector_dim))

    #sim_cb.run_sim()
    return sim_model, vectorizer

def token_to_text(array, tokenizer):
    try: reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    except: reverse_word_map = tokenizer
    reverse_word_map[0] = 'END'
    array = [reverse_word_map[int(x)] for x in array]
    array = ''.join(str(x)+' ' for x in array).split('END')[0]
    return array

class TokenList:
    def __init__(self, token_list):
        self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>'] + token_list
        self.t2id = {v:k for k,v in enumerate(self.id2t)}
    def id(self, x):	return self.t2id.get(x, 1)
    def token(self, x):	return self.id2t[x]
    def num(self):		return len(self.id2t)
    def startid(self):  return 2
    def endid(self):    return 3

def pad_to_longest(xs, tokens, max_len=999):
    longest = min(len(max(xs, key=len))+2, max_len)
    X = np.zeros((len(xs), longest), dtype='int32')
    X[:,0] = tokens.startid()
    for i, x in enumerate(xs):
        x = x[:max_len-2]
        for j, z in enumerate(x):
            X[i,1+j] = tokens.id(z)
        X[i,1+len(x)] = tokens.endid()
    return X

def LoadCSVgOLD(fn):
    with open(fn, encoding='utf-8') as fin:
        for line in fin:
            lln = line.rstrip('\r\n').split('\t')
            yield lln
            
def LoadList(fn):
    with open(fn, encoding = "utf-8") as fin:
        st = list(ll for ll in fin.read().split('\n') if ll != "")
    return st

            
def MakeS2SDict(fn=None, min_freq=5, delimiter=' ', dict_file=None):
    if dict_file is not None and os.path.exists(dict_file):
        print('loading', dict_file)
        lst = LoadList(dict_file)
        midpos = lst.index('<@@@>')
        itokens = TokenList(lst[:midpos])
        otokens = TokenList(lst[midpos+1:])
        return itokens, otokens
    data = LoadCSV(fn)
    wdicts = [{}, {}]
    for ss in data:
        for seq, wd in zip(ss, wdicts):
            #print(seq)
            for w in seq.split(delimiter): 
                wd[w] = wd.get(w, 0) + 1
    wlists = []
    for wd in wdicts:	
        wd = ljqpy.FreqDict2List(wd)
        wlist = [x for x,y in wd if y >= min_freq]
        wlists.append(wlist)
    print('seq 1 words:', len(wlists[0]))
    print('seq 2 words:', len(wlists[1]))
    itokens = TokenList(wlists[0])
    otokens = TokenList(wlists[1])
    if dict_file is not None:
        SaveList(wlists[0]+['<@@@>']+wlists[1], dict_file)
    return itokens, otokens

def MakeS2SData(fn=None, itokens=None, otokens=None, delimiter=' ', h5_file=None, max_len=200):
    if h5_file is not None and os.path.exists(h5_file):
        print('loading', h5_file)
        with h5py.File(h5_file) as dfile:
            X, Y = dfile['X'][:], dfile['Y'][:]
        return X, Y
    data = ljqpy.LoadCSVgOLD(fn)
    Xs = [[], []]
    for ss in data:
        if len(ss) < 2: continue
        for seq, xs in zip(ss, Xs):
            xs.append(list(seq.split(delimiter)))
    X, Y = pad_to_longest(Xs[0], itokens, max_len), pad_to_longest(Xs[1], otokens, max_len)
    if h5_file is not None:
        with h5py.File(h5_file, 'w') as dfile:
            dfile.create_dataset('X', data=X)
            dfile.create_dataset('Y', data=Y)
    return X, Y