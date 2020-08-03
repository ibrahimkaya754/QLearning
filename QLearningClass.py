"""
Created on Wed Nov  8 15:13:04 2017
@author: ikaya
"""
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append('/media/veracrypt1/DigitalTwin/modules2import/')
from import_modules import *
from helper_functions import *
from keras.losses import mean_squared_error

print('Done!')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class neuralnet():
    def __init__(self, numberofstate, numberofaction, 
                 activation_func, trainable_layer, initializer,
                 list_nn, load_saved_model, numberofmodels):
        
        self.activation_func  = activation_func
        self.trainable_layer  = trainable_layer
        self.init             = initializer
        self.opt              = Adam(lr=0.001)
        self.regularization   = 0.0
        self.description      = ''
        self.numberofstate    = numberofstate
        self.numberofaction   = numberofaction
        self.list_nn          = list_nn
        self.load_saved_model = load_saved_model
        self.total_layer_no   = len(self.list_nn)+1
        self.numberofmodels   = numberofmodels
        self.loss             = mean_squared_error
        self.model            = {}
        self.input            = Input(shape=(self.numberofstate,), name='states')

        for ii in range(self.numberofmodels):
            model_name = 'model'+str(ii+1)
            model_path = os.getcwd()+"/" + model_name + '.hdf5'
            L1 = Dense(self.list_nn[0], activation=self.activation_func,
                       kernel_initializer=self.init, trainable = self.trainable_layer,
                       kernel_regularizer=regularizers.l2(self.regularization))(self.input)

            for ii in range(1,len(self.list_nn)):
                L1 = Dense(self.list_nn[ii], activation=self.activation_func, trainable = self.trainable_layer,
                           kernel_initializer=self.init,
                           kernel_regularizer=regularizers.l2(self.regularization))(L1)    


            LOut  = Dense(self.numberofaction, activation='linear', name='action'+str(ii+1),
                          kernel_initializer=self.init,
                          kernel_regularizer=regularizers.l2(self.regularization))(L1)
            
            model = Model(inputs=self.input, outputs=LOut)
            plot_model(model,to_file=model_name+'.png', show_layer_names=True,show_shapes=True)
            print('\n%s with %s params created' % (model_name,model.count_params()))

            self.model[model_name] = { 'model_name'    : model_name,
                                       'model_path'    : model_path,
                                       'model_network' : model,
                                       'numberofparams': model.count_params(),
                                       'compile'       : model.compile(optimizer=self.opt, 
                                                         loss=self.loss, metrics=['mse']) }
            if self.load_saved_model:
                if not os.path.exists(self.model['model1']['model_path']):
                    print('There is no model saved to the related directory!')
                else:
                    self.model[model_name]['model_network'] = load_model(self.model['model1']['model_path'])
                    if os.path.exists(os.getcwd()+"/" + 'best_model_msd' + '.hdf5'):
                        self.model['model1']['best']['model_path']['maxscore']    = os.getcwd()+"/" \
                                                                                    + 'best_model_msd' + '.hdf5'
                        self.model['model1']['best']['model_network']['maxscore'] = load_model(os.getcwd()+"/" \
                                                                                    + 'best_model_msd' + '.hdf5')
                    if os.path.exists(os.getcwd()+"/" + 'best_model_mtd' + '.hdf5'):
                        self.model['model1']['best']['model_path']['maxtime']     = os.getcwd()+"/" \
                                                                                    + 'best_model_mtd' + '.hdf5'
                        self.model['model1']['best']['model_network']['maxtime']  = load_model(os.getcwd()+"/" \
                                                                                    + 'best_model_mtd' + '.hdf5')
    
    def __describe__(self):
        return self.description
     
    def summary(self):
        for key in self.model.keys():
            self.model[key]['model_network'].summary()
            self.listOfmodels = [key for key in self.model.keys()]
            print('\nModel Name is: ',self.model[key]['model_name'])
            print('\nModel Path is: ',self.model[key]['model_path'])
            print('\nActivation Function is: ',self.activation_func)
            print('\n*******************************************************************************')
        if self.description != '':
            print('\nModel Description: '+self.__describe__())

class agent(neuralnet):
    def __init__(self, numberofstate, numberofaction, activation_func='elu', 
                 trainable_layer= True, initializer= 'he_normal', list_nn= [250,150], 
                 load_saved_model= False, location='./', buffer= 50000, annealing= 1000, 
                 batchSize= 100, gamma= 0.95, tau= 0.001, numberofmodels= 5, dimension= 2):
        
        super().__init__(numberofstate, numberofaction, activation_func, trainable_layer, initializer,
                         list_nn, load_saved_model, numberofmodels)
        
        self.epsilon                  = 1.0
        self.location                 = location
        self.gamma                    = gamma
        self.batchSize                = batchSize
        self.buffer                   = buffer
        self.annealing                = annealing
        self.replay                   = []
        self.sayac                    = 0
        self.tau                      = tau
        self.dimension                = dimension
        
    def replay_list(self, state, action, reward, newstate, done):
        if len(self.replay) < self.buffer: #if buffer not filled, add to it
            self.replay.append((state, action, reward, newstate, done))
            print("buffer_size = ",len(self.replay))
        else: #if buffer full, overwrite old values
            if (self.sayac < (self.buffer-1)):
                self.sayac = self.sayac + 1
            else:
                self.sayac = 0
            self.replay[self.sayac] = (state, action, reward, newstate, done)
            print("sayac = ",self.sayac)

    def remember(self,main_model,target_model):
        model        = self.model[main_model]
        target_model = self.model[target_model]
        minibatch    = random.sample(self.replay, self.batchSize)
        X_train      = []
        y_train      = []
        action       = {}
        maxQ         = {}
        Qval         = {}
        state        = {}
        update       = {}
        for memory in minibatch:
            #Get max_Q(S',a)
            state['old'], actionn, reward, state['new'], done = memory
            for key in state.keys():
                Qval[key] = model.predict(state[key].reshape(1,self.numberofstate), batch_size= 1)
            y             = np.zeros((1,self.numberofaction))
            y[:]          = Qval['old'][:]
            dim_          = 0
            Qval['trgt']  = target_model.predict(state['new'].reshape(1,self.numberofstate), batch_size=1)
            for act in self.dimension:
                action[act] = np.argmax(Qval['new'][0][(dim_/self.dimension)*self.numberofaction : \
                              (dim_+1/self.dimension)*self.numberofaction])
                maxQ[act]   = Qval['trgt'][0][action[act]+(dim_/self.dimension)*self.numberofaction]

                if not done:
                    update[act] = reward + self.gamma * maxQ[act]
                else:
                    update[act] = reward
        
                y[0][actionn[0,dim_]+(dim_/self.dimension)*self.numberofaction] = update[act]
                dim_                                                            = dim_ + 1
    
            X_train.append(state['old'].reshape(self.numberofstate,))
            y_train.append(y.reshape(self.numberofaction,))
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        model.fit(X_train, y_train, batch_size=self.batchSize, nb_epoch=1, verbose=1)
        return model

    def train_model(self, epoch, training_mode):
        def training_mode1():
            self.remember('model1','model2')
            if epoch % 20 == 0:
                counter1 = 0
                counter2 = counter1 + 1
                for _ in range(self.numberofmodels):
                    if counter2 >= self.numberofmodels:
                        counter2 = 0
                    self.remember(self.listOfmodels[counter1],self.listOfmodels[counter2])
                    counter1 = counter1 + 1
                    counter2 = counter2 + 1          
            if epoch % 99 == 0:
                if os.path.exists(os.getcwd()+"/" + 'best_model_msd' + '.hdf5'):   
                    self.model['model1']['model_network'].load_weights(self.model['model1']['best']['model_path']['maxscore'])
                if os.path.exists(os.getcwd()+"/" + 'best_model_mtd' + '.hdf5'):   
                    self.model['model1']['model_network'].load_weights(self.model['model1']['best']['model_path']['maxtime'])
            
        def training_mode2():
            self.remember('model1','model2')
            main_weights   = self.model['model1']['model_network'].get_weights()
            target_weights = self.model['model2']['model_network'].get_weights()
            for i, layer_weights in enumerate(main_weights):
                target_weights[i] = target_weights[i] * (1-self.tau) + self.tau * layer_weights
            if len(self.replay)>=(0.9*self.buffer):
                if os.path.exists(os.getcwd()+"/" + 'best_model_msd' + '.hdf5'):
                    best_weights = self.model['model1']['best']['model_network']['maxscore'].get_weights()
                if os.path.exists(os.getcwd()+"/" + 'best_model_mtd' + '.hdf5'):
                    best_weights = self.model['model1']['best']['model_network']['maxtime'].get_weights()
                for i, layer_weights in enumerate(best_weights):
                    target_weights[i] = target_weights[i] * (1-self.tau) + self.tau * layer_weights
            if epoch % 20 == 0:
                if os.path.exists(os.getcwd()+"/" + 'best_model_msd' + '.hdf5'):
                    best_weights = self.model['model1']['best']['model_network']['maxscore'].get_weights()
                if os.path.exists(os.getcwd()+"/" + 'best_model_mtd' + '.hdf5'):
                    best_weights = self.model['model1']['best']['model_network']['maxtime'].get_weights()
                for i, layer_weights in enumerate(best_weights):
                    main_weights[i] = main_weights[i] * 0.5 + 0.5 * layer_weights
            
            self.model['model1']['model_network'].set_weights(main_weights)
            self.model['model2']['model_network'].set_weights(target_weights)

        if len(self.replay) >= self.annealing:            
            if training_mode == 1:
                training_mode1()
            elif training_mode == 2:
                training_mode2()

    def save_replay(self):
        return self.replay
        
    def save(self, time, target_time, score, target_score, mtd= False,msd= False):
        self.model['model1']['model_network'].save(self.model['model1']['model_path'])

        self.model['model1']['best'] = { 'model_path'    : {'maxscore' : os.getcwd()+"/" + 'best_model_msd' + '.hdf5',
                                                            'maxtime'  : os.getcwd()+"/" + 'best_model_mtd' + '.hdf5'},
                                         'model_network' : {'maxscore' : '','maxtime'  : ''},
                                         'mtd'           : False,
                                         'msd'           : False,
                                         'maxtime'       : None,
                                         'maxscore'      : None }
        if mtd:
            self.model['model1']['best']['mtd']                      = True
            self.model['model1']['best']['maxtime']                  = time
            self.model['model1']['best']['model_network']['maxtime'] = load_model(self.model['model1']['model_path'])
            self.model['model1']['best']['model_network']['maxtime'].save(self.model['model1']['best']['model_path']['maxtime'])

        if msd:
            self.model['model1']['best']['msd']                       = True
            self.model['model1']['best']['maxscore']                  = score
            self.model['model1']['best']['model_network']['maxscore'] = load_model(self.model['model1']['model_path'])
            self.model['model1']['best']['model_network']['maxscore'].save(self.model['model1']['best']['model_path']['maxscore'])

        