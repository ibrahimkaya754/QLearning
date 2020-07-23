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

print('Done!')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# BUILD CUSTOM NETWORK
class neuralnet():
    def __init__(self, numberofstate, numberofaction, 
                 activation_func, trainable_layer, initializer,
                 list_nn, load_weights, numberofmodels):
        
        self.activation_func = activation_func
        self.trainable_layer = trainable_layer
        self.init            = initializer
        self.opt             = Yogi(lr=0.001)
        self.losses          = {}
        self.lossWeights     = {}
        self.regularization  = 0.0
        self.numberofstate   = numberofstate
        self.numberofaction  = numberofaction
        self.list_nn         = list_nn
        self.load_weights    = load_weights
        self.total_layer_no  = len(self.list_nn)+1
        self.numberofmodels  = numberofmodels
        self.model_          = {}
        self.input           = Input(shape=(self.numberofstate,), name='states')

        for ii in range(self.numberofmodels):
            mdl_name = 'model'+str(ii+1)
            self.L1 = Dense(self.list_nn[0], activation=self.activation_func,
                        kernel_initializer=self.init, trainable = self.trainable_layer,
                        kernel_regularizer=regularizers.l2(self.regularization))(self.input)

            for ii in range(1,len(self.list_nn)):
                self.L1 = Dense(self.list_nn[ii], activation=self.activation_func, trainable = self.trainable_layer,
                            kernel_initializer=self.init,
                            kernel_regularizer=regularizers.l2(self.regularization))(self.L1)    


            self.LOut = Dense(self.numberofaction, activation='linear', name='action'+str(ii+1),
                            kernel_initializer=self.init,
                            kernel_regularizer=regularizers.l2(self.regularization))(self.L1)

            self.__build_model__(model_name= mdl_name)
            self.model_[mdl_name] = {'model_name': self.model_name,
                                     'model_path': self.model_path,
                                     'network'   : self.model}
                        
    def __build_model__(self,model_name):
        self.model                         = Model(inputs=self.input, outputs=self.LOut)
        self.model_name                    = model_name
        self.description                   = ''
        self.losses['all_targets']         = huber_loss
        self.lossWeights['all_targets']    = 1.0
        self.model_path                    = os.getcwd()+"/" + self.model_name + '.hdf5'
        self.learning_rate_decrease_factor = 0.97
        self.learning_rate_patience        = 5
        self.number_of_params              = self.model.count_params()
        self.reduce_lr                     = ReduceLROnPlateau(monitor='val_loss', 
                                                               factor=self.learning_rate_decrease_factor,
                                                               patience=self.learning_rate_patience, 
                                                               min_lr=0.0000001, mode='min', verbose=1)
        self.checkpoint                    = ModelCheckpoint(self.model_path, 
                                                             monitor='val_loss', verbose=1, 
                                                             save_best_only=True, period=1, 
                                                             mode='min',save_weights_only=False)
        self.model.compile(optimizer=self.opt, loss=self.losses['all_targets'], metrics=['mse'])
        plot_model(self.model,to_file=self.model_name+'.png', show_layer_names=True,show_shapes=True)
        print('\n%s with %s params created' % (self.model_name,self.number_of_params))
        if os.path.exists(self.model_path):
            if self.load_weights:
                print('weights loaded for %s' % (self.model_name))
                self.model.load_weights(self.model_path)

    def __describe__(self):
        return self.description
     
    def summary(self):
        for key in self.model_.keys():
            self.model_[key]['network'].summary()
            print('\nModel Name is: ',self.model_[key]['model_name'])
            print('\nModel Path is: ',self.model_[key]['model_path'])
            print('\nActivation Function is: ',self.activation_func)
            print('\nLearning Rate Decreases by a factor of %s with patience of %s' % (self.learning_rate_decrease_factor,
                                                                                self.learning_rate_patience))
            print('\n*******************************************************************************')
        if self.description != '':
            print('\nModel Description: '+self.__describe__())

class agent(neuralnet):
    def __init__(self, numberofstate, numberofaction, activation_func='elu', 
                 trainable_layer=True, initializer='he_normal', list_nn=[250,150], 
                 load_weights=False, location='./', buffer=50000, annealing= 1000, 
                 batchSize= 100,gamma= 0.95, tau = 0.001, numberofmodels=5):
        
        super().__init__(numberofstate, numberofaction, activation_func, trainable_layer, initializer,
                         list_nn, load_weights, numberofmodels)
        
        self.epsilon                  = 1.0
        self.location                 = location
        self.gamma                    = gamma
        self.batchSize                = batchSize
        self.buffer                   = buffer
        self.annealing                = annealing
        self.replay                   = []
        self.sayac                    = 0
        self.tau                      = tau
        
    
    
      
    
    
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
    
    def remember(self,model,target_model):
        minibatch  = random.sample(self.replay, self.batchSize)
        X_train    = []
        y_train    = []
        for memory in minibatch:
            #Get max_Q(S',a)
            oldstate, actionn, rewardd, new_state, done = memory
            old_qval = model.predict(oldstate.reshape(1,self.numberofstate), batch_size=1)
            new_qval = model.predict(new_state.reshape(1,self.numberofstate), batch_size=1)
            ax       = np.argmax(new_qval[0][0:int(self.action_number)])
            ay       = np.argmax(new_qval[0][int(self.action_number/2):int(self.action_number)])
            newQ     = target_model.predict(new_state.reshape(1,self.numberofstate), batch_size=1)
            maxQ_x   = newQ[0][ax]
            maxQ_y   = newQ[0][int(self.action_number/2)+ay]
            y        = np.zeros((1,self.action_number))
            y[:]     = old_qval[:]
            if not done: #non-terminal state
                update_x = (rewardd + (self.gamma * maxQ_x))
                update_y = (rewardd + (self.gamma * maxQ_y))
            else: #terminal state
                update_x = rewardd
                update_y = rewardd
                
            y[0][int(actionn)]    = update_x
            y[0][int(self.action_number/2)+int(actionn[0,1])] = update_y
            X_train.append(oldstate.reshape(self.numberofstate,))
            y_train.append(y.reshape(self.action_number,))
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        model.fit(X_train, y_train, batch_size=100, nb_epoch=1, verbose=1)
        return model

    def train_model(self,epch,choose):
        if len(self.replay) >= self.annealing:            
            if choose == 1:
                self.train_model1(epch)
            elif choose == 2:
                self.train_model2(epch)
            elif choose == 3:
                self.train_model3(epch)
            elif choose == 4:
                self.train_model4(epch)
            elif choose == 5:
                self.train_model5(epch)
            
    def train_model1(self, epch):
        if len(self.replay) >= self.annealing:
            self.remember(self.model1,self.model2)
            if epch % 20 == 0:
                self.remember(self.model5,self.model1)
                self.remember(self.model4,self.model5)
                self.remember(self.model3,self.model4)
                self.remember(self.model2,self.model3)          
            if epch % 50 == 0:     
                self.model2 = self.model1
                self.model2.load_weights(self.location + "/model1.h5")
            if epch % 99 == 0:
                self.model1 = self.model_best
                self.model1.load_weights(self.location + "/model_best.h5")
                
    def train_model2(self, epch):
        if len(self.replay) >= self.annealing:
            if epch % 10 == 0:
                self.remember(self.model5,self.model1)
                self.remember(self.model4,self.model5)
                self.remember(self.model3,self.model4)
                self.remember(self.model2,self.model3) 
            self.remember(self.model1,self.model2)            
            if epch % 25 == 0:     
                self.model2 = self.model1
                self.model2.load_weights(self.location + "/model1.h5")
            if epch % 99 == 0:
                self.model1 = self.model_best
                self.model1.load_weights(self.location + "/model_best.h5")

    def train_model3(self, epch):
        if len(self.replay) >= self.annealing:         
            self.remember(self.model1,self.model2)
            if epch % 15 == 0:
                self.model2 = self.model1
                self.model2.load_weights(self.location + "/model1.h5")
            if epch % 99 == 0:
                self.model1 = self.model_best
                self.model1.load_weights(self.location + "/model_best.h5")
       
    def train_model4(self, epch):
        if len(self.replay) >= self.annealing:
            self.remember(self.model1,self.model2)
            if epch % 32 == 0:             
                self.remember(self.model2,self.model3)        
            if epch % 16 == 0: 
                self.remember(self.model3,self.model4)        
            if epch % 8 == 0: 
                self.remember(self.model4,self.model5)        
            if epch % 4 == 0: 
                self.remember(self.model5,self.model1)
            if epch % 64 == 0:     
                self.model2 = self.model1
                self.model2.load_weights(self.location + "/model1.h5")   
            if epch % 99 == 0:
                self.model1 = self.model_best
                self.model1.load_weights(self.location + "/model_best.h5")
            
    def train_model5(self, epch):
        if len(self.replay) >= self.annealing:
            self.remember(self.model1,self.model2)
            main_weights = self.model1.get_weights()
            target_weights = self.model2.get_weights()
            for i, layer_weights in enumerate(main_weights):
                target_weights[i] = target_weights[i] * (1-self.tau) + self.tau * layer_weights
#                target_weights[i] = target_weights[i] + self.tau * layer_weights
            if len(self.replay)>=(0.9*self.buffer):
                best_weights = self.model_best.get_weights()
                for i, layer_weights in enumerate(best_weights):
                    target_weights[i] = target_weights[i] * (1-self.tau) + self.tau * layer_weights
#                    target_weights[i] = target_weights[i] + (1-self.tau) * layer_weights
            if epch % 20 == 0:
                main_weights = self.model1.get_weights()
                best_weights = self.model_best.get_weights()
                for i, layer_weights in enumerate(best_weights):
                    main_weights[i] = main_weights[i] * 0.5 + 0.5 * layer_weights
            
            self.model1.set_weights(main_weights)
            self.model2.set_weights(target_weights)

    def saved_data(self,use_saved_data,eps,saved_replay):
        self.replay                   = saved_replay
        if use_saved_data == True:
            print("saved data is used, epsilon is = ", eps)
            self.epsilon                  = eps
            
            json_file                     = open(self.location + '/model1.json','r')
            loaded_model_json             = json_file.read()
            json_file.close()
            self.model1                   = model_from_json(loaded_model_json)
            self.model1.load_weights(self.location + "/model1.h5")
            
            
            json_file                     = open(self.location + '/model_best.json','r')
            loaded_model_json             = json_file.read()
            json_file.close()
            self.model_best                   = model_from_json(loaded_model_json)
            self.model_best.load_weights(self.location + "/model_best.h5")
            
            self.model1.compile(loss = huber_loss,optimizer=Adam(lr=0.001))
      
            self.model_best.compile(loss = huber_loss,optimizer=Adam(lr=0.001))
  
    def save_replay(self):
        return self.replay
        
    def save(self, time, score, max_score_done, max_time_done):
        model_json = self.model1.to_json()
        with open(self.location + "/model1.json","w") as json_file:
            json_file.write(model_json)
        self.model1.save_weights(self.location + "/model1.h5")

        if max_time_done:
            model_best_json = self.model_best.to_json()
            with open(self.location + "/model_best" + ".json","w") as json_file:
                json_file.write(model_best_json)
            self.model1.save_weights(self.location + "/model_best" + ".h5",overwrite=True)

        if time >= 100:
            model1_json = self.model1.to_json()
            with open(self.location + "/model1" + "_" + str(time) + "_score_" + str(score) + ".json","w") as json_file:
                json_file.write(model1_json)
            self.model1.save_weights(self.location + "/model1" + "_" + str(time) + "_score_" + str(score) + ".h5",overwrite=True)
            
            model2_json = self.model2.to_json()
            with open(self.location + "/model2" + "_" + str(time) + "_score_" + str(score) + ".json","w") as json_file:
                json_file.write(model2_json)
            self.model2.save_weights(self.location + "/model2" + "_" + str(time) + "_score_" + str(score) + ".h5",overwrite=True)
            
            model3_json = self.model3.to_json()
            with open(self.location + "/model3" + "_" + str(time) + "_score_" + str(score) + ".json","w") as json_file:
                json_file.write(model3_json)
            self.model3.save_weights(self.location + "/model3" + "_" + str(time) + "_score_" + str(score) + ".h5",overwrite=True)
            
            model4_json = self.model4.to_json()
            with open(self.location + "/model4" + "_" + str(time) + "_score_" + str(score) + ".json","w") as json_file:
                json_file.write(model4_json)
            self.model4.save_weights(self.location + "/model4" + "_" + str(time) + "_score_" + str(score) + ".h5",overwrite=True)
            
            model5_json = self.model5.to_json()
            with open(self.location + "/model5" + "_" + str(time) + "_score_" + str(score) + ".json","w") as json_file:
                json_file.write(model5_json)
            self.model5.save_weights(self.location + "/model5" + "_" + str(time) + "_score_" + str(score) + ".h5",overwrite=True)