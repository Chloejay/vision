import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import sklearn.linear_model
from sphinx.util.inspect import object_description

#load the data 
oced_bli=pd.read_csv('oced_bli.csv', thousands=',')
gdp_per_capita=pd.read_csv('gdp_per_capita.csv', thousands=','
delimiter()='\t',
    encoding='lation1', 
    na_values='n/a') 

#prepare the data
country_stats= prepare_country_stats(oced_bli, gdp_percapita)
X= np.c_[country_stats['GDP per capita']] #use np.c_ to contact the scaler and matrix 
y= np.c_[country_stats['Life statisfication']] 

#visulize the data
country_stats.plot(kind='scatter', x='GDP per capita', y='Life statisfication')
plt.show() 

#select a linear model then construct the model
model= sklearn.linear_model.LinearRegression() 

#train the model 
model.fit(X,y) 

#make the prediction for the new data 
X_new=[[22587]] 
print(model.predict(X_new)) 

#write the function to fetch the data 
import os
import tarfile 
from six.moves import urllib 

#automate is the best way to get thecode done and more efficient and reproduce the processure
#awlays try to make the relationable path for the directory that I need to use, use sys module, sys.path &os.path.insert 

DOWNLOAD_ROOT='http://raw.githubusercontent.com/ageron/handson-ml2/master/'
HOUSING_PATH= os.path.join('datasets','housing') 
HOUSING_URL= DOWNLOAD_ROOT+ "datasets/housing/housing.tgz"  

def fetch_housing_data(housing_url= HOUSING_URL,housing_path= HOUSING_PATH): 
    #the default value for the parameters as the 2nd variables 
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path) 
    tgz_path= os.path.join(housing_path, 'housing.tgz') 
    urllib.request.urlretrieve(housing_url,tgz_path) 
    #use tarfile to extract the zip file 
    housing_tgz= tarfile.open(tgz_path) 
    housing_tgz.extractall(path=housing_path) 
    housing_tgz.close()  

import pandas as pd     
#write the function to load the data 
def load_housing_data(housing_path= HOUSING_PATH):
    #use os.path.join to join the directory that link two different files 
    csv_path=os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path) 

housing= load_housing_data() 
housing.head() 

#-------------------------------------------------------------------------------------------
#smaplize one example that directly write one function to call the files by the pandas
#one way to best in the programming is to abstract the thoughts, the way of thinking 
def load_job_data(job_list= JOB_PATH):
    csv_path= os.path.join(job_list, 'job_list.csv')
    return pd.read_csv(csv_path) 
job_list= load_job_data() 
job_list.tail() 
#--------------------------------------------------------------------------------------------

from zlib import crc32 

def test_set_check(identifier,test_ratio):
    return crc32.(np.int64(identifier))**0xffffff <test_ratio*2*32 
def split_train_test_by_id(data, test_ratio, id_column):
    ids=data[id_column]
    in_test_set= ids.apply(lambda id_:test_set_check(id_, test_ratio))
    return data.iloc[~in_test_set], data.iloc[in_test_set] 

housing_with_id= house.reset_index() #add the index as the new column 
train_set, test_set= split_train_test_by_id(housing_with_id, 0.2, 'index') 

#built arch for the hyperparamters, so that have some space to undrstand how to optimize the model to get 
#the result with min (loss), for the loss function, besides h is one of the model function, the final one 
#as the convention, will use the g, which is the final decision function we use and the objective/goal 

#start to prepare the resume and update the github that can be usaed for the porject case  

#in the end there is no one that you can reply on, so better to just make the plan that full on me and I can fully managed
#the only problem I have is I need to be fully cleared(can't walk along well in vague), for that is the way I can fully understand myself! 
#I know I was silly for couple months, but its a journey that I growup 

class Growuup:
    def __init__(self, age, object_description, time= 2019):
        '''
        give the docsting for the mistake I have already took and which made me feel so bad, and even I tried to ask help from the 
        so called friends, then there is NONE, for no one can really help me if I don't understand myself well and I don't take the time 
        to save myself, so it's the time I should do it and it's the time I should give myself a jump and an adventure truly, for 
        what seed grows what fruits, so I need the good one, so better to give myself a good seeds that I imagined so much before,
        for I know I was always looking for that one. so I should just follow it and try hard with it, for one day I will truly 
        appreciate for the person I was trying so hard to be. 
        '''
        self._age= age
        self.object_description= object_description
        self.time=time
    def makeChange(self):
        if self.time>2019:
            print('move to berlin at {}'.format(self._age+1))
        else:
            print('work hard and live hard in shanghai in {} when {} is at age of {}'.format(self.time, self.object_description,self._age))
            
    def buildHabit(self, action):
        '''understand very well that the self control or discipline is a good habit I should do, so I will start and back 
        to the journey and just observe and experiment, to gain some knowledge'''
        if t in range(self.time):
                print(t) 
        else:
            print('keep trying in {}'.format(self.time)) 

#tuple can't be changed once it created by () but just tuple don't have methods that list have, for it's can't be changed 
#set are the bag of the unique values, will execute no-op if add/update(similar to the append/extend) the value is already existed in the set 
#union, intersection and difference are the method for the two sets 

def reflection(name):
    '''
    I know I'm silly but I just wanted to laugh and want to laugn just the life what happened in the past and in the future,
    and sometimes life is ugly but also beautiful, depends on how the way I react to it
    '''
    if name!='Chloe':
        print('end')
    else:
        print('{} know now and will not be stupid any more, just live as a decent human and live in a rule'.format(name))

class LazyRules:
    rules_filename = 'plural6-rules.txt'

    def __init__(self):
        self.pattern_file = open(self.rules_filename, encoding='utf-8')
        self.cache = []

    def __iter__(self):
        self.cache_index = 0
        return self

    def __next__(self):
        self.cache_index += 1
        if len(self.cache) >= self.cache_index:
            return self.cache[self.cache_index - 1]
        if self.pattern_file.closed:
            raise StopIteration
            line = self.pattern_file.readline()
        if not line:
            self.pattern_file.close()
            raise StopIteration
        
        pattern, search, replace = line.split(None, 3)
        funcs = build_match_and_apply_functions(
        pattern, search, replace)
        self.cache.append(funcs)
        return funcs 

rules = LazyRules() 

import multiprocessing 
import os    
import time       


import numpy as np   
import pandas as pd     
 
from typing import Any, Optional, Tuple
import torch    
import torch.nn as nn  
import torchvision

def makeProject(x):
    if date_<day:
        result= x
        return result 

class beginBraveEnough(object):
    '''make the leading and trailing  variables for name convention, for private usage or magic method''' 
    def __init__(self, name, time, location):
        self._name= name
        self.time= time 
        self.location= location 
    
        def callMyDream(self):
            print('{} is calling {} her name and bury something  in a small and quite room, where called accpetance'.format(self.name)) 

#create the loss function in the tensorflow to set the customzied threshold
def create_hub(threshold=1.0):
    '''kind of write the closure to pass the args for the wrapped fns''' 
    def huber_fn(y_true, y_pred):
        error= y_true- y_pred
        is_small_error=tf.abs(error)<threshold
        squared_loss= tf.squared(error)/2
        linear_loss=threshold*tf.abs(error)-threshold**2/2

        return tf.where(is_small_error,squared_loss, linear_loss)
    return huber_fn 
'''also upload the model need to specify the model loss function name and threshold value''' 

model.compile(loss_function= create_hub(2.0), optimizer='adam') 

#make a subcalss and call some function of method by the parent class 
super().__init__(args) 

class BuildStuff(partentClass):
    def __init__(self, name, list_):
        self.name= name
        self.lits_= list_ 
        self.hidden_layer= [use list_comprhension for _ in  range(5)] #sample MLP function for the 5 hidden layers 
    def call(self, other_variables):
        for x in self._hidden_layer:
            Z= layer(Z) #the layer is the function that in the keras.Layer function 








