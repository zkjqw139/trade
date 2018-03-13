# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:50:51 2018

@author: hasee
"""

import pandas as pd
import numpy  as np
from   sklearn.svm import SVR
import matplotlib.pyplot  as plt
from   sklearn     import preprocessing

from   keras.layers.core      import Dense,Activation,Dropout
from   keras.layers.recurrent import LSTM
from   keras.models           import Sequential
import visualize
import time

def load_data(filename):
    stock=pd.read_csv(filename).values
    stock=remove_data(stock)
    norm_stock=normalized_data(stock)
    return norm_stock
    
    
    
def show_data(stock):
    print(stock)
        
def remove_data(stock):
    stock=stock[:,0::3]
    return stock

def normalized_data(stock):
    scaler=preprocessing.MinMaxScaler()
    for index in range(stock.shape[1]):
        stock[:,index]=scaler.fit_transform(stock[:,index])
    return stock

def visualize_plot(stock):
    item=np.zeros(stock.shape[0])
    new_stock=np.zeros([stock.shape[0],3],np.float32)
    for index in range(item.shape[0]):
        item[index]=index
    new_stock[:,0]=item
    new_stock[:,1]=stock[:,0]
    new_stock[:,2]=stock[:,1]
    df=pd.DataFrame(new_stock,index=item,columns=['Item','Open','Close'])
    visualize.plot_basic(df)

def get_train_label(stock,train=True):
    item=np.zeros(stock.shape[0])
    new_stock=np.zeros([stock.shape[0],3],np.float32)
    for index in range(item.shape[0]):
        item[index]=index
    new_stock[:,0]=item
    new_stock[:,1]=stock[:,0]
    new_stock[:,2]=stock[:,1]
    feature=new_stock[:,1::2]
    label  =new_stock[:,2]
    return feature,label    
    
    
def build_imporved_model(input_dim,output_dim,return_sequences):
    model=Sequential()
    model.add(LSTM(input_shape=(None,input_dim),
                   units=output_dim,
                   return_sequences=return_sequences))
    model.add(Dropout(0.2))
    model.add(LSTM(128,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    return model

def unroll(data, sequence_length=24):

    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)    

def train_lstm(stock):
    unroll_length= 50
    batch_size   = 512
    epochs       = 50
    feature,label=get_train_label(stock,True)
    
    feature = unroll(feature, unroll_length)
    label = label[-feature.shape[0]:]

    print("feature", feature.shape)
    print("label", label.shape) 
  
    
    
    
    model = build_imporved_model(feature.shape[-1],output_dim = unroll_length, return_sequences=True)
    start=time.time()
    model.compile(loss='mean_squared_error', optimizer='adam')
    print('compilation time : ', time.time() - start)

    model.fit(feature, label, batch_size=batch_size,epochs=epochs,verbose=2,validation_split=0.05)

    return model


def test_lstm(model,test_stock):
    unroll_length= 50
    batch_size   = 512
    feature,label=get_train_label(test_stock,False)
    
    feature = unroll(feature, unroll_length)
    label = label[-feature.shape[0]:]
    print("feature", feature.shape)
    print("label", label.shape)
    
    
    predictions = model.predict(feature, batch_size=batch_size)
    visualize.plot_lstm_prediction(predictions, label)
    return predictions

if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
#    train_data='C:/Users/hasee/desktop/hw1/training_data.csv'
#    test_data ='C:/Users/hasee/Desktop/hw1/testing_data.csv'
#    output    ='C:/Users/hasee/Desktop/hw1/output.csv'
    train_stoke = load_data(args.training)
    test_stoke  = load_data(args.testing)
    
   
    model=train_lstm(train_stoke)
    predictions=test_lstm(model,test_stoke)
    new_pre=np.zeros(len(predictions))
    
    for index in range(predictions.shape[0]):
        p=0
        if index+1<predictions.shape[0]:
            p=predictions[index+1]-predictions[index]
            if p>=0:
                p=1
            else:
                p=-1
            new_pre[index]=p
        else:
            p=0
    with open(args.output, 'w') as output_file:
             # We will perform your action as the open price in the next day.
            state=0
            action=0
            for index in range(test_stoke.shape[0]):
                if index ==0:
                    action=1
                    state =1
                elif index<50:
                    action=0
                elif index>=50:
                     p=new_pre[index-50]
                     if p==1 and state==1:
                         action=0
                     elif p==1 and state==0:
                         action=1
                         state=1
                     elif p==-1 and state==1:
                         action=-1
                         state=0
                     elif p==-1 and state==0:
                         action=0
                         state =0
                     elif p==0  and state==1:
                         action=0
                     elif p==0  and state==0:
                         action=0
                output_file.write(str(action)+str('\n'))
