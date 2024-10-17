import pandas as pd
from configparser import ConfigParser
import matplotlib.pyplot as plt
import sys
import ast
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle

from datetime import datetime

def buildingDataset(df,reader,component_signal_tuple):
    activities_col_name = eval(reader['DATASET']['activities_col'])[0]
    timestamps_col_name = eval(reader['DATASET']['timestamp_col'])[0]
    #df[timestamps_col_name] = pd.to_datetime(df[timestamps_col_name])
    df_measures = df[df[activities_col_name].str.startswith(component_signal_tuple[1])]
    #df_measures = df_measures.head(100)
    df_errors = df[df[activities_col_name] == component_signal_tuple[0] + '_is_down']
    #df_errors = df_errors.head(1)
    plottingTrends(df_measures,t[0])
    col = eval(reader['DATASET']['new_colums'])
    df_result = pd.DataFrame(columns=col)
    counter_event_fault = 0
    dim = df_measures.shape[0]
    for i, row in df_measures.iterrows():
        duration = (df_errors[timestamps_col_name].iloc[counter_event_fault] - row[timestamps_col_name])/60
        while (duration < 0 and counter_event_fault < dim - 1):
            counter_event_fault = counter_event_fault + 1
            duration = (df_errors[timestamps_col_name].iloc[counter_event_fault] - row[timestamps_col_name])/60
        if (duration >= 0):
            new_line = dict()
            new_line[col[0]] = row['Event_code']
            new_line[col[1]] = df_errors['Event_code'].iloc[counter_event_fault]
            new_line[col[2]] = pd.to_numeric(row['Value'], errors='coerce')
            new_line[col[3]] = duration
            df_result.loc[len(df_result)] = new_line

    df_result.to_csv(component_signal_tuple[0] + "_lstm_dataset.csv", index=False)
    return df_result

def plottingTrends(df,component):
    plt.figure(figsize=(10, 6))
    plt.plot(df['time:timestamp'], df['Value'], marker='o', label='RUL trend')
    plt.title(component+' trends')
    plt.ylabel('Values recorded by signal')
    plt.xlabel('times')

    # Ruotare le etichette dell'asse x
    plt.xticks(rotation=45)
    plt.savefig(component+'_trends.pdf')
    plt.close()


def split_sequence(df, n_steps):
    X, y = list(), list()
    seq_x=list()
    seq_y=list()
    end_ix=n_steps
    for i,row in df.iterrows():
        # find the end of this pattern

        if(i< end_ix):
            seq_x.append(row['delta_time'])
            old_delta=row['measure_value']
        else:
            end_ix=end_ix+n_steps
            X.append(seq_x)
            seq_y.append(old_delta)
            y.append(seq_y)
            seq_x=[row['delta_time']]
            seq_y=list()

    return np.array(X), np.array(y)

def best_cv(model, x, y):
    accuracy = 0
    X_train_best = []
    y_train_best = []
    X_test_best = []
    y_test_best = []
    kf = KFold(n_splits=7)  # Define the split - into 2 folds
    kf.get_n_splits()  # returns the number of splitting iterations in the cross-validator
    x, y = shuffle(x, y)
    for train_index, test_index in kf.split(x):
        # print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, np.ravel(y_train, order='C'),epochs=200, verbose=0)
        # print(X_train.shape,X_test.shape)


        # Eseguire le previsioni sul set di test utilizzando il modello addestrato
        predictions = model.predict(X_test)

        # Calcolare MAEs
        mae = mean_absolute_error(y_test, predictions)
        print("MAE:", mae)

        # Calcolare MAPE
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        print("MAPE:", mape)

        # Calcolare RMSE
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print("RMSE:", rmse)
        if (mae > accuracy):
            accuracy = mae
            X_train_best = X_train
            y_train_best = y_train
            X_test_best = X_test
            y_test_best = y_test

    best_model = model.fit(X_train_best, np.ravel(y_train_best, order='C'))
    return best_model, X_test_best, y_test_best

def LSTMAlgorithm(df):
    n_steps = 5
    X, y = split_sequence(df, n_steps)
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    best_model, X_test_best, y_test_best=best_cv(model, X, y)
    predictions = model.predict(X_test_best)

    mae = mean_absolute_error(y_test_best, predictions)
    file_report.write("MAE: "+str(mae)+'\n')

    # Calcolare MAPE
    mape = np.mean(np.abs((y_test_best - predictions) / y_test_best)) * 100
    file_report.write("MAPE:"+str(mape)+'\n')

    # Calcolare RMSE
    rmse = np.sqrt(mean_squared_error(y_test_best, predictions))
    file_report.write("RMSE: "+str(rmse)+'\n')
    for i, p in enumerate(predictions):
        file_report.write(
            'Delta times observed:\n ' + str(X_test_best[i]) + '\nValue predicted: ' + str(p) + '\n\n\n\n')

    delta_times=np.array([[[380], [285], [190], [95], [0]]])
    threeshold_prediction = model.predict(delta_times)
    file_report.write('Delta times observed:\n ' + str(delta_times) + '\nValue predicted: ' + str(threeshold_prediction) + '\n\n\n\n')
    return model

if __name__ == "__main__":
    if len(sys.argv) == 4:
        config = sys.argv[1]
        dataset=sys.argv[2]
        file_report_name=sys.argv[3]
        reader = ConfigParser()
        reader.read(config)
        file_report=open(file_report_name,'w')
        df = pd.read_csv(dataset, sep=',')
        activities_col_name = eval(reader['DATASET']['activities_col'])[0]
        component_signals_list_str = reader.get('DATASET', 'component_signals_list')
        component_signals_list = ast.literal_eval(component_signals_list_str)
        timestamps_col_name = eval(reader['DATASET']['timestamp_col'])[0]
        #df[timestamps_col_name] = pd.to_datetime(df[timestamps_col_name])
        file_report.write('\n\n\n')
        for j,t in enumerate(component_signals_list):
            file_report.write('RUL estimation for component '+t[0]+':\n')
            df_lstm=buildingDataset(df,reader,t)
            #df_lstm=pd.read_csv(t[0]+'_lstm_dataset.csv',sep=',')
            model=LSTMAlgorithm(df_lstm)



        file_report.close()




