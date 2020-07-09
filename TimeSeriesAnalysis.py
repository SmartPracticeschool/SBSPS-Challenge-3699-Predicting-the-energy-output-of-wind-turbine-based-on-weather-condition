import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
#feature scaling
from sklearn.preprocessing import MinMaxScaler
# applying time series generator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# Creating the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
# Early stopping
from tensorflow.keras.callbacks import EarlyStopping
#######################################################################

df = pd.read_csv('T1.csv',index_col='Date/Time',infer_datetime_format=True)
df.columns=['LV ActivePower (kW)','Wind Speed (m/s)','Theoretical_Power_Curve (KWh)','Wind Direction (°)']

df=df.iloc[46086:,:]
df=df.round(2)

sns.heatmap(df.corr(),annot=True,cmap='viridis')
plt.title('Heatmap of co-relation between variables',fontsize=16)
plt.show()


groups = [0, 1, 2, 3]
values = df.values
fig,sub = plt.subplots(2,2)
plt.subplots_adjust(wspace=1, hspace=1)

for ax, i in zip(sub.flatten(),groups):
    ax.plot(values[:,i])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(df.columns[i])
plt.show()


#splitting into training set and test set
test_in_3days=432
                    # how many rows per day? we know that it is every 10 min
                    #(24*60)/10=144
                    #test_in 72hr i.e in 3 days is 144*3=432

train = df.iloc[:-test_in_3days]
test = df.iloc[-test_in_3days:]
print(train)
print(test)
# Feature Scaling
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.fit_transform(test)

# applying time series generator
generator = TimeseriesGenerator(scaled_train, scaled_train,length=72,batch_size=2)

#creating the RNN model
model = Sequential()
# creating RNN layer
model.add(LSTM(150,input_shape=(144,scaled_train.shape[1]),return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(150,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(150))
model.add(Dropout(0.3))


model.add(Dense(scaled_train.shape[1]))
model.compile(optimizer='adam', loss='mae')

validation_generator = TimeseriesGenerator(scaled_test,scaled_test,length=72,batch_size=64)
early_stopping=EarlyStopping(monitor='val_loss',patience=6)
model.fit_generator(generator,epochs=15,validation_data=validation_generator,callbacks=[early_stopping])

print(model.summary())


losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

# evaluating the test result
first_eval_batch = scaled_train[-144:] #length=144458777

first_eval_batch = first_eval_batch.reshape((1, 144, scaled_train.shape[1]))
#print(first_eval_batch)

result=model.predict(first_eval_batch)
print(result)


print(scaled_test[0])
n_features = scaled_train.shape[1]
print(n_features)
test_predictions = []

first_eval_batch = scaled_train[-144:]
current_batch = first_eval_batch.reshape((1,144, n_features))


print(current_batch)
for i in range(432):
    current_pred = model.predict(current_batch)[0]

    test_predictions.append(current_pred)

    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
print(test_predictions)

# Inverse Transform and compare
true_predictions = scaler.inverse_transform(test_predictions)
true_predictions = pd.DataFrame(data=true_predictions,columns=test.columns)




true_predictions.to_csv('Output predction.csv',index=False,header=True)
print(true_predictions)

plt.show()
model.save('my_model_01.h5py')
model.save("final_prediction.hdf5")