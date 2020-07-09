from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import h5py


model = load_model('my_model_01.h5py')
print(model.summary())
df = pd.read_csv('T1.csv',index_col='Date/Time',infer_datetime_format=True)
df=df.iloc[46086:,:].values.round(2)
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df)


# evaluating the test result



first_eval_batch = scaled_df.reshape((1, 4444, scaled_df.shape[1]))
#print(first_eval_batch)
result=model.predict(first_eval_batch)
print(result)


print(scaled_df[0])
n_features = scaled_df.shape[1]
print(n_features)
test_predictions = []


current_batch = first_eval_batch.reshape((1,4444, n_features))


print(current_batch)
for i in range(432):
    current_pred = model.predict(current_batch)[0]

    test_predictions.append(current_pred)

    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
print(test_predictions)

# Inverse Transform and compare
true_predictions = scaler.inverse_transform(test_predictions)
true_predictions = pd.DataFrame(data=true_predictions,columns=['LV ActivePower (kW)','Wind Speed (m/s)','Theoretical_Power_Curve (KWh)','Wind Direction (°)'])
print(true_predictions)
true_predictions.to_csv('Forcast predction.csv',index=False,header=True)
print(true_predictions)