import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Read data

housing = pd.read_csv('cal_housing_clean.csv')



# split labels and features

x_data = housing.drop(['medianHouseValue'], axis=1)
y_label = housing['medianHouseValue']


# Split test and train data

x_train, x_test, y_train, y_test = train_test_split(x_data, y_label, test_size=0.3, random_state=101)


# Scaling the data

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = pd.DataFrame(data=scaler.transform(x_train), columns=x_train.columns, index=x_train.index)
x_test = pd.DataFrame(data=scaler.transform(x_test), columns=x_test.columns, index=x_test.index)


# Create input functions


# training input
train_input_fn = tf.estimator.inputs.pandas_input_fn(x_train, y=y_train, batch_size=10, num_epochs=100, shuffle=True)

# Estimator input
predict_input_fn = tf.estimator.inputs.pandas_input_fn(x_test, batch_size=10, num_epochs=1, shuffle=False)



# ---------- Creating model ----------


# Adding list of features to the model

age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

feature_cols = [age, rooms, bedrooms, population, households, income]



# Creating the DNN regression model


optimizer = tf.train.ProximalAdagradOptimizer(
              learning_rate=0.1,
              l1_regularization_strength=0.001
            )
model = tf.estimator.DNNRegressor(hidden_units=[6,6,6],
                                  feature_columns=feature_cols,
                                  optimizer=optimizer)


# Training the model using input

model.train(train_input_fn, steps=25000)



# DEFINITIONS: epoch, batch, size
#
# Epoch is when your model goes through your whole training data once. Step is when your model trains on a single batch
# (or a single sample if you send samples one by one).
#
# Training for 5 epochs on a 1000 samples 10 samples per batch will take 500 steps.



# Predict using the model and input_fn

predict_gen = model.predict(predict_input_fn)
predictions = list(predict_gen)
final_predictions = []

for pred in predictions:
    final_predictions.append(pred['predictions'])


# Checking the error in predictions

error = mean_squared_error(y_test,final_predictions)

print(error)