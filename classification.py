import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



# Read data

census = pd.read_csv('census_data.csv')


# convert label from character to number


print(census['income_bracket'].unique())
# output: ['< 50k', ' >50k']


def label_fix(label):
    if label == ' <50k':
        return 0
    else:
        return 1


census['income_bracket'] = census['income_bracket'].apply(label_fix)


# split labels and features

x_data = census.drop(['income_bracket'], axis=1)
y_label = census['income_bracket']



# Split test and train data

x_train, x_test, y_train, y_test = train_test_split(x_data, y_label, test_size=0.3, random_state=101)


# Create input functions


# training input
train_input_fn = tf.estimator.inputs.pandas_input_fn(x_train, y=y_train, batch_size=100, num_epochs=100, shuffle=True)

# Estimator input
predict_input_fn = tf.estimator.inputs.pandas_input_fn(x_test, batch_size=10, num_epochs=1, shuffle=False)




# ---------- Creating model ----------


# Adding list of features to the model

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)


# If you want to use DNNClassifier, convert categorical coloumns to embedding_column or indicator_column
# general practise is embedding_dimensions =  number_of_categories**0.25

gender = tf.feature_column.embedding_column(gender, dimension=1)
occupation = tf.feature_column.embedding_column(occupation, dimension=1)
marital_status = tf.feature_column.embedding_column(marital_status, dimension=1)
relationship = tf.feature_column.embedding_column(relationship, dimension=1)
education = tf.feature_column.embedding_column(education, dimension=1)
workclass = tf.feature_column.embedding_column(workclass, dimension=1)
native_country = tf.feature_column.embedding_column(native_country, dimension=1)



age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")


feature_cols = [gender,occupation,marital_status,relationship,education,workclass,native_country,
            age,education_num,capital_gain,capital_loss,hours_per_week]

# Creating classifier model

# Linear classifier
# model = tf.estimator.LinearClassifier(feature_columns=feature_cols)

# DNN classifier
model = tf.estimator.DNNClassifier(hidden_units=[6,6,6], feature_columns=feature_cols)

# Train the model

model.train(train_input_fn, steps=5000)

# Predict using the model and input_fn

predict_gen = model.predict(predict_input_fn)
predictions = list(predict_gen)
final_predictions = []

for pred in predictions:
    final_predictions.append(pred['class_ids'][0])


# Checking the error in predictions

report = classification_report(y_test,final_predictions)

print(report)