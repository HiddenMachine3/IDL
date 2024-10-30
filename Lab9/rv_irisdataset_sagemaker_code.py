###############################################################################################################################################
# Data Preparation
# ================
import sagemaker
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
import os
# Initialize SageMaker session and specify bucket
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
role = sagemaker.get_execution_role()
# Load Iris dataset
iris_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
header=None)
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
# Convert species to numeric values
iris_data['species'] = iris_data['species'].astype('category').cat.codes
print(iris_data['species'].unique()) # Should output [0, 1, 2]
# Split data into train and validation sets
train_data, val_data = train_test_split(iris_data, test_size=0.2, random_state=42)
# Move the label column to the front of the dataframe
train_data = train_data[['species', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
val_data = val_data[['species', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
# Save locally as CSV
train_data.to_csv('train.csv', header=False, index=False)
val_data.to_csv('validation.csv', header=False, index=False)
# one can print the train_data and the val_data that would be the 80% training data 20% validation data
###############################################################################################################################################

# upload the tain_data and val_data to S3 bucket
s3_train_path = sagemaker_session.upload_data(path='train.csv', bucket=bucket, key_prefix='xgboost-iris/train')
s3_validation_path = sagemaker_session.upload_data(path='validation.csv', bucket=bucket, key_prefix='xgboost-iris/validation')
print(f"Training data uploaded to: {s3_train_path}")
print(f"Validation data uploaded to: {s3_validation_path}")

###############################################################################################################################################

# Specify the XGBoost container
container = sagemaker.image_uris.retrieve("xgboost", boto3.Session().region_name, "1.5-1")
# Set XGBoost hyperparameters
xgboost = sagemaker.estimator.Estimator(
container,
role,
instance_count=1,
instance_type='ml.m5.large',
output_path=f's3://{bucket}/xgboost-iris/output', sagemaker_session=sagemaker_session)
xgboost.set_hyperparameters(
objective="multi:softmax",
num_class=3, # There are 3 classes in the Iris dataset
num_round=100
)
# Specify input data
train_input = sagemaker.inputs.TrainingInput(s3_data=s3_train_path, content_type='csv')
validation_input = sagemaker.inputs.TrainingInput(s3_data=s3_validation_path, content_type='csv')
# Start the training job
xgboost.fit({"train": train_input, "validation": validation_input})

###############################################################################################################################################


# Deploy the XGBoost model to an endpoint
xgboost_predictor = xgboost.deploy(initial_instance_count=1, instance_type='ml.m5.large')
# Prepare a sample input for prediction
test_data = val_data.drop(columns=['species']).iloc[2].values # Assuming 'species' is the label column
test_data = ','.join(map(str, test_data)) # Convert features to a CSV row format
print(f"Test data for prediction: {test_data}")
# Make a prediction
response = xgboost_predictor.predict(
test_data,
initial_args={'ContentType': 'text/csv'} # Specify content type as text/csv
)
# The response might need decoding, depending on format
predicted_class = response.decode('utf-8') # Decode the response to a readable format (if required)
print(f"Predicted class: {predicted_class}")
# Clean up by deleting the endpoint
xgboost_predictor.delete_endpoint()
