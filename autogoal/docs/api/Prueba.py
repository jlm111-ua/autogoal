# # AutoGOAL Example: basic usage of the AutoML class
# from autogoal.datasets import cars
# from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
# from autogoal.ml import AutoML
# from sklearn.feature_extraction import DictVectorizer
# # Load dataset
# X, y = cars.load()

# # Instantiate AutoML and define input/output types
# automl = AutoML(
# 	input=(MatrixContinuousDense, Supervised[VectorCategorical]),
# 	output=VectorCategorical,
# )

# # Run the pipeline search process
# automl.fit(X, y)

# # Open the file with write mode
# file = open('Resultado.txt', 'w')

# # Write some text to the file
# file.write(str(automl.best_pipeline_) + "\n")
# file.write(str(automl.best_score_) + "\n")

# # Close the file
# file.close()
from scipy import sparse as sp
import numpy as np

train_data = open("/home/coder/autogoal/autogoal/docs/api/train_data.data", "r")
train_labels = open("/home/coder/autogoal/autogoal/docs/api/train_labels.data", "r")
valid_data = open("/home/coder/autogoal/autogoal/docs/api/test_data.data", "r")
valid_labels = open("/home/coder/autogoal/autogoal/docs/api/test_labels.data", "r")

# Count the number of lines in each file
num_lines_train_data = sum(1 for line in train_data)
num_lines_valid_data = sum(1 for line in valid_data)
ytrain = []

print(num_lines_train_data)

# Reset the file pointer to the beginning of the file
train_data.seek(0)

Xtrain = sp.lil_matrix((num_lines_train_data, 7), dtype=int)
print(Xtrain.shape)

# How can I add the train_data into the Xtrain matrix?
for row, line in enumerate(train_data):
	column = 0
	for col in line.strip().split():
		Xtrain[int(row), column] = int(col)
		column += 1

for line in train_labels:
    ytrain.append(int(line))

print(ytrain)
	


