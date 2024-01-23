# AutoGOAL Example: basic usage of the AutoML class
from autogoal.datasets import cars
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal.ml import AutoML
from sklearn.feature_extraction import DictVectorizer

# Load dataset
X, y = cars.load()

# Instantiate AutoML and define input/output types
automl = AutoML(
	input=(MatrixContinuousDense, Supervised[VectorCategorical]),
	output=VectorCategorical,
)

# Run the pipeline search process
automl.fit(X, y)

# Open the file with write mode
file = open('Resultado.txt', 'w')

# Write some text to the file
file.write(str(automl.best_pipeline_) + "\n")
file.write(str(automl.best_score_) + "\n")

# Close the file
file.close()

