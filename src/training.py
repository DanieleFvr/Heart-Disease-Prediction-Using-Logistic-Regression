import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mccv = joblib.load("../data/mccv.pkl")

# initializing the model
logReg = LogisticRegression()

metrics = {
	"accuracy": [],
	"precision": [],
	"recall": [],
	"f1": []
}

# loading preprocessed splits
for i in range(mccv):
	X_train = pd.read_csv(f"../data/{i}/X_train.csv")
	X_test = pd.read_csv(f"../data/{i}/X_test.csv")
	y_train = pd.read_csv(f"../data/{i}/y_train.csv").values.ravel()
	y_test = pd.read_csv(f"../data/{i}/y_test.csv").values.ravel()

	# fitting the model on training data
	logReg.fit(X_train, y_train)
	print(f"Logistic regression model fit complete	({i})")

	# evaluating model performance on test data
	y_prediction = logReg.predict(X_test)
	print(f"Prediction on test data complete		({i})")

	metrics["accuracy"].append(accuracy_score(y_test, y_prediction))
	metrics["precision"].append(precision_score(y_test, y_prediction))
	metrics["recall"].append(recall_score(y_test, y_prediction))
	metrics["f1"].append(f1_score(y_test, y_prediction))

print("\nMETRICS:")
print(f"Accuracy:	{np.mean(metrics['accuracy'])}")
print(f"Precision:	{np.mean(metrics['precision'])}")
print(f"Recall:		{np.mean(metrics['recall'])}")
print(f"F1 score:	{np.mean(metrics['f1'])}")