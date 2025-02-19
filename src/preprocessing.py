import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------- DATASET LOADING

# defining column names
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

# loading dataset
dataFrame = pd.read_csv("../data/dataset.data", names=columns, delimiter=",")

# ----------------------------------- FINDING MISSING VALUES

# defining expected values for numerical features (minimum and maximum)
rules_numerical = {
	"age":		[20, 110],
	"trestbps":	[90, 200],
	"chol":		[100, 600],
	"thalach":	[60, 250],
	"oldpeak":	[0.0, 6.2]
}

# defining expected values for categorical features
rules_categorical = {
	"sex":		[0, 1],
	"cp":		[1, 2, 3, 4],
	"fbs":		[0, 1],
	"restecg":	[0, 1, 2],
	"exang":	[0, 1],
	"slope":	[1, 2, 3],
	"ca":		[0, 1, 2, 3],
	"thal":		[3, 6, 7],
	"target":	[0, 1, 2, 3, 4]
}

# detecting invalid values in numerical columns
numericalCols_status = {}
for col in rules_numerical:
	invCounter = 0
	for value in dataFrame[col]:
		if isinstance(value, int) or isinstance(value, float): # conditions for validity
			if value >= rules_numerical[col][0] and value <= rules_numerical[col][1]: # conditions for validity
				pass
			else:
				invCounter += 1
				dataFrame.replace(value, np.nan, inplace=True) # converting value to NaN
		else:
			invCounter += 1
			dataFrame.replace(value, np.nan, inplace=True) # converting value to NaN
	numericalCols_status[col] = invCounter

# detecting invalid values in categorical columns
categoricalCols_status = {}
for col in rules_categorical:
	invCounter = 0
	for value in dataFrame[col]:
		isValid = False
		for i in rules_categorical[col]:
			if str(value) == str(i) or str(value) == str(float(i)): # conditions for validity
				isValid = True
				break
		if isValid == False:
			invCounter += 1
			dataFrame.replace(value, np.nan, inplace=True) # converting value to NaN
	categoricalCols_status[col] = invCounter

# ----------------------------------- HANDLING MISSING VALUES

# dropping rows with missing values
dataFrame_cleaned = dataFrame.dropna()

# ----------------------------------- TARGET COLUMN VALUE CONVERSION

dataFrame_cleaned.loc[:, "target"] = (dataFrame_cleaned["target"] > 0).astype(int)
print("Data cleaned successfully.")

# ----------------------------------- ASSESSING CLASS DISTRIBUTION

classes = dataFrame_cleaned["target"].value_counts()
print(f"\nClass distribution:\n{classes}\n")

# ----------------------------------- DATA SPLITTING

# separating features from target labels
X = dataFrame_cleaned.drop(columns=["target"])
y = dataFrame_cleaned["target"]

mccv = 50 # number of random splits for MCCV
joblib.dump(mccv, "../data/mccv.pkl")

for i in range(mccv):
	# splitting into training (80%) and testing (20%) sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i, stratify=y)
	print(f"Data split successfully			({i})")

	# ----------------------------------- ONE-HOT ENCODING

	# defining categorical features to one-hot encode
	columns_OHE = ["cp", "restecg", "slope", "thal"]

	# applying OHE to both feature sets
	X_train_OHE = pd.get_dummies(X_train, columns=columns_OHE, drop_first=True)
	X_test_OHE = pd.get_dummies(X_test, columns=columns_OHE, drop_first=True)
	print(f"Data encoded successfully		({i})")

	# reindexing to make sure X_test_OHE has the same columns as X_train_OHE
	X_test_OHE = X_test_OHE.reindex(columns=X_train_OHE.columns, fill_value=0)

	# ----------------------------------- SCALING

	# defining categorical features to scale
	columns_scaling = ["age", "trestbps", "chol", "thalach", "oldpeak"]

	# applying Z-score normalization to both feature sets while avoiding data leakage
	scaler = StandardScaler()
	X_train_OHE_scaled = scaler.fit_transform(X_train_OHE[columns_scaling])
	X_test_OHE_scaled = scaler.transform(X_test_OHE[columns_scaling])
	print(f"Data scaled successfully		({i})")

	# converting back to dataframes
	X_train_OHE[columns_scaling] = pd.DataFrame(X_train_OHE_scaled, columns=columns_scaling, index=X_train_OHE.index)
	X_test_OHE[columns_scaling] = pd.DataFrame(X_test_OHE_scaled, columns=columns_scaling, index=X_test_OHE.index)
	X_train_preprocessed = X_train_OHE
	X_test_preprocessed = X_test_OHE

	# ----------------------------------- SAVING PREPROCESSED SPLITS TO DATA DIRECTORY

	folderPath = f"../data/{i}"
	os.makedirs(folderPath, exist_ok=True)

	X_train_preprocessed.to_csv(f"{folderPath}/X_train.csv", index=False)
	X_test_preprocessed.to_csv(f"{folderPath}/X_test.csv", index=False)
	y_train.to_csv(f"{folderPath}/y_train.csv", index=False)
	y_test.to_csv(f"{folderPath}/y_test.csv", index=False)

	print(f"Data splits saved successfully	({i})")