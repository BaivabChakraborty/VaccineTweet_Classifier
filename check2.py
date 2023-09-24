# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

# Load the training and testing data
train_val_data = pd.read_csv("train_val.csv")
test_data = pd.read_csv("test.csv")

# Extract the features and labels from the training data
X_text = train_val_data["tweet"]
y_labels = train_val_data["labels"].str.split()

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2),
)
X = vectorizer.fit_transform(X_text)

# Convert the labels to binary matrix representation
mlb = MultiLabelBinarizer()
y_labels_binarized = mlb.fit_transform(y_labels)

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y_labels_binarized, test_size=0.1, random_state=42
)

# Initialize the MultiOutputClassifier with Naive Bayes
model = MultiOutputClassifier(MultinomialNB(alpha=1, fit_prior=False))

# Fit the model to the multi-label training data
model.fit(X_train, y_train)

# Predict the labels for the validation set
y_pred_val = model.predict(X_val)

# Calculate the accuracy for each label
label_accuracies = []
for i in range(y_val.shape[1]):
    label_accuracy = accuracy_score(y_val[:, i], y_pred_val[:, i])
    label_accuracies.append(label_accuracy)

# Print the accuracy for each label
for i, accuracy in enumerate(label_accuracies):
    print(f"Accuracy for label {i}: {accuracy:.2f}")

# Now, let's predict the labels for the test data
X_test_text = test_data["tweet"]
X_test = vectorizer.transform(X_test_text)
y_pred_test = model.predict(X_test)

# Save the predicted labels to the test_data DataFrame
y_pred_test_labels = mlb.inverse_transform(y_pred_test)
test_data["pred_labels"] = [" ".join(labels) for labels in y_pred_test_labels]

# Save the test_data DataFrame with predictions to a new CSV file
test_data.drop("tweet", axis=1, inplace=True)
test_data.to_csv("prediction_file.csv", index=False)

# Opening the CSV file with predictions
result = pd.read_csv("prediction_file.csv")
print(result)

