
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = [
    iris.target_names[i] for i in iris.target
]

print("=" * 50)
print("IRIS FLOWER CLASSIFICATION")
print("=" * 50)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nSpecies distribution:")
print(df["species"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

sns.pairplot(df, hue="species", diag_kind="hist")
plt.suptitle("Iris Dataset - Feature Relationships", y=1.02)
plt.show()

df.hist(figsize=(10, 8), bins=15, edgecolor="black")
plt.suptitle("Distribution of Iris Features")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    s=80
)
plt.title("Petal Length vs Petal Width")
plt.tight_layout()
plt.show()

X = df.drop("species", axis=1)
y = df["species"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("\nModel training completed successfully!")
print("Model classes:", model.classes_)

y_pred = model.predict(X_test)

print("\nFirst 10 actual species:")
print(y_test.head(10).to_list())

print("\nFirst 10 predicted species:")
print(y_pred[:10])

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

report_text = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(report_text)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

cm_display.plot(cmap="Blues")
plt.title("Iris Classification - Confusion Matrix")
plt.tight_layout()
plt.show()

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(
    results_dir / "classification_report.csv"
)

cm_df = pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
)

cm_df.index.name = "Actual"
cm_df.columns.name = "Predicted"

cm_df.to_csv(results_dir / "confusion_matrix.csv")

with open(
    results_dir / "model_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("Iris Flower Classification\n")
    file.write("Algorithm: Logistic Regression\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write("Classification Report:\n")
    file.write(report_text)

print("\n" + "=" * 50)
print("PROJECT EXECUTION COMPLETED")
print("=" * 50)
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Results saved in:", results_dir.resolve())
print("Files created:")
print("- classification_report.csv")
print("- confusion_matrix.csv")
print("- model_summary.txt")
