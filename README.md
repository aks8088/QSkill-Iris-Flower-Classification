# Iris Flower Classification

## Project Overview

This project uses machine learning to classify iris flowers into three species: Setosa, Versicolor, and Virginica. The classification is based on sepal and petal measurements.

The project was developed as part of the QSkill Artificial Intelligence and Machine Learning Internship.

## Objectives

- Load and explore the Iris dataset.
- Visualize relationships between flower measurements.
- Split the dataset into training and testing sets.
- Train a Logistic Regression classification model.
- Evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Save evaluation results as CSV and text files.

## Dataset

The project uses the Iris dataset available through scikit-learn.

- Total samples: 150
- Input features: 4
- Target classes: 3

### Input Features

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

### Target Classes

- Setosa
- Versicolor
- Virginica

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Workflow

1. Load the Iris dataset.
2. Explore the dataset and check for missing values.
3. Visualize feature distributions and relationships.
4. Separate input features and target labels.
5. Split the data into 80% training and 20% testing sets.
6. Train a Logistic Regression model.
7. Predict species for the test samples.
8. Evaluate classification performance.
9. Save the evaluation results.

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

In the initial evaluation, the model correctly classified 29 out of 30 test samples, achieving 96.67% accuracy.

The final evaluation results are available in the `results` folder.

## Project Structure

```text
QSkill_AI_ML_Internship/
├── iris_classification.py
├── README.md
└── results/
    ├── classification_report.csv
    ├── confusion_matrix.csv
    └── model_summary.txt
```

## How It Works

The model learns patterns between the four flower measurements and the corresponding species labels using the training dataset. It then predicts species for the unseen test samples.

The confusion matrix helps identify correct classifications and cases where one species is mistaken for another.

## Conclusion

This project demonstrates a complete beginner-level supervised machine learning workflow, including data exploration, visualization, model training, prediction, and performance evaluation.

It provides practical experience with classification algorithms and common Python data science libraries.