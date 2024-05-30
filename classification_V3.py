#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import datetime
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df = df.dropna()

    # Feature Engineering
    df['emmState_total'] = df['emmState_registered_subscriber_count'] + df['emmState_deregistered_subscriber_count']
    df['ecm_total'] = df['ecmState_connected_subscriber_count'] + df['ecmState_idle_subscriber_count']
    df['amfRmState_total'] = df['amfRmState_registered_subscriber_count'] + df['amfRmState_deregistered_subscriber_count']
    df['amfCmState_total'] = df['amfCmState_connected_subscriber_count'] + df['amfCmState_idle_subscriber_count']

    columns_to_remove = [
        'Sl. No.', 'Timestamp', 'Node Name', 'Date', 'Time', 'Day of the week',
        'emmState_registered_subscriber_count', 'emmState_deregistered_subscriber_count',
        'ecmState_connected_subscriber_count', 'ecmState_idle_subscriber_count',
        'amfRmState_registered_subscriber_count', 'amfRmState_deregistered_subscriber_count',
        'amfCmState_connected_subscriber_count', 'amfCmState_idle_subscriber_count',
        'N2_enabled', 'N2_disabled', 'N12_enabled', 'N12_disabled', 'N8_enabled', 'N8_disabled',
        'N11_enabled', 'N11_disabled', 'N15_enabled', 'N15_disabled', 's1mme_enabled',
        's1mme_disabled', 's6ad_enabled', 's6ad_disabled', 's11_enabled', 's11_disabled'
    ]
    df.drop(columns=columns_to_remove, inplace=True)

    return df

def visualize_data_distribution(target_column):
    plt.figure(figsize=(10, 6))
    sns.countplot(target_column)
    plt.title('Class Distribution')
    plt.show()

def split_and_scale_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def visualize_confusion_matrices(confusion_matrices, class_labels):
    for name, conf_matrix in confusion_matrices.items():
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {name}')
        plt.show()

def main():
    df = load_and_preprocess_data('Node_Classifier_Data_S.csv')

    target_column = 'Resultant_Priority'
    if target_column not in df.columns:
        raise KeyError(f"The target column '{target_column}' is not found in the dataset. Please check the column names.")

    visualize_data_distribution(df[target_column])

    X, y = split_and_scale_data(df, target_column)
    
    # Splitting the dataset for validation
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Using the same classifiers for training
    classifiers = [
        ('RandomForest', RandomForestClassifier(n_estimators=100)),
        ('ExtraTrees', ExtraTreesClassifier(n_estimators=100)),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=100)),
        ('SVC', SVC(kernel='rbf', C=1)),
        ('Multi Layer Perceptron', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000))
    ]

    confusion_matrices = {}
    accuracy_scores = {}
    classification_reports = {}

    for name, model in classifiers:
        start_time = datetime.datetime.now()
        print(f"\nModel Training ({name}) Starts at:", start_time)
        model.fit(X_train, y_train)
        end_time = datetime.datetime.now()
        print(f"Model Training ({name}) Ends at:", end_time)

        # Model evaluation on the entire dataset
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"\nAccuracy Score ({name}) on the entire dataset:", accuracy)

        print(f"\nConfusion Matrix ({name}) on the entire dataset:")
        conf_matrix = confusion_matrix(y, predictions)
        class_labels = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']
        conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
        print(conf_matrix_df)

        print(f"\nClassification Report ({name}) on the entire dataset:")
        report = classification_report(y, predictions)
        print(report)

        # Save the confusion matrix, accuracy score, and classification report
        confusion_matrices[name] = conf_matrix_df
        accuracy_scores[name] = accuracy
        classification_reports[name] = report

        # Save the trained model
        filename = f'{name}_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        print(f"Model ({name}) Saved Successfully!\n")
        print("==============================================================================")

    # Visualize confusion matrices using heatmaps
    class_labels = ['Blocker', 'Critical', 'Major', 'Minor', 'Trivial']
    visualize_confusion_matrices(confusion_matrices, class_labels)

    # Visualize accuracy scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
    plt.title('Accuracy Scores on the Entire Dataset')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.show()

    # Print classification reports
    for name, report in classification_reports.items():
        print(f"Classification Report for {name} on the Entire Dataset:")
        print(report)
        print("--------------------------------------------------")

if __name__ == "__main__":
    main()
