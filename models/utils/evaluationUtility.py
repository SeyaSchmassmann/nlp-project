import wandb
import json
import pandas as pd
import os
import glob
import math
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from transformers.trainer_utils import PredictionOutput

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    report = classification_report(y, y_pred, output_dict=True)

    wandb.init(project="nlp-lantsch-schmassmann-wigger", entity="nlp-lantsch-schmassmann-wigger")
    wandb.log({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
    })
    wandb.finish()

    return precision, recall, f1

def calculate_metrics(model, validation_texts, validation_labels, test_texts, test_labels, classifier_name, vectorizer_name, model_name, training_duration):

    val_preds = model.predict(validation_texts)
    if isinstance(val_preds, PredictionOutput):
        val_preds = val_preds.predictions.argmax(-1)
    val_acc = accuracy_score(validation_labels, val_preds)
    val_report = classification_report(validation_labels, val_preds, output_dict=True)
    val_conf_matrix = confusion_matrix(validation_labels, val_preds)

    test_preds = model.predict(test_texts)
    if isinstance(test_preds, PredictionOutput):
        test_preds = test_preds.predictions.argmax(-1)
    test_acc = accuracy_score(test_labels, test_preds)
    test_report = classification_report(test_labels, test_preds, output_dict=True)
    test_conf_matrix = confusion_matrix(test_labels, test_preds)

    result = {
        'classifier': classifier_name,
        'vectorizer': vectorizer_name,
        'val_accuracy': val_acc,
        'val_precision': val_report['1']['precision'],
        'val_recall': val_report['1']['recall'],
        'val_f1': val_report['1']['f1-score'],
        'test_accuracy': test_acc,
        'test_precision': test_report['1']['precision'],
        'test_recall': test_report['1']['recall'],
        'test_f1': test_report['1']['f1-score'],
        'training_duration': training_duration,
    }

    results_name = f"executions/{model_name}/{classifier_name}_{vectorizer_name}"
    result_df = pd.DataFrame([result])
    result_df.to_csv(f"{results_name}.csv", index=False)

    with open(f"{results_name}_val_confusion.json", 'w') as f:
        json.dump(val_conf_matrix.tolist(), f, indent=4)
    with open(f"{results_name}_test_confusion.json", 'w') as f:
        json.dump(test_conf_matrix.tolist(), f, indent=4)

    print(f"\nClassifier: {classifier_name} | Vectorizer: {vectorizer_name}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Training Duration: {training_duration:.2f} seconds")


def evaluate_classifier(classifier, classifier_name, vectorizer, vectorizer_name, train_texts, train_labels, validation_texts, validation_labels, test_texts, test_labels, model_name):
    model = make_pipeline(vectorizer, classifier)
    start_time = time.time()
    model.fit(train_texts, train_labels)
    end_time = time.time()
    training_duration = end_time - start_time
    calculate_metrics(model, validation_texts, validation_labels, test_texts, test_labels, classifier_name, vectorizer_name, model_name, training_duration)
    evaluate_model(model, test_texts, test_labels)

def analyze_all_results(results_dir):
    result_records = []

    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))

    for csv_file in csv_files:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]

        # Parse classifier and vectorizer from filename
        if "_" not in base_name:
            continue  # skip malformed files

        classifier, vectorizer = base_name.split("_", 1)

        # Read metrics
        try:
            df = pd.read_csv(csv_file)
            record = df.iloc[0].to_dict()
        except Exception as e:
            print(f"Failed to read {csv_file}: {e}")
            continue

        record["Classifier"] = classifier
        record["Vectorizer"] = vectorizer

        # Read confusion matrices
        try:
            with open(os.path.join(results_dir, f"{base_name}_val_confusion.json")) as f:
                record["Validation Confusion"] = json.load(f)
            with open(os.path.join(results_dir, f"{base_name}_test_confusion.json")) as f:
                record["Test Confusion"] = json.load(f)
        except Exception as e:
            print(f"Failed to load confusion matrix for {base_name}: {e}")

        result_records.append(record)

    all_results = pd.DataFrame(result_records)
    all_results["Model"] = all_results["Classifier"] + " + " + all_results["Vectorizer"]

    # Plot Metrics
    metrics = [
        "val_accuracy", "val_precision", "val_recall", "val_f1",
        "test_accuracy", "test_precision", "test_recall", "test_f1",
        "training_duration"
    ]

    # Filter valid metrics that exist in your DataFrame
    valid_metrics = [m for m in metrics if m in all_results.columns]
    num_metrics = len(valid_metrics)
    cols = 2
    rows = math.ceil(num_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = axes.flatten()

    for idx, metric in enumerate(valid_metrics):
        ax = axes[idx]
        df_sorted = all_results.sort_values(metric, ascending=False)

        sns.barplot(x=metric, y="Model", data=df_sorted, palette="viridis", ax=ax)
        ax.set_title(f"Model Comparison: {metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel("")

        # Add value labels
        for bar in ax.patches:
            width = bar.get_width()
            ax.text(
                width + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=8
            )

    # Hide unused subplots
    for i in range(len(valid_metrics), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

    # Plot Confusion Matrices
    models_with_matrices = [r for r in result_records if "Validation Confusion" in r and "Test Confusion" in r]
    num_models = len(models_with_matrices)
    cols = 3  # 3 matrices per row
    rows = math.ceil(num_models * 2 / cols)  # 2 matrices (val + test) per model

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    i = 0
    for row in models_with_matrices:
        for phase in ["Validation", "Test"]:
            matrix = row[f"{phase} Confusion"]
            model_name = f"{row['Classifier']} + {row['Vectorizer']}"
            ax = axes[i]
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"{model_name} - {phase}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            i += 1

    # Hide unused subplots
    for j in range(i, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return all_results