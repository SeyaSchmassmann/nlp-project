import wandb
import json
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

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

def evaluate_classifier(classifier, classifier_name, vectorizer, vectorizer_name, train_texts, train_labels, validation_texts, validation_labels, test_texts, test_labels, model_name):
    model = make_pipeline(vectorizer, classifier)
    model.fit(train_texts, train_labels)
    
    val_preds = model.predict(validation_texts)
    val_acc = accuracy_score(validation_labels, val_preds)
    val_report = classification_report(validation_labels, val_preds, output_dict=True)
    val_conf_matrix = confusion_matrix(validation_labels, val_preds)

    test_preds = model.predict(test_texts)
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

    
    evaluate_model(model, test_texts, test_labels)