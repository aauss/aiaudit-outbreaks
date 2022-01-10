import random
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")

    print(kwargs["submission_metadata"])
    gt = pd.read_json(test_annotation_file)
    gt = gt.replace(to_replace="retinopathy",value="1")
    gt = gt.replace(to_replace="normal",value="0")
    pred = pd.read_json(user_submission_file)
    pred = pred.replace(to_replace="retinopathy",value="1")
    pred = pred.replace(to_replace="normal",value="0")

    gt_vector = gt['Diagnosis'].values.astype(int)
    #for ground truth id find matching prediciton:
    p_vector = []
    for id in gt['Id']:
        val = pred[pred['Id']==id]["Prediction"].values
        p_vector.append(int(val[0]))
    p_vector = np.array(p_vector)

    acc = accuracy_score(gt_vector,p_vector)
    f1 = f1_score(gt_vector,p_vector)
    prec = precision_score(gt_vector,p_vector)
    recall = recall_score(gt_vector,p_vector)

    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "train_split": {
                    "Accuracy": acc,
                    "F1": f1,
                    "Precision": prec,
                    "Recall": recall,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "train_split": {
                    "Accuracy": acc,
                    "F1": f1,
                    "Precision": prec,
                    "Recall": recall,
                }
            },
            {
                "test_split": {
                    "Accuracy": acc,
                    "F1": f1,
                    "Precision": prec,
                    "Recall": recall,
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    print(output)
    return output
