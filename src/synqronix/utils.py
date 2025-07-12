import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from qbraid.runtime.aws import BraketProvider 
import pennylane as qml
import torch


def setup_quantum_device(num_features, api_key, quantum_device, shots=None):
    """Set up the quantum device for PennyLane."""
    if quantum_device == "default":
        # Use the default PennyLane device
        pl_dev = qml.device("default.qubit", wires=num_features, shots=shots)
        return pl_dev

    elif quantum_device == "ionq_aria":
        device_name = "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-2"

    elif quantum_device == "ionq_forte":
        device_name = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"


    provider      = BraketProvider(api_key)
    ionq_device   = provider.get_device(
        device_name
    )

    aws_sess      = provider._get_aws_session()                

    pl_dev = qml.device(
        "braket.aws.qubit",
        device_arn = ionq_device._device.arn,
        aws_session = aws_sess,
        s3_destination_folder=("qb-braket-results", "ionq_jobs"),
        wires = num_features,
        shots = shots,
        _run_kwargs = {"disable_error_mitigation": True},  # ≤ 30 + 3×10 = 60 credits
    )
    return pl_dev

def calculate_class_weights(train_dataset, device):
    """ Calculate class weights based on the frequency of each class in the training dataset."""
    train_labels = torch.cat([data.y for data in train_dataset])
    class_freq = torch.bincount(train_labels)
    class_wt = 1.0 / class_freq.float() 
    class_wt = class_wt / (class_wt.sum() * len(class_freq)) 
    class_wt = class_wt.to(device)