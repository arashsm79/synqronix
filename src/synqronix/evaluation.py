import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                            confusion_matrix, classification_report, roc_curve, auc,
                            roc_auc_score)
from sklearn.preprocessing import label_binarize
import os
from tqdm import tqdm
import pandas as pd

def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Compute comprehensive evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (for ROC/AUC)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    if y_probs is not None:
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 2:
                y_true_bin = label_binarize(y_true, classes=unique_classes)
                metrics['roc_auc'] = roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr')
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
        except:
            metrics['roc_auc'] = 0.0
    
    return metrics

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
    
    Returns:
        dict: Evaluation results
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_probs = np.array(all_probs)
    
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = avg_loss
    
    return {
        'metrics': metrics,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, val_f1s, save_path=None):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Validation Loss', color='red')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(train_accs, label='Train Accuracy', color='blue')
    axes[1].plot(val_accs, label='Validation Accuracy', color='red')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(val_f1s, label='Validation F1 Score', color='green')
    axes[2].set_title('Validation F1 Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(y_true, y_probs, class_names=None, save_path=None):
    """Plot ROC curves for multiclass classification"""
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
    else:
        # Multiclass classification
        y_true_bin = label_binarize(y_true, classes=unique_classes)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            # Handle NaN values by replacing them with 0
            y_true_clean = y_true_bin[:, i]
            y_probs_clean = np.nan_to_num(y_probs[:, i], nan=0.0)
            
            fpr, tpr, _ = roc_curve(y_true_clean, y_probs_clean)
            roc_auc = auc(fpr, tpr)
            
            class_name = class_names[i] if class_names else f'Class {unique_classes[i]}'
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_distribution(y_true, y_pred, class_names=None, save_path=None):
    """Plot class distribution comparison"""
    unique_classes = np.unique(y_true)
    
    true_counts = [np.sum(y_true == cls) for cls in unique_classes]
    pred_counts = [np.sum(y_pred == cls) for cls in unique_classes]
    
    x = np.arange(len(unique_classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
    bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution: True vs Predicted')
    
    if class_names:
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {cls}' for cls in unique_classes])
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def full_evaluation(model, test_loader, device, save_dir=None, class_names=None):
    """
    Perform comprehensive evaluation and generate all plots
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device for evaluation
        save_dir: Directory to save plots and results
        class_names: Names of classes for plotting
    
    Returns:
        dict: Complete evaluation results
    """
    print("Starting comprehensive evaluation...")
    
    results = evaluate_model(model, test_loader, device)
    
    print("\nEvaluation Metrics:")
    print("-" * 40)
    for metric, value in results['metrics'].items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    print("\nClassification Report:")
    print("-" * 40)
    target_names = class_names if class_names else None
    print(classification_report(results['labels'], results['predictions'], 
                              target_names=target_names))
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        plot_confusion_matrix(results['labels'], results['predictions'], 
                            class_names=class_names,
                            save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        
        plot_roc_curves(results['labels'], results['probabilities'],
                       class_names=class_names,
                       save_path=os.path.join(save_dir, 'roc_curves.png'))
        
        plot_class_distribution(results['labels'], results['predictions'],
                              class_names=class_names,
                              save_path=os.path.join(save_dir, 'class_distribution.png'))
        
        results_df = pd.DataFrame({
            'true_labels': results['labels'],
            'predicted_labels': results['predictions'],
            'probabilities': [prob.tolist() for prob in results['probabilities']]
        })
        results_df.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)
        
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
    
    return results

def load_and_evaluate(checkpoint_path, model_class, model_kwargs, test_loader, device, save_dir=None):
    """
    Load a model from checkpoint and evaluate it
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model initialization
        test_loader: Test data loader
        device: Device for evaluation
        save_dir: Directory to save evaluation results
    
    Returns:
        dict: Evaluation results
    """
    model = model_class(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    
    results = full_evaluation(model, test_loader, device, save_dir)
    
    return results