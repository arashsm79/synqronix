import torch
import argparse
import os
import sys

from synqronix.dataproc import dataloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from torch.nn import LeakyReLU, Linear, ReLU, Dropout
from synqronix.dataproc.dataloader import NeuralGraphDataLoader, ColumnarNeuralGraphDataLoader
from synqronix.models.gnn import NeuralGNN, NeuralGNNWithAttention
from synqronix.trainer import GNNTrainer
from synqronix.evaluation import full_evaluation, plot_training_curves, load_and_evaluate
from synqronix.models.qgcn import QGCN
from synqronix.config import define_parameters
from synqronix.utils import setup_quantum_device


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_data(args):
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
    eval_dir = os.path.join(args.save_dir, 'evaluation')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    print("Loading and processing data...")
    if args.dataset == 'functional':
        dataloader = NeuralGraphDataLoader(
            data_dir=args.data_dir,
            k=args.k,
            connectivity_threshold=args.connectivity_threshold,
            batch_size=args.batch_size
        )
    elif args.dataset == 'anatomical':
        dataloader = ColumnarNeuralGraphDataLoader(
            data_dir=args.data_dir,
            k=args.k,
            column_width=args.col_width,
            column_height=args.col_height,
            batch_size=args.batch_size
        )
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset}")
    
    train_loader, val_loader, test_loader = dataloader.get_dataloaders()
    
    print(f"Data loaded successfully!")
    print(f"Number of features: {dataloader.get_num_features()}")
    print(f"Number of classes: {dataloader.get_num_classes()}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    return dataloader, train_loader, val_loader, test_loader, device, checkpoint_dir, eval_dir

def main(args, dataloader, train_loader, val_loader, test_loader, device, checkpoint_dir, eval_dir):
    model_kwargs = {
        'num_features': dataloader.get_num_features(),
        'num_classes': dataloader.get_num_classes(),
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout_rate': args.dropout_rate
    }
    
    if args.model_type == 'AttentionGNN':
        model = NeuralGNNWithAttention(**model_kwargs)

    elif args.model_type == 'QGCN':
        quantum_device = setup_quantum_device(num_features=model_kwargs['num_features'], api_key=args.api_key,
                                               quantum_device=args.quantum_device)
        model = QGCN(input_dims=model_kwargs['num_features'], q_depths=args.q_depths, 
                     output_dims=model_kwargs['num_classes'], activ_fn=LeakyReLU(0.2),
                     dropout_rate=args.dropout_rate, hidden_dim=args.hidden_dim, 
                     readout=False, quantum_device=quantum_device)
    else:
        model_kwargs['model_type'] = args.model_type
        model = NeuralGNN(**model_kwargs)
    
    print(f"Model created: {args.model_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.evaluate_only:
        print("Evaluation only mode...")
        model_class = NeuralGNNWithAttention if args.model_type == 'AttentionGNN' else NeuralGNN
        results = load_and_evaluate(
            args.evaluate_only, model_class, model_kwargs, 
            test_loader, device, eval_dir
        )
        return
    
    trainer = GNNTrainer(
        model=model,
        device=device,
        save_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq
    )
    
    trainer.setup_optimizer(
        optimizer_type='Adam',
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    print("Starting training...")
    train_losses, val_losses, train_accs, val_accs = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        resume_from=args.resume_from
    )
    
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs, trainer.val_f1_scores,
        save_path=os.path.join(eval_dir, 'training_curves.png')
    )
    
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print("\nLoading best model for final evaluation...")
        trainer.load_checkpoint(best_model_path)
    
    print("Performing final evaluation on test set...")
    
    class_names = [f"BF_{cls}" for cls in dataloader.label_encoder.classes_]
    
    test_results = full_evaluation(
        model=trainer.model,
        test_loader=test_loader,
        device=device,
        save_dir=eval_dir,
        class_names=class_names
    )
    
    print(f"\nTraining completed!")
    print(f"Results saved in: {args.save_dir}")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Final test accuracy: {test_results['metrics']['accuracy']:.4f}")
    print(f"Final test F1 score: {test_results['metrics']['f1_score']:.4f}")
    print(f"Final test ROC AUC: {test_results['metrics']['roc_auc']:.4f}")

if __name__ == "__main__":
    args = define_parameters()
    (data_loader, train_loader, val_loader, 
    test_loader, device, checkpoint_dir, eval_dir) = build_data(args)
    main(args, data_loader, train_loader, val_loader, test_loader, device, checkpoint_dir, eval_dir)
