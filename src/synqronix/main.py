import torch
import argparse
import os
import numpy as np
from synqronix.dataproc.dataloader import NeuralGraphDataLoader, ColumnarNeuralGraphDataLoader
from synqronix.models.gnn import NeuralGNN, NeuralGNNWithAttention
from synqronix.trainer import GNNTrainer
from synqronix.evaluation import full_evaluation, plot_training_curves, load_and_evaluate


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Neural Graph Classification with GNN')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing .mat files')
    parser.add_argument('--save_dir', type=str, default='./results',
                      help='Directory to save results and checkpoints')
    parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN', 'GAT', 'AttentionGNN'],
                      help='Type of GNN model to use')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                      help='Number of GNN layers')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                      help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=500,
                      help='Number of training epochs')
    parser.add_argument('--k', type=int, default=20,
                      help='Number of top connected neurons for each neuron')
    parser.add_argument('--connectivity_threshold', type=float, default=0.5,
                      help='Threshold for functional connectivity')
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--resume_from', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    parser.add_argument('--evaluate_only', type=str, default=None,
                      help='Path to checkpoint for evaluation only')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--col_width', type=int, default=50,
                      help='Column width for anatomical dataset')
    parser.add_argument('--col_height', type=int, default=30,
                      help='Column height for anatomical dataset')
    parser.add_argument('--dataset', type=str, default='functional', choices=['functional', 'anatomical'],
                      help='Type of dataset to use (functional or anatomical)')
    
    args = parser.parse_args()
    
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
    
    model_kwargs = {
        'num_features': dataloader.get_num_features(),
        'num_classes': dataloader.get_num_classes(),
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout_rate': args.dropout_rate
    }
    
    if args.model_type == 'AttentionGNN':
        model = NeuralGNNWithAttention(**model_kwargs)
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
    main()