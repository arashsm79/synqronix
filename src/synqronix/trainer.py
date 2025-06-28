import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
from tqdm import tqdm
import numpy as np
from synqronix.evaluation import evaluate_model, compute_metrics
from synqronix.models.gnn import NeuralGNN, NeuralGNNWithAttention

class GNNTrainer:
    def __init__(self, model, device, save_dir='checkpoints', checkpoint_freq=5):
        """
        Trainer for Graph Neural Networks
        
        Args:
            model: The GNN model to train
            device: Device to run training on
            save_dir: Directory to save checkpoints
            checkpoint_freq: Save checkpoint every N epochs
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.checkpoint_freq = checkpoint_freq
        self.grace_period = 5
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_f1_scores = []
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def setup_optimizer(self, optimizer_type='Adam', lr=0.001, weight_decay=1e-4):
        """Setup optimizer and scheduler"""
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10)
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        train_bar = tqdm(train_loader, desc="Training")
        
        for batch in train_bar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            loss = self.criterion(out, batch.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            total_correct += (pred == batch.y).sum().item()
            total_samples += batch.y.size(0)
            
            current_acc = total_correct / total_samples if total_samples > 0 else 0
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, avg_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = batch.to(self.device)
                
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                loss = self.criterion(out, batch.y)
                total_loss += loss.item()
                
                probs = torch.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
        }
        
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved at epoch {epoch} with validation accuracy: {self.best_val_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        self.val_f1_scores = checkpoint['val_f1_scores']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader, num_epochs=100, resume_from=None):
        """Main training loop"""
        start_epoch = 0
        
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resuming training from epoch {start_epoch}")
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(train_loader)
            
            val_loss, val_metrics = self.validate_epoch(val_loader)
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['f1_score']
            
            self.scheduler.step(val_acc)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            if (epoch + 1) % self.checkpoint_freq == 0 or is_best and epoch > self.grace_period:
                self.save_checkpoint(epoch, is_best)
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f} | Time: {epoch_time:.2f}s")
            
            if epoch - self.best_epoch > 25:  # No improvement for 25 epochs
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies