# Train, valid and test in Website Attack scenario of HiFi WF
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda.amp as amp
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.special import expit
import time
import sys
from pathlib import Path
import copy
import math

HiFi_WF_path = Path('')             # Input HiFi WF modle
sys.path.append(str(HiFi_WF_path))
from HiFi_WF import HiFi_WF

def calculate_mapk(y_true, y_pred, k):
    """Calculate mAP@k"""
    map_scores = []
    for true_labels, pred_scores in zip(y_true, y_pred):
        num_relevant = min(true_labels.sum(), k)
        if num_relevant == 0:
            continue
            
        top_k_indices = np.argsort(pred_scores)[-k:][::-1]
        precision_at_k = []
        correct = 0
        
        for i, idx in enumerate(top_k_indices, 1):
            if true_labels[idx] == 1:
                correct += 1
                precision_at_k.append(correct / i)
        
        ap = sum(precision_at_k) / num_relevant
        map_scores.append(ap)
    
    return np.mean(map_scores) if map_scores else 0.0

def calculate_pak(y_true, y_pred, k):
    """Calculate P@k"""
    p_scores = []
    for true_labels, pred_scores in zip(y_true, y_pred):
        top_k_indices = np.argsort(pred_scores)[-k:][::-1]
        correct = true_labels[top_k_indices].sum()
        p_scores.append(correct / k)
    return np.mean(p_scores)

def focal_loss_with_logits(logits, targets, alpha=0.75, gamma=5.0):
    """
    Multi-label Focal Loss to address class imbalance issues
    Address the class imbalance problem using Focal Loss. 
    After multiple experiments, 0.75 and 5.0 are selected as the parameter values for alpha and gamma, respectively.
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none')
    
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    modulating_factor = (1.0 - p_t).pow(gamma)
    
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss = modulating_factor * alpha_factor * bce_loss
    
    return focal_loss.mean()

def main_loss(main_preds, main_labels):
    """
    Loss function focusing only on main pages
    Since there is no need for subpage recognition in this scenario, hierarchical constraints are not required, and only Focal Loss is employed
    """
    if main_labels.dim() == 2:
        main_labels = main_labels.unsqueeze(1).expand(-1, 2, -1)
    
    return focal_loss_with_logits(main_preds, main_labels)

def find_optimal_threshold_by_f1(y_true, y_prob, num_thresholds=100):
    """Find optimal threshold based on F1 score"""
    thresholds = np.linspace(0, 1, num_thresholds)
    best_threshold = 0.5
    best_f1 = 0
    
    for th in thresholds:
        y_pred = (y_prob > th).astype(int)
        f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
    
    return best_threshold

def main():
    # 1. Data preparation
    print("Loading dataset...")
    data = np.load('')     # Input dataset
    X = data['X']
    y_main = data['y_main']
    y_sub = data['y_sub']

    X = torch.FloatTensor(X)
    y_main = torch.FloatTensor(y_main)
    
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.6667, random_state=42)
    
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_main_train, y_main_val, y_main_test = y_main[train_idx], y_main[val_idx], y_main[test_idx]
    
    batch_size = 32
    num_workers = 4
    
    train_dataset = TensorDataset(X_train, y_main_train)
    val_dataset = TensorDataset(X_val, y_main_val)
    test_dataset = TensorDataset(X_test, y_main_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # 2. Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = HAMSIC(num_main=50, num_sub=500, in_channels=2).to(device)
    
    # Adjust optimizer parameters
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001,  
        weight_decay=1e-5, 
        betas=(0.9, 0.999) 
    )
    
    scaler = torch.cuda.amp.GradScaler()

    # 3. Training function
    def train_one_epoch(epoch):
        model.train()
        total_loss = 0
        main_targets = []
        main_outputs = []
        start_time = time.time()
    
        # Use cosine annealing + warmup
        if epoch < 5:  
            lr = 0.001 * (epoch + 1) / 5
        else: 
            lr = 0.001 * 0.5 * (1 + math.cos(math.pi * (epoch - 5) / 95))
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
        for batch_X, batch_y_main in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y_main = batch_y_main.to(device, non_blocking=True)
        
            with torch.cuda.amp.autocast():
                main_preds, _ = model(batch_X)
                # Use weighted loss function to balance precision and recall to get best F1-Score
                loss = main_loss(main_preds, batch_y_main)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
            total_loss += loss.item()
            main_targets.append(batch_y_main.detach().cpu().numpy())
            main_outputs.append(main_preds.detach().cpu().numpy())
    
        current_lr = optimizer.param_groups[0]['lr']
        main_targets = np.concatenate(main_targets)
        main_outputs = np.concatenate(main_outputs)
        main_probs = expit(main_outputs)
        
        # Extract main page probabilities from model outputs (fuse dual branches)
        sample_main_probs = np.max(main_probs, axis=1)
        
        # Use dynamic threshold
        optimal_th = find_optimal_threshold_by_f1(main_targets, sample_main_probs)

        # Generate predictions using optimal threshold
        main_preds = (sample_main_probs > optimal_th).astype(int)
    
        main_recall = recall_score(main_targets, main_preds, average='micro', zero_division=0)
        main_map_at_k = calculate_mapk(main_targets, sample_main_probs, k=2) 
    
        epoch_time = time.time() - start_time
    
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f} - "
              f"Time: {epoch_time:.2f}s - "
              f"Main page recall: {main_recall:.4f} - "
              f"Main page mAP@K: {main_map_at_k:.4f} - "  
              f"Optimal threshold: {optimal_th:.4f} - "
              f"LR: {current_lr:.6f}")
    
        return total_loss/len(train_loader), main_recall

    # 4. Evaluation function
    def evaluate(loader, set_name="Test set"):
        model.eval()
        main_targets = []
        main_outputs = []
        
        with torch.no_grad():
            for batch_X, batch_y_main in loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y_main = batch_y_main.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    main_preds, _ = model(batch_X)
                
                main_targets.append(batch_y_main.cpu().numpy())
                main_outputs.append(main_preds.cpu().numpy())
        
        main_targets = np.concatenate(main_targets)
        main_outputs = np.concatenate(main_outputs)
        main_probs = expit(main_outputs)
        
        # Extract main page probabilities from model outputs (fuse dual branches)
        sample_main_probs = np.max(main_probs, axis=1)

        # Dynamically find optimal threshold for current training set
        optimal_th_main = find_optimal_threshold_by_f1(main_targets, sample_main_probs)
        
        # Generate predictions using optimal threshold
        main_preds = (sample_main_probs > optimal_th_main).astype(int)
        
        main_accuracy = accuracy_score(main_targets, main_preds)
        main_precision = precision_score(main_targets, main_preds, average='micro', zero_division=0)
        main_recall = recall_score(main_targets, main_preds, average='micro', zero_division=0)
        main_f1 = f1_score(main_targets, main_preds, average='micro', zero_division=0)
        main_p_at_k = calculate_pak(main_targets, sample_main_probs, k=2)  
        main_map_at_k = calculate_mapk(main_targets, sample_main_probs, k=2)  
        
        print(f"\n{set_name} results:")
        print("Main page metrics:")
        print(f"Accuracy: {main_accuracy:.4f} - Precision: {main_precision:.4f} - "
              f"Recall: {main_recall:.4f} - F1 score: {main_f1:.4f} - "
              f"P@K: {main_p_at_k:.4f} - mAP@K: {main_map_at_k:.4f} - Optimal threshold: {optimal_th_main:.4f}")  # Print text: P@2/mAP@2 → P@K/mAP@K
        
        return {
            'main': {
                'accuracy': main_accuracy,
                'precision': main_precision,
                'recall': main_recall,
                'f1': main_f1,
                'p_at_k': main_p_at_k, 
                'map_at_k': main_map_at_k,  
                'optimal_th': optimal_th_main
            }
        }

    # 5. Training and testing
    print("\nStarting training...")
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(80): 
        train_loss, train_recall = train_one_epoch(epoch)
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'model_epoch{epoch+1}.pth')
            print(f"\nValidation set evaluation (Epoch {epoch+1}):")
            val_results = evaluate(val_loader, "Validation set")
            
            if val_results['main']['f1'] > best_f1:
                best_f1 = val_results['main']['f1']
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, 'model_best_f1.pth')
                print(f"Best model saved, F1 score: {best_f1:.4f}")
    
    print("\nTraining completed, starting testing...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_results = evaluate(test_loader, "Test set")
    torch.save(model.state_dict(), 'model_final.pth')

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    main()