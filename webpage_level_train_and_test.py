# Train, valid and test in subpage recognition scenario of HiFi WF
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda.amp as amp
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.special import expit
import time
import sys
from pathlib import Path
import copy

HiFi_WF_path = Path('.../modle')             # Input the path of HiFi WF modle
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

def hierarchical_constraint_loss(main_logits, sub_logits):
    """Hierarchical Constraint Loss function with bidirectional constraints"""
    main_probs = torch.sigmoid(main_logits)
    sub_probs = torch.sigmoid(sub_logits)
    
    constraint_mask = torch.repeat_interleave(main_probs, 10, dim=-1)
    
    # Constraint 1: When parent class is inactive, sub-classes should be close to 0
    inactive_parent = (constraint_mask < 0.3).float()
    inactive_loss = F.binary_cross_entropy_with_logits(
        sub_logits, 
        torch.zeros_like(sub_logits),
        reduction='none'
    )
    loss_inactive = torch.mean(inactive_loss * inactive_parent)
    
    # Constraint 2: When parent class is active, sub-classes should maintain reasonable values
    active_parent = (constraint_mask > 0.7).float()
    loss_active = F.mse_loss(
        sub_probs * active_parent, 
        constraint_mask * active_parent, 
        reduction='mean'
    )
    
    # Constraint 3: When sub-class is active, parent class must be active
    active_child = (sub_probs > 0.5).float()
    main_logits_expanded = torch.repeat_interleave(main_logits, 10, dim=-1)
    loss_consistency = F.binary_cross_entropy_with_logits(
        main_logits_expanded,
        active_child,
        reduction='mean'
    )
    
    return 0.3 * loss_inactive + 0.2 * loss_active + 0.3 * loss_consistency

def hierarchical_loss(main_preds, sub_preds, main_labels, sub_labels):
    """
    Multi-task hierarchical loss function
    Tips: Through our extensive experiments, we recommend that the parameters for Hierarchical Focal Loss used in all scenarios be as follows:
    Scenario 1: 2-tab Chrome datasets: main_loss: 2.5, sub_loss: 1.0, constraint_loss: 0.2
    Scenario 2: 3-tab Chrome datasets: main_loss: 1.5, sub_loss: 1.0, constraint_loss: 0.2
    Scenario 3: 2-tab Tor datasets: main_loss: 0.5, sub_loss: 1.0, constraint_loss: 0.2
    """
    if main_labels.dim() == 2:
        main_labels = main_labels.unsqueeze(1).expand(-1, 2, -1)
    if sub_labels.dim() == 2:
        sub_labels = sub_labels.unsqueeze(1).expand(-1, 2, -1)
    
    main_loss = focal_loss_with_logits(main_preds, main_labels)
    sub_loss = focal_loss_with_logits(sub_preds, sub_labels)
    constraint_loss = hierarchical_constraint_loss(main_preds, sub_preds)
    
    return 2.5 * main_loss + 1.0 * sub_loss + 0.2 * constraint_loss

def find_optimal_threshold_by_f1(y_true, y_prob, num_thresholds=100):
    """Find the optimal threshold based on F1 score"""
    thresholds = np.linspace(0, 1, num_thresholds)
    best_threshold = 0.5
    best_f1 = 0
    
    for th in thresholds:
        y_pred = (y_prob > th).astype(int)
        f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
            
    # Threshold selection logic optimized for validation-set F1, consistent with the reasoning in the Appendix C
    return best_threshold
    
def main():
    # 1. Data preparation
    print("Loading dataset...")
    data = np.load('.../dataset.npz')    # Input dataset
    X = data['X']
    y_main = data['y_main']
    y_sub = data['y_sub']

    X = torch.FloatTensor(X)
    y_main = torch.FloatTensor(y_main)
    y_sub = torch.FloatTensor(y_sub)
    
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.6667, random_state=42)
    
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_main_train, y_main_val, y_main_test = y_main[train_idx], y_main[val_idx], y_main[test_idx]
    y_sub_train, y_sub_val, y_sub_test = y_sub[train_idx], y_sub[val_idx], y_sub[test_idx]
    
    batch_size = 64
    num_workers = 4
    
    train_dataset = TensorDataset(X_train, y_main_train, y_sub_train)
    val_dataset = TensorDataset(X_val, y_main_val, y_sub_val)
    test_dataset = TensorDataset(X_test, y_main_test, y_sub_test)
    
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
    
    model = HiFi_WF(num_main=50, num_sub=500, in_channels=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    # 3. Training function
    def train_one_epoch(epoch):
        model.train()
        total_loss = 0
        main_targets = []
        main_outputs = []
        sub_targets = []
        sub_outputs = []
        start_time = time.time()
    
        lr = 0.001 * (0.95 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
        for batch_X, batch_y_main, batch_y_sub in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y_main = batch_y_main.to(device, non_blocking=True)
            batch_y_sub = batch_y_sub.to(device, non_blocking=True)
        
            with torch.cuda.amp.autocast():
                main_preds, sub_preds = model(batch_X)
                loss = hierarchical_loss(main_preds, sub_preds, batch_y_main, batch_y_sub)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
            total_loss += loss.item()
            main_targets.append(batch_y_main.detach().cpu().numpy())
            main_outputs.append(main_preds.detach().cpu().numpy())
            sub_targets.append(batch_y_sub.detach().cpu().numpy())
            sub_outputs.append(sub_preds.detach().cpu().numpy())
    
        current_lr = optimizer.param_groups[0]['lr']
        main_targets = np.concatenate(main_targets)
        main_outputs = np.concatenate(main_outputs)
        main_probs = expit(main_outputs) 
        main_probs_fused = np.max(main_probs, axis=1)
        main_preds = (main_probs_fused > 0.5).astype(int)
    
        main_recall = recall_score(main_targets, main_preds, average='micro', zero_division=0)
        main_map_at_k = calculate_mapk(main_targets, main_probs_fused, k= )  # Set k as per scenario (see Table 2 in the paper)
        
        sub_targets = np.concatenate(sub_targets)
        sub_outputs = np.concatenate(sub_outputs)
        sub_probs = expit(sub_outputs) 
        sub_probs_fused = np.max(sub_probs, axis=1) 
        sub_preds = (sub_probs_fused > 0.5).astype(int)
    
        sub_recall = recall_score(sub_targets, sub_preds, average='micro', zero_division=0)
        sub_map_at_k = calculate_mapk(sub_targets, sub_probs_fused, k= )  # Set k as per scenario (see Table 2 in the paper)
    
        epoch_time = time.time() - start_time
    
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f} - "
              f"Time: {epoch_time:.2f}s - "
              f"Homepage recall: {main_recall:.4f} - Subpage recall: {sub_recall:.4f} - "
              f"Homepage mAP@k: {main_map_at_k:.4f} - Subpage mAP@k: {sub_map_at_k:.4f} - "
              f"LR: {current_lr:.6f}")
    
        return total_loss/len(train_loader), main_recall, sub_recall

    # 4. Evaluation function
    def evaluate(loader, set_name="Test set"):
        model.eval()
        main_targets = []
        main_outputs = []
        sub_targets = []
        sub_outputs = []
        
        with torch.no_grad():
            for batch_X, batch_y_main, batch_y_sub in loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y_main = batch_y_main.to(device, non_blocking=True)
                batch_y_sub = batch_y_sub.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    main_preds, sub_preds = model(batch_X)
                
                main_targets.append(batch_y_main.cpu().numpy())
                main_outputs.append(main_preds.cpu().numpy())
                sub_targets.append(batch_y_sub.cpu().numpy())
                sub_outputs.append(sub_preds.cpu().numpy())
        
        main_targets = np.concatenate(main_targets)
        main_outputs = np.concatenate(main_outputs)
        main_probs = expit(main_outputs)  # [N,2,C]
        
        # Max-pooling for probability fusion is adopted
        main_probs_fused = np.max(main_probs, axis=1)  # [N,C]
        optimal_th_main = find_optimal_threshold_by_f1(main_targets, main_probs_fused)
        main_preds = (main_probs_fused > optimal_th_main).astype(int)  # [N,C]
        
        main_precision = precision_score(main_targets, main_preds, average='micro', zero_division=0)
        main_recall = recall_score(main_targets, main_preds, average='micro', zero_division=0)
        main_f1 = f1_score(main_targets, main_preds, average='micro', zero_division=0)
        main_p_at_k = calculate_pak(main_targets, main_probs_fused, k= )          # Set k as per scenario (see Table 2 in the paper)
        main_map_at_k = calculate_mapk(main_targets, main_probs_fused, k= )       # Set k as per scenario (see Table 2 in the paper)
        
        sub_targets = np.concatenate(sub_targets)
        sub_outputs = np.concatenate(sub_outputs)
        sub_probs = expit(sub_outputs)  # [N,2,C]

        # Max-pooling for probability fusion is adopted
        sub_probs_fused = np.max(sub_probs, axis=1)  # [N,C]
        optimal_th_sub = find_optimal_threshold_by_f1(sub_targets, sub_probs_fused)
        sub_preds = (sub_probs_fused > optimal_th_sub).astype(int)  # [N,C]
        
        sub_precision = precision_score(sub_targets, sub_preds, average='micro', zero_division=0)
        sub_recall = recall_score(sub_targets, sub_preds, average='micro', zero_division=0)
        sub_f1 = f1_score(sub_targets, sub_preds, average='micro', zero_division=0)
        sub_p_at_k = calculate_pak(sub_targets, sub_probs_fused, k= )                 # Set k as per scenario (see Table 2 in the paper)
        sub_map_at_k = calculate_mapk(sub_targets, sub_probs_fused, k= )              # Set k as per scenario (see Table 2 in the paper)
        
        print(f"\n{set_name} results:")
        print("Homepage metrics:")
        print(f"Precision: {main_precision:.4f} - "
              f"Recall: {main_recall:.4f} - F1 score: {main_f1:.4f} - "
              f"P@k: {main_p_at_k:.4f} - mAP@k: {main_map_at_k:.4f}")
        
        print("\nSubpage metrics:")
        print(f"Optimal threshold: {optimal_th_sub:.4f}")
        print(f"Precision: {sub_precision:.4f} - "
              f"Recall: {sub_recall:.4f} - F1 score: {sub_f1:.4f} - "
              f"P@k: {sub_p_at_k:.4f} - mAP@k: {sub_map_at_k:.4f}")
        
        return {
            'main': {
                'precision': main_precision,
                'recall': main_recall,
                'f1': main_f1,
                'p_at_k': main_p_at_k, 
                'map_at_k': main_map_at_k,  
                'optimal_th': optimal_th_main
            },
            'sub': {
                'precision': sub_precision,
                'recall': sub_recall,
                'f1': sub_f1,
                'p_at_k': sub_p_at_k,  
                'map_at_k': sub_map_at_k,  
                'optimal_th': optimal_th_sub
            }
        }

    # 5. Start training and testing (add saving of the best model)
    print("\nStarting training...")
    best_f1 = 0
    best_model_state = None
    
    for epoch in range():  # Input the number of epochs
        train_one_epoch(epoch)
        
    """
    Tips: Through our extensive experiments, we recommend that the parameters in all scenarios be as follows:
    Scenario 1: 2-tab Chrome datasets: 100 epochs
    Scenario 2: 3-tab Chrome datasets: 200 epochs
    Scenario 3: 2-tab Tor datasets: 150 epochs
    """
    
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'model_epoch{epoch+1}.pth')
            print(f"\nValidation set evaluation (Epoch {epoch+1}):")
            val_results = evaluate(val_loader, "Validation set")
            
            # Save the model with the highest F1 score of homepages or subpages
            if val_results['main']['f1'] > best_f1:
               best_f1 = val_results['main']['f1']
               best_model_state = model.state_dict()
               torch.save(best_model_state, 'model_best_f1.pth')
    
    print("\nTraining completed, starting testing...")
    model.load_state_dict(torch.load('model_best_f1.pth'))  # Load the best model
    test_results = evaluate(test_loader, "Test set")
    torch.save(model.state_dict(), 'model_final.pth')

if __name__ == "__main__":
    torch.manual_seed(42)  
    np.random.seed(42)    # Set random seeds to ensure reproducibility
    torch.backends.cudnn.benchmark = True
    main()