# Train, valid and test in open-world scenario
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support  # New import
from sklearn.model_selection import train_test_split  
import time
import sys
import multiprocessing
from pathlib import Path
import copy

if sys.platform.startswith('win'):
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

HiFi_WF_path = Path('')      # Input HiFi WF model
sys.path.append(str(HiFi_WF_path))
from HiFi_WF import HiFi_WF

def calculate_pak(y_true, y_pred, k):
    """Calculate P@k"""
    p_scores = []
    for true_labels, pred_scores in zip(y_true, y_pred):
        top_k_indices = np.argsort(pred_scores)[-k:][::-1]
        correct = true_labels[top_k_indices].sum()
        p_scores.append(correct / k)
    return np.mean(p_scores)

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

def calculate_pre_rec_f1(y_true, y_prob, threshold=0.5):
    """Calculate Precision, Recall and F1-Score based on threshold inference method"""
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='samples', zero_division=0
    )
    return precision, recall, f1

def focal_loss_with_logits(logits, targets, alpha=0.75, gamma=5.0):
    """
    Address the class imbalance problem using Focal Loss. 
    """
    if logits.dim() == 3:
        logits = logits.mean(dim=1) 
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
    if main_logits.dim() == 3:
        main_logits = main_logits.mean(dim=1)
    if sub_logits.dim() == 3:
        sub_logits = sub_logits.mean(dim=1)
        
    main_probs = torch.sigmoid(main_logits)
    sub_probs = torch.sigmoid(sub_logits)
    constraint_mask = torch.repeat_interleave(main_probs, 10, dim=-1)
    
    # Constraint 1: When parent class is inactive, sub-classes should be close to 0
    inactive_parent = (constraint_mask < 0.3).float()
    inactive_loss = F.binary_cross_entropy_with_logits(
        sub_logits, torch.zeros_like(sub_logits), reduction='none')
    loss_inactive = torch.mean(inactive_loss * inactive_parent)
    
    # Constraint 2: When parent class is active, sub-classes should maintain reasonable values
    active_parent = (constraint_mask > 0.7).float()
    loss_active = F.mse_loss(
        sub_probs * active_parent, constraint_mask * active_parent, reduction='mean')
    
    # Constraint 3: When sub-class is active, parent class must be active
    active_child = (sub_probs > 0.5).float()
    main_logits_expanded = torch.repeat_interleave(main_logits, 10, dim=-1)
    loss_consistency = F.binary_cross_entropy_with_logits(
        main_logits_expanded, active_child, reduction='mean')
    
    return 0.3 * loss_inactive + 0.2 * loss_active + 0.3 * loss_consistency

def hierarchical_loss(main_preds, sub_preds, main_labels, sub_labels):
    """Multi-task hierarchical loss function"""
    main_loss = focal_loss_with_logits(main_preds, main_labels)
    sub_loss = focal_loss_with_logits(sub_preds, sub_labels)
    constraint_loss = hierarchical_constraint_loss(main_preds, sub_preds)
    return 2.5 * main_loss + 1.0 * sub_loss + 0.3 * constraint_loss

def calculate_binary_auc(y_true, y_score):
    """Calculate binary AUC"""
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_score)

def calculate_multi_class_auc(y_true, y_prob):
    """Calculate multi-class AUC in homepage scenario and subpage scenario, respectively"""
    if len(y_true) == 0:
        return 0.0
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(1, -1)
    aucs = []
    for class_idx in range(y_true.shape[1]):
        class_true = y_true[:, class_idx]
        if np.sum(class_true) in (0, len(y_true)):
            continue
        try:
            class_auc = roc_auc_score(class_true, y_prob[:, class_idx])
            aucs.append(class_auc)
        except:
            continue
    return np.mean(aucs) if aucs else 0.0

def main():
    print("Loading dataset...")
    train_data = np.load(' ')     # Input training dataset
    temp_data = np.load(' ')     # Input temporary dataset, which will be split into validating dataset and testing dataset

    X_train = torch.FloatTensor(train_data['X']) 
    y_main_train = torch.FloatTensor(train_data['y_main'])
    y_sub_train = torch.FloatTensor(train_data['y_sub'])
    
    X_temp = torch.FloatTensor(temp_data['X']) 
    y_main_temp = torch.FloatTensor(temp_data['y_main'])
    y_sub_temp = torch.FloatTensor(temp_data['y_sub'])
    
    print("...")
    X_val, X_test, y_main_val, y_main_test, y_sub_val, y_sub_test = train_test_split(
        X_temp, y_main_temp, y_sub_temp, 
        test_size=0.7,
        random_state=42,
        stratify=None
    )

    batch_size = 32
    num_workers = 0 if sys.platform.startswith('win') else min(4, multiprocessing.cpu_count())
    pin_memory = torch.cuda.is_available() and num_workers > 0
    
    train_dataset = TensorDataset(X_train, y_main_train, y_sub_train)
    val_dataset = TensorDataset(X_val, y_main_val, y_sub_val)
    test_dataset = TensorDataset(X_test, y_main_test, y_sub_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = HiFi_WF(num_main=50, num_sub=500, in_channels=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Consistent LR with TMWF
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)  # Consistent scheduler
    scaler = torch.cuda.amp.GradScaler()


    best_aucm = 0.0
    best_model_path = "hifiwf_best_aucm.pth"

    # 6. Training function
    def train_one_epoch(epoch):
        model.train()
        total_loss = 0
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
    
        for batch_X, batch_y_main, batch_y_sub in train_loader:
            # Confirm and correct the input data dimensions
            if batch_X.ndim == 2:
                batch_X = batch_X.unsqueeze(1)
            if batch_X.shape[1] == 1 and model.in_channels == 2: 
                batch_X = batch_X.repeat(1, 2, 1) 
            
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
    
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f} - "
              f"Time: {epoch_time:.2f}s - LR: {current_lr:.6f}")
        return total_loss / len(train_loader)

    # 7. Evaluate function
    def evaluate(loader, set_name="Test set"):
        model.eval()
        all_max_score = []
        all_is_monitored = []
        main_targets = []
        sub_targets = []
        main_probs_all = []
        sub_probs_all = []
        
        with torch.no_grad():
            for batch_X, batch_y_main, batch_y_sub in loader:
            # Confirm and correct the input data dimensions
                if batch_X.ndim == 2:
                    batch_X = batch_X.unsqueeze(1)
                if batch_X.shape[1] == 1 and model.in_channels == 2:
                    batch_X = batch_X.repeat(1, 2, 1)
                
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y_main = batch_y_main.to(device, non_blocking=True)
                batch_y_sub = batch_y_sub.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    main_preds, sub_preds = model(batch_X)
            
                main_probs = torch.sigmoid(main_preds)
                sub_probs = torch.sigmoid(sub_preds)
                if main_probs.dim() == 3:
                    main_probs = main_probs.mean(dim=1)  
                if sub_probs.dim() == 3:
                    sub_probs = sub_probs.mean(dim=1)
                
                # Convert to numpy for metric calculation
                main_probs_np = main_probs.cpu().numpy()
                sub_probs_np = sub_probs.cpu().numpy()
                
                # Open-world metric calculation
                batch_max_score = np.maximum(
                    np.max(main_probs_np, axis=1),
                    np.max(sub_probs_np, axis=1)
                )
                batch_is_monitored = np.any(
                    np.concatenate([
                        batch_y_main.cpu().numpy(), 
                        batch_y_sub.cpu().numpy()
                    ], axis=1),
                    axis=1
                ).astype(int)
                
                # Collect data
                all_max_score.extend(batch_max_score)
                all_is_monitored.extend(batch_is_monitored)
                main_targets.append(batch_y_main.cpu().numpy())
                sub_targets.append(batch_y_sub.cpu().numpy())
                main_probs_all.append(main_probs_np)
                sub_probs_all.append(sub_probs_np)
        
        # Calculate metrics
        all_max_score = np.array(all_max_score)
        all_is_monitored = np.array(all_is_monitored)
        
        # Calculate AUCM via function "calculate_binary_auc"
        AUCM = calculate_binary_auc(all_is_monitored, all_max_score)
        
        # Calculate AUCN via function "calculate_binary_auc"
        AUCN = calculate_binary_auc(1 - all_is_monitored, 1 - all_max_score)
        
        main_targets = np.concatenate(main_targets)
        sub_targets = np.concatenate(sub_targets)
        main_probs_all = np.concatenate(main_probs_all)
        sub_probs_all = np.concatenate(sub_probs_all)
        
        main_auc = calculate_multi_class_auc(main_targets, main_probs_all)
        main_p_at_2 = calculate_pak(main_targets, main_probs_all, k=2)
        main_map_at_2 = calculate_mapk(main_targets, main_probs_all, k=2)
        # Calculate Precision, Recall, F1 for main page
        main_pre, main_rec, main_f1 = calculate_pre_rec_f1(main_targets, main_probs_all)
        
        sub_auc = calculate_multi_class_auc(sub_targets, sub_probs_all)
        sub_p_at_2 = calculate_pak(sub_targets, sub_probs_all, k=2)
        sub_map_at_2 = calculate_mapk(sub_targets, sub_probs_all, k=2)
        # Calculate Precision, Recall, F1 for sub page
        sub_pre, sub_rec, sub_f1 = calculate_pre_rec_f1(sub_targets, sub_probs_all)
        
        print(f"\n{set_name} results:")
        print("=== Core Open-world Metrics ===")
        print(f"AUCM (AUC for monitored class recognition): {AUCM:.4f}")
        print(f"AUCN (AUC for non-monitored class exclusion): {AUCN:.4f}")
        print("\n=== Homepage Level Metrics ===")
        print(f"Class Discrimination AUC: {main_auc:.4f} - P@2: {main_p_at_2:.4f} - mAP@2: {main_map_at_2:.4f}")
        print(f"Precision: {main_pre:.4f} - Recall: {main_rec:.4f} - F1: {main_f1:.4f}") 
        
        print("\n=== Subpage Level Metrics ===")
        print(f"Class Discrimination AUC: {sub_auc:.4f} - P@2: {sub_p_at_2:.4f} - mAP@2: {sub_map_at_2:.4f}")
        print(f"Precision: {sub_pre:.4f} - Recall: {sub_rec:.4f} - F1: {sub_f1:.4f}")  
        
        return {
            'open_world': {'AUCM': AUCM, 'AUCN': AUCN},
            'main': {'auc': main_auc, 'p_at_2': main_p_at_2, 'map_at_2': main_map_at_2, 
                     'precision': main_pre, 'recall': main_rec, 'f1': main_f1}, 
            'sub': {'auc': sub_auc, 'p_at_2': sub_p_at_2, 'map_at_2': sub_map_at_2, 
                    'precision': sub_pre, 'recall': sub_rec, 'f1': sub_f1}       
        }

    # 8. Training and testing
    print("\nStarting training...")
    epochs = 70 
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(epoch)
        
        # Validate every epoch
        print(f"\nValidation set evaluation (Epoch {epoch+1}):")
        val_results = evaluate(val_loader, "Validation set")
        
        # Save the best model based on AUCM
        current_aucm = val_results['open_world']['AUCM']
        if current_aucm > best_aucm:
            best_aucm = current_aucm
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_aucm': current_aucm
            }, best_model_path)
            print(f"Saved best model! Current AUCM: {current_aucm:.4f}, Best AUCM: {best_aucm:.4f}")

        # Cosine annealing scheduling
        scheduler.step()

        # Save intermediate model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'hamsic_model_epoch{epoch+1}.pth')
            print(f"Saved model at epoch {epoch+1}")

    # 9. Evaluate best model on test set
    print("\n" + "=" * 70)
    print("Evaluating best model on test set...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_results = evaluate(test_loader, "Test set")

    # Save test results
    np.save("test_results_openworld.npy", test_results)
    print("Test results saved to test_results_openworld.npy")
    torch.save(model.state_dict(), 'model_final.pth')

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    main()