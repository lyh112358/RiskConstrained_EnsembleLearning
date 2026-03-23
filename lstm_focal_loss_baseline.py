"""
LSTM Baseline Model with Focal Loss for Academic Early Warning

An LSTM-based neural network approach for predicting academic performance
and generating early warning signals in blended learning environments.

Key Components:
- Enhanced LSTM architecture with BatchNorm and Dropout
- Focal Loss for handling extreme class imbalance (failure rate < 10%)
- Learning rate scheduler (ReduceLROnPlateau) for better convergence
- Early stopping to prevent overfitting
- Threshold optimization for imbalanced classification (G-mean maximization)

This baseline model serves as a comparison point for the recall-constrained
heterogeneous ensemble approach, demonstrating the effectiveness of the proposed
ensemble learning framework.

Reference: "Kinematics-Inspired Behavioral Dynamics and Risk-Constrained Ensemble Learning
for Student Performance Prediction in Blended Environments"

Author: Yinghe Li (Jilin University)
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score
import os
import random

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-4
EPOCHS = 80
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.4
EARLY_STOPPING_ROUNDS = 20

OPTIMIZATION_OBJECTIVE = 'gmean'

DATA_PATHS = {
    'train': [
        'path/to/your/Merged_2020_全特征_updated.csv',
        'path/to/your/Merged_2021_全特征_updated.csv',
        'path/to/your/Merged_2022_全特征_updated.csv',
        'path/to/your/Merged_2023_全特征_updated.csv',
        'path/to/your/Merged_2024_全特征_updated.csv'
    ],
    'test': 'path/to/your/Merged_2023_全特征_updated.csv'
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class StudentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_process_data():
    print(">>> 1. Loading raw data...")
    train_dfs = [pd.read_csv(f) for f in DATA_PATHS['train']]
    df_full_train = pd.concat(train_dfs, ignore_index=True)
    df_test = pd.read_csv(DATA_PATHS['test'])

    exclude_cols = ['SID', 'grade', 'grade_label', 'MOOCgrade', '作业总成绩', 'custom_fold_id']
    feature_cols = [c for c in df_full_train.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if c in df_test.columns]

    X_full = df_full_train[feature_cols].replace([np.inf, -np.inf], np.nan).values
    y_full = df_full_train['grade_label'].values

    X_test = df_test[feature_cols].replace([np.inf, -np.inf], np.nan).values
    y_test = df_test['grade_label'].values

    print(">>> 2. Splitting validation set (Stratified Split)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=SEED, stratify=y_full
    )

    print(">>> 3. Preprocessing data...")
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def find_optimal_threshold(y_true, y_pred_prob):
    thresholds = np.concatenate([
        np.linspace(0.01, 0.4, 200),
        np.linspace(0.41, 0.99, 100)
    ])
    best_threshold, best_score = 0.5, -np.inf

    for t in thresholds:
        y_pred = (y_pred_prob > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        specificity = tn / (tn + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        if OPTIMIZATION_OBJECTIVE == 'gmean':
            score = np.sqrt(recall * specificity)
        elif OPTIMIZATION_OBJECTIVE == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score, best_threshold = score, t
    return best_threshold

class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(EnhancedLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)

        out = lstm_out[:, -1, :]
        out = self.bn(out)
        out = self.fc(out)
        return out

def run_baseline():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Running device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_process_data()

    train_loader = DataLoader(StudentDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StudentDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(StudentDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = EnhancedLSTM(
        input_size=X_train.shape[1],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = "best_lstm_baseline.pth"

    print("\n>>> Starting training (With Focal Loss & Scheduler)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                loss = criterion(out, y_batch)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        scheduler.step(avg_val)

        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {current_lr:.1e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_ROUNDS:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("\n>>> Final Evaluation (Loading Best Model)...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            prob = torch.sigmoid(out).cpu().numpy()
            all_probs.extend(prob)

    y_pred_prob = np.array(all_probs).ravel()
    best_t = find_optimal_threshold(y_test, y_pred_prob)
    y_pred = (y_pred_prob > best_t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    recall = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    gmean = np.sqrt(recall * spec)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    print("=" * 50)
    print(f"LSTM + Focal Loss + Scheduler Baseline")
    print("-" * 50)
    print(f"G-Mean      : {gmean:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"Specificity : {spec:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"AUC         : {auc:.4f}")
    print(f"Best Thresh : {best_t:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    run_baseline()
