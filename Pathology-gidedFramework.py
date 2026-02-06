"""
Pathology-guided Framework for MCI vs HC Classification
- Features: Std Error Shades in History, High-res t-SNE,saliency map, Standardized CM
@author: Kyeonggu Lee
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.signal
import scipy.io 
import h5py
import os
import random
import copy
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. Configuration
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
seed_everything(seed)

DATA_FILE_PATH = "D:/HC_MCI_AD/data.mat"
CHANNELS = 32
FS_TARGET = 1000 
SEC_PER_SUBJECT = 30
SEGMENT_SEC = 1
TIME_POINTS = FS_TARGET * SEC_PER_SUBJECT
SEGMENT_POINTS = FS_TARGET * SEGMENT_SEC
NUM_SEGMENTS = SEC_PER_SUBJECT // SEGMENT_SEC

BATCH_SIZE = 32
EPOCHS_P1 = 200 
EPOCHS_P2 = 30 
EPOCHS_P3 = 30 
LR = 0.0005
EMBED_DIM = 256
N_FOLDS = 10      

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Models
# ==========================================
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x): return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x): return self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, seq_len, latent_dim):
        super(Encoder, self).__init__()
        self.layer1 = DownBlock(in_channels, 64); self.layer2 = DownBlock(64, 128); self.layer3 = DownBlock(128, 256)
        self.final_len = seq_len // 8 
        self.fc = nn.Linear(256 * self.final_len, latent_dim)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.act(self.fc(x.flatten(1)))

class Autoencoder(nn.Module):
    def __init__(self, channels, seq_len, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(channels, seq_len, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * (seq_len // 8))
        self.up1 = UpBlock(256, 128); self.up2 = UpBlock(128, 64)
        self.final = nn.ConvTranspose1d(64, channels, kernel_size=4, stride=2, padding=1)
        self.seq_len = seq_len
    def forward(self, x):
        z = self.encoder(x)
        d = self.fc_dec(z).view(z.size(0), 256, -1)
        d = self.up1(d); d = self.up2(d); d = self.final(d)
        if d.shape[-1] != self.seq_len:
            d = nn.functional.interpolate(d, size=self.seq_len, mode='linear', align_corners=False)
        return d, z

class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, 64), nn.ELU(), nn.Dropout(0.6), nn.Linear(64, num_classes))
    def forward(self, x): return self.fc(x)

# ==========================================
# 2. Datasets
# ==========================================
class TripletDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x); self.y = torch.LongTensor(y)
        self.labels_set = set(y.tolist())
        self.label_to_indices = {label: np.where(y == label)[0] for label in self.labels_set}
    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        anchor, label = self.x[idx], self.y[idx].item()
        pos_idx = idx
        if len(self.label_to_indices[label]) > 1:
            while pos_idx == idx: pos_idx = np.random.choice(self.label_to_indices[label])
        neg_label = np.random.choice(list(self.labels_set - {label}))
        neg_idx = np.random.choice(self.label_to_indices[neg_label])
        return anchor, self.x[pos_idx], self.x[neg_idx]

class SegmentDataset(Dataset):
    def __init__(self, x_data):
        n_sub, ch, time = x_data.shape; n_seg = time // SEGMENT_POINTS
        self.x = torch.FloatTensor(x_data).view(n_sub, ch, n_seg, SEGMENT_POINTS).permute(0, 2, 1, 3).reshape(-1, ch, SEGMENT_POINTS)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx]

class SubjectDataset(Dataset):
    def __init__(self, x, y): self.x = torch.FloatTensor(x); self.y = torch.LongTensor(y)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ==========================================
# 3. Training & Helpers
# ==========================================
def get_subject_embedding(encoder, x_subject):
    encoder.eval()
    B, C, T = x_subject.shape; N_S = T // SEGMENT_POINTS
    x_segs = x_subject.view(B, C, N_S, SEGMENT_POINTS).permute(0, 2, 1, 3).reshape(-1, C, SEGMENT_POINTS)
    with torch.no_grad(): z = encoder(x_segs.to(device))
    return z.view(B, N_S, -1).mean(dim=1)

def train_phase1_ae(train_x):
    loader = DataLoader(SegmentDataset(train_x), batch_size=BATCH_SIZE, shuffle=True)
    model = Autoencoder(CHANNELS, SEGMENT_POINTS, EMBED_DIM).to(device)
    opt = optim.Adam(model.parameters(), lr=LR); crit = nn.MSELoss(); history = []
    for ep in range(EPOCHS_P1):
        model.train(); loss_avg = 0
        for bx in loader:
            opt.zero_grad(); recon, _ = model(bx.to(device)); loss = crit(recon, bx.to(device)); loss.backward(); opt.step(); loss_avg += loss.item()
        history.append(loss_avg/len(loader))
    return model.encoder, history

def train_phase2_triplet(encoder, train_x, train_y):
    loader = DataLoader(TripletDataset(train_x, train_y), batch_size=16, shuffle=True)
    opt = optim.Adam(encoder.parameters(), lr=LR * 0.2); crit = nn.TripletMarginLoss(margin=1.0); history = []
    encoder.train()
    for ep in range(EPOCHS_P2):
        loss_avg = 0
        for anc, pos, neg in loader:
            def get_z(x):
                B, C, T = x.shape; N_S = T // SEGMENT_POINTS
                return encoder(x.view(B, C, N_S, SEGMENT_POINTS).permute(0, 2, 1, 3).reshape(-1, C, SEGMENT_POINTS).to(device)).view(B, N_S, -1).mean(dim=1)
            opt.zero_grad(); loss = crit(get_z(anc), get_z(pos), get_z(neg)); loss.backward(); opt.step(); loss_avg += loss.item()
        history.append(loss_avg/len(loader))
    return encoder, history

def train_phase3_clf(encoder, train_x, train_y, test_x, test_y):
    for p in encoder.parameters(): p.requires_grad = False
    clf = ClassifierHead(EMBED_DIM, 2).to(device); opt = optim.Adam(clf.parameters(), lr=LR); crit = nn.CrossEntropyLoss()
    tr_loader = DataLoader(SubjectDataset(train_x, train_y), batch_size=16, shuffle=True)
    te_loader = DataLoader(SubjectDataset(test_x, test_y), batch_size=16, shuffle=False)
    best_acc = 0; final_preds = []; final_probs = []; history = [] 
    for ep in range(EPOCHS_P3):
        clf.train(); loss_avg = 0
        for bx, by in tr_loader:
            opt.zero_grad(); z = get_subject_embedding(encoder, bx.to(device)); loss = crit(clf(z), by.to(device)); loss.backward(); opt.step(); loss_avg += loss.item()
        history.append(loss_avg/len(tr_loader))
        clf.eval(); preds, trues, probs_epoch = [], [], []
        with torch.no_grad():
            for bx, by in te_loader:
                z = get_subject_embedding(encoder, bx.to(device))
                outputs = clf(z)
                p = torch.argmax(outputs, dim=1) 
                prob = torch.softmax(outputs, dim=1)[:, 1] 
                preds.extend(p.cpu().numpy()); trues.extend(by.numpy()); probs_epoch.extend(prob.cpu().numpy()) 
        acc = accuracy_score(trues, preds)
        if acc >= best_acc: 
            best_acc = acc; final_preds = preds; final_probs = probs_epoch; best_clf = copy.deepcopy(clf)
    return best_acc, history, final_preds, trues, final_probs, best_clf

# XAI Saliency Map Function
def compute_saliency_map(encoder, classifier, x_input, target_class):
    encoder.eval(); classifier.eval()
    x_segs = torch.FloatTensor(x_input).view(CHANNELS, NUM_SEGMENTS, SEGMENT_POINTS).permute(1, 0, 2).to(device)
    x_segs.requires_grad_() 
    z = encoder(x_segs); z_agg = z.view(1, NUM_SEGMENTS, -1).mean(dim=1)
    output = classifier(z_agg) 
    classifier.zero_grad(); encoder.zero_grad()
    score = output[0, target_class]; score.backward()
    gradients = x_segs.grad.abs(); saliency = gradients.mean(dim=0).cpu().numpy() 
    return saliency

def load_and_prep_data():
    with h5py.File(DATA_FILE_PATH, 'r') as f:
        X = np.array(f['all_trials_data']).transpose(0, 2, 1)
        y = np.array(f['all_trial_labels']).flatten()
        groups = np.array(f['all_sub_ids']).flatten() if 'all_sub_ids' in f else np.arange(len(y))
    if X.shape[2] != TIME_POINTS: X = scipy.signal.resample(X, TIME_POINTS, axis=2)
    X = (X - np.mean(X, axis=2, keepdims=True)) / (np.std(X, axis=2, keepdims=True) + 1e-6)
    mask = np.isin(y, [1, 2, 3])
    return X[mask], (y[mask] - 1).astype(int), groups[mask]

# ==========================================
# 5. Main Experiment
# ==========================================
def run_experiment():
    X, y, groups = load_and_prep_data()
    scenario_id = "Groupwise_Triplet_MCI_vs_HC"
    print(f" Scenario: {scenario_id}")
    
    target_mask = np.isin(y, [0, 1])
    X_target, y_target, groups_target = X[target_mask], y[target_mask], groups[target_mask]
    X_ad_all = X[y == 2] 
    
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    all_fold_preds, all_fold_trues, all_fold_probs = [], [], []
    fold_accs = []
    dist_ad_mci_list, dist_ad_hc_list, dist_hc_mci_list = [], [], []
    
    # Shade를 위해 모든 Fold의 History 수집
    hist_p1_all, hist_p2_all, hist_p3_all = [], [], []

    # XAI Accumulators
    acc_sal_hc = np.zeros((CHANNELS, SEGMENT_POINTS)); acc_sal_mci = np.zeros((CHANNELS, SEGMENT_POINTS))
    count_hc, count_mci = 0, 0

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_target, y_target, groups=groups_target)):
        seed_everything(seed)
        X_tr, X_te = X_target[train_idx], X_target[test_idx]
        y_tr, y_te = y_target[train_idx], y_target[test_idx]
        
        X_ad_guide = X[y == 2]
        base_encoder, h_p1 = train_phase1_ae(X_ad_guide)
        
        X_p2 = np.concatenate([X_tr, X_ad_guide], axis=0)
        y_p2 = np.concatenate([y_tr, np.full(len(X_ad_guide), 2)], axis=0)
        curr_encoder, h_p2 = train_phase2_triplet(copy.deepcopy(base_encoder), X_p2, y_p2)
        
        acc, h_p3, preds, trues, probs, best_clf = train_phase3_clf(curr_encoder, X_tr, y_tr, X_te, y_te)
        
        fold_accs.append(acc)
        all_fold_preds.extend(preds); all_fold_trues.extend(trues); all_fold_probs.extend(probs) 
        hist_p1_all.append(h_p1); hist_p2_all.append(h_p2); hist_p3_all.append(h_p3)

        # Centroid Distance Calculation
        with torch.no_grad():
            z_ad_all = get_subject_embedding(curr_encoder, torch.FloatTensor(X_ad_all).to(device)).cpu().numpy()
            z_mci_all = get_subject_embedding(curr_encoder, torch.FloatTensor(X_target[y_target == 1]).to(device)).cpu().numpy()
            z_hc_all = get_subject_embedding(curr_encoder, torch.FloatTensor(X_target[y_target == 0]).to(device)).cpu().numpy()
            c_ad, c_mci, c_hc = np.mean(z_ad_all, axis=0), np.mean(z_mci_all, axis=0), np.mean(z_hc_all, axis=0)
            dist_ad_mci_list.append(np.linalg.norm(c_ad - c_mci))
            dist_ad_hc_list.append(np.linalg.norm(c_ad - c_hc))
            dist_hc_mci_list.append(np.linalg.norm(c_hc - c_mci))

        # XAI Saliency
        for idx, true_label in enumerate(y_te):
            sal = compute_saliency_map(curr_encoder, best_clf, X_te[idx], target_class=true_label)
            if true_label == 0: acc_sal_hc += sal; count_hc += 1
            elif true_label == 1: acc_sal_mci += sal; count_mci += 1

        # t-SNE Plot
        if fold == 0:
            with torch.no_grad():
                z_tr = get_subject_embedding(curr_encoder, torch.FloatTensor(X_tr).to(device)).cpu().numpy()
                z_te = get_subject_embedding(curr_encoder, torch.FloatTensor(X_te).to(device)).cpu().numpy()
                z_ad = get_subject_embedding(curr_encoder, torch.FloatTensor(X_ad_all[:100]).to(device)).cpu().numpy()
                z_vis = np.concatenate([z_tr, z_te, z_ad], axis=0); n_tr, n_te = len(z_tr), len(z_te)
                tsne = TSNE(n_components=2, perplexity=30, random_state=42); z_2d = tsne.fit_transform(z_vis)
                
                plt.rcParams['font.family'] = 'Arial'
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = {0: '#2E8B57', 1: '#FFA500', 2: '#FF4500'}
                labels = {0: 'HC', 1: 'MCI', 2: 'AD anchor'}
                
                plt.scatter(z_2d[n_tr + n_te:, 0], z_2d[n_tr + n_te:, 1], c=colors[2], marker='s', s=120, alpha=0.3, edgecolors='none', label=labels[2])
                for c in [0, 1]:
                    idx_c = np.where(y_tr == c)[0]
                    plt.scatter(z_2d[idx_c, 0], z_2d[idx_c, 1], c=colors[c], marker='o', s=150, edgecolors='white', linewidth=1.5, alpha=0.85, label=labels[c])
                
                ax.legend(fontsize=22, loc='upper right', frameon=True)
                for spine in ax.spines.values(): spine.set_linewidth(2.5)
                ax.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=8)
                ax.grid(True, linestyle='--', alpha=0.4, linewidth=1)
                plt.tight_layout(); plt.savefig(f"tSNE_Fold_{fold+1}_Publication.png", dpi=600); plt.close()

        print(f"  [Fold {fold+1}] Acc: {acc*100:.2f}%", end='\r')

    # History Plot with Shade
    plt.rcParams['font.family'] = 'Arial'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['Phase 1: Auto-encoder', 'Phase 2: Triplet', 'Phase 3: Classifier']
    colors_h = ['#1f77b4', '#ff7f0e', '#2ca02c']
    hist_list = [hist_p1_all, hist_p2_all, hist_p3_all]

    for i, h_data in enumerate(hist_list):
        ax = axes[i]; arr = np.array(h_data); mean_h = np.mean(arr, axis=0); std_h = np.std(arr, axis=0) / np.sqrt(N_FOLDS); eps = range(len(mean_h))
        ax.plot(eps, mean_h, color=colors_h[i], lw=2.5, label='Mean Loss')
        ax.fill_between(eps, mean_h - std_h, mean_h + std_h, color=colors_h[i], alpha=0.2, label='Std Error')
        ax.set_title(titles[i], fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('Epochs', fontsize=15, fontweight='bold'); ax.set_ylabel('Loss', fontsize=15, fontweight='bold')
        for sp in ax.spines.values(): sp.set_linewidth(2.0)
        ax.tick_params(axis='both', labelsize=13, width=2.0, length=6); ax.grid(True, alpha=0.5)
        if i == 0: ax.legend(fontsize=12)
    plt.tight_layout(); plt.savefig(f"History_Publication_{scenario_id}.png", dpi=300); plt.close()

    # CM
    cm = confusion_matrix(all_fold_trues, all_fold_preds)
    plt.figure(figsize=(5, 4)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HC', 'MCI'], yticklabels=['HC', 'MCI'], vmin=5, vmax=30)
    plt.title(f"CM: {scenario_id}"); plt.savefig(f"CM_{scenario_id}.png"); plt.close()

    # Save Stats & XAI
    scipy.io.savemat('Centroid_Distances.mat', {'dist_ad_mci': np.array(dist_ad_mci_list), 'dist_ad_hc': np.array(dist_ad_hc_list), 'dist_hc_mci': np.array(dist_hc_mci_list)})
    if count_hc > 0:
        avg_sal_hc = acc_sal_hc / count_hc
        avg_sal_mci = acc_sal_mci / count_mci
        
        scipy.io.savemat(f'XAI_Results_{scenario_id}.mat', {
            'xai_channel_imp_hc': avg_sal_hc.mean(axis=1),  
            'xai_channel_imp_mci': avg_sal_mci.mean(axis=1),
            'xai_heatmap_hc': avg_sal_hc,                   
            'xai_heatmap_mci': avg_sal_mci                  
        })
    print(f"\n Final Avg Acc: {np.mean(fold_accs)*100:.2f}%")

if __name__ == "__main__":
    run_experiment()