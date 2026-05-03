import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import librosa
import json
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from collections import Counter
import math
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    # Paths
    "train_csv": "HOME_DIR/classification_model/sg_dataset/balanced_train.csv",
    "test_csv": "HOME_DIR/classification_model/sg_dataset/balanced_test.csv",
    "base_model": "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
    "save_dir": "HOME_DIR/classification_model/model/trained_classifier",
    
    # Parameters
    "sample_rate": 32000,
    "chunk_batch_size": 16,
    "batch_size": 8,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    
    # Early stopping
    "patience": 5,
    "min_delta": 0.001,
    "lambda_radius":0.1,
    "protos_per_class":5,
    
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ==========================================
# 2. DATASET
# ==========================================

def build_label_mapping(train_csv, test_csv):
    """Build mapping from CSV dataset to get num classes and label info."""
    # Load both train and test to get all unique categories
    df_train = pd.read_csv(train_csv, delimiter=';')
    df_test = pd.read_csv(test_csv, delimiter=';')
    
    # Get all unique categories
    all_categories = set(df_train['categories'].unique()) | set(df_test['categories'].unique())
    all_categories = sorted(list(all_categories))
    
    # Create mappings
    category2id = {cat: idx for idx, cat in enumerate(all_categories)}
    id2category = {idx: cat for cat, idx in category2id.items()}
    
    num_classes = len(all_categories)
    
    print(f"✓ Dataset has {num_classes} classes")
    print(f"  Train samples: {len(df_train)}")
    print(f"  Test samples:  {len(df_test)}")
    
    return num_classes, category2id, id2category

class TrainingDataset(Dataset):
    def __init__(self, csv_file, category2id, train_mode=True):
        """
        Args:
            csv_file: Path to CSV file (train or test)
            category2id: Dict mapping category -> id
            train_mode: If True, use random crops; else sliding windows
        """
        self.df = pd.read_csv(csv_file, delimiter=';')
        self.category2id = category2id
        self.sr = CONFIG["sample_rate"]
        self.target_len = self.sr * 5  # 5 seconds
        self.train_mode = train_mode
        
        self.has_segments = 'start_time' in self.df.columns and 'end_time' in self.df.columns
        
        print(f"  Loaded {len(self.df)} samples from {csv_file}")
        if self.has_segments:
            print(f"    Dataset has segment information (start_time, end_time)")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            
            # Get audio path from fullfilename column
            path = row['fullfilename']
            
            # Get category and convert to id
            category = row['categories']
            label_id = self.category2id[category]

            # Get segment times if available
            if self.has_segments:
                start_time = row['start_time']
                end_time = row['end_time']
                segment_duration = row['segment_duration']
            else:
                start_time = 0.0
                end_time = None
                segment_duration = None
            
        except Exception as e:
            print(f"\n Error at index {idx}: {e}")
            print(f"   Row: {row}")
            raise e
        
        # Load and process audio
        chunks = []
        try:
            # Load audio (with segment bounds if available)
            if self.has_segments and end_time is not None:
                # Load only the specific segment
                y, _ = librosa.load(path, sr=self.sr, mono=True, 
                                   offset=start_time, duration=segment_duration)
            else:
                # Load full audio
                y, _ = librosa.load(path, sr=self.sr, mono=True)
            if len(y) > 0:
                if self.train_mode:
                    # TRAINING: Random crop or loop
                    if len(y) < self.target_len:
                        n_repeats = int(np.ceil(self.target_len / len(y)))
                        y = np.tile(y, n_repeats)[:self.target_len]
                    elif len(y) > self.target_len:
                        start = np.random.randint(0, len(y) - self.target_len)
                        y = y[start:start+self.target_len]
                    chunks.append(y)
                else:
                    # INFERENCE: Sliding windows
                    if len(y) < self.target_len:
                        padded = np.zeros(self.target_len, dtype=np.float32)
                        padded[:len(y)] = y
                        chunks.append(padded)
                    else:
                        stride = int(self.target_len * 0.5)
                        for start in range(0, len(y) - self.target_len + 1, stride):
                            chunks.append(y[start : start + self.target_len])
                        if len(chunks) == 0:
                            chunks.append(y[-self.target_len:])
            else:
                chunks.append(np.zeros(self.target_len, dtype=np.float32))
                
        except Exception as e:
            print(f"\n Audio loading error at {idx} for {path}: {e}")
            chunks.append(np.zeros(self.target_len, dtype=np.float32))
        
        return {"audio": chunks, "label": label_id}

# ==========================================
# 3. MODEL
# ==========================================

def extract_centroids(model, train_loader, device, num_classes):
    """
    Computes the average feature vector for each class.
    """
    model.eval()
    # 1024 is the output dim of the ConvNeXt backbone
    class_sums = torch.zeros(num_classes, 1024).to(device)
    class_counts = torch.zeros(num_classes).to(device)
    
    print("📊 Extracting class centroids for prototype initialization...")
    with torch.no_grad():
        for batch in tqdm(train_loader):
            inputs = batch["audio"].to(device)
            labels = batch["label"].to(device)
            
            # Get features from the backbone only
            features = model.backbone(inputs)
            
            # Global Average Pooling (if backbone hasn't flattened it)
            if features.dim() > 2:
                features = features.mean(dim=(2, 3)) if features.dim() == 4 else features.mean(dim=1)
            
            # Accumulate features per class
            for i in range(len(labels)):
                label = labels[i]
                if label != -1:  # Ignore unknown/noise labels if any
                    class_sums[label] += features[i]
                    class_counts[label] += 1
    
    # Calculate means
    # Add small epsilon to avoid division by zero for rare classes
    centroids = class_sums / (class_counts.unsqueeze(1) + 1e-8)
    
    return centroids

def init_multiprototypes_kmeans(model, train_loader, feature_extractor, device, num_classes, protos_per_class=5):
    print("📊 Running K-Means Warm Start for Prototypes...")
    model.eval()
    
    # Dictionary to hold features for each class
    features_per_class = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting Training Features"):
            audio_arrays, labels = batch
        
            inputs = feature_extractor(audio_arrays, padding=True, return_tensors="pt")
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Extract the 512D projected features
            features = model.extract_features(inputs)
            
            for i in range(len(labels)):
                lbl = labels[i].item()
                if lbl != -1: # Skip unknowns if any exist in train
                    features_per_class[lbl].append(features[i].cpu().numpy())
    
    # Initialize a tensor to hold our new centroids
    actual_dim = model.proto_layer.prototypes.shape[-1]
    centroids = torch.zeros(num_classes, protos_per_class, actual_dim)
    
    for c in range(num_classes):
        feats = np.array(features_per_class[c])
        
        if len(feats) >= protos_per_class:
            # Run K-Means to find the distinct call types
            kmeans = KMeans(n_clusters=protos_per_class, n_init=10, random_state=42)
            kmeans.fit(feats)
            centroids[c] = torch.tensor(kmeans.cluster_centers_)
            
        elif len(feats) > 0:
            # Fallback: if a bird has fewer recordings than protos_per_class, just duplicate them
            idx = np.random.choice(len(feats), protos_per_class, replace=True)
            print(f"Bird {c} has less than 5 recordings!!!!")
            centroids[c] = torch.tensor(feats[idx])
            
        else:
            # Fallback: Empty class, leave as random
            nn.init.kaiming_uniform_(centroids[c].unsqueeze(0), a=math.sqrt(5))
            
    # Inject the calculated centroids directly into the model's parameters
    with torch.no_grad():
        model.proto_layer.prototypes.copy_(centroids.to(device))
        
    print(" Prototypes successfully initialized!")

def initialize_prototypes(model, centroids):
    """
    Replaces random weights in the PrototypeLayer with the calculated centroids.
    """
    # Navigate to the prototype layer inside your sequential head
    with torch.no_grad():
        model.classifier[-1].prototypes.copy_(centroids)
    print(" Prototypes initialized with class centroids.")

class MultiPrototypeLayer(nn.Module):
    def __init__(self, feature_dim, num_classes, protos_per_class=5):
        super().__init__()
        self.num_classes = num_classes
        self.protos_per_class = protos_per_class
        
        # Shape: [num_classes, protos_per_class, feature_dim]
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, protos_per_class, feature_dim))
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))

    def forward(self, x):
        # x: [Batch, feature_dim] -> [Batch, 1, 1, feature_dim]
        x_expanded = x.unsqueeze(1).unsqueeze(1)
        
        # prototypes: -> [1, num_classes, protos_per_class, feature_dim]
        p_expanded = self.prototypes.unsqueeze(0)
        
        # 1. Squared Euclidean Distance to ALL prototypes
        # distances: [Batch, num_classes, protos_per_class]
        distances = torch.sum((x_expanded - p_expanded) ** 2, dim=-1)
        
        # 2. Min-Pooling: Find the closest prototype for each class
        # min_distances: [Batch, num_classes]
        min_distances, _ = torch.min(distances, dim=2)
        
        # Return negative distances as logits (for CE Loss), AND raw distances (for Radius Loss)
        return -min_distances, min_distances

class AudioProtoPNetClassifier(nn.Module):
    """
    Custom classifier head on frozen AudioProtoPNet features.
    """
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        
        # Load the full pretrained AudioProtoPNet
        print(f"Loading pretrained model: {base_model_name}")
        full_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)

        if hasattr(full_model, "backbone"):
            print("✓ Found internal '.backbone' module. Using it as feature extractor.")
            self.backbone = full_model.backbone
        elif hasattr(full_model, "base_model"):
            print("✓ Found internal '.base_model' module. Using it as feature extractor.")
            self.backbone = full_model.base_model
        else:
            print(" Could not find internal backbone. Using full model (might output logits).")
            self.backbone = full_model

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Determine feature dimension automatically
        self.feature_dim = self._get_feature_dim()
        print(f"✓ Detected feature dimension: {self.feature_dim}")

        self.projection_dim = 512
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.projection_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.projection_dim),
            nn.Dropout(0.2)
        )

        self.proto_layer = MultiPrototypeLayer(self.projection_dim, num_classes, CONFIG["protos_per_class"])
        
        print(f" Created classifier head: {self.feature_dim} -> {num_classes}")
    
    def _get_feature_dim(self):
        """Run a dummy input to find the output shape."""
        # Create dummy audio input (batch 1, 32000 samples)
        try:
            dummy_input = torch.randn(1, 16000)
            with torch.no_grad():
                if hasattr(self.backbone.config, "hidden_size"):
                    return self.backbone.config.hidden_size
                if hasattr(self.backbone.config, "num_features"):
                    return self.backbone.config.num_features
        except:
            pass
        return 1024

    def extract_features(self, inputs):
        """Gets the 512D features just before the prototype layer."""
        with torch.no_grad():
            outputs = self.backbone(inputs, output_hidden_states=True)
            
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                features = outputs.hidden_states[-1]
            else:
                features = outputs

        # Case A: (Batch, Time, Dim) -> e.g. (8, 64, 1024)
        if features.dim() == 3:
            features = features.mean(dim=1) # Average over time -> (8, 1024)
            
        # Case B: (Batch, Dim, Time, Freq) -> e.g. (8, 1024, 8, 8)
        elif features.dim() == 4:
            features = features.mean(dim=(2, 3)) #
                
        # We DO want gradients flowing through the projection layer during training
        return self.projection(features)

    def forward(self, x):
        feats = self.extract_features(x)
        logits, min_distances = self.proto_layer(feats)
        return logits, min_distances

# ==========================================
# 4. TRAINING
# ==========================================

def train_collate_fn(batch):
    """Collate function for training."""
    audio_arrays = []
    all_labels = []
    
    for item in batch:
        audio_chunks = item["audio"]
        label = item["label"]
        
        if len(audio_chunks) == 0:
            continue
        
        # Use first chunk for training
        audio_arrays.append(audio_chunks[0])
        all_labels.append(label)
    
    if len(audio_arrays) == 0:
        return None, None
    
    return audio_arrays, torch.tensor(all_labels, dtype=torch.long)


def train_epoch(model, train_loader, optimizer, criterion, feature_extractor, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_data in pbar:
        if batch_data[0] is None:
            continue
        
        audio_arrays, labels = batch_data
        
        # Extract features
        inputs = feature_extractor(audio_arrays, padding=True, return_tensors="pt")
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs)

        # --- LogitNorm Modification ---
        tau = 0.04  # Temperature hyperparameter
        norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = logits / norms / tau
        # -----------------------------
        
        loss = criterion(logit_norm, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Stats
        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100.0 * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def train_epoch_multiprototype(model, train_loader, feature_extractor, optimizer, criterion_ce, device, lambda_radius=0.1):
    model.train()
    total_loss, total_ce, total_radius = 0.0, 0.0, 0.0
    correct_predictions, total_samples = 0, 0
    
    for batch in tqdm(train_loader, desc="Training Epoch"):
        audio_arrays, labels = batch

        inputs = feature_extractor(audio_arrays, padding=True, return_tensors="pt")
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass returns both logits (for CE) and raw positive distances (for Radius)
        logits, min_distances = model(inputs)
        
        # 1. Cross-Entropy Loss (Operates on logits to separate classes)
        loss_ce = criterion_ce(logits, labels)
        
        # 2. Radius Loss (Pulls samples tightly to their true class's closest prototype)
        # min_distances shape: [Batch, num_classes]
        batch_idx = torch.arange(len(labels)).to(device)
        
        # Extract only the distance to the CORRECT class
        true_class_distances = min_distances[batch_idx, labels]
        loss_radius = true_class_distances.mean()
        
        # 3. Combine Losses
        loss = loss_ce + (lambda_radius * loss_radius)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_radius += loss_radius.item()

        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
    num_batches = len(train_loader)
    avg_loss = total_loss/num_batches
    accuracy = 100.0 * correct_predictions / total_samples
    print(f"  Avg Loss: {avg_loss:.4f} | CE: {total_ce/num_batches:.4f} | Radius: {total_radius/num_batches:.4f} | Acc: {accuracy:.2%}")
    return avg_loss, accuracy
 

def validate(model, val_loader, criterion, feature_extractor, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation"):
            if batch_data[0] is None:
                continue
            
            audio_arrays, labels = batch_data
            
            # Extract features
            inputs = feature_extractor(audio_arrays, padding=True, return_tensors="pt")
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc

def validate_epoch_multiprototype(model, feature_extractor, val_loader, criterion_ce, device, lambda_radius=0.1, sample_rate=32000):
    # Set model to evaluation mode (disables dropout, stabilizes BatchNorm)
    model.eval()
    
    total_loss, total_ce, total_radius = 0.0, 0.0, 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Disable gradient calculation for speed and memory savings
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation Epoch"):
            audio_arrays, labels = batch

            inputs = feature_extractor(audio_arrays, padding=True, return_tensors="pt")
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 2. Forward pass (unpack dictionary if necessary using **inputs)
            logits, min_distances = model(inputs)
            
            # 3. Calculate Losses (to track if the model is diverging)
            loss_ce = criterion_ce(logits, labels)
            
            batch_idx = torch.arange(len(labels)).to(device)
            true_class_distances = min_distances[batch_idx, labels]
            loss_radius = true_class_distances.mean()
            
            loss = loss_ce + (lambda_radius * loss_radius)
            
            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_radius += loss_radius.item()
            
            # 4. Calculate Accuracy
            # Since logits are negative distances, argmax finds the closest prototype
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
    # Calculate Averages
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_ce = total_ce / num_batches
    avg_radius = total_radius / num_batches
    accuracy = 100.0 * correct_predictions / total_samples
    
    print(f"  [Val] Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | Radius: {avg_radius:.4f} | Acc: {accuracy:.2%}")
    
    return avg_loss, accuracy

def run_inference(model, test_loader, feature_extractor, num_classes, device):
    """Run inference, compute metrics, and plot AUROC & PR curves."""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    print("\nRunning inference on test set...")
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Testing"):
            if batch_data[0] is None:
                continue
            
            audio_arrays, labels = batch_data
            
            batch_probs = []
            
            for i, audio_chunks in enumerate(audio_arrays):
                if not isinstance(audio_chunks, list):
                    audio_chunks = [audio_chunks]
                
                chunk_probs = []
                for chunk in audio_chunks:
                    inputs = feature_extractor([chunk], padding=True, return_tensors="pt")
                    inputs = inputs.to(device)
                    
                    outputs = model(inputs)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    probs = torch.softmax(logits, dim=1)
                    chunk_probs.append(probs.cpu())
                
                # Max voting across sliding windows
                final_probs, _ = torch.max(torch.cat(chunk_probs), dim=0)
                batch_probs.append(final_probs)
            
            batch_probs = torch.stack(batch_probs)
            all_probs.append(batch_probs.numpy())
            all_preds.extend(torch.argmax(batch_probs, dim=1).tolist())
            all_labels.extend(labels.tolist())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Compute base metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Top-3 accuracy
    top3_preds = np.argsort(all_probs, axis=1)[:, -3:][:, ::-1]
    top3_correct = np.sum([all_labels[i] in top3_preds[i] for i in range(len(all_labels))])
    top3_acc = top3_correct / len(all_labels)
    
    # Binarize labels for cmAP and AUROC
    y_true_bin = label_binarize(all_labels, classes=range(num_classes))
    
    try:
        cmAP = average_precision_score(y_true_bin, all_probs, average='macro')
    except:
        cmAP = 0.0
    
    try:
        auroc = roc_auc_score(y_true_bin, all_probs, multi_class='ovr', average='macro')
        
        # ==========================================
        # 1. Plot Multi-Class AUROC Curve
        # ==========================================
        print("  📊 Generating AUROC curve plot...")
        fpr = dict()
        tpr = dict()
        
        for i in range(num_classes):
            if np.sum(y_true_bin[:, i]) > 0: 
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        
        if len(fpr) > 0:
            all_fpr = np.unique(np.concatenate([fpr[i] for i in fpr.keys()]))
            mean_tpr = np.zeros_like(all_fpr)
            
            for i in fpr.keys():
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            
            mean_tpr /= len(fpr)
            macro_auc = auc(all_fpr, mean_tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(all_fpr, mean_tpr, color='#2196F3', lw=3, label=f'Macro-average ROC curve (AUROC = {macro_auc:.4f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('Multi-Class ROC Curve (Macro-OVR)', fontsize=14)
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(True, alpha=0.3)
            
            roc_plot_path = os.path.join(CONFIG.get("save_dir", "."), 'test_auroc_curve.png')
            fig.savefig(roc_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Saved AUROC curve to {roc_plot_path}")

        # ==========================================
        # 2. Plot Multi-Class Precision-Recall Curve (cmAP)
        # ==========================================
        print("  📊 Generating Precision-Recall (cmAP) curve plot...")
        precision = dict()
        recall = dict()
        
        for i in range(num_classes):
            if np.sum(y_true_bin[:, i]) > 0:
                precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
                
        if len(precision) > 0:
            # We create a standardized Recall axis to interpolate the Precision values
            mean_recall = np.linspace(0, 1, 100)
            mean_precision = np.zeros_like(mean_recall)
            
            for i in precision.keys():
                # Note: precision_recall_curve returns values in decreasing order of recall.
                # np.interp requires the x-axis to be increasing, so we reverse the arrays ([::-1])
                mean_precision += np.interp(mean_recall, recall[i][::-1], precision[i][::-1])
            
            mean_precision /= len(precision)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(mean_recall, mean_precision, color='#4CAF50', lw=3, 
                    label=f'Macro-average PR curve (cmAP = {cmAP:.4f})')
            
            # Baseline is the ratio of positive samples to total samples
            baseline = np.sum(y_true_bin) / y_true_bin.size
            ax.plot([0, 1], [baseline, baseline], 'k--', lw=2, label=f'Random (Baseline = {baseline:.4f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title('Multi-Class Precision-Recall Curve (Macro-Averaged)', fontsize=14)
            ax.legend(loc="lower left", fontsize=11)
            ax.grid(True, alpha=0.3)
            
            pr_plot_path = os.path.join(CONFIG.get("save_dir", "."), 'test_cmap_pr_curve.png')
            fig.savefig(pr_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Saved PR/cmAP curve to {pr_plot_path}")
            
    except Exception as e:
        print(f"  ⚠️ Could not calculate or plot curves: {e}")
        auroc = 0.0
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS:")
    print(f"  Top-1 Accuracy: {accuracy:.4f}")
    print(f"  Top-3 Accuracy: {top3_acc:.4f}")
    print(f"  cmAP:           {cmAP:.4f}")
    print(f"  AUROC:          {auroc:.4f}")
    print(f"{'='*60}\n")
    
    return accuracy, top3_acc, cmAP, auroc
    
# def run_inference(model, test_loader, feature_extractor, num_classes, device):
#     """Run inference and compute metrics."""
#     model.eval()
    
#     all_labels = []
#     all_preds = []
#     all_probs = []
    
#     print("\nRunning inference on test set...")
#     with torch.no_grad():
#         for batch_data in tqdm(test_loader, desc="Testing"):
#             if batch_data[0] is None:
#                 continue
            
#             audio_arrays, labels = batch_data
            
#             # For each sample, we may have multiple chunks (sliding windows)
#             # We'll process them and take the max probability
#             batch_probs = []
            
#             for i, audio_chunks in enumerate(audio_arrays):
#                 if not isinstance(audio_chunks, list):
#                     audio_chunks = [audio_chunks]
                
#                 chunk_probs = []
#                 for chunk in audio_chunks:
#                     inputs = feature_extractor([chunk], padding=True, return_tensors="pt")
#                     inputs = inputs.to(device)
                    
#                     outputs = model(inputs)
#                     logits = outputs[0] if isinstance(outputs, tuple) else outputs
#                     probs = torch.softmax(logits, dim=1)
#                     chunk_probs.append(probs.cpu())
                
#                 # Max voting
#                 final_probs, _ = torch.max(torch.cat(chunk_probs), dim=0)
#                 batch_probs.append(final_probs)
            
#             batch_probs = torch.stack(batch_probs)
#             all_probs.append(batch_probs.numpy())
#             all_preds.extend(torch.argmax(batch_probs, dim=1).tolist())
#             all_labels.extend(labels.tolist())
    
#     all_probs = np.vstack(all_probs)
#     all_labels = np.array(all_labels)
#     all_preds = np.array(all_preds)
    
#     # Compute metrics
#     accuracy = accuracy_score(all_labels, all_preds)
    
#     # Top-3 accuracy
#     top3_preds = np.argsort(all_probs, axis=1)[:, -3:][:, ::-1]
#     top3_correct = np.sum([all_labels[i] in top3_preds[i] for i in range(len(all_labels))])
#     top3_acc = top3_correct / len(all_labels)
    
#     # For multi-class single label, convert to binary format for cmAP and AUROC
#     from sklearn.preprocessing import label_binarize
#     y_true_bin = label_binarize(all_labels, classes=range(num_classes))
    
#     try:
#         cmAP = average_precision_score(y_true_bin, all_probs, average='macro')
#     except:
#         cmAP = 0.0
    
#     try:
#         auroc = roc_auc_score(y_true_bin, all_probs, multi_class='ovr', average='macro')
#     except:
#         auroc = 0.0
    
#     print(f"\n{'='*60}")
#     print(f"TEST RESULTS:")
#     print(f"  Top-1 Accuracy: {accuracy:.4f}")
#     print(f"  Top-3 Accuracy: {top3_acc:.4f}")
#     print(f"  cmAP:           {cmAP:.4f}")
#     print(f"  AUROC:          {auroc:.4f}")
#     print(f"{'='*60}\n")
    
#     return accuracy, top3_acc, cmAP, auroc


def test_collate_fn(batch):
    """Collate function for testing (keeps all chunks)."""
    audio_arrays = []
    all_labels = []
    
    for item in batch:
        audio_chunks = item["audio"]
        labels = item["label"]
        
        if labels is None or labels < 0:
            continue
        
        # Keep all chunks for sliding window inference
        audio_arrays.append(audio_chunks)
        all_labels.append(labels)
    
    if len(audio_arrays) == 0:
        return None, None
    
    return audio_arrays, torch.tensor(all_labels, dtype=torch.long)


# ==========================================
# 5. MAIN
# ==========================================

def main():
    print("\n" + "="*60)
    print("TRAINING CUSTOM CLASSIFIER ON AUDIOPROTOPNET FEATURES")
    print("="*60 + "\n")
    
    # Get number of classes and mappings
    print(" Loading dataset info...")
    num_classes, category2id, id2category = build_label_mapping(
        CONFIG["train_csv"],
        CONFIG["test_csv"]
    )
    
    # Load feature extractor
    print("\n Loading feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        CONFIG["base_model"],
        trust_remote_code=True
    )
    
    # Create model
    print("\n Creating model...")
    model = AudioProtoPNetClassifier(CONFIG["base_model"], num_classes)
    model = model.to(CONFIG["device"])
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Model Info:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {total_params - trainable_params:,}")
    
    # Create datasets from CSV files
    print("\n Creating datasets...")
    
    # Load train CSV and create validation split
    df_train_full = pd.read_csv(CONFIG["train_csv"], delimiter=';')

    species_counts = df_train_full['categories'].value_counts()
    min_samples = species_counts.min()
    
    print(f"  Samples per species - Min: {min_samples}, Max: {species_counts.max()}, Mean: {species_counts.mean():.1f}")
    
    if min_samples < 2:
        # Some species have only 1 sample - can't stratify
        print(f" Some species have < 2 samples, using simple split (no stratification)")
        train_indices, val_indices = train_test_split(
            range(len(df_train_full)),
            test_size=0.2,
            random_state=42
        )
        return
    else:
        # Enough samples per species - can stratify
        print(f" All species have >= 2 samples, using stratified split")
        try:
            train_indices, val_indices = train_test_split(
                range(len(df_train_full)),
                test_size=0.2,
                stratify=df_train_full['categories'],
                random_state=42
            )
        except ValueError as e:
            # Fallback to simple split if stratification fails
            print(f"  Stratification failed: {e}")
            print(f"  Using simple split instead")
            train_indices, val_indices = train_test_split(
                range(len(df_train_full)),
                test_size=0.2,
                random_state=42
            )

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    # Save temporary train/val CSVs
    train_csv_temp = os.path.join(CONFIG["save_dir"], "temp_train.csv")
    val_csv_temp = os.path.join(CONFIG["save_dir"], "temp_val.csv")
    
    df_train_full.iloc[train_indices].to_csv(train_csv_temp, sep=';', index=False)
    df_train_full.iloc[val_indices].to_csv(val_csv_temp, sep=';', index=False)
    
    train_dataset = TrainingDataset(train_csv_temp, category2id, train_mode=True)
    val_dataset = TrainingDataset(val_csv_temp, category2id, train_mode=True)
    test_dataset = TrainingDataset(CONFIG["test_csv"], category2id, train_mode=False)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Test samples:  {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                             shuffle=True, collate_fn=train_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                           shuffle=False, collate_fn=train_collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1,
                            shuffle=False, collate_fn=test_collate_fn, num_workers=0)
    # print(f"category2id: {category2id}")
    all_labels = []
    for batch in test_loader:
        if batch[0] is not None:
            audio_arrays, labels = batch
            all_labels.extend(labels.tolist())
    print(f"Unique labels in loaded test data: {len(set(all_labels))}")
    print(f"Label distribution: {Counter(all_labels)}")


    # Training setup
    init_multiprototypes_kmeans(
        model, train_loader, feature_extractor, CONFIG["device"], 
        num_classes, CONFIG["protos_per_class"]
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print("\n Starting training...\n")
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*60}")
        
        # Train
        # train_loss, train_acc = train_epoch(
        #     model, train_loader, optimizer, criterion,
        #     feature_extractor, CONFIG["device"]
        # )

        train_loss, train_acc = train_epoch_multiprototype(
            model, train_loader, feature_extractor, optimizer, 
            criterion, CONFIG["device"], CONFIG["lambda_radius"])
        
        # Validate
        # val_loss, val_acc = validate(
        #     model, val_loader, criterion,
        #     feature_extractor, CONFIG["device"]
        # )
        
        val_loss, val_acc = validate_epoch_multiprototype(
            model, feature_extractor, val_loader, criterion, 
            CONFIG["device"], CONFIG["lambda_radius"])

        # Update scheduler
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"\n Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss - CONFIG["min_delta"]:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(CONFIG["save_dir"], "best_new_model.pth"))
            
            print(f" Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f" Patience: {patience_counter}/{CONFIG['patience']}")
        
        # Early stopping
        if patience_counter >= CONFIG["patience"]:
            print(f"\n Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training curves
    print("\n Plotting training curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "training_curves.png"), dpi=150)
    print(f"  ✓ Saved to {CONFIG['save_dir']}/training_curves.png")
    
    # Load best model for testing
    print("\n Loading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(CONFIG["save_dir"], "best_new_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f" Loaded model from epoch {checkpoint['epoch']+1}")
    
    # Final evaluation
    test_acc, test_top3, test_cmAP, test_auroc = run_inference(
        model, test_loader, feature_extractor, num_classes, CONFIG["device"]
    )
    
    # Save final results
    results = {
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': checkpoint['val_loss'],
        'best_val_acc': checkpoint['val_acc'],
        'test_top1_acc': test_acc,
        'test_top3_acc': test_top3,
        'test_cmAP': test_cmAP,
        'test_auroc': test_auroc,
        'num_classes': num_classes,
        'config': CONFIG
    }
    
    import json
    with open(os.path.join(CONFIG["save_dir"], "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Training complete! Results saved to {CONFIG['save_dir']}")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    main()