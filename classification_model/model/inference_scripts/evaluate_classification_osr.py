import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import librosa
import os
import math
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoFeatureExtractor
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Model & Data Paths
    "model_checkpoint": "HOME_DIR/classification_model/model/checkpoints/best_new_model.pth",
    "base_model": "DBD-research-group/AudioProtoPNet-5-BirdSet-XCL",
    "known_species_csv": "HOME_DIR/classification_model/sg_dataset/balanced_train.csv",
    "test_csv": "HOME_DIR/classification_model/sg_dataset/open_test1.csv",
    
    # Threshold Configuration
    "threshold_json_path": "HOME_DIR/classification_model/model/open_set_results/metrics.json", 
    "use_min_distance": True,
    "use_max_softmax": True,
    
    # Output file
    "output_dir": OUTPUT_DIR,
    "output_csv": OUTPUT_CSV,
    
    # Model parameters (must match training)
    "protos_per_class": 5,
    "sample_rate": 32000,
    "batch_size": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ==========================================
# MODEL DEFINITIONS
# ==========================================


class MultiPrototypeLayer(nn.Module):
    def __init__(self, feature_dim, num_classes, protos_per_class=5):
        super().__init__()
        self.num_classes = num_classes
        self.protos_per_class = protos_per_class
        
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, protos_per_class, feature_dim))
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))

    def forward(self, x):
        x_expanded = x.unsqueeze(1).unsqueeze(1)
        p_expanded = self.prototypes.unsqueeze(0)
        
        distances = torch.sum((x_expanded - p_expanded) ** 2, dim=-1)
        min_distances, _ = torch.min(distances, dim=2)
        
        return -min_distances, min_distances


class AudioProtoPNetClassifier(nn.Module):
    def __init__(self, base_model_name, num_classes):
        super().__init__()
        
        print(f"Loading pretrained model: {base_model_name}")
        full_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        
        if hasattr(full_model, "backbone"):
            self.backbone = full_model.backbone
        elif hasattr(full_model, "base_model"):
            self.backbone = full_model.base_model
        else:
            self.backbone = full_model

        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.feature_dim = self._get_feature_dim()
        
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2)
        )
        
        self.projected_dim = 512
        self.proto_layer = MultiPrototypeLayer(
            self.projected_dim, num_classes, CONFIG["protos_per_class"]
        )
    
    def _get_feature_dim(self):
        try:
            if hasattr(self.backbone.config, "hidden_size"):
                return self.backbone.config.hidden_size
            if hasattr(self.backbone.config, "num_features"):
                return self.backbone.config.num_features
        except:
            pass
        return 1024

    def extract_features(self, inputs):
        with torch.no_grad():
            outputs = self.backbone(inputs, output_hidden_states=True)
            
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                features = outputs.hidden_states[-1]
            else:
                features = outputs

        if features.dim() == 3:
            features = features.mean(dim=1)
        elif features.dim() == 4:
            features = features.mean(dim=(2, 3))
                
        return self.projection(features)
    
    def forward(self, x, return_features=False):
        feats = self.extract_features(x)
        logits, min_distances = self.proto_layer(feats)
        
        if return_features:
            return logits, min_distances, feats
        return logits, min_distances

# ==========================================
# DATASET
# ==========================================

class InferenceDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, delimiter=';')
        self.sr = CONFIG["sample_rate"]
        self.target_len = self.sr * 5
        self.has_segments = 'start_time' in self.df.columns
        print(f"Loaded {len(self.df)} samples for inference.")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['fullfilename']
        
        # Load audio (with segment bounds if available)
        try:
            if self.has_segments and 'start_time' in row and not pd.isna(row['start_time']):
                y, _ = librosa.load(path, sr=self.sr, mono=True,
                                   offset=row['start_time'], duration=row['segment_duration'])
            else:
                y, _ = librosa.load(path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            y = np.zeros(self.target_len, dtype=np.float32)
        
        # Split into chunks
        chunks = []
        if len(y) > 0:
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
        
        # Keep original category if we want to save it to output
        true_category = row['categories'] if 'categories' in row else "Unknown"
            
        return {
            "path": path,
            "audio": chunks,
            "true_category": true_category
        }

def collate_fn(batch):
    return {
        "path": [item["path"] for item in batch],
        "audio": [item["audio"] for item in batch],
        "true_category": [item["true_category"] for item in batch]
    }

# ==========================================
# UTILS
# ==========================================

def compute_min_distance_score(min_distances):
    """
    Returns negative distance: higher = closer to a known prototype = more likely known.
    """
    closest_dist, _ = torch.min(min_distances, dim=1)
    return -closest_dist.cpu().numpy()

def load_threshold(json_path):
    """
    Load OSR thresholds from the metrics JSON file.
    Returns (min_distance_threshold, max_softmax_threshold).
    """
    print(f"Loading thresholds from {json_path}...")
    min_dist_thresh = -25.0  # fallback
    max_soft_thresh = 0.1    # fallback
    try:
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'min_distance' in data:
            min_dist_thresh = data['min_distance']['threshold_95tpr']
        if 'max_softmax' in data:
            max_soft_thresh = data['max_softmax']['threshold_95tpr']
    except Exception as e:
        print(f"Failed to load thresholds: {e}")
    return min_dist_thresh, max_soft_thresh

def compute_oscr(known_scores, known_correct_flags, unknown_scores):
    """
    Computes CCR and FPR for the OSCR curve.
    - known_scores: Scores of known classes.
    - known_correct_flags: Boolean array (True if species ID was correct).
    - unknown_scores: Scores of unknown/noise samples.
    """
    # Combine and sort all unique scores to use as threshold candidates
    thresholds = np.unique(np.concatenate([known_scores, unknown_scores]))
    thresholds = np.sort(thresholds)[::-1]  # High to low
    
    ccr = []
    fpr = []
    n_k = len(known_scores)
    n_u = len(unknown_scores)
    
    for t in thresholds:
        # CCR: Correct ID AND Accepted by threshold
        ccr.append(np.sum((known_scores >= t) & known_correct_flags) / n_k)
        # FPR: Unknown incorrectly accepted by threshold
        fpr.append(np.sum(unknown_scores >= t) / n_u)
        
    return np.array(fpr), np.array(ccr)


# ==========================================
# MAIN INFERENCE
# ==========================================

def main():
    print("="*60)
    print("OPEN-SET INFERENCE SCRIPT (MIN DISTANCE THRESHOLD)")
    print("="*60)
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 1. Load known species to map predictions back to text
    print("\n📂 Loading known species mapping...")
    df_train = pd.read_csv(CONFIG["known_species_csv"], delimiter=';')
    known_species = sorted(list(set(df_train['categories'].unique())))
    id2category = {idx: cat for idx, cat in enumerate(known_species)}
    category2id = {cat: idx for idx, cat in enumerate(known_species)}
    num_classes = len(known_species)
    print(f"  ✓ {num_classes} known species identified")
    
    # 2. Load Thresholds
    min_dist_threshold, max_soft_threshold = load_threshold(CONFIG["threshold_json_path"])
    
    active_methods = []
    if CONFIG["use_min_distance"]:
        active_methods.append(f"min_distance (threshold={min_dist_threshold:.4f})")
    if CONFIG["use_max_softmax"]:
        active_methods.append(f"max_softmax (threshold={max_soft_threshold:.4f})")
    
    print(f"  ✓ Active OSR methods: {', '.join(active_methods) if active_methods else 'NONE (no rejection)'}")

    # 3. Load Model
    print(f"\n🤖 Loading model...")
    model = AudioProtoPNetClassifier(CONFIG["base_model"], num_classes)
    checkpoint = torch.load(CONFIG["model_checkpoint"], map_location=CONFIG["device"])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(CONFIG["device"])
    model.eval()
    print("  ✓ Model loaded")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        CONFIG["base_model"], trust_remote_code=True
    )
    
    # 4. Load Dataset
    test_dataset = InferenceDataset(CONFIG["test_csv"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                             shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # 5. Run Inference
    results = []
    
    print("\n🚀 Starting inference on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            if batch is None:
                continue
                
            paths = batch["path"]
            audio_arrays = batch["audio"]
            true_categories = batch["true_category"]
            
            for i in range(len(audio_arrays)):
                audio_chunks = audio_arrays[i]
                if not isinstance(audio_chunks, list):
                    audio_chunks = [audio_chunks]
                
                chunk_logits = []
                chunk_min_dists = []
                
                # Process all chunks (sliding windows) for this single file
                for chunk in audio_chunks:
                    inputs = feature_extractor([chunk], padding=True, return_tensors="pt")
                    inputs = inputs.to(CONFIG["device"])
                    
                    logits, min_distances = model(inputs)
                    chunk_logits.append(logits)
                    chunk_min_dists.append(min_distances)
                
                # Aggregate across chunks (average the logits/distances)
                final_logits = torch.cat(chunk_logits, dim=0).mean(dim=0, keepdim=True)
                final_min_dists = torch.cat(chunk_min_dists, dim=0).mean(dim=0, keepdim=True)
                
                # Calculate the min_distance score (higher is more confident)
                min_dist_score = compute_min_distance_score(final_min_dists)[0]
                
                # Get softmax probabilities for metrics
                probs = torch.softmax(final_logits, dim=1).cpu().numpy().squeeze()
                max_softmax_score = float(np.max(probs))
                
                # Predict the class
                pred_idx = torch.argmax(final_logits, dim=1).item()
                predicted_category = id2category[pred_idx]
                
                # Apply OSR — sample must pass ALL enabled checks
                reject_reasons = []
                if CONFIG["use_min_distance"] and min_dist_score < min_dist_threshold:
                    reject_reasons.append("min_distance")
                if CONFIG["use_max_softmax"] and max_softmax_score < max_soft_threshold:
                    reject_reasons.append("max_softmax")
                
                is_unknown = len(reject_reasons) > 0
                
                # Final prediction logic
                final_prediction = "Unknown (Open-Set)" if is_unknown else predicted_category
                
                # True label index (-1 if not a known species)
                true_cat = true_categories[i]
                true_idx = category2id.get(true_cat, -1)
                is_known_gt = true_idx >= 0
                
                results.append({
                    "filename": paths[i],
                    "true_category": true_cat,
                    "predicted_category": predicted_category,
                    "min_distance_score": min_dist_score,
                    "max_softmax_score": max_softmax_score,
                    "is_unknown": is_unknown,
                    "reject_reasons": ",".join(reject_reasons) if reject_reasons else "",
                    "final_prediction": final_prediction,
                    "true_idx": true_idx,
                    "pred_idx": pred_idx,
                    "is_known_gt": is_known_gt,
                    "probs": probs,
                })

    # 6. Save results
    print(f"\n💾 Saving results to {CONFIG['output_csv']}...")
    # Drop internal fields before saving
    save_df = pd.DataFrame([{
        k: v for k, v in r.items() if k not in ('true_idx', 'pred_idx', 'is_known_gt', 'probs')
    } for r in results])
    save_df.to_csv(CONFIG["output_csv"], index=False)
    
    # Print summary statistics
    total_samples = len(results)
    unknown_samples = sum(1 for r in results if r["is_unknown"])
    print(f"  ✓ Processed {total_samples} samples.")
    print(f"  ✓ Flagged {(unknown_samples/total_samples)*100:.2f}% ({unknown_samples}) as Unknown.")
    
    # ========================================
    # 7. Compute Metrics
    # ========================================
    known_results = [r for r in results if r["is_known_gt"]]
    unknown_results = [r for r in results if not r["is_known_gt"]]
    num_k = len(known_results)
    num_u = len(unknown_results)

    if num_k > 0 and num_u > 0:
        # 1. Prepare Data for Isolated Curves
        k_correct = np.array([r["pred_idx"] == r["true_idx"] for r in known_results])
        
        # Min-Distance Scores
        k_scores_dist = np.array([r["min_distance_score"] for r in known_results])
        u_scores_dist = np.array([r["min_distance_score"] for r in unknown_results])
        
        # MSP Scores
        k_scores_msp = np.array([r["max_softmax_score"] for r in known_results])
        u_scores_msp = np.array([r["max_softmax_score"] for r in unknown_results])

        # 2. Calculate Isolated OSCR Curves
        fpr_dist, ccr_dist = compute_oscr(k_scores_dist, k_correct, u_scores_dist)
        fpr_msp, ccr_msp = compute_oscr(k_scores_msp, k_correct, u_scores_msp)
        
        auoscr_dist = auc(fpr_dist, ccr_dist)
        auoscr_msp = auc(fpr_msp, ccr_msp)

        # 3. Calculate Hybrid System "Single Point" (Dual-Gate)
        
        # True Positives (TP): Known sample, correctly classified, AND accepted
        tp = sum(1 for r in known_results if not r["is_unknown"] and r["pred_idx"] == r["true_idx"])
        
        # True Negatives (TN): Unknown sample, correctly rejected as "Unknown"
        tn = sum(1 for r in unknown_results if r["is_unknown"])
        
        # False Positives (FP): 
        # Type 1: Unknown noise that was accidentally accepted
        fp_noise = sum(1 for r in unknown_results if not r["is_unknown"])
        # Type 2: Known bird that was accepted, but the species ID was WRONG
        fp_misclass = sum(1 for r in known_results if not r["is_unknown"] and r["pred_idx"] != r["true_idx"])
        total_fp = fp_noise + fp_misclass
        
        # False Negatives (FN): Known bird that was rejected by the OSR gate
        fn = sum(1 for r in known_results if r["is_unknown"])

        # OSCR Specific Metrics
        hybrid_ccr = tp / num_k  # This is the same as Recall
        hybrid_fpr = fp_noise / num_u

        # Standard Classification Metrics (Open-Set Adapted)
        hybrid_precision = tp / (tp + total_fp) if (tp + total_fp) > 0 else 0.0
        hybrid_recall = tp / num_k
        hybrid_f1 = 2 * (hybrid_precision * hybrid_recall) / (hybrid_precision + hybrid_recall) if (hybrid_precision + hybrid_recall) > 0 else 0.0
        hybrid_accuracy = (tp + tn) / (num_k + num_u)

        print(f"\n{'='*60}")
        print(f"OSCR PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"  Isolated Min-Distance AUOSCR: {auoscr_dist:.4f}")
        print(f"  Isolated Max-Softmax AUOSCR:  {auoscr_msp:.4f}")
        print(f"  -------------------------------------------")
        print(f"  HYBRID SYSTEM (Dual-Gate) Point:")
        print(f"    CCR (Recall):               {hybrid_ccr:.4f}")
        print(f"    FPR (Noise Acceptance):     {hybrid_fpr:.4f}")
        print(f"  -------------------------------------------")
        print(f"  HYBRID SYSTEM OVERALL METRICS:")
        print(f"    Precision:                  {hybrid_precision:.4f}")
        print(f"    Recall:                     {hybrid_recall:.4f}")
        print(f"    F1-Score:                   {hybrid_f1:.4f}")
        print(f"    Total System Accuracy:      {hybrid_accuracy:.4f}")
        print(f"{'='*60}")
        # 4. Generate Combined OSCR Plot
        plot_dir = CONFIG["output_dir"]
        os.makedirs(plot_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot Isolated Curves
        ax.plot(fpr_dist, ccr_dist, color='#2196F3', lw=2, label=f'Min-Distance (AUOSCR: {auoscr_dist:.4f})')
        ax.plot(fpr_msp, ccr_msp, color='#FF9800', lw=2, linestyle='--', label=f'Max-Softmax (AUOSCR: {auoscr_msp:.4f})')
        
        # Plot the Hybrid Single Point
        ax.plot(hybrid_fpr, hybrid_ccr, marker='*', markersize=15, color='red', 
                linestyle='None', label='Hybrid System (Dual-Gate)', markeredgecolor='black')

        # Formatting
        ax.set_xscale('log') # Standard for OSR to see low FPR performance
        ax.set_xlabel('False Positive Rate (log scale)', fontsize=12)
        ax.set_ylabel('Correct Classification Rate', fontsize=12)
        ax.set_title('OSCR Curves: Min-Dist vs. MSP with Hybrid Operating Point', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        
        oscr_path = os.path.join(plot_dir, 'combined_oscr_comparison.png')
        fig.savefig(oscr_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  📊 Combined OSCR Plot saved to {oscr_path}")

        # ========================================
        # 5. Score Distribution Plots
        # ========================================
        
        # --- Plot A: Min-Distance Distribution ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Percentile clipping to ignore extreme negative outliers (e.g., -10000)
        all_dist = np.concatenate([k_scores_dist, u_scores_dist])
        lower_limit = np.percentile(all_dist, 1) 
        upper_limit = max(all_dist) * 1.05 if max(all_dist) > 0 else 0.5

        ax.hist(k_scores_dist, bins=50, alpha=0.6, color='#2196F3', range=(lower_limit, upper_limit), label=f'Known ({len(k_scores_dist)})', density=True)
        ax.hist(u_scores_dist, bins=50, alpha=0.6, color='#F44336', range=(lower_limit, upper_limit), label=f'Unknown ({len(u_scores_dist)})', density=True)
        
        if CONFIG.get("use_min_distance", True):
            ax.axvline(x=min_dist_threshold, color='black', linestyle='--', lw=2, label=f'Threshold = {min_dist_threshold:.4f}')
            
        ax.set_xlabel('Min Distance Score (higher = more confident)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Score Distribution (Min-Distance): Known vs Unknown', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        dist_path = os.path.join(plot_dir, 'osr_score_dist_mindist.png')
        fig.savefig(dist_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  📊 Min-Distance Dist saved to {dist_path}")

        # --- Plot B: Max-Softmax Distribution ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Softmax is inherently bounded between 0.0 and 1.0, so no dynamic limits needed
        ax.hist(k_scores_msp, bins=50, alpha=0.6, color='#FF9800', range=(0, 1), label=f'Known ({len(k_scores_msp)})', density=True)
        ax.hist(u_scores_msp, bins=50, alpha=0.6, color='#9C27B0', range=(0, 1), label=f'Unknown ({len(u_scores_msp)})', density=True)
        
        if CONFIG.get("use_max_softmax", True):
            ax.axvline(x=max_soft_threshold, color='black', linestyle='--', lw=2, label=f'Threshold = {max_soft_threshold:.4f}')
            
        ax.set_xlabel('Max Softmax Probability (0.0 to 1.0)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Score Distribution (Max Softmax): Known vs Unknown', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        msp_path = os.path.join(plot_dir, 'osr_score_dist_maxsoftmax.png')
        fig.savefig(msp_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  📊 Max-Softmax Dist saved to {msp_path}")
    
    print(f"{'='*60}")
    
    print("\nDONE!")

if __name__ == "__main__":
    main()
