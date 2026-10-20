"""Milestone 5.5 — Generate Publication Evidence

Performs a rigorous, deterministic re-evaluation pass over the test sets
using the exact frozen artifacts and identical random seeds to generate
publication-quality metrics, confusion matrices, and classification reports.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

import os
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT))

print("[DEBUG] Bootstrapping evidence script...")

from evaluate_cross_domain import load_split, get_component_predictions, DATASETS, read_json

PROFILE = "in_domain"

def log(msg: str) -> None:
    print(f"[EVIDENCE] {time.strftime('%H:%M:%S')} {msg}", flush=True)

def generate_confusion_matrix_plot(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['BENIGN', 'ATTACK'],
                yticklabels=['BENIGN', 'ATTACK'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    # Normalized
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=['BENIGN', 'ATTACK'],
                yticklabels=['BENIGN', 'ATTACK'])
    plt.title(f"{title} (Normalized)")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(str(out_path).replace('.png', '_normalized.png'), dpi=300)
    plt.close()

def run():
    fig_dir = ROOT / "reports" / "figures" / "confusion_matrices"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir = ROOT / "reports" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    
    in_domain_metrics = read_json(ROOT / "reports" / "experiments" / "in_domain_evaluation.json")
    
    classification_reports_md = "# Detailed Classification Reports\n\n"
    summary_md = "# Test Performance Summary\n\n"
    summary_md += "| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Support |\n"
    summary_md += "|---|---|---|---|---|---|---|---|---|\n"
    
    for ds in DATASETS:
        if os.environ.get("SMOKE_TEST") == "1" and ds != os.environ.get("SMOKE_DATASET", "N-BaIoT"):
            continue
            
        log(f"[{ds}] Loading frozen models and deterministic test set...")
        import joblib
        import torch
        
        loaded_models = {}
        loaded_metrics = {}
        
        rf_path = ROOT / "models" / PROFILE / "random_forest" / f"{ds}_rf.joblib"
        if rf_path.exists():
            loaded_models["rf"] = joblib.load(rf_path)
            loaded_metrics["rf"] = read_json(ROOT / "models" / PROFILE / "random_forest" / f"{ds}_rf_metrics.json")
            
        mlp_path = ROOT / "models" / PROFILE / "neural_network" / f"{ds}_mlp.pt"
        if mlp_path.exists():
            loaded_models["mlp"] = torch.load(mlp_path)
            loaded_metrics["mlp"] = read_json(ROOT / "models" / PROFILE / "neural_network" / f"{ds}_mlp_metrics.json")
            
        ae_path = ROOT / "models" / PROFILE / "autoencoder" / f"{ds}_ae.joblib"
        if ae_path.exists():
            loaded_models["ae"] = joblib.load(ae_path)
            loaded_metrics["ae"] = read_json(ROOT / "models" / PROFILE / "autoencoder" / f"{ds}_ae_metrics.json")
            
        X_test, y_test = load_split(ds, "test")
        if len(y_test) == 0:
            continue
            
        test_preds = get_component_predictions(loaded_models, loaded_metrics, X_test)
        
        if ds not in in_domain_metrics or ds not in in_domain_metrics[ds]:
            continue
            
        ds_configs = in_domain_metrics[ds][ds]
        
        # Diagnostic extraction
        benign_cnt = np.sum(y_test == 0)
        attack_cnt = np.sum(y_test == 1)
        tot = len(y_test)
        log(f"[{ds}] Test Set Diagnosis: Total={tot}, Benign={benign_cnt} ({benign_cnt/tot:.2%}), Attack={attack_cnt} ({attack_cnt/tot:.2%})")
        
        # Models to evaluate explicitly based on Task 2: RF, MLP, AE, RF+MLP, RF+MLP+AE
        eval_targets = ["rf", "mlp", "ae", "rf+mlp", "rf+mlp+ae"]
        
        for m_name in eval_targets:
            if m_name not in ds_configs:
                continue
                
            config = ds_configs[m_name]
            fused_proba = np.zeros(len(y_test))
            for comp_name, weight in config["frozen_weights"].items():
                if comp_name in test_preds:
                    fused_proba += weight * test_preds[comp_name]
                    
            threshold = config["frozen_threshold"]
            y_pred = (fused_proba >= threshold).astype(int)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            generate_confusion_matrix_plot(
                y_test, y_pred, 
                title=f"{ds} - {m_name.upper()}",
                out_path=fig_dir / f"{ds}_{m_name.replace('+', '_')}.png"
            )
            
            # Classification report
            clf_rep = classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"], digits=4)
            classification_reports_md += f"## {ds} - {m_name.upper()}\n\n```text\n{clf_rep}\n```\n\n"
            
            # Additional metrics
            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
            
            # AUC calculation (guard against single-class edge cases, though none should exist in these test sets)
            try:
                roc = roc_auc_score(y_test, fused_proba)
                pr = average_precision_score(y_test, fused_proba)
                roc_str, pr_str = f"{roc:.4f}", f"{pr:.4f}"
            except ValueError:
                roc_str, pr_str = "N/A", "N/A"
                
            summary_md += f"| **{ds}** | {m_name.upper()} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {roc_str} | {pr_str} | {tot} |\n"
            
            if m_name == "rf+mlp+ae":
                log(f"[{ds}] {m_name.upper()} Diagnostics -> TP:{tp} TN:{tn} FP:{fp} FN:{fn} | Rec:{rec:.4f} Prec:{prec:.4f}")

    with open(table_dir / "classification_reports.md", "w") as f:
        f.write(classification_reports_md)
        
    with open(table_dir / "test_performance_summary.md", "w") as f:
        f.write(summary_md)
        
    log("Evidence generation complete.")

if __name__ == "__main__":
    run()
