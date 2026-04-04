import os
import cv2
import numpy as np
from tp04_pca import FaceRecognitionPCA

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset")
    test_dir = os.path.join(script_dir, "test_images")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=== EXPERIENCE 1: Évaluation de l'effet de k (Nombre de composantes) ===")
    
    X, y = FaceRecognitionPCA(n_components=1).load_dataset(dataset_path)
    if len(X) == 0:
        print("Erreur: Dataset introuvable.")
        return
        
    for k in [10, 20, 50]:
        pca_k = FaceRecognitionPCA(n_components=min(k, len(X)))
        pca_k.labels = y
        pca_k.compute_pca(X)
        print(f"Modèle PCA entraîné avec succès : k={pca_k.n_components}/{k}")
        
    print("\n=== EXPERIENCE 2: Analyse du Taux de Faux Rejets et Acceptations (FAR/FRR) ===")
    pca_eval = FaceRecognitionPCA(n_components=min(10, len(X)))
    pca_eval.labels = y
    pca_eval.compute_pca(X)
    
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    y_true = []
    y_scores = []
    
    for f in test_files:
        path = os.path.join(test_dir, f)
        img = cv2.imread(path)
        if img is None: continue
        res_rect = pca_eval.detect_face(img)
        if res_rect is None: continue
            
        face_img, rect = res_rect
        proj = pca_eval.project(face_img.flatten())
        
        p1_idx = [i for i, lbl in enumerate(pca_eval.labels) if lbl == 'person1']
        p1_projs = pca_eval.projections[p1_idx]
        if len(p1_projs) == 0: continue
            
        distances1 = [np.linalg.norm(proj - tp) for tp in p1_projs]
        min_d1 = min(distances1)
        
        is_genuine = 1 if "p1" in f else 0 
        y_true.append(is_genuine)
        y_scores.append(min_d1)
        
    if not y_true:
        print("Aucun fichier valide dans test_images.")
        return
        
    thresholds = np.linspace(max(0, min(y_scores)-500), max(y_scores)+500, 50)
    far_list = []
    frr_list = []
    
    genuine_c = sum(1 for yt in y_true if yt == 1)
    impostor_c = sum(1 for yt in y_true if yt == 0)
    genuine_c = max(1, genuine_c)
    impostor_c = max(1, impostor_c)
    
    for th in thresholds:
        far = sum(1 for yt, s in zip(y_true, y_scores) if yt == 0 and s <= th) / impostor_c
        frr = sum(1 for yt, s in zip(y_true, y_scores) if yt == 1 and s > th) / genuine_c
        far_list.append(far)
        frr_list.append(frr)
        
    diff = [abs(far - frr) for far, frr in zip(far_list, frr_list)]
    min_diff_idx = np.argmin(diff)
    opt_th = thresholds[min_diff_idx]
    opt_far = far_list[min_diff_idx]
    opt_frr = frr_list[min_diff_idx]
    accuracy = 1.0 - (opt_far + opt_frr) / 2.0
    
    metrics_path = os.path.join(results_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"=== Experiment Metrics (PCA - k={pca_eval.n_components}) ===\n")
        f.write(f"Optimal Threshold (EER approximation): {opt_th:.2f}\n")
        f.write(f"FAR (False Acceptance Rate): {opt_far*100:.2f}%\n")
        f.write(f"FRR (False Rejection Rate): {opt_frr*100:.2f}%\n")
        f.write(f"Global Accuracy Evaluation: {accuracy*100:.2f}%\n")
        
    print(f"L'export métrologique des seuils (Optimal FAR/FRR) est achevé : {metrics_path}")

if __name__ == "__main__":
    main()
