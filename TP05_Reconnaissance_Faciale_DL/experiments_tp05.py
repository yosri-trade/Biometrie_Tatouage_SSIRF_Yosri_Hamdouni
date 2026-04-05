import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tp05_dl import FaceRecognitionDL

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset")
    test_dir = os.path.join(script_dir, "test_images")
    results_dir = os.path.join(script_dir, "results")
    
    sys = FaceRecognitionDL()
    print("Analyse de la Base de Données SFace ONNX...")
    sys.build_database(dataset_path)
    
    print("\n=== EXPERIENCE A: Cosine vs Euclidean ===")
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    
    y_true = []
    y_scores_euclid = []
    y_scores_cosine = []
    
    for f in test_files:
        path = os.path.join(test_dir, f)
        img = cv2.imread(path)
        if img is None: continue
        
        face_data, rect = sys.detect_face(img)
        if face_data is None: continue
            
        test_emb = sys.extract_embedding(img, rect)
        
        p1_embs = sys.database.get('person1', [])
        if not p1_embs: continue
        
        distances = [sys.euclidean_distance(test_emb, r) for r in p1_embs]
        similarities = [sys.cosine_similarity(test_emb, r) for r in p1_embs]
        
        min_euclid = min(distances)
        max_cos = max(similarities)
        
        is_genuine = 1 if "p1" in f else 0
        y_true.append(is_genuine)
        y_scores_euclid.append(min_euclid)
        y_scores_cosine.append(max_cos)
        
        print(f"Test {f} | Authenticité: {is_genuine} | Dist Euclidienne: {min_euclid:.4f} | Sim Cosinus: {max_cos:.4f}")
        
    print("\n=== EXPERIENCE B/C: Analyse des Seuils (FAR/FRR) sur P1 vs P1 / P1 vs P2 ===")
    thresholds = [0.4, 0.6, 0.8]
    
    genuine_c = max(1, sum(1 for yt in y_true if yt == 1))
    impostor_c = max(1, sum(1 for yt in y_true if yt == 0))
    
    metrics_path = os.path.join(results_dir, "metrics_dl.txt")
    with open(metrics_path, "w") as f:
        f.write("=== Analyse Deep Learning Metrics (Cosine) ===\n")
        f.write(f"Cibles testees => Authentiques(Person1): {genuine_c} | Imposteurs(Person2): {impostor_c}\n\n")
        
        for th in thresholds:
            far = sum(1 for yt, s in zip(y_true, y_scores_cosine) if yt == 0 and s >= th) / impostor_c
            frr = sum(1 for yt, s in zip(y_true, y_scores_cosine) if yt == 1 and s < th) / genuine_c
            
            res_str = f"Seuil Cosinus = {th} -> FAR: {far*100:.1f}% | FRR: {frr*100:.1f}%"
            print(res_str)
            f.write(res_str + "\n")
            
    print(f"\nRésultats d'expérimentation exportés dans {metrics_path}")

if __name__ == "__main__":
    main()
