import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image, ImageOps, ImageFilter

def preprocess(image_path: str) -> np.ndarray:
    # 1. Ouverture et conversion en niveaux de gris
    img = Image.open(image_path).convert("L")
    
    # 2. Redimensionnement en 300x300
    img = img.resize((300, 300))
    
    # 3. Égalisation de l'histogramme
    img = ImageOps.equalize(img)
    
    # 4. Binarisation avec un seuil de 128
    img = img.point(lambda p: 255 if p > 128 else 0, mode="L")
    
    # 5. Extraction des contours
    img = img.filter(ImageFilter.FIND_EDGES)
    
    # 6. Conversion finale en tableau numpy
    arr = np.array(img, dtype=np.uint8)
    return arr

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    p1 = os.path.join(script_dir, "empreinte1.jpg")
    p2 = os.path.join(script_dir, "empreinte2.jpg")
    
    # Prétraitement des images
    img1_arr = preprocess(p1)
    img2_arr = preprocess(p2)
    
    # Calculer le score SSIM
    score, _ = compare_ssim(img1_arr, img2_arr, data_range=255, full=True)
    print(f"SSIM Score: {score:.4f}")
    
    # Implémente la logique de décision : Si SSIM >= 0.75 afficher "ACCEPTÉE", sinon "REJETÉE"
    if score >= 0.75:
        decision = "ACCEPTÉE"
    else:
        decision = "REJETÉE"
    
    print(f"Décision finale : {decision}")
    
    # Génération d'une figure Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Méthode SSIM | Score: {score:.4f} | Résultat: {decision}", fontsize=14, fontweight='bold', color='blue' if decision=='ACCEPTÉE' else 'red')
    
    axes[0].imshow(img1_arr, cmap='gray')
    axes[0].set_title("Empreinte 1 (Filtres Prêtés)")
    axes[0].axis("off")
    
    axes[1].imshow(img2_arr, cmap='gray')
    axes[1].set_title("Empreinte 2 (Filtres Prêtés)")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparaison_ssim.png"))
    plt.close()
    
    print("--- TRITEMENT TERMINE. Résultats dans le dossier 'results/' ---")

if __name__ == "__main__":
    main()
