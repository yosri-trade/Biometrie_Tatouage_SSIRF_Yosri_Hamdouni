import os
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib.pyplot as plt

def save_comparison(img1, title1, img2, title2, results_dir, filename, cmap1=None, cmap2=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    if cmap1:
        axes[0].imshow(img1, cmap=cmap1)
    else:
        axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis("off")
    
    if cmap2:
        axes[1].imshow(img2, cmap=cmap2)
    else:
        axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Partie 1 : Lecture de l'image et sauvegarde
    print("Partie 1: Lecture ...")
    input_path = os.path.join(script_dir, "input_image.jpg")
    img = Image.open(input_path).convert("RGB")
    img.save(os.path.join(results_dir, "image_originale.png"))
    
    # Partie 2 : Redimensionnement
    print("Partie 2: Redimensionnement ...")
    img_resized = img.resize((300, 150))
    save_comparison(img, "Originale", img_resized, "Redimensionnée (300x150)", results_dir, "image_redimensionnee.png")
    
    # Partie 3 : Augmentation de la luminosité
    print("Partie 3: Luminosité ...")
    enhancer = ImageEnhance.Brightness(img)
    img_bright = enhancer.enhance(1.5)
    save_comparison(img, "Originale", img_bright, "Luminosité x1.5", results_dir, "image_luminosite_augmente.png")
    
    # Partie 4 : Conversion en niveaux de gris
    print("Partie 4: Niveaux de gris ...")
    img_gray = img.convert("L")
    save_comparison(img, "Originale RGB", img_gray, "Niveaux de gris (Mode L)", results_dir, "image_gris.png", cmap2="gray")
    
    # Partie 5 : Binarisation avec seuil de 128
    print("Partie 5: Binarisation ...")
    threshold = 128
    img_bin = img_gray.point(lambda p: 255 if p > threshold else 0, mode="1")
    save_comparison(img_gray, "Niveaux de gris", img_bin, "Binarisée (Seuil 128)", results_dir, "image_binarisee.png", cmap1="gray", cmap2="gray")
    
    # Partie 6 : Détection de contours
    print("Partie 6: Contours ...")
    img_edges = img_gray.filter(ImageFilter.FIND_EDGES)
    save_comparison(img_gray, "Niveaux de gris", img_edges, "Contours (FIND_EDGES)", results_dir, "image_contours.png", cmap1="gray", cmap2="gray")
    
    # Partie 7 : Flou gaussien (rayons 1, 2 et 3)
    print("Partie 7: Flou Gaussien ...")
    img_blur1 = img.filter(ImageFilter.GaussianBlur(1))
    img_blur2 = img.filter(ImageFilter.GaussianBlur(2))
    img_blur3 = img.filter(ImageFilter.GaussianBlur(3))
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img)
    axes[0].set_title("Originale")
    axes[0].axis("off")
    axes[1].imshow(img_blur1)
    axes[1].set_title("Flou (r=1)")
    axes[1].axis("off")
    axes[2].imshow(img_blur2)
    axes[2].set_title("Flou (r=2)")
    axes[2].axis("off")
    axes[3].imshow(img_blur3)
    axes[3].set_title("Flou (r=3)")
    axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "image_flou_gaussien.png"))
    plt.close()
    
    # Partie 8 : Calcul et tracé de l'histogramme
    print("Partie 8: Histogramme ...")
    hist = img_gray.histogram()
    plt.figure(figsize=(8, 5))
    plt.plot(hist, color="black")
    plt.title("Histogramme de l'image en niveaux de gris")
    plt.xlabel("Niveau de gris (0-255)")
    plt.ylabel("Nombre de pixels")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "histogramme.png"))
    plt.close()
    
    # Partie 9 : Égalisation de l'histogramme
    print("Partie 9: Egalisation de l'histogramme ...")
    img_eq = ImageOps.equalize(img_gray)
    save_comparison(img_gray, "Originale Grise", img_eq, "Égalisée", results_dir, "image_egalisee.png", cmap1="gray", cmap2="gray")
    
    print("--- TRITEMENT TERMINE. Résultats dans le dossier 'results/' ---")

if __name__ == "__main__":
    main()
