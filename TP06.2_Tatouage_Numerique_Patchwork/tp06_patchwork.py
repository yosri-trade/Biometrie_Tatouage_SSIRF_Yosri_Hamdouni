import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def generate_pixel_pairs(image_shape, N, key):
    """
    Génère N paires de coordonnées aléatoires avec une clé secrète.
    """
    random.seed(key)
    h, w = image_shape[:2]
    
    pairs = []
    for _ in range(N):
        # Pixel A
        y_a, x_a = random.randint(0, h - 1), random.randint(0, w - 1)
        # Pixel B
        y_b, x_b = random.randint(0, h - 1), random.randint(0, w - 1)
        pairs.append(((y_a, x_a), (y_b, x_b)))
    return pairs

def apply_patchwork_grayscale(img, pairs, delta):
    """
    Applique le tatouage Patchwork sur une image en niveaux de gris.
    """
    watermarked = img.copy().astype(np.float32)
    for (ya, xa), (yb, xb) in pairs:
        watermarked[ya, xa] += delta
        watermarked[yb, xb] -= delta
    
    # Clip pour garder les pixels dans [0, 255]
    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
    return watermarked

def detect_patchwork(img, pairs, threshold):
    """
    Calcule S_n et retourne la décision (Tatouée = Vrai/Faux).
    """
    S_n = 0
    img_float = img.astype(np.float32)
    for (ya, xa), (yb, xb) in pairs:
        # Expected without watermark : approx 0.
        # Expected with watermark : approx 2*N*delta.
        S_n += (img_float[ya, xa] - img_float[yb, xb])
    
    return S_n > threshold, S_n

# -- Fonctions d'attaques
def add_noise(img):
    noise = np.random.normal(0, 5, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def apply_compression(img, filename="compressed.jpg", quality=50):
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imread(filename, cv2.IMREAD_COLOR)

def apply_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def main():
    # Variables globales pour le Patchwork
    N = 10000        # Nombre de paires (nombre élevé pour de meilleures stats)
    delta = 5        # Force de modification
    key = 42         # Clé secrète pour retrouver les paires
    threshold = N * delta  # Seuil de décision (entre 0 et 2*N*delta)
    
    input_path = "input.png"

    print("\n" + "="*60)
    print("PARTIE 1 -- Patchwork Niveau de Gris")
    print("="*60)

    # 1. Chargement de l'image
    img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Erreur : Impossible de lire '{input_path}'")
        return

    # 2. Génération paires avec clé
    pairs = generate_pixel_pairs(img_gray.shape, N, key)

    # 3. Application Patchwork
    watermarked_gray = apply_patchwork_grayscale(img_gray, pairs, delta)

    # 4. Sauvegarde de l'image tatouée
    cv2.imwrite("watermarked_gray.png", watermarked_gray)

    # 5. Détection
    decision_orig, S_n_orig = detect_patchwork(img_gray, pairs, threshold)
    decision_water, S_n_water = detect_patchwork(watermarked_gray, pairs, threshold)

    print(f"Statistiques | Valeur attendue: {2 * N * delta} | Seuil: {threshold}\n")
    print(f"--> S_n (Image Originale) : {S_n_orig:7.0f} | Décision: {'Tatouée' if decision_orig else 'Non tatouée'}")
    print(f"--> S_n (Image Tatouée)   : {S_n_water:7.0f} | Décision: {'Tatouée' if decision_water else 'Non tatouée'}")

    # 6. Affichage avant / après (Niveau de gris)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Avant : Originale")
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Après : Tatouée\nS_n = {S_n_water:.0f} > {threshold}")
    plt.imshow(watermarked_gray, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("part1_grayscale_compare.png")
    plt.close()

    print("\n" + "="*60)
    print("PARTIE 2 -- Patchwork RGB et Simulation d'attaques")
    print("="*60)

    # 1. Appliquer Patchwork sur le canal R
    img_rgb = cv2.imread(input_path, cv2.IMREAD_COLOR) # BGR
    watermarked_rgb = img_rgb.copy()
    
    R_channel = watermarked_rgb[:, :, 2]
    watermarked_R = apply_patchwork_grayscale(R_channel, pairs, delta)
    watermarked_rgb[:, :, 2] = watermarked_R
    cv2.imwrite("watermarked_rgb.png", watermarked_rgb)

    # 2. Visualiser la différence (Canal R tatoué - Canal R original)
    diff = cv2.absdiff(img_rgb[:, :, 2], watermarked_R)
    plt.figure(figsize=(6, 5))
    plt.title("Différence absolue sur le canal R (Patchwork)")
    plt.imshow(diff, cmap='hot')
    plt.colorbar()
    plt.savefig("part2_diff_canal_R.png")
    plt.close()

    # 3. Simulate Attacks
    attacked_noise = add_noise(watermarked_rgb)
    attacked_jpeg = apply_compression(watermarked_rgb, "compressed.jpg", 50)
    attacked_blur = apply_blur(watermarked_rgb)

    # Helper function for checking detection on the R channel
    def detect_rgb(img):
        return detect_patchwork(img[:, :, 2], pairs, threshold)

    dec_noise, S_n_noise = detect_rgb(attacked_noise)
    dec_jpeg, S_n_jpeg = detect_rgb(attacked_jpeg)
    dec_blur, S_n_blur = detect_rgb(attacked_blur)

    print("Résultats des Attaques :")
    print(f"1. Bruit gaussien -- S_n : {S_n_noise:7.0f} | Résultat : {'Tatouage DÉTECTÉ' if dec_noise else 'Tatouage PERDU'}")
    print(f"2. Image JPEG 50% -- S_n : {S_n_jpeg:7.0f} | Résultat : {'Tatouage DÉTECTÉ' if dec_jpeg else 'Tatouage PERDU'}")
    print(f"3. Flou (Blur)    -- S_n : {S_n_blur:7.0f} | Résultat : {'Tatouage DÉTECTÉ' if dec_blur else 'Tatouage PERDU'}")
    print("\n--- Comparaison avec LSB ---")
    print("Contrairement à LSB (qui se concentre sur le bit de poids faible de chaque pixel),")
    print("Patchwork est une méthode additive diffuse qui modifie faiblement, mais statistiquement, un")
    print("grand nombre de pixels. Une modification du bit LSB due au bruit, à la compression JPEG ou au")
    print("flou le détruit irrémédiablement, alors que la moyenne statistique du Patchwork préserve S_n")
    print("au-dessus du seuil, rendant cette méthode considérablement plus robuste face aux attaques.")

    # Affichage des 3 attaques
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Original Tatoué RGB", "Attaque Bruit", "Attaque JPEG", "Attaque Flou"]
    images = [watermarked_rgb, attacked_noise, attacked_jpeg, attacked_blur]
    decs = [True, dec_noise, dec_jpeg, dec_blur]

    for i in range(4):
        # Convert BGR to RGB for matplotlib
        axs[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        status = "Détecté" if decs[i] else "Perdu"
        axs[i].set_title(f"{titles[i]}\n{status}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig("part2_attaques_result.png")
    plt.close()
    
    print("\nImages générées :")
    print("- watermarked_gray.png")
    print("- part1_grayscale_compare.png")
    print("- watermarked_rgb.png")
    print("- part2_diff_canal_R.png")
    print("- compressed.jpg")
    print("- part2_attaques_result.png")

if __name__ == "__main__":
    main()
