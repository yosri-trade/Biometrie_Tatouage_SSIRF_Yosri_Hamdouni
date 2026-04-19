import cv2
import numpy as np
import random
import os

def message_to_bin(msg):
    """Convertit un texte en séquence binaire."""
    return ''.join([format(ord(c), "08b") for c in msg])

def bin_to_message(binary_str):
    """Convertit une séquence binaire en texte."""
    chars = [chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8)]
    return ''.join(chars)

# ================= PARTIE 1 : LSB Niveau de Gris =================

def embed_lsb_gray(image_path, message, output_path):
    """
    Étapes :
    - Lire image en niveaux de gris
    - Convertir message en binaire
    - Aplatir image (flatten)
    - Modifier LSB des pixels
    - Reconstruire image
    - Sauvegarder
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
        
    binary_message = message_to_bin(message)
    msg_len = len(binary_message)
    
    flatten_img = img.flatten()
    
    if msg_len > len(flatten_img):
        raise ValueError("Le message est trop long pour cette image.")
        
    for i in range(msg_len):
        # Mettre le LSB à 0 et ajouter le bit du message
        flatten_img[i] = (flatten_img[i] & ~1) | int(binary_message[i])
        
    stego_img = flatten_img.reshape(img.shape)
    cv2.imwrite(output_path, stego_img)

def extract_lsb_gray(image_path, msg_len):
    """
    Étapes :
    - Lire image
    - Extraire LSB des pixels
    - Reconstruire le message
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
        
    flatten_img = img.flatten()
    bit_len = msg_len * 8
    
    extracted_bits = [str(flatten_img[i] & 1) for i in range(bit_len)]
    binary_str = "".join(extracted_bits)
    
    return bin_to_message(binary_str)

# ================= PARTIE 2 : LSB RGB =================

def embed_lsb_rgb(image_path, message, output_path):
    """
    Étapes :
    - Lire image couleur
    - Parcourir pixels (R,G,B)
    - Insérer bits dans chaque canal
    - Sauvegarder image
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
        
    binary_message = message_to_bin(message)
    msg_len = len(binary_message)
    
    flatten_img = img.flatten() # Équivalent de parcourir séquentiellement B, G, R
    
    if msg_len > len(flatten_img):
        raise ValueError("Le message est trop long pour cette image.")
        
    for i in range(msg_len):
        flatten_img[i] = (flatten_img[i] & ~1) | int(binary_message[i])
        
    stego_img = flatten_img.reshape(img.shape)
    cv2.imwrite(output_path, stego_img)

def extract_lsb_rgb(image_path, msg_len):
    """
    Étapes :
    - Lire image
    - Extraire LSB de chaque canal
    - Reconstituer message
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
        
    flatten_img = img.flatten()
    bit_len = msg_len * 8
    
    extracted_bits = [str(flatten_img[i] & 1) for i in range(bit_len)]
    binary_str = "".join(extracted_bits)
    
    return bin_to_message(binary_str)

# ================= PARTIE 3 : LSB avec Clé Secrète =================

def embed_lsb_key(image_path, message, output_path, key):
    """
    Étapes :
    - Lire image grayscale
    - Générer positions aléatoires avec seed = key
    - Insérer bits dans positions sélectionnées
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
        
    binary_message = message_to_bin(message)
    msg_len = len(binary_message)
    
    flatten_img = img.flatten()
    
    if msg_len > len(flatten_img):
        raise ValueError("Le message est trop long pour cette image.")
        
    # Génération des positions aléatoires avec graine (seed)
    random.seed(key)
    positions = random.sample(range(len(flatten_img)), msg_len)
    
    for i in range(msg_len):
        pos = positions[i]
        flatten_img[pos] = (flatten_img[pos] & ~1) | int(binary_message[i])
        
    stego_img = flatten_img.reshape(img.shape)
    cv2.imwrite(output_path, stego_img)

def extract_lsb_key(image_path, msg_len, key):
    """
    Étapes :
    - Générer mêmes positions
    - Extraire LSB
    - Reconstituer message
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
        
    flatten_img = img.flatten()
    bit_len = msg_len * 8
    
    # Génération des mêmes positions aléatoires avec la même graine
    random.seed(key)
    positions = random.sample(range(len(flatten_img)), bit_len)
    
    extracted_bits = [str(flatten_img[pos] & 1) for pos in positions]
    binary_str = "".join(extracted_bits)
    
    return bin_to_message(binary_str)

# ================= PROGRAMME PRINCIPAL =================

def main():
    message = "bonjour"

    # Vérification ou Création de l'image de test 'input.png'
    if not os.path.exists("input.png"):
        print("Création de l'image de test 'input.png' (100x100 RGB)...")
        # Image couleur 100x100
        test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite("input.png", test_img)
    
    # PARTIE 1
    print("\n--- PARTIE 1 : LSB Niveau de Gris ---")
    embed_lsb_gray("input.png", message, "gray_output.png")
    res_gray = extract_lsb_gray("gray_output.png", len(message))
    print(f"Message extrait (Gray) : {res_gray}")
    
    # PARTIE 2
    print("\n--- PARTIE 2 : LSB RGB ---")
    embed_lsb_rgb("input.png", message, "rgb_output.png")
    res_rgb = extract_lsb_rgb("rgb_output.png", len(message))
    print(f"Message extrait (RGB) : {res_rgb}")
    
    # PARTIE 3
    print("\n--- PARTIE 3 : LSB avec Clé Secrète ---")
    embed_lsb_key("input.png", message, "key_output.png", key=42)
    res_key = extract_lsb_key("key_output.png", len(message), key=42)
    print(f"Message extrait (Clé=42) : {res_key}")
    
    # Test d'extraction avec une mauvaise clé
    try:
        res_wrong_key = extract_lsb_key("key_output.png", len(message), key=99)
        print(f"Message extrait (Clé incorrecte=99) : {res_wrong_key}")
    except Exception as e:
        print(f"Erreur avec la clé 99 : {e}")

if __name__ == "__main__":
    main()
