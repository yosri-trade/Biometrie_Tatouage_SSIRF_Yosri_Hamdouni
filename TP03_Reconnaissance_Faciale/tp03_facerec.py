import os
import cv2
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

class FaceVerificationSystem:
    def __init__(self, cascade_path):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.reference_features = None

    def detect_face(self, image):
        """
        Convert to grayscale, use Viola-Jones. Return coordinates of largest face.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        if len(faces) == 0:
            return None
        
        # S'il y a plusieurs visages, retourner celui avec la plus grande surface (w * h)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        return largest_face

    def extract_lbp_features(self, face_image):
        """
        Resize (128x128). Classique LBP. Normalised 256-bins hist.
        """
        resized = cv2.resize(face_image, (128, 128))
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
            
        lbp = np.zeros_like(gray)
        rows, cols = gray.shape
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j]   >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1]   >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j]   >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1]   >= center) << 0
                lbp[i, j] = code
                
        # Calculate histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        
        # Normalize
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def setup_reference(self, image_path):
        """
        Charge image, détecte visage, extrait LBP.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")
        
        face_rect = self.detect_face(img)
        if face_rect is None:
            raise ValueError(f"Aucun visage détecté sur l'image de référence: {image_path}")
            
        x, y, w, h = face_rect
        face_crop = img[y:y+h, x:x+w]
        
        self.reference_features = self.extract_lbp_features(face_crop)
        print("Image de référence configurée avec succès.")

    def verify_face(self, image_path, threshold=0.75):
        """
        Extrait LBP et compare à la ref via Euclidean distance. similarity = 1 - dist.
        """
        if self.reference_features is None:
            raise ValueError("Veuillez d'abord configurer l'image de référence.")
            
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image introuvable : {image_path}")
            
        face_rect = self.detect_face(img)
        if face_rect is None:
            return {"error": f"Aucun visage détecté sur l'image test: {image_path}"}
            
        x, y, w, h = face_rect
        face_crop = img[y:y+h, x:x+w]
        
        test_features = self.extract_lbp_features(face_crop)
        dist = euclidean(self.reference_features, test_features)
        
        similarity = max(0.0, 1.0 - dist)
        
        decision = "Match" if similarity >= threshold else "No Match"
        
        return {
            "distance": dist,
            "similarity": similarity,
            "decision": decision,
            "face_rect": face_rect
        }


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    
    cascade_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
    ref_path = os.path.join(script_dir, "reference_face.jpg")
    test_path = os.path.join(script_dir, "test_face.jpg")
    
    # 1. Instancier le système.
    sys = FaceVerificationSystem(cascade_path)
    
    # 2. Lancer setup_reference('reference_face.jpg').
    try:
        sys.setup_reference(ref_path)
    except Exception as e:
        print(f"Erreur Ref: {e}")
        exit()
        
    # 3. Lancer verify_face('test_face.jpg').
    res = sys.verify_face(test_path, threshold=0.75)
    
    # 4. Afficher dans la console les résultats
    if "error" in res:
        print(f"Erreur Test: {res['error']}")
        exit()
        
    print(f"Distance euclidienne : {res['distance']:.4f}")
    print(f"Similarité : {res['similarity']:.4f}")
    print(f"Décision : {res['decision']}")
    
    # 5. Utiliser matplotlib pour afficher côte à côte
    img_ref = cv2.imread(ref_path)
    img_test = cv2.imread(test_path)
    
    # Convert BGR to RGB for matplotlib
    img_ref_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    img_test_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
    
    # Add rectangle to ref
    ref_rect = sys.detect_face(img_ref)
    if ref_rect is not None:
        rx, ry, rw, rh = ref_rect
        cv2.rectangle(img_ref_rgb, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 3)
        
    # Add rectangle to test
    tx, ty, tw, th = res["face_rect"]
    cv2.rectangle(img_test_rgb, (tx, ty), (tx+tw, ty+th), (0, 255, 0) if res['decision'] == 'Match' else (255, 0, 0), 3)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Verification: {res['decision']} (Sim: {res['similarity']:.3f})", fontsize=14, fontweight='bold', color='green' if res['decision']=='Match' else 'red')
    
    axes[0].imshow(img_ref_rgb)
    axes[0].set_title("Référence")
    axes[0].axis("off")
    
    axes[1].imshow(img_test_rgb)
    axes[1].set_title("Test")
    axes[1].axis("off")
    
    # 6. Sauvegarder
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "resultat_verification.png"))
    plt.close()
    
    print("Résultat sauvegardé dans results/resultat_verification.png")
