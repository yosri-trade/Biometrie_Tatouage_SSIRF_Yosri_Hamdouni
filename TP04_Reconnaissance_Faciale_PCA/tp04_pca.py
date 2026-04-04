import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class FaceRecognitionPCA:
    def __init__(self, n_components=30):
        self.n_components = n_components
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(script_dir, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.mean = None
        self.eigenvectors = None
        self.projections = None
        self.labels = None

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        if len(faces) == 0:
            return None
            
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        face_crop = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (100, 100))
        return face_resized, largest_face

    def load_dataset(self, dataset_path):
        X = []
        y = []
        for person_name in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                result = self.detect_face(img)
                if result is not None:
                    face_img, _ = result
                    # Vectorisation : aplatissement 1D
                    face_vector = face_img.flatten()
                    X.append(face_vector)
                    y.append(person_name)
                    
        return np.array(X), np.array(y)

    def compute_pca(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Astuce Eigenfaces : SVD evite la creation d'un Cov 10000x10000
        # qui bloquerait par complexité O(D^3).
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        eigenvectors = Vt.T
        
        # Conserver n_components vectoriels
        self.eigenvectors = eigenvectors[:, :self.n_components]
        
        # Projections PCA base d'entraînement matricielle
        self.projections = np.dot(X_centered, self.eigenvectors)

    def project(self, face_vector):
        centered = face_vector - self.mean
        return np.dot(centered, self.eigenvectors)

    def recognize(self, image_path, threshold):
        img = cv2.imread(image_path)
        if img is None:
            return None, None, "File Not Found", None
            
        result = self.detect_face(img)
        if result is None:
            return None, None, "No Face Detectée", None
            
        face_img, face_rect = result
        face_vect = face_img.flatten()
        test_proj = self.project(face_vect)
        
        # Distance Euclienne absolue dans l'hyper-sphère PCA
        distances = [np.linalg.norm(test_proj - train_proj) for train_proj in self.projections]
        min_dist = min(distances)
        min_idx = np.argmin(distances)
        matched_identity = self.labels[min_idx]
        
        decision = "Match" if min_dist <= threshold else "No Match"
        return matched_identity, min_dist, decision, face_rect

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset")
    test_dir = os.path.join(script_dir, "test_images")
    results_dir = os.path.join(script_dir, "results")
    
    print("Initialisation du pipeline Eigenfaces PCA (k=10)...")
    pca_system = FaceRecognitionPCA(n_components=10)
    
    X, y = pca_system.load_dataset(dataset_path)
    if len(X) == 0:
        print("Dataset introuvable ou visages non détéctés.")
        exit()
        
    pca_system.labels = y
    # Cap n_components pour les petits datasets (ex: len=10)
    pca_system.n_components = min(10, len(X)) 
    print(f"Bases d'entraînement récupérée : {len(X)} images (dim {X.shape[1]}).")
    pca_system.compute_pca(X)
    
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    if not test_files:
        print("Dossier de tests vide.")
        exit()
        
    test_image_path = os.path.join(test_dir, test_files[0])
    threshold = 4000.0  # Seuil de démonstration PCA Euclidienne
    matched_id, min_dist, decision, rect = pca_system.recognize(test_image_path, threshold)
    
    if min_dist is None:
        print("Rapport d'échec : Reconnaissance nulle.")
        exit()
        
    print(f"Identite Trouvée : {matched_id}")
    print(f"Distance Min : {min_dist:.2f}")
    print(f"Décision finale : {decision}")
    
    img_bgr = cv2.imread(test_image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    x, y_box, w, h = rect
    color = (0, 255, 0) if decision == "Match" else (255, 0, 0)
    cv2.rectangle(img_rgb, (x, y_box), (x+w, y_box+h), color, 3)
    cv2.putText(img_rgb, f"{matched_id} [{min_dist:.0f}]", (x, y_box-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"PCA Eigenfaces | {matched_id} | Dist: {min_dist:.0f} | {decision}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "test_result.png"))
    plt.close()
    
    print("Graphique sauvegardé sous results/test_result.png")
