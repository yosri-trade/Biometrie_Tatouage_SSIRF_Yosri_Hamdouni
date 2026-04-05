import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_recognition

class FaceRecognitionDL:
    def __init__(self):
        self.database = {}

    def detect_face(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)
        
        if not face_locations:
            return None, None
            
        best_face_loc = max(face_locations, key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]))
        top, right, bottom, left = best_face_loc
        
        face_crop = img_rgb[top:bottom, left:right]
        if face_crop.size == 0:
            return None, None
            
        face_resized = cv2.resize(face_crop, (160, 160))
        rect = (left, top, right-left, bottom-top)
        return face_resized, rect

    def extract_embedding(self, image, face_info): # signature maintained for compat with experiments.py
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        left, top, w, h = face_info
        # face_recognition takes (top, right, bottom, left)
        locs = [(top, left+w, top+h, left)]
        encodings = face_recognition.face_encodings(img_rgb, known_face_locations=locs)
        if len(encodings) > 0:
            return encodings[0]
        return None

    def build_database(self, dataset_path):
        for person_name in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            if person_name not in self.database:
                self.database[person_name] = []
                
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is None: continue
                
                face_data, rect = self.detect_face(img)
                if face_data is not None:
                    emb = self.extract_embedding(img, rect)
                    if emb is not None:
                        self.database[person_name].append(emb)

    def euclidean_distance(self, emb1, emb2):
        return np.linalg.norm(emb1 - emb2)

    def cosine_similarity(self, emb1, emb2):
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0: return 0.0
        return np.dot(emb1, emb2) / (n1 * n2)

    def recognize(self, image_path, threshold=0.5, metric='euclidean'):
        img = cv2.imread(image_path)
        if img is None:
            return None, None, "File Not Found", None
            
        face_data, rect = self.detect_face(img)
        if face_data is None:
            return None, None, "No Face Detectée", None
            
        test_emb = self.extract_embedding(img, rect)
        if test_emb is None: return None, None, "No DL Embedding", None
             
        best_match_label = "Inconnu"
        best_score = float('inf') if metric == 'euclidean' else -float('inf')
        
        for label, embeddings in self.database.items():
            for ref_emb in embeddings:
                if metric == 'euclidean':
                    score = self.euclidean_distance(test_emb, ref_emb)
                    if score < best_score:
                        best_score = score
                        best_match_label = label
                else: 
                    score = self.cosine_similarity(test_emb, ref_emb)
                    if score > best_score:
                        best_score = score
                        best_match_label = label
                        
        if metric == 'euclidean':
            decision = "Match" if best_score <= threshold else "No Match"
        else:
            decision = "Match" if best_score >= threshold else "No Match"
            
        if decision == "No Match":
            best_match_label = "Inconnu"
            
        return best_match_label, best_score, decision, rect

def main():
    import os
    import cv2
    import matplotlib.pyplot as plt

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset")
    test_dir = os.path.join(script_dir, "test_images")
    results_dir = os.path.join(script_dir, "results")
    
    print("Initialisation du modele DL Face Recognition (Dlib 128D)...")
    sys = FaceRecognitionDL()
    print("Construction base de données...")
    sys.build_database(dataset_path)
    
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    if not test_files: return
    test_img = os.path.join(test_dir, test_files[0])
    
    print(f"Lancement sur l'image : {test_files[0]}")
    label, score, decision, rect = sys.recognize(test_img, threshold=0.5, metric='euclidean')
    
    if rect is None:
         print("Echec detect")
         return
         
    print(f"Identité : {label} | Score: {score:.4f} | Dec: {decision}")
    
    img_bgr = cv2.imread(test_img)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    x, y, w, h = rect
    color = (0, 255, 0) if decision == "Match" else (255, 0, 0)
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 3)
    cv2.putText(img_rgb, f"{label} [{score:.2f}]", (x, max(15, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"Face DL | ID: {label} | Dist: {score:.2f} | {decision}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "test_result_dl.png"))
    plt.close()
    print("Graphique sauvegardé sous results/test_result_dl.png")

if __name__ == "__main__":
    main()
