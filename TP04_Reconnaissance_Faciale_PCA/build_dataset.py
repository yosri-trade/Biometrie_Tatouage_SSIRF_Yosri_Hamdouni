import os
import urllib.request
from PIL import Image, ImageEnhance
import stat

base_dir = r"C:\Users\yhamd\OneDrive\Desktop\Projet biometrie\Biometrie_Tatouage_SSIRF_Yosri_Hamdouni\TP04_Reconnaissance_Faciale_PCA"

def augment_and_save(src_path, dst_folder, prefix, count):
    if not os.path.exists(src_path):
        print(f"Skipping {src_path}, not found.")
        return
    img = Image.open(src_path).convert("RGB")
    for i in range(count):
        out = img.copy()
        # Variations for robust eigenfaces
        if i % 3 == 1:
            out = ImageEnhance.Brightness(out).enhance(1.2)
        elif i % 3 == 2:
            out = ImageEnhance.Brightness(out).enhance(0.8)
        
        if i % 4 == 0:
            out = ImageEnhance.Contrast(out).enhance(1.1)

        # Subtle noise or shifts can also be done, but brightness/contrast is enough for PCA tests
        out.save(os.path.join(dst_folder, f"{prefix}_{i}.jpg"))

def main():
    # 1. Download Lenna for person2
    person2_src = os.path.join(base_dir, "lenna.jpg")
    try:
        urllib.request.urlretrieve("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png", person2_src)
    except:
        pass
    
    person1_src = r"C:\Users\yhamd\OneDrive\Desktop\Projet biometrie\Biometrie_Tatouage_SSIRF_Yosri_Hamdouni\TP03_Reconnaissance_Faciale\reference_face.jpg"
    test_src_1 = r"C:\Users\yhamd\OneDrive\Desktop\Projet biometrie\Biometrie_Tatouage_SSIRF_Yosri_Hamdouni\TP03_Reconnaissance_Faciale\test_face.jpg"
    
    # 2. Populate dataset/person1
    p1_dir = os.path.join(base_dir, "dataset", "person1")
    augment_and_save(person1_src, p1_dir, "p1_train", 5)
    
    # 3. Populate dataset/person2
    p2_dir = os.path.join(base_dir, "dataset", "person2")
    augment_and_save(person2_src, p2_dir, "p2_train", 5)
    
    # 4. Populate test_images
    test_dir = os.path.join(base_dir, "test_images")
    augment_and_save(test_src_1, test_dir, "test_p1", 2)
    augment_and_save(person2_src, test_dir, "test_p2", 2)
    
    print("Dataset generation completed.")

if __name__ == "__main__":
    main()
