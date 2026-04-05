import requests
import os

print("Downloading SFace via HuggingFace mirror...")
url_sface = "https://huggingface.co/csn-tt/sface/resolve/main/face_recognition_sface_2021dec.onnx"
r1 = requests.get(url_sface)
with open("face_recognition_sface_2021dec.onnx", "wb") as f:
    f.write(r1.content)

print("Downloading YuNet via HuggingFace...")
url_yunet = "https://huggingface.co/csn-tt/sface/resolve/main/face_detection_yunet_2023mar.onnx"
r2 = requests.get(url_yunet)
with open("face_detection_yunet_2023mar.onnx", "wb") as f:
    f.write(r2.content)

print("Downloads finished successfully. File Sizes:")
print(f"Sface: {os.path.getsize('face_recognition_sface_2021dec.onnx')} bytes")
print(f"Yunet: {os.path.getsize('face_detection_yunet_2023mar.onnx')} bytes")
