"""Download the original third-party model weights; verify their SHA-256 hashes."""
import hashlib
from pathlib import Path
from urllib.request import urlopen, Request

MODELS = Path(__file__).resolve().parent / "models"
ASSETS = (
    ("face_detection_yunet_2023mar.onnx", "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx", "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    ("face_recognition_sface_2021dec.onnx", "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx", "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
    ("gesture_recognizer.task", "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task", "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482"),
)

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    MODELS.mkdir(exist_ok=True)
    for name, url, expected in ASSETS:
        target = MODELS / name
        if target.is_file() and digest(target) == expected:
            print(f"Verified {name}")
            continue
        temp = target.with_suffix(".part")
        try:
            print(f"Downloading {name}", flush=True)
            with urlopen(Request(url, headers={"User-Agent": "FaceVision/1.0"}), timeout=90) as response, temp.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if digest(temp) != expected:
                raise RuntimeError(f"SHA-256 mismatch: {name}")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
