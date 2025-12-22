import cv2
import os
import glob

# -----------------------------
# CONFIG
IMG_DIR = "track/forest/IMG"
OUTPUT_VIDEO = "./videos/udacity_forest_drive.mp4"
FPS = 20  # Udacity is ~20 FPS

# -----------------------------
# LOAD IMAGE FILES
# -----------------------------
image_paths = sorted(
    glob.glob(os.path.join(IMG_DIR, "center_*.jpg"))
)

assert len(image_paths) > 0, "No images found!"

# Read first image to get size
first_img = cv2.imread(image_paths[0])
height, width, _ = first_img.shape

# -----------------------------
# VIDEO WRITER
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (width, height))

print(f"[INFO] Writing {len(image_paths)} frames...")

# -----------------------------
# WRITE FRAMES
# -----------------------------
for img_path in image_paths:
    img = cv2.imread(img_path)
    video.write(img)

video.release()
print("[DONE] Video saved as:", OUTPUT_VIDEO)
