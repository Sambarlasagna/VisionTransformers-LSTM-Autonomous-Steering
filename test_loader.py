from data_loader import UdacityDrivingDataset
from torch.utils.data import DataLoader

DATASET_DIR = r"D:\Programming\Projects\VisionTransformers-LSTM-Autonomous-Steering\udacity_dataset\self_driving_car_dataset_make"
CSV_PATH = DATASET_DIR + r"\driving_log.csv"
IMG_DIR = DATASET_DIR + r"\IMG"

dataset = UdacityDrivingDataset(CSV_PATH, IMG_DIR)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

images, angles = next(iter(loader))

print("Image batch shape:", images.shape)   # (B, 3, 224, 224)
print("Steering angles:", angles)
