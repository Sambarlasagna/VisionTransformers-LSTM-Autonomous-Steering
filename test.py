import numpy as np
import pandas as pd

# -------- CONFIG --------
CSV_PATH = r"D:\Programming\Projects\VisionTransformers-LSTM-Autonomous-Steering\udacity_dataset\self_driving_car_dataset_make\driving_log.csv"
# ------------------------

def main():
    # Load CSV
    df = pd.read_csv(CSV_PATH, header=None)
    df.columns = [
        "center", "left", "right",
        "steering", "throttle", "brake", "speed"
    ]

    steering_values = df["steering"].values.astype(np.float32)

    # Compute deltas
    deltas = np.zeros_like(steering_values)
    deltas[1:] = steering_values[1:] - steering_values[:-1]

    print("Total samples:", len(deltas))
    print("Mean Δ:", deltas.mean())
    print("Std Δ:", deltas.std())
    print("Min Δ:", deltas.min())
    print("Max Δ:", deltas.max())

    # Optional: percentiles
    print("Percentiles (5, 50, 95):",
          np.percentile(deltas, [5, 50, 95]))

if __name__ == "__main__":
    main()
