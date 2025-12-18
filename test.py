import pandas as pd

df = pd.read_csv("D:\Programming\Projects\VisionTransformers-LSTM-Autonomous-Steering\Run\driving_log.csv", header=None)
print(df[3].describe())
