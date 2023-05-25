# Reorganize the data folder to use HuggingFace ImageFolder
import pandas as pd
import shutil
import os
from datasets import load_dataset

if __name__ == "__main__":
    move = False
    split = False
    upload = True

    if move:
        print("Move the files")
        path = "data/train.txt"
        train = pd.read_csv(path, sep = '\t', header = None, names = ['image','label', 'genre'])
        train = train.astype({"label": int, "genre": int})
        train['label'] = train['label'].map(lambda x: 0 if x == -1 else x)
        for index, row in train.iterrows():
            src = os.path.join('data', '_train', row['image'])
            dst = os.path.join('data', 'train', str(row['label']), row['image'])
            shutil.copy(src, dst)

    if split:
        print("Split into train test")
        # do the train test split

    if upload:
        '''path = "_data/train.txt"
        train = pd.read_csv(path, sep = '\t', header = None, names = ['file_name','label', 'genre'])
        train = train.astype({"label": int, "genre": int})
        train['label'] = train['label'].map(lambda x: 0 if x == -1 else x)
        train.to_csv("data/metadata.csv", index=False)'''

        print("Upload on HuggingFace")
        dataset = load_dataset("imagefolder", data_dir="data", cache_dir=".cache/huggingface/datasets")
        dataset.push_to_hub("JoffreyMa/BGDIA704_faces", token="your_write_token")
        print("finito")