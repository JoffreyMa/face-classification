import os
from transformers import pipeline
from datasets import load_dataset, load_from_disk
import pandas as pd
import pickle


if __name__ == "__main__":
    print("inference")
    '''
    dataset_path = os.path.join("data", "test", "dataset_test.hf")
    #dataset = load_dataset("imagefolder", data_dir="data/test", cache_dir=".cache/huggingface/datasets", split="test")
    #dataset.save_to_disk(dataset_path)

    dataset = load_from_disk(dataset_path)
    images = dataset["image"]
    classifier = pipeline("image-classification", model="models/checkpoint-7221")
    results = classifier(images)
    '''
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)

    converted_data = []
    for d in results:
        item = d[0]
        converted_data.append(int(item['label'][-1]))

    submission=pd.DataFrame(converted_data, columns= ['score'])
    submission['score'] = submission['score'].map(lambda x: -1 if x == 0 else x)
    submission.to_csv("data_challenge_joffreyma.csv", header = None, index = None)