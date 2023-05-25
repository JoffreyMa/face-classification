import os
from dataset import Faceset

from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import DefaultDataCollator
import evaluate
import numpy as np

from datasets import Dataset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
from datasets.utils import DownloadMode

if __name__ == "__main__":
    dataset_path = os.path.join("data", "train", "dataset.hf")

    #dataset = load_dataset("imagefolder", data_dir="data/train", cache_dir=".cache/huggingface/datasets", download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS, split="train")
    #dataset.save_to_disk(dataset_path)

    dataset = load_from_disk(dataset_path)

    faces = dataset.train_test_split(test_size=0.2)

    checkpoint = "google/vit-base-patch16-224-in21k" # change it
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples
    
    faces = faces.with_transform(transforms)

    image_dataset = faces.remove_columns(["genre"])
    extra_features_dataset = faces.remove_columns(["image", "label"])


    data_collator = DefaultDataCollator()

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir="models",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=image_dataset["train"],
        eval_dataset=image_dataset["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()