
import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision.transforms import v2
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

parser = argparse.ArgumentParser()
parser.add_argument('--huggingface_token', dest='huggingface_token', type=str,
                    default=os.getenv("HUGGINGFACE_TOKEN"))
parser.add_argument('--model_checkpoint', dest='model_checkpoint', type=str,
                    default="Marquez-Group-EMBL/AXIS-foundation")
parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=15)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10)
parser.add_argument('--lr', dest='lr', type=float, default=5e-6)
parser.add_argument('--train_dataset', dest='train_dataset',
                    type=str)
parser.add_argument('--test_dataset', dest='test_dataset',
                    type=str)
parser.add_argument('--image_type', dest='image_type', type=str, default="vis")
parser.add_argument('--image_size', dest='image_size', type=int, default=518)
parser.add_argument('--color_mode', dest='color_mode',
                    type=str, default="grayscale")

parser.add_argument('--wandb_entity', dest='wandb_entity',
                    type=str, default=os.getenv("WANDB_ENTITY"))
parser.add_argument('--wandb_project', dest='wandb_project',
                    type=str, default=os.getenv("WANDB_PROJECT"))

args = parser.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device != "cpu":
    print(f"CUDA device: {torch.cuda.get_device_name()}")

run_name = f"AXIS-{args.train_dataset}-{args.image_type}-{args.image_size}-b{args.batch_size}-lr{args.lr}-n_epochs{args.n_epoch}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Wandb login
if args.wandb_entity and args.wandb_project:
    wandb.login()
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_name)
    os.environ["WANDB_PROJECT"] = args.wandb_project 
    os.environ["WANDB_LOG_MODEL"] = "false"  
    run.config["model_checkpoint"] = args.model_checkpoint
    run.config["train_dataset"] = args.train_dataset
    run.config["test_dataset"] = args.test_dataset
    run.config["image_type"] = args.image_type
    run.config["lr"] = args.lr
    run.config["batch_size"] = args.batch_size
    run.config["n_epoch"] = args.n_epoch
    run.config["image_size"] = args.image_size
    run.config["color_mode"] = args.color_mode

# Huggingface login
login(token=args.huggingface_token)

# Load datasets
train_dataset = load_dataset("imagefolder", data_dir=f"{args.train_dataset}/{args.image_type}")
test_dataset = load_dataset("imagefolder", data_dir=f"{args.test_dataset}/{args.image_type}")

# Create label2id and id2label mappings
labels = test_dataset["train"].features["label"].names
print(f"Labels: {labels}")
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# Define image transformations
image_processor = AutoImageProcessor.from_pretrained(args.model_checkpoint, size={
                                                     "height": args.image_size, "width": args.image_size})
normalize = v2.Normalize(mean=image_processor.image_mean,
                         std=image_processor.image_std)
if args.color_mode == 'grayscale':
    load_image = [
            v2.ToImage(),
            v2.Grayscale(3)
            ]
else:
    load_image = [
            v2.ToImage()
            ]
train_transforms = v2.Compose(
    load_image +
    [
        v2.Resize((args.image_size, args.image_size)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ]
)
val_transforms = v2.Compose(
    load_image +
    [
        v2.Resize((args.image_size, args.image_size)),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ]
)

# Set dataset transforms
def preprocess_train(example_batch):
    example_batch["pixel_values"] = [train_transforms(
        image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    example_batch["pixel_values"] = [val_transforms(
        image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

train_ds = train_dataset["train"]
train_ds.set_transform(preprocess_train)
val_ds = test_dataset["train"]
val_ds.set_transform(preprocess_val)
# Calculate class weights to handle class imbalance
class_counts = np.array([res["counts"].as_py()
                   for res in train_dataset["train"].data["label"].value_counts()])

loss_weights = torch.tensor(len(train_dataset["train"].data)/(2 * class_counts)).float().to(device)

# Load model and apply LoRA
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable parameters: {trainable_params} || all parameters: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

model = AutoModelForImageClassification.from_pretrained(
    args.model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
try:
    print('LOADING PEFT MODEL')
    config = PeftConfig.from_pretrained(args.model_checkpoint)
    lora_model = PeftModel.from_pretrained(
        model, args.model_checkpoint, is_trainable=True)
    print("PEFT model loaded successfully")
    print_trainable_parameters(lora_model)
except Exception as e:
    print(f"Error loading PEFT model: {e}")
    print("Creating new LoRA model")
    config = LoraConfig(
        r=25,
        lora_alpha=20,
        target_modules="all-linear",
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    lora_model = get_peft_model(model, config)
    print("New LoRA model created") 
    print_trainable_parameters(lora_model)

model.to(device)
lora_model.to(device)


# Define training arguments
model_name = args.model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    run_name,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=args.batch_size,
    fp16=True,
    num_train_epochs=args.n_epoch,
    logging_steps=1,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="balanced_accuracy",
    push_to_hub=True,
    label_names=["labels"]
)

# Define metrics
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    y_pred = [abs(prediction - 1) for prediction in predictions]
    y_true = [abs(label - 1) for label in eval_pred.label_ids]
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    true_other = cm[0, 0]
    false_other = cm[1, 0]
    true_crystal = cm[1, 1]
    false_crystal = cm[0, 1]
    return {
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1, "accuracy": accuracy,
        "true_crystal": true_crystal,
        "false_crystal": false_crystal,
        "true_other": true_other,
        "false_other": false_other
    }

# Data collator
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"]
                               for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    if "google" in args.model_checkpoint:
        return {"pixel_values": pixel_values, "labels": labels, "interpolate_pos_encoding": True}
    else:
        return {"pixel_values": pixel_values, "labels": labels}

# Custom Trainer to incorporate class weights in loss computation
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.CrossEntropyLoss(weight=loss_weights)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Initialize Trainer and start training
trainer = CustomTrainer(
    lora_model,
    training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn
)
train_results = trainer.train()


print(train_results)
