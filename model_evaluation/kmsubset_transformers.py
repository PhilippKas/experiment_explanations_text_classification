# Load Libs
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, f1_score, accuracy_score

from transformers import AutoTokenizer, DataCollatorWithPadding, TrainerCallback, Trainer

from datasets import Dataset

import json

# Load preprocessing script and its vars 
from preprocessing import *

n_to_test = [250,500,750,1000,1250]

accuracy = []
f1_micro = []
f1_macro = []

for al_samplesize in n_to_test:
    # Parameters of the run
    model_name = "distilbert-base-uncased"

    # Based on this:
    # https://huggingface.co/transformers/v3.2.0/custom_datasets.html
    # https://lajavaness.medium.com/multiclass-and-multilabel-text-classification-in-one-bert-model-95c54aab59dc

    # For Model Selection purpose
    # https://huggingface.co/transformers/v3.3.1/pretrained_models.html

    kmeans = KMeans(
        init="random",
        n_clusters=al_samplesize,
        n_init=10,
        max_iter=300,
        random_state=42)

    kmeans.fit(embeddings_train)

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings_train)

    texts_train_selected = [texts_train[i] for i in closest]

    labels_onehot_train_selected = [labels_onehot_train[i] for i in closest]

    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True)

    input_ids_train = [tokenizer(val, truncation=True) for val in texts_train_selected]
    input_ids_test = [tokenizer(val, truncation=True) for val in texts_test]

    # Create Dataset 
    ds_train = Dataset.from_dict({"text": texts_train_selected, "label": np.array(labels_onehot_train_selected).astype("float64")})
    ds_test = Dataset.from_dict({"text": texts_test, "label": np.array(labels_onehot_test).astype("float64")})

    def tokenization(example):
        return tokenizer(example["text"], truncation=True)

    ds_train = ds_train.map(tokenization, batched=True)
    ds_test = ds_test.map(tokenization, batched=True)

    def get_preds_from_logits(logits):
        ret = np.zeros(logits.shape)
        for i in range(logits.shape[1]):
            ret[:, i] = (logits[:, i] >= 0).astype(int)
        print(ret)
        return ret

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        
        predictions = get_preds_from_logits(logits)
        accuracy = accuracy_score(labels, predictions)
        f1_micro = f1_score(labels, predictions, average="micro")
        f1_macro = f1_score(labels, predictions, average="macro")
        
        
        return {"accuracy": accuracy, "f1_micro": f1_micro, "f1_macro": f1_macro}

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set a specific Loss

    class MultiTaskClassificationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs[0]
            
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            
            return (loss, outputs) if return_outputs else loss
        
    EPOCHS = 50

    class PrinterCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, logs=None, **kwargs):
            print(f"\nEpoch {state.epoch}: ")

    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=90, id2label=int_to_labels, label2id=labels_to_int)

    # !!! Cuda Memory becomes a big problem 
    # Because Matrix Multiplication is fucking wild see: https://christopher5106.github.io/deep/learning/2018/10/28/understand-batch-matrix-multiplication.html
    # https://huggingface.co/docs/transformers/v4.20.1/en/performance

    training_args = TrainingArguments(
        #output_dir="F:/huggingface_models/my_first_model",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=EPOCHS,
        weight_decay=.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        metric_for_best_model="accuracy",
        bf16=True, 
        seed=42
    )

    trainer = MultiTaskClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[PrinterCallback]
    )

    trainer.train()

    train_logs = np.zeros((len(texts_test), 90))
    with torch.no_grad():
        for i, text in enumerate(texts_test):
            tok = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cuda:0")
            logs = model(**tok).logits.cpu().detach().numpy()
            train_logs[i,:] = logs
            del tok, logs
            torch.cuda.empty_cache() 
        
    # Decode the result
    preds = get_preds_from_logits(train_logs)
    
    accuracy.append(accuracy_score(labels_onehot_test, preds))
    f1_micro.append(f1_score(labels_onehot_test, preds, average="micro"))
    f1_macro.append(f1_score(labels_onehot_test, preds, average="macro"))

    print(f"Results of the {al_samplesize}n run:")
    print(accuracy[-1])
    print(f1_micro[-1])
    print(f1_macro[-1])

results = {
    "n":n_to_test,
    "accuracy":accuracy,
    "f1_micro":f1_micro,
    "f1_macro":f1_macro}

# Serializing json
json_object = json.dumps(results, indent=4)

# Writing to sample.json
with open("results/kmsubset_transformer_50.json", "w") as outfile:
    outfile.write(json_object)