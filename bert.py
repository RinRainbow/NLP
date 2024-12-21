from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =============================================================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# =============================================================


class DisasterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_model(model, train_loader, val_loader, num_epochs=2):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * num_epochs,
    )

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device),
            )
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

        # エポックごとの損失と精度を表示
        train_acc = accuracy_score(all_labels, all_preds)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}"
        )

        validate_model(model, val_loader)


def validate_model(model, val_loader):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device),
            )
            loss = outputs.loss
            val_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")


# =============================================================

# import train data
tarin_data = pd.read_csv("train.csv")
texts = tarin_data["text"].tolist()
labels = tarin_data["target"].tolist()

# import test data
test_data = pd.read_csv("test.csv")
test_texts = test_data["text"].tolist()

# split data (80%: train, 20%: validation)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# prerare dataset
train_dataset = DisasterDataset(train_texts, train_labels, tokenizer)
val_dataset = DisasterDataset(val_texts, val_labels, tokenizer)
test_dataset = DisasterDataset(test_texts, [0] * len(test_texts), tokenizer)

# craete data loader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
train_model(model, train_loader, val_loader, num_epochs=2)


# predict
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
        )
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())

# create submission file
submission = pd.DataFrame({"id": test_data["id"], "target": predictions})
submission.to_csv("submission.csv", index=False)
