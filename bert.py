# # Best hyperparameters: {'learning_rate': 1.3392029532959121e-05, 'batch_size': 8, 'max_length': 64}
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import (
#     BertForSequenceClassification,
#     AdamW,
#     get_scheduler,
#     BertTokenizer,
# )
# from torch.optim import AdamW
# from torch.nn import CrossEntropyLoss
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import re
# import matplotlib.pyplot as plt
# import optuna

# # =============================================================


# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs")
#     model = torch.nn.DataParallel(model)
# # =============================================================


# class DisasterDataset(Dataset):
#     # def __init__(self, texts, locations, labels, tokenizer, max_length=64):
#     #     self.texts = texts
#     #     # self.locations = locations
#     #     self.labels = labels
#     #     self.tokenizer = tokenizer
#     #     self.max_length = max_length
#     def __init__(self, texts, keywords, labels, tokenizer, max_length=64):
#         self.texts = texts
#         self.keywords = keywords
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         combined_text = f"{self.texts[idx]} [KEYWORD] {self.keywords[idx]}"
#         # combined_text = self.texts[idx]
#         encoding = self.tokenizer(
#             combined_text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         )
#         item = {key: val.squeeze(0).to(device) for key, val in encoding.items()}
#         item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long).to(device)
#         return item


# def train_model(model, train_loader, val_loader, num_epochs=3):
#     # optimizer = AdamW(model.parameters(), lr=1.3392029532959121e-5)
#     optimizer = AdamW(model.parameters(), lr=2e-5)
#     scheduler = get_scheduler(
#         "linear",
#         optimizer=optimizer,
#         num_warmup_steps=0,
#         num_training_steps=len(train_loader) * num_epochs,
#     )

#     for epoch in range(num_epochs):

#         model.train()
#         train_loss = 0
#         all_preds = []
#         all_labels = []

#         for batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(
#                 input_ids=batch["input_ids"].to(device),
#                 attention_mask=batch["attention_mask"].to(device),
#                 labels=batch["labels"].to(device),
#             )
#             loss = outputs.loss
#             train_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(batch["labels"].cpu().numpy())

#         # エポックごとの損失と精度を表示
#         train_acc = accuracy_score(all_labels, all_preds)
#         print(
#             f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}"
#         )

#         validate_model(model, val_loader)


# def validate_model(model, val_loader):
#     model.eval()
#     val_loss = 0
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in val_loader:
#             outputs = model(
#                 input_ids=batch["input_ids"].to(device),
#                 attention_mask=batch["attention_mask"].to(device),
#                 labels=batch["labels"].to(device),
#             )
#             loss = outputs.loss
#             val_loss += loss.item()

#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(batch["labels"].cpu().numpy())

#     val_acc = accuracy_score(all_labels, all_preds)
#     print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")


# def clean_text(text):
#     if not isinstance(text, str):  # 确保输入是字符串
#         return ""  # 对于非字符串值，返回空字符串
#     # text = re.sub(r"http\S+", "http", text)  # 移除 URLs
#     text = re.sub(r"[^a-zA-Z0-9#!?\s]", "", text)  # 移除非字母字符
#     # text = text.lower()  # 轉換為小寫
#     return text


# # =============================================================

# # import train data
# train_data = pd.read_csv("train.csv")


# # miss count
# # print(
# #     f"train data 'keyword' miss num: {train_data["keyword"].isnull().sum()}, Percentage: {train_data["keyword"].isnull().sum()/len(train_data["keyword"])*100}"
# # )
# # print(
# #     f"train data 'location' miss num: {train_data["location"].isnull().sum()}, Percentage: {train_data["location"].isnull().sum()/len(train_data["location"])*100}"
# # )
# # keyword_target_count = (
# #     train_data.groupby(["keyword", "target"]).size().unstack(fill_value=0)
# # )
# # # 根據 target=0 的數量排序（由少到多）
# # keyword_target_count = keyword_target_count.sort_values(by=0, ascending=True)
# # # 畫長條圖
# # keyword_target_count.plot(
# #     kind="bar", stacked=False, figsize=(12, 6), color=["blue", "orange"]
# # )

# # # 設定標題和座標標籤
# # plt.title("Relationship between Keyword and Target", fontsize=16)
# # plt.xlabel("Keyword", fontsize=14)
# # plt.ylabel("Count", fontsize=14)
# # plt.xticks(rotation=90)  # 關鍵字長的話旋轉標籤
# # plt.legend(["Target = 0", "Target = 1"], fontsize=12)
# # plt.tight_layout()

# # # 顯示圖表
# # plt.show()


# # train_data["text"] = train_data["text"].apply(clean_text)
# # train_data["location"] = train_data["location"].apply(clean_text).fillna("")
# train_data["text"] = train_data["text"]
# train_data["location"] = train_data["location"].fillna("")
# # 计算每个 location 的出现次数
# location_counts = train_data["location"].value_counts()
# # 保留出现频率小于等于 2 的 location，其他置为空字符串
# locations = (
#     train_data["location"]
#     .apply(lambda x: x if x != "" and location_counts[x] <= 2 else "")
#     .tolist()
# )
# texts = train_data["text"].tolist()
# labels = train_data["target"].tolist()
# keywords = train_data["keyword"].fillna("").tolist()
# # import test data
# test_data = pd.read_csv("test.csv")

# # miss count
# # print(
# #     f"test data 'keyword' miss num: {test_data["keyword"].isnull().sum()}, Percentage: {test_data["keyword"].isnull().sum()/len(test_data["keyword"])*100}"
# # )
# # print(
# #     f"test data 'location' miss num: {test_data["location"].isnull().sum()}, Percentage: {test_data["location"].isnull().sum()/len(test_data["location"])*100}"
# # )
# # 假設你的資料是 data
# # 使用 groupby 計算每個 keyword 在 target = 0 和 target = 1 的出現次數

# # test_data["text"] = test_data["text"].apply(clean_text)
# # test_data["location"] = test_data["location"].apply(clean_text).fillna("")
# test_data["text"] = test_data["text"]
# test_data["location"] = test_data["location"].fillna("")
# test_data["keyword"] = test_data["keyword"].fillna("").tolist()
# # 计算每个 location 的出现次数
# location_counts = test_data["location"].value_counts()
# # 保留出现频率小于等于 2 的 location，其他置为空字符串
# test_locations = (
#     test_data["location"]
#     .apply(lambda x: x if x != "" and location_counts[x] <= 2 else "")
#     .tolist()
# )
# test_texts = test_data["text"].tolist()
# test_keywords = test_data["keyword"]
# # split data (80%: train, 20%: validation)
# train_texts, val_texts, train_keywords, val_keywords, train_labels, val_labels = (
#     train_test_split(texts, keywords, labels, test_size=0.2, random_state=42)
# )

# # prerare dataset
# train_dataset = DisasterDataset(train_texts, train_keywords, train_labels, tokenizer)
# val_dataset = DisasterDataset(val_texts, val_keywords, val_labels, tokenizer)
# test_dataset = DisasterDataset(
#     test_texts, test_keywords, [0] * len(test_texts), tokenizer
# )

# # craete data loader
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8)
# test_loader = DataLoader(test_dataset, batch_size=8)

# train_model(model, train_loader, val_loader, num_epochs=2)

# # predict
# model.eval()
# predictions = []
# with torch.no_grad():
#     for batch in test_loader:
#         outputs = model(
#             input_ids=batch["input_ids"].to(model.device),
#             attention_mask=batch["attention_mask"].to(model.device),
#         )
#         logits = outputs.logits
#         preds = torch.argmax(logits, dim=1)
#         predictions.extend(preds.cpu().numpy())

# # create submission file
# submission = pd.DataFrame({"id": test_data["id"], "target": predictions})
# submission.to_csv("submission.csv", index=False)


# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import (
#     BertForSequenceClassification,
#     BertTokenizer,
#     AdamW,
#     get_scheduler,
# )
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import re

# # 加載模型和設備
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


# # 定義數據集類
# class DisasterDataset(Dataset):
#     def __init__(self, texts, keywords, labels, tokenizer, max_length=64):
#         self.texts = texts
#         self.keywords = keywords
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         combined_text = f"{self.texts[idx]} [KEYWORD] {self.keywords[idx]}"
#         encoding = self.tokenizer(
#             combined_text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         )
#         item = {key: val.squeeze(0).to(device) for key, val in encoding.items()}
#         item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long).to(device)
#         return item


# # 清理文本
# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = re.sub(r"http\S+", "http", text)  # 移除 URLs
#     text = re.sub(r"[^a-zA-Z0-9#\s]", "", text)  # 移除非字母數字字符
#     return text


# # 加載數據
# train_data = pd.read_csv("train.csv")
# test_data = pd.read_csv("test.csv")

# # 清理數據
# train_data["text"] = train_data["text"].apply(clean_text)
# train_data["keyword"] = train_data["keyword"].fillna("")
# test_data["text"] = test_data["text"].apply(clean_text)
# test_data["keyword"] = test_data["keyword"].fillna("")

# # 準備數據集和加載器
# train_texts = train_data["text"].tolist()
# train_keywords = train_data["keyword"].tolist()
# train_labels = train_data["target"].tolist()
# test_texts = test_data["text"].tolist()
# test_keywords = test_data["keyword"].tolist()

# train_dataset = DisasterDataset(train_texts, train_keywords, train_labels, tokenizer)
# test_dataset = DisasterDataset(
#     test_texts, test_keywords, [0] * len(test_texts), tokenizer
# )

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=8)


# # 訓練模型
# def train_model(model, train_loader, num_epochs=2):
#     optimizer = AdamW(model.parameters(), lr=2e-5)
#     scheduler = get_scheduler(
#         "linear",
#         optimizer=optimizer,
#         num_warmup_steps=0,
#         num_training_steps=len(train_loader) * num_epochs,
#     )

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0
#         all_preds = []
#         all_labels = []

#         for batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(
#                 input_ids=batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 labels=batch["labels"],
#             )
#             loss = outputs.loss
#             train_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(batch["labels"].cpu().numpy())

#         train_acc = accuracy_score(all_labels, all_preds)
#         print(
#             f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}"
#         )


# train_model(model, train_loader)

# # 預測測試集
# model.eval()
# predictions = []
# with torch.no_grad():
#     for batch in test_loader:
#         outputs = model(
#             input_ids=batch["input_ids"],
#             attention_mask=batch["attention_mask"],
#         )
#         logits = outputs.logits
#         preds = torch.argmax(logits, dim=1)
#         predictions.extend(preds.cpu().numpy())

# # 確保測試集與預測結果長度一致
# if len(test_data) != len(predictions):
#     raise ValueError(
#         f"Mismatch in test_data and predictions length: {len(test_data)} vs {len(predictions)}"
#     )

# # 創建提交檔案
# submission = pd.DataFrame({"id": test_data["id"], "target": predictions})
# submission.to_csv("submission.csv", index=False)
# print("Submission file created: submission.csv")


# 找最佳超參數，Best hyperparameters: Best hyperparameters: {'learning_rate': 8.369990782077691e-06, 'batch_size': 8, 'max_length': 64}
# import optuna
# from transformers import (
#     BertTokenizer,
#     BertForSequenceClassification,
#     AdamW,
#     get_scheduler,
# )
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import torch
# import pandas as pd
# import re

# # =============================================================

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # =============================================================


# class DisasterDataset(Dataset):
#     def __init__(self, texts, locations, labels, tokenizer, max_length=64):
#         self.texts = texts
#         # self.locations = locations
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         # combined_text = f"{self.texts[idx]} [LOCATION] {self.locations[idx]}"
#         combined_text = self.texts[idx]
#         encoding = self.tokenizer(
#             combined_text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         )
#         item = {key: val.squeeze(0) for key, val in encoding.items()}
#         item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
#         return item


# def train_and_validate_model(
#     model, train_loader, val_loader, optimizer, scheduler, num_epochs
# ):
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0
#         all_preds, all_labels = [], []

#         for batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(
#                 input_ids=batch["input_ids"].to(model.device),
#                 attention_mask=batch["attention_mask"].to(model.device),
#                 labels=batch["labels"].to(model.device),
#             )
#             loss = outputs.loss
#             train_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#             scheduler.step()

#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(batch["labels"].cpu().numpy())

#         train_acc = accuracy_score(all_labels, all_preds)
#         print(
#             f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}"
#         )

#         validate_model(model, val_loader)


# def validate_model(model, val_loader):
#     model.eval()
#     val_loss = 0
#     all_preds, all_labels = [], []

#     with torch.no_grad():
#         for batch in val_loader:
#             outputs = model(
#                 input_ids=batch["input_ids"].to(model.device),
#                 attention_mask=batch["attention_mask"].to(model.device),
#                 labels=batch["labels"].to(model.device),
#             )
#             loss = outputs.loss
#             val_loss += loss.item()

#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(batch["labels"].cpu().numpy())

#     val_acc = accuracy_score(all_labels, all_preds)
#     print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
#     return val_acc


# def objective(trial):
#     # Hyperparameters to tune
#     learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 3e-5)
#     batch_size = trial.suggest_categorical("batch_size", [8])
#     max_length = trial.suggest_categorical(
#         "max_length",
#         [
#             64,
#         ],
#     )

#     train_dataset = DisasterDataset(
#         train_texts, train_locations, train_labels, tokenizer, max_length=max_length
#     )
#     val_dataset = DisasterDataset(
#         val_texts, val_locations, val_labels, tokenizer, max_length=max_length
#     )

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)

#     # Load model
#     model = BertForSequenceClassification.from_pretrained(
#         "bert-base-uncased", num_labels=2
#     )
#     model.to("cuda")

#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     scheduler = get_scheduler(
#         "linear",
#         optimizer=optimizer,
#         num_warmup_steps=0,
#         num_training_steps=len(train_loader) * 2,  # Assuming 2 epochs
#     )

#     # Train and validate
#     train_and_validate_model(
#         model, train_loader, val_loader, optimizer, scheduler, num_epochs=2
#     )
#     val_acc = validate_model(model, val_loader)
#     return val_acc


# # =============================================================
# # 数据处理
# def clean_text(text):
#     if not isinstance(text, str):  # 确保输入是字符串
#         return ""  # 对于非字符串值，返回空字符串
#     # text = re.sub(r"http\S+", "", text)  # 移除 URLs
#     text = re.sub(r"[^a-zA-Z0-9#\s]", "", text)  # 移除非字母字符
#     # text = text.lower()  # 转换为小写
#     return text


# # 加载数据
# train_data = pd.read_csv("train.csv")
# train_data["text"] = train_data["text"].apply(clean_text)
# train_data["location"] = train_data["location"].apply(clean_text).fillna("")
# location_counts = train_data["location"].value_counts()
# locations = (
#     train_data["location"]
#     .apply(lambda x: x if x != "" and location_counts[x] <= 2 else "")
#     .tolist()
# )
# texts = train_data["text"].tolist()
# labels = train_data["target"].tolist()

# train_texts, val_texts, train_locations, val_locations, train_labels, val_labels = (
#     train_test_split(texts, locations, labels, test_size=0.2, random_state=42)
# )

# # =============================================================
# # 超参数搜索
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=30)  # 执行 10 次搜索

# # 打印最佳超参数
# print("Best hyperparameters:", study.best_params)


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
    def __init__(self, texts, keywords, labels, tokenizer, max_length=64):
        self.texts = texts
        self.keywords = keywords
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        combined_text = f"{self.texts[idx]} [KEYWORD] {self.keywords[idx]}"
        encoding = self.tokenizer(
            combined_text,
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
train_data = pd.read_csv("train.csv")
texts = train_data["text"].tolist()
keywords = train_data["keyword"].tolist()
labels = train_data["target"].tolist()

# import test data
test_data = pd.read_csv("test.csv")
test_texts = test_data["text"].tolist()
test_keywords = test_data["keyword"].tolist()

# split data (80%: train, 20%: validation)
train_texts, val_texts, train_keywords, val_keywords, train_labels, val_labels = (
    train_test_split(texts, keywords, labels, test_size=0.2, random_state=42)
)

# prerare dataset
train_dataset = DisasterDataset(train_texts, train_keywords, train_labels, tokenizer)
val_dataset = DisasterDataset(val_texts, val_keywords, val_labels, tokenizer)
test_dataset = DisasterDataset(
    test_texts, test_keywords, [0] * len(test_texts), tokenizer
)

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
