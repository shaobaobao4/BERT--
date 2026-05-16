"""
bert.py - 使用BERT对20newsgroups二分类 (alt.atheism vs soc.religion.christian)
冻结BERT主体层，仅训练分类头，大幅提升CPU训练速度
"""
import os, glob, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import BertTokenizer, BertModel

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

MAX_LEN = 128; BATCH_SIZE = 16; EPOCHS = 5; LR = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 自定义BERT分类器（冻结BERT主体，只训练分类头）
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', local_files_only=True)
        for p in self.bert.parameters(): p.requires_grad = False  # 冻结
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)
    def forward(self, input_ids, attention_mask):
        pooled = self.bert(input_ids, attention_mask).pooler_output
        return self.fc(self.dropout(pooled))

# 加载本地数据
def load_data():
    categories = ['alt.atheism', 'soc.religion.christian']
    data_home = os.path.join(os.path.expanduser('~'), 'scikit_learn_data', '20news_home')
    X, y = [], []
    for label, cat in enumerate(categories):
        for subset in ['20news-bydate-train', '20news-bydate-test']:
            for fp in glob.glob(os.path.join(data_home, subset, cat, '*')):
                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                        X.append(f.read())
                    y.append(label)
                except: pass
    print(f"加载 {len(X)} 条数据")
    return X, y

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts; self.labels = labels; self.tokenizer = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length',
                             max_length=MAX_LEN, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(),
                'attention_mask': enc['attention_mask'].flatten(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

def evaluate(model, loader):
    model.eval(); preds, true = [], []
    with torch.no_grad():
        for b in loader:
            out = model(b['input_ids'].to(device), b['attention_mask'].to(device))
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            true.extend(b['label'].numpy())
    return (accuracy_score(true, preds), precision_score(true, preds, zero_division=0),
            recall_score(true, preds, zero_division=0), f1_score(true, preds, zero_division=0),
            confusion_matrix(true, preds))

def main():
    X, y = load_data()
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.2, random_state=42, stratify=y_tv)
    print(f"训练: {len(y_train)}, 验证: {len(y_val)}, 测试: {len(y_test)}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    train_loader = DataLoader(NewsDataset(X_train, y_train, tokenizer), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(NewsDataset(X_val, y_val, tokenizer), BATCH_SIZE)
    test_loader = DataLoader(NewsDataset(X_test, y_test, tokenizer), BATCH_SIZE)

    model = BertClassifier().to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)  # 只优化分类头
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    print("开始训练（冻结BERT主体，仅训练分类头）...")
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train(); total_loss = 0
        for b in train_loader:
            ids, mask, labels = b['input_ids'].to(device), b['attention_mask'].to(device), b['label'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask), labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        val_acc, val_p, val_r, val_f1, _ = evaluate(model, val_loader)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_bert.pth')
        print(f"E{epoch+1}/{EPOCHS} | Loss:{total_loss/len(train_loader):.4f} | "
              f"Val Acc:{val_acc:.4f} P:{val_p:.4f} R:{val_r:.4f} F1:{val_f1:.4f} | {time.time()-t0:.1f}s")

    print("\n=== 测试结果 ===")
    model.load_state_dict(torch.load('best_bert.pth', weights_only=True))
    test_acc, test_p, test_r, test_f1, test_cm = evaluate(model, test_loader)
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_p:.4f}")
    print(f"Recall:    {test_r:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"Confusion Matrix:\n{test_cm}")

if __name__ == '__main__':
    main()
