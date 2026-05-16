import re
import string
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader


# 锁死随机种子
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

# 超参数
MAX_VOCAB_SIZE = 10000
MAX_SEQ_LENGTH = 300
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 文本清洗
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    stop_words = {'the', 'and', 'for', 'that', 'you', 'from', 'with', 'this', 'was', 'are'}
    words = [w for w in text.split() if w not in stop_words and len(w) > 3]
    return ' '.join(words)


# 数据加载与处理
def load_and_preprocess_data():
    categories = ['alt.atheism', 'soc.religion.christian']
    print("从本地文件加载数据...")
    
    data_home = os.path.join(os.path.expanduser('~'), 'scikit_learn_data', '20news_home')
    X_all = []
    y_all = []
    for label, cat in enumerate(categories):
        for subset in ['20news-bydate-train', '20news-bydate-test']:
            for fp in glob.glob(os.path.join(data_home, subset, cat, '*')):
                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                        X_all.append(f.read())
                    y_all.append(label)
                except: pass
    
    X_all = [preprocess_text(doc) for doc in X_all]
    y_all = np.array(y_all)

    # 数据集切分 (stratify确保类别均衡)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)
    
    print(f"加载 {len(y_all)} 条数据，训练:{len(y_train)} 验证:{len(y_val)} 测试:{len(y_test)}")

    # 构建词汇表
    word_freq = Counter()
    for text in X_train:
        word_freq.update(text.split())
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_freq.most_common(MAX_VOCAB_SIZE - 2):
        word_to_idx[word] = len(word_to_idx)

    # 文本转序列
    def text_to_sequence(texts):
        seqs = []
        for text in texts:
            idx_seq = [word_to_idx.get(w, 1) for w in text.split()]
            idx_seq = idx_seq[:MAX_SEQ_LENGTH] if len(idx_seq) >= MAX_SEQ_LENGTH else idx_seq + [0] * (
                        MAX_SEQ_LENGTH - len(idx_seq))
            seqs.append(idx_seq)
        return torch.tensor(seqs, dtype=torch.long)

    # 构建DataLoader
    train_data = TensorDataset(text_to_sequence(X_train), torch.tensor(y_train, dtype=torch.long))
    val_data = TensorDataset(text_to_sequence(X_val), torch.tensor(y_val, dtype=torch.long))
    test_data = TensorDataset(text_to_sequence(X_test), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader, len(word_to_idx)


# GRU模型
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Xavier初始化
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, hidden = self.gru(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)


# 训练与评估
def train_and_evaluate():
    train_loader, val_loader, test_loader, vocab_size = load_and_preprocess_data()

    # 超参数
    EMBED_DIM = 256
    HIDDEN_DIM = 256
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 5e-4
    EPOCHS = 40

    model = GRUClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, 2, 0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss().to(device)

    def evaluate(loader):
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device)
                out = model(bx)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                true.extend(by.numpy())
        return (accuracy_score(true, preds), precision_score(true, preds, zero_division=0),
                recall_score(true, preds, zero_division=0), f1_score(true, preds, zero_division=0),
                confusion_matrix(true, preds))

    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        # 验证
        val_acc, val_p, val_r, val_f1, _ = evaluate(val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_gru_model.pth')



        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Val Acc: {val_acc:.4f}, P:{val_p:.4f} R:{val_r:.4f} F1:{val_f1:.4f}")
        
    # 如果40轮太慢，每10轮显示进度

    # 测试
    model.load_state_dict(torch.load('best_gru_model.pth', weights_only=True))
    test_acc, test_p, test_r, test_f1, test_cm = evaluate(test_loader)
    print(f"\n=== GRU测试结果 ===")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_p:.4f}")
    print(f"Recall:    {test_r:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"Confusion Matrix:\n{test_cm}")


if __name__ == '__main__':
    train_and_evaluate()