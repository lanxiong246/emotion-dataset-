# emotion-dataset-
处理中英文数据集，用RNN和DEEPSEEK模型对比参数
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from sklearn.preprocessing import label_binarize

# 1. 加载训练集和测试集数据
train_file_path = r"E:\ukm master course\project2\English dataset\tweet_emotions_train.csv"
test_file_path = r"E:\ukm master course\project2\English dataset\tweet_emotions_test.csv"

# 尝试不同编码加载数据
try:
    train_df = pd.read_csv(train_file_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        train_df = pd.read_csv(train_file_path, encoding='gbk')
    except UnicodeDecodeError:
        train_df = pd.read_csv(train_file_path, encoding='latin1')

try:
    test_df = pd.read_csv(test_file_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        test_df = pd.read_csv(test_file_path, encoding='gbk')
    except UnicodeDecodeError:
        test_df = pd.read_csv(test_file_path, encoding='latin1')

# 2. 处理缺失值
train_df.dropna(subset=['content', 'sentiment'], inplace=True)
test_df.dropna(subset=['content', 'sentiment'], inplace=True)

# 3. 标签编码
label_encoder = LabelEncoder()

# 对训练集进行标签编码
train_df['sentiment_encoded'] = label_encoder.fit_transform(train_df['sentiment'])

# 确保测试集中的标签也能够适配训练集中的标签
test_df['sentiment_encoded'] = label_encoder.transform(test_df['sentiment'])

# 4. 文本分词 + 序列化 + Padding
max_vocab = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_vocab, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df['content'])

X_train_seq = tokenizer.texts_to_sequences(train_df['content'])
X_test_seq = tokenizer.texts_to_sequences(test_df['content'])

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# 5. One-hot 编码标签
y_train_cat = to_categorical(train_df['sentiment_encoded'])
y_test_cat = to_categorical(test_df['sentiment_encoded'])
num_classes = y_train_cat.shape[1]
# 调参 inputsize outputsize input_length ,128,64
# 6. 构建LSTM模型
model = Sequential([
    Embedding(input_dim=max_vocab, output_dim=9, input_length=max_len),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. 模型训练
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train_pad, y_train_cat, validation_split=0.2, epochs=2, batch_size=64, callbacks=[early_stop])

# 8. 模型评估
loss, acc = model.evaluate(X_test_pad, y_test_cat)
print(f"✅ Test Accuracy: {acc:.2f}")

# 9. 分类报告 & 混淆矩阵
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)

# 修改：确保 `target_names` 是正确的字符串类型
target_names = ['fun', 'happiness', 'hate', 'neutral', 'relief', 'sadness', 'surprise', 'worry', 'love']

# 分类报告
print("\n📊 Classification Report:")
print(classification_report(test_df['sentiment_encoded'], y_pred_classes, target_names=target_names))

# 混淆矩阵
cm = confusion_matrix(test_df['sentiment_encoded'], y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# 10. ROC曲线
y_test_bin = label_binarize(test_df['sentiment_encoded'], classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])  # Assuming 9 classes

fpr, tpr, thresholds = roc_curve(y_test_bin.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 11. 输出准确率
print(f"Test Accuracy: {acc:.2f}")
