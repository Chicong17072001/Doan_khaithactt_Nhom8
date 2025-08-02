# ĐỀ TÀI: WORD2VEC VÀ DOC2VEC CHO PHÂN LOẠI VĂN BẢN TIẾNG VIỆT

## 📋 Tổng quan đề tài

Đề tài này nghiên cứu và áp dụng các mô hình **Word2Vec** và **Doc2Vec** để học biểu diễn văn bản tiếng Việt, sau đó sử dụng các thuật toán phân loại để thực hiện bài toán phân loại văn bản.

## 🎯 Mục tiêu

1. **Nghiên cứu lý thuyết**: Tìm hiểu mô hình Word2Vec, Doc2Vec và word embedding
2. **Cài đặt thực nghiệm**: Triển khai các mô hình biểu diễn văn bản
3. **Xây dựng dataset**: Thu thập ~500 văn bản tiếng Việt với 4-5 nhãn phân loại
4. **So sánh hiệu quả**: Đánh giá TF-IDF vs Word2Vec vs Doc2Vec
5. **Phân loại văn bản**: Áp dụng Rocchio, K-NN, Naive Bayes

## 🧠 Lý thuyết cơ bản

### Word Embedding là gì?

**Word Embedding** là kỹ thuật biểu diễn từ dưới dạng vector số thực trong không gian nhiều chiều, giúp máy tính hiểu được ý nghĩa và mối quan hệ giữa các từ.

### TF-IDF (Term Frequency - Inverse Document Frequency)

- **Nguyên lý**: Đo trọng số của từ dựa trên tần suất xuất hiện trong tài liệu và độ hiếm trong toàn bộ corpus
- **Công thức**: `TF-IDF(t,d) = TF(t,d) × IDF(t)`
- **Ưu điểm**: Đơn giản, hiệu quả với dữ liệu nhỏ
- **Nhược điểm**: Không nắm bắt được ngữ cảnh và ý nghĩa của từ

### Word2Vec

- **Nguyên lý**: Học biểu diễn vector của từ dựa trên ngữ cảnh xung quanh
- **Kiến trúc**:
  - **CBOW** (Continuous Bag of Words): Dự đoán từ trung tâm từ ngữ cảnh
  - **Skip-gram**: Dự đoán ngữ cảnh từ từ trung tâm
- **Ưu điểm**: Nắm bắt được ý nghĩa ngữ nghĩa, từ tương tự có vector gần nhau
- **Nhược điểm**: Không xử lý được từ ngoài vocab, cần dữ liệu lớn

### Doc2Vec (Paragraph Vector)

- **Nguyên lý**: Mở rộng Word2Vec để học biểu diễn toàn bộ tài liệu
- **Kiến trúc**:
  - **PV-DM** (Distributed Memory): Tương tự CBOW nhưng có thêm vector tài liệu
  - **PV-DBOW** (Distributed Bag of Words): Tương tự Skip-gram cho tài liệu
- **Ưu điểm**: Biểu diễn trực tiếp tài liệu, giữ được thông tin ngữ cảnh
- **Nhược điểm**: Cần nhiều dữ liệu, tính toán phức tạp

## 📊 Cấu trúc dữ liệu

```
Dataset: ~500 văn bản tiếng Việt
Nhãn: 4-5 loại chủ đề/thể loại
Chia dữ liệu: 80% train - 20% test (stratified)
```

## 🔬 Chi tiết từng bước thực nghiệm

### Cell 12: Tiền xử lý dữ liệu

```python
def preprocess(text):
    text = re.sub(f"[{string.punctuation}]", " ", text)
    return text.lower()
```

**Mục đích**:

- Loại bỏ dấu câu
- Chuyển về chữ thường
- Chuẩn hóa văn bản cho mô hình

### Cell 13-15: TF-IDF Vectorization

```python
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(sentences)
df["tf_idf"] = df["sentence"].apply(lambda x: tfidf_vectorizer.transform([x]).toarray()[0])
```

**Nguyên lý TF-IDF**:

1. **TF (Term Frequency)**: Đếm tần suất từ trong tài liệu
2. **IDF (Inverse Document Frequency)**: Tính độ hiếm của từ trong corpus
3. **Kết hợp**: Từ quan trọng = xuất hiện nhiều trong tài liệu nhưng ít trong corpus

### Cell 16-17: Word2Vec Implementation

```python
sentences = [word_tokenize(s) for s in sentences]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=50)
```

**Tham số quan trọng**:

- `vector_size=100`: Kích thước vector biểu diễn từ
- `window=5`: Kích thước cửa sổ ngữ cảnh (xem 5 từ trước/sau)
- `min_count=1`: Từ phải xuất hiện ít nhất 1 lần
- `epochs=50`: Số lần lặp qua toàn bộ dữ liệu

**Cách tạo vector câu**:

```python
def sentence_to_vec(sent):
    return np.mean([model.wv[w] for w in word_tokenize(sent)], axis=0)
```

Lấy **trung bình** vector của tất cả từ trong câu

### Cell 18-19: Doc2Vec Implementation

```python
sentences = [TaggedDocument(words=word_tokenize(row['sentence']), tags=[str(i)])
             for i, row in df.iterrows()]
model = Doc2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=50)
```

**Khác biệt với Word2Vec**:

- Mỗi tài liệu có một **tag** duy nhất
- Mô hình học **trực tiếp** vector cho tài liệu
- Không cần trung bình vector từ

### Cell 20-21: Chia dữ liệu Train/Test

```python
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=2025
)
```

**Stratified Split**: Đảm bảo tỷ lệ các nhãn giống nhau trong train và test

### Cell 22: Huấn luyện các mô hình phân loại

#### 1. Rocchio Classifier (NearestCentroid)

**Nguyên lý**:

- Tính **centroid** (trung tâm) của mỗi class
- Phân loại bằng cách tìm centroid gần nhất
- Phù hợp với dữ liệu có các class tách biệt rõ ràng

#### 2. K-Nearest Neighbors (KNN)

**Nguyên lý**:

- Tìm **k=10** điểm gần nhất trong không gian feature
- Phân loại theo **đa số** của k neighbors
- Không cần training, lazy learning

#### 3. Naive Bayes

**Nguyên lý**:

- Áp dụng **định lý Bayes** với giả thiết độc lập
- `P(class|features) ∝ P(features|class) × P(class)`
- Hiệu quả với dữ liệu nhỏ

### Cell 23-30: Đánh giá mô hình

```python
def evaluate(model, X_test, y_test, feature, algorithm):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
```

**Metrics đánh giá**:

- **Precision**: Độ chính xác dự đoán
- **Recall**: Khả năng phát hiện
- **F1-Score**: Trung bình điều hòa của Precision và Recall
- **Confusion Matrix**: Ma trận nhầm lẫn

## 📈 Kết quả và phân tích

### So sánh hiệu quả các phương pháp

| Phương pháp  | Ưu điểm                                   | Nhược điểm                            | Phù hợp                      |
| ------------ | ----------------------------------------- | ------------------------------------- | ---------------------------- |
| **TF-IDF**   | Đơn giản, nhanh, hiệu quả với dữ liệu nhỏ | Không hiểu ngữ nghĩa                  | Dataset nhỏ, từ vựng ổn định |
| **Word2Vec** | Hiểu ngữ nghĩa, từ tương tự gần nhau      | Mất thông tin thứ tự, cần dữ liệu lớn | Dataset lớn, cần semantic    |
| **Doc2Vec**  | Biểu diễn trực tiếp tài liệu              | Cần rất nhiều dữ liệu, phức tạp       | Dataset rất lớn              |

### Nguyên nhân TF-IDF hoạt động tốt hơn

1. **Kích thước dữ liệu**: ~500 văn bản quá ít cho Word2Vec/Doc2Vec
2. **Chất lượng training**: Word2Vec cần hàng triệu từ để học tốt
3. **Mất thông tin**: Word2Vec lấy trung bình → mất ngữ cảnh
4. **Overfitting**: Doc2Vec dễ overfit với dữ liệu nhỏ
