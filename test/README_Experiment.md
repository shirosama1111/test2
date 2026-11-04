# Hướng dẫn thống kê dataset và chạy thực nghiệm

Tài liệu áp dụng cho `test/experiment_pipeline.py` và `test/run_experiments.py` trên branch `dataset-evaluation-pipeline`.

## 1. Dataset

Dataset mặc định là `test/data/bug_dataset.jsonl`. Schema và giới hạn được mô tả trong `test/data/DATASET_CARD.md`. Theo card, tập hiện có 1.000 mẫu synthetic, nhãn chưa được con người xác minh, 3 lớp bug (`UI`, `API`, `LOGIC`), 4 lớp severity và split train/val/test.

**Lưu ý:** kết quả đo mức khớp với annotation hiện có, không chứng minh nhãn là ground truth đúng. Dùng LLM như một người đánh giá độc lập để phát hiện trường hợp cần kiểm tra thủ công; không tự động thay nhãn chỉ vì LLM bất đồng.

## 2. Cài đặt

Tại thư mục gốc repo:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scikit-learn
```

Để dùng LLM qua API tương thích OpenAI:

```bash
pip install openai
# Linux/macOS
export OPENAI_API_KEY="your-key"
# Windows PowerShell
$env:OPENAI_API_KEY="your-key"
```

Đặt tên model đánh giá (nên chọn model khác với model chính cần đối chiếu):

```bash
export SECOND_OPINION_MODEL="gpt-4o-mini"
```

Hoặc truyền `--model <model-name>`. Nếu endpoint tương thích OpenAI không phải mặc định, đặt `OPENAI_BASE_URL`. Dữ liệu gửi tới endpoint bên ngoài chỉ nên gồm nội dung report được pipeline chọn; hãy kiểm tra chính sách dữ liệu và quyền chia sẻ trước khi gọi API.

Excel là tùy chọn:

```bash
pip install openpyxl
```

## 3. Kịch bản thực nghiệm

### Kịch bản 0 — thống kê / kiểm tra dữ liệu

```bash
python test/run_experiments.py --scenario stats
```

Báo cáo số mẫu, phân bố nhãn/severity/split, annotation status, độ dài văn bản, trường thiếu, duplicate ID/text và nhãn sai schema. Kết quả được lưu trong `test/runs/<timestamp>/dataset_stats.json`.

### Kịch bản A — baseline luật

```bash
python test/run_experiments.py --scenario rule
```

Chạy rule-based classifier trên test split. Đây là baseline minh bạch, dựa trên luật/keyword.

### Kịch bản B — baseline học máy

```bash
python test/run_experiments.py --scenario ml --repeats 3 --seed 42
```

Huấn luyện TF-IDF + Logistic Regression trên train split và đánh giá trên test split. Pipeline không dùng test để huấn luyện.

### Kịch bản C — LLM đánh giá độc lập (second opinion)

```bash
python test/run_experiments.py --scenario second-opinion --model gpt-4o-mini --llm-limit 20
```

Mỗi lần gọi, LLM chỉ nhận report text để dự đoán `bug_type` và `severity`; không đưa nhãn gold, root cause, resolution hoặc ví dụ train có nhãn vào prompt. Mặc định gửi toàn bộ test set; `--llm-limit N` giúp thử trước với N mẫu để kiểm soát chi phí.

### So sánh A/B/C trên cùng test set

```bash
python test/run_experiments.py --scenario all --repeats 3 --seed 42
```

`all` lần lượt chạy rule, ML và LLM second-opinion trên test split đã khai báo. Cần `OPENAI_API_KEY` cho model C. Có thể chạy không LLM bằng cách chọn riêng `rule` hoặc `ml`.

Lưu ý: `--repeats` chạy lại cùng test set, không bootstrap/resample test để tránh thay đổi đối tượng so sánh. Với model rule/ML deterministic, các lần chạy có thể giống nhau; các lần gọi LLM có thể khác và phát sinh chi phí.

## 4. Metric

Metric được tính riêng cho loại bug và severity:

- Accuracy
- Balanced accuracy
- Macro precision, macro recall, macro F1
- Weighted F1
- Confusion matrix và precision/recall/F1 từng lớp
- Severity ordinal MAE: khoảng cách trung bình theo thứ tự LOW < MEDIUM < HIGH < CRITICAL

Khi `--repeats` lớn hơn 1, pipeline ghi mean và standard deviation của các metric được chọn. Dùng macro-F1 và balanced accuracy để quan sát hiệu năng theo lớp, nhất là khi lớp không cân bằng; đọc confusion matrix để hiểu loại nhầm lẫn.

## 5. Kết quả, CSV và Excel

Chạy CSV và JSON:

```bash
python test/run_experiments.py --scenario all --repeats 3 --seed 42
```

Xuất thêm Excel:

```bash
python test/run_experiments.py --scenario all --repeats 3 --seed 42 --excel
```

Mỗi lần chạy tạo `test/runs/<timestamp>/` (có thể đổi bằng `--output-dir`) với:

- `summary.json`: cấu hình, hash SHA-256 dataset, thống kê, metric từng run và tổng hợp.
- `metrics.csv`: metric theo từng hệ thống/run.
- `model_comparison.csv`: bảng tổng hợp để so sánh A/B/C.
- `experiment_report.xlsx`: tùy chọn, gồm các sheet so sánh model, kết quả từng run và thống kê dataset.

Ví dụ chỉ định output:

```bash
python test/run_experiments.py \\
  --dataset test/data/bug_dataset.jsonl \\
  --scenario all --repeats 3 --seed 42 \\
  --output-dir test/runs/ablation-01 --excel
```

Có thể gọi pipeline không qua wrapper:

```bash
python test/experiment_pipeline.py --scenario all --repeats 3 --seed 42 --excel
```

## 6. Cách rà soát bất đồng nhãn

1. Chạy `stats` để kiểm tra phân bố, duplicate, trường thiếu và nhãn không hợp lệ.
2. Chạy A/B/C với cùng test split; giới hạn LLM trong lần thử đầu.
3. Mở `model_comparison.csv` để xem metric tổng hợp; dùng `metrics.csv` và confusion matrix trong JSON để phân tích lớp.
4. Với second opinion, lấy các trường hợp bất đồng để người gán nhãn rà soát độc lập, ghi lại quyết định và lý do.
5. Nếu sửa nhãn, quản lý phiên bản dataset và chạy lại pipeline trên cùng cấu hình để so sánh.
6. Với dataset template/synthetic, cân nhắc tạo test split theo kịch bản gốc/template để kiểm tra khả năng tổng quát hóa; split ngẫu nhiên có thể làm điểm số lạc quan do các mẫu gần giống nhau xuất hiện ở cả train và test.

Không chọn model dựa trên test set rồi xem chính test đó là đánh giá cuối không thiên lệch. Dùng validation để chọn cấu hình và giữ test cho báo cáo cuối. LLM second opinion là tín hiệu hỗ trợ kiểm tra chéo, không phải người gán nhãn chân lý.
