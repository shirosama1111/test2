# Bug Report Dataset (1000 mẫu, synthetic)

## Cấu trúc (khớp 10 thành phần của form)
| # | Thành phần | Trường trong JSONL |
|---|---|---|
| 1 | Bug Description | `bug_description` |
| 2 | Screenshot | `screenshot.file`, `screenshot.description` |
| 3 | Console Logs | `console_logs` (list) |
| 4 | Error Context | `error_context` (browser, os, viewport, url, occurred_at, user_role) |
| 5 | Steps to Reproduce | `steps_to_reproduce` (list) |
| 6 | Ground-truth Labels | `ground_truth_label` (UI / API / LOGIC) |
| 7 | Severity Labels | `severity` (LOW / MEDIUM / HIGH / CRITICAL) |
| 8 | Root Cause | `root_cause` |
| 9 | Resolution | `resolution` |
| 10 | Metadata | `metadata` (id, source, created_at, app_version, annotation_status, language, split, seed) |

## Thống kê
- 1000 mẫu: UI 338, API 337, LOGIC 325
- Severity: MEDIUM 491, HIGH 247, LOW 188, CRITICAL 74
- Split: train 800 / val 100 / test 100 (ngẫu nhiên, seed 2026)
- 39 kịch bản lỗi gốc (13 mỗi nhãn), 911 mô tả khác nhau

## Giới hạn quan trọng
- Dữ liệu **tổng hợp từ template**, chưa phải lỗi thật và chưa được con người xác minh (`annotation_status = synthetic-unverified`).
- Ảnh chụp màn hình là ảnh mô phỏng, không phải giao diện thật.
- Mỗi kịch bản lặp lại nhiều lần với giá trị khác nhau, nên nhãn dễ học hơn dữ liệu thật. Chia train/test theo kịch bản gốc nếu cần đánh giá khắt khe.
- Chỉ có tiếng Anh.
