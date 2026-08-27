# Lab 25 — GPU FinOps: Write-up

> Sinh viên: Vũ Tiến Dung · MSV 2A202602009 · AICB Phase 2 · Track 2 (Infrastructure)
> Dữ liệu: seed cố định 25, tái tạo được bằng `python data/generate.py`. Mọi số liệu dưới đây lấy trực tiếp từ output của missions.

## 1. Baseline vs. Optimized

| Chỉ số | Baseline | Optimized | Thay đổi |
|---|---|---|---|
| Tổng chi phí GPU /tháng | $27,133 | $14,626 | **−$12,507 (−46.1%)** |
| Chi phí inference /ngày | $48.87 | $8.48 | −82.6% |
| **$/1M-token** | **$6.488** | **$1.126** | **−82.6%** |

Điểm đáng chú ý nhất: nếu chỉ nhìn `$/GPU-giờ`, đòn bẩy lớn nhất là **Purchasing** ($10,040 = 80% tổng số tiền tiết kiệm); nhưng nếu nhìn **`$/1M-token`** — đơn vị đo "trả tiền nhận được gì" — đòn bẩy inference (cascade + cache + batch) mới mạnh nhất (−82.6%). Hai thước đo này cho kết quả trái ngược về "cái gì quan trọng nhất", đúng như bài học cốt lõi của lab: `$/GPU-giờ` cho biết bạn **trả bao nhiêu**, `$/1M-token` cho biết bạn **nhận được gì**.

## 2. Phân tích từng đòn bẩy

| Lever | Tiết kiệm/tháng | Cơ chế |
|---|---|---|
| **Purchasing (spot/reserved)** | **$10,040 (80%)** | 5 job interruptible → spot + checkpoint (mô phỏng: +3% overhead ghi checkpoint + rework do gián đoạn, nhưng vẫn rẻ hơn ~40%); 3 job duty-cycle ≥ 55% (điểm hòa vốn của chiết khấu reserved 45%) → reserved |
| **Inference (cascade/cache/batch)** | $1,212 | Cascade: request đơn giản → model nhỏ (rẻ 15×); prompt cache: input đã cache tính 10% giá; batch: −50%. Chiết khấu **nhân** nhau: batch + 100% cache = 0.05 giá gốc |
| **Right-size util-lies** | $655 | Hạ 2 GPU "nói dối" xuống 1 tier |
| **Kill idle** | $600 | `gpu-h100-5` bỏ không 8h/ngày → $20/ngày |

## 3. GPU-Util Lie — GPU nào "nói dối" và thiệt hại bao nhiêu

Hai GPU bị flag: **`gpu-h100-4`** (util 98.2%, MFU 0.194) và **`gpu-a10g-1`** (util 96.9%, MFU 0.268).

**Cơ chế:** `nvidia-smi` chỉ báo cáo *có kernel nào đang resident* trong khoảng sample — nó đo hoạt động clock, không đo thông lượng FLOP. Decode LLM là **memory-bound** (arithmetic intensity ~1–2 FLOP/byte, trong khi ridge point H100 ~295 FLOP/byte): các SM **stall chờ HBM**, GPU "bận" 98% nhưng chỉ tính ~20% công suất đỉnh.

**Tác động tài chính:** trả giá H100 đỉnh nhưng nhận 1/5 FLOPs hữu ích → **giá thực trên mỗi FLOP hữu ích = 1/MFU ≈ 5.2×** giá niêm yết với `gpu-h100-4` (3.7× với `gpu-a10g-1`). Đây là lý do phải giám sát bằng **MFU/MBU**, không phải GPU-Util.

## 4. Hai extension đã làm (có đo lường, có unit test)

### Ext 4 — Ngân sách Reasoning
Tách chi phí `$` và năng lượng `Wh` theo cờ `is_reasoning` (hàm `reasoning_budget()` trong `missions/m2_inference_levers.py`, 3 cap-scenarios, greedy reroute):

| Cap reasoning | Reroute | Tiết kiệm $ | Tiết kiệm Wh |
|---|---|---|---|
| 10% | 0 | $0.00 | 0 |
| 5% | 81 requests | $0.73 | 15,778 |
| 2% | 153 requests | $0.96 | 25,556 |

- Reasoning = **8.4% requests / 16.5% tokens → 16.5% chi phí nhưng 94.0% năng lượng** (hệ số ~80×/query): reasoning là **bom năng lượng** hơn là vấn đề chi phí ở traffic hiện tại.
- Cap 10% (mức gợi ý chung) **không ràng buộc** vì traffic hiện chỉ 8.4% — bài học: *đo trước khi đặt cap*.
- **Routing rule:** giới hạn reasoning theo ngân sách; vượt cap thì reroute **thinker lớn nhất trước** (cắt offender đắt nhất); chỉ giữ reasoning ở đâu nó tự trả tiền cho mình.

### Ext 5 — Carbon-aware Scheduling
File mới `missions/ext5_carbon_aware.py`: tính năng lượng 5 job interruptible (1,789 kWh/tháng) và so sánh 5 vùng:

| Vùng | $/kWh | gCO2/kWh | Tiền điện | Carbon (gCO2e) |
|---|---|---|---|---|
| us-east-1 (hiện tại) | $0.120 | 380 | $214.68 | 679,820 |
| **europe-north1 (sạch nhất)** | $0.090 | 30 | $161.01 | **53,670** |
| us-east-wa (rẻ nhất) | $0.055 | 90 | $98.39 | 161,010 |
| europe-central2 | $0.180 | 660 | $322.02 | 1,180,740 |

- Chuyển 5 job sang `europe-north1`: **giảm 626,150 gCO2e/tháng (−92.1%)** và tiền điện còn **rẻ hơn $54/tháng** — cắt carbon gần như miễn phí.
- `us-east-wa` là lựa chọn **dominated-win** (rẻ nhất + sạch nhì); `europe-central2` tệ cả hai chiều → loại.
- **Trade-off latency:** vùng sạch nhất (Na Uy) chỉ phù hợp job training/batch interruptible; inference nhạy latency phải ở gần users (us-east-wa / us-west-2).

## 5. Ba hành động đầu tiên nếu là FinOps lead của NimbusAI

1. **Đổi purchasing tier ngay trong tháng này** — 80% tổng tiết kiệm ($10,040/tháng), chỉ là thay contract, không cần sửa code. Đàm phán spot + checkpoint cho 5 job interruptible; reserved cho 3 job duty ≥ 55%.
2. **Right-size 2 GPU "lie" + tắt GPU idle trong tuần** — $1,255/tháng, effort gần như bằng không. Đồng thời thay dashboard GPU-Util bằng MFU/MBU để không bị lỏa lần nữa.
3. **Lộ trình tối ưu inference theo thứ tự cascade → cache → batch**, kèm ngân sách reasoning (cap + reroute largest-first). Tag coverage đã 92% → chuyển showback → chargeback quý tới để các team tự chịu trách nhiệm chi phí.

*Kiểm chứng: `python verify.py` → 11/11 checks · `pytest -q` → 29 passed (15 gốc + 14 test extension viết theo TDD).*
