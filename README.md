
# 📈 VN Stock Watchlist & Alerts (v3.3)

**Bổ sung mới:**
- 🔄 **Nút Refresh thủ công** (bên cạnh auto refresh 60s – có nếu cài `streamlit-autorefresh`).
- 💹 **Định dạng giá có dấu phẩy** (`23,450`).
- 🟢/🔴 **Xu hướng** so với **giá đóng cửa phiên trước** (lấy từ `vnstock` daily).
- ✅ UI cập nhật ngay khi **Thêm/Xoá mã**, **Bật/Tắt/Xoá alert** (`st.rerun()`).

## Chạy nhanh
```bash
python -m venv .venv
# Windows: . .venv\Scriptsctivate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# .env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
ALERT_TO=your_email@gmail.com
# SMTP_DEBUG=1   # tuỳ chọn

streamlit run app.py
```

## Docker
Xem `Dockerfile` và `docker-compose.yml` để chạy container.
