
# app.py
# VN Watchlist + Email Alerts (1p) – vnstock==0.1.1 + Fallback VNDirect snapshot + Daily

import os, json, time, smtplib, ssl, csv, uuid
from email.message import EmailMessage
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import requests
import certifi

# (Tuỳ chọn) Auto refresh mỗi 60 giây.
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTORF = True
except Exception:
    HAS_AUTORF = False

# --- Data provider: vnstock 0.1.1 ---
try:
    from vnstock import stock_intraday_data, stock_historical_data
    VNSTOCK_OK = True
except Exception:
    VNSTOCK_OK = False

# --- Storage files ---
WATCHLIST_FILE = "watchlist.json"
ALERTS_FILE    = "alerts.json"
EMAIL_LOG_FILE = "email_log.csv"

# --- SMTP / ENV ---
load_dotenv()
SMTP_SERVER = os.getenv("SMTP_SERVER", "")
SMTP_PORT   = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER   = os.getenv("SMTP_USER", "")
SMTP_PASS   = os.getenv("SMTP_PASS", "")
ALERT_TO    = os.getenv("ALERT_TO", "")

# --- Common schema for alerts table ---
ALERT_COLUMNS = ["id", "symbol", "op", "target", "email", "enabled", "last_sent_at"]

# --- Helpers for storage ---
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --- Normalize alerts ---
def normalize_alerts(alerts_list):
    """Đảm bảo mỗi alert có đủ trường và kiểu hợp lệ."""
    normalized = []
    for a in alerts_list or []:
        sym = (a.get("symbol") or "").upper()
        op  = a.get("op") if a.get("op") in (">=", "<=") else ">="
        try:
            tgt = float(a.get("target") or 0)
        except Exception:
            tgt = 0.0
        norm = {
            "id": a.get("id") or f"{sym}-{op}-{tgt}-{int(time.time())}",
            "symbol": sym,
            "op": op,
            "target": tgt,
            "email": a.get("email") or os.getenv("ALERT_TO", ""),
            "enabled": bool(a.get("enabled", True)),
            "last_sent_at": a.get("last_sent_at") or ""
        }
        normalized.append(norm)
    return normalized

def update_watchlist_prices(watchlist):
    """Update watchlist with today's prices."""
    today = datetime.now().strftime("%Y-%m-%d")
    updated = False
    
    # Create a new list to store the updated watchlist
    updated_watchlist = []
    
    for item in watchlist:
        # Convert string items to dict format
        if not isinstance(item, dict):
            item = {"symbol": str(item).strip().upper()}
        
        # Ensure symbol exists and is not empty
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            continue
            
        # Create a copy of the item to avoid modifying the original
        item = item.copy()
        
        # Initialize price history if not exists
        if "price_history" not in item or not isinstance(item["price_history"], dict):
            item["price_history"] = {}
        else:
            # Convert price_history to a new dict to ensure it's mutable
            item["price_history"] = dict(item["price_history"])
            
        # Skip if price already updated today
        if today in item["price_history"]:
            updated_watchlist.append(item)
            continue
            
        try:
            # Get current price
            price_data = get_latest_price(symbol)
            if price_data and "price" in price_data and price_data["price"] is not None:
                item["price_history"][today] = price_data["price"]
                updated = True
        except Exception as e:
            print(f"Error getting price for {symbol}: {str(e)}")
            
        updated_watchlist.append(item)
    
    return updated_watchlist, updated

# --- Fallback providers ---
def fetch_price_vndirect_snapshot(symbol: str) -> dict:
    """Fallback HTTP snapshot từ VNDirect (không chính thức)."""
    symbol = symbol.strip().upper()
    try:
        url = "https://price-api.vndirect.com.vn/stocks/snapshot"
        r = requests.get(url, params={"symbols": symbol}, timeout=6)
        r.raise_for_status()
        js = r.json()
        items = js.get("data") or js.get("stockData") or js.get("stocks") or []
        if not items:
            return {"symbol": symbol, "price": None, "time": None, "err": "snapshot empty"}
        it = items[0]
        val = None
        for k in ("matchPrice", "lastPrice", "price", "close"):
            if k in it and it[k] is not None:
                try:
                    val = float(it[k])
                    break
                except (ValueError, TypeError):
                    pass
        if val is None:
            return {"symbol": symbol, "price": None, "time": None, "err": "no price in snapshot"}
        return {"symbol": symbol, "price": val, "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "note": "VNDirect snapshot"}
    except Exception as e:
        return {"symbol": symbol, "price": None, "time": None, "err": f"snapshot err: {e}"}

def fetch_price_daily_close(symbol: str) -> dict:
    """Fallback daily close gần nhất bằng vnstock (ổn định)."""
    symbol = symbol.strip().upper()
    try:
        if not VNSTOCK_OK:
            return {"symbol": symbol, "price": None, "time": None, "err": "vnstock not available"}
        df = stock_historical_data(symbol, (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"),
                                   datetime.now().strftime("%Y-%m-%d"))
        if df is None or df.empty:
            return {"symbol": symbol, "price": None, "time": None, "err": "daily empty"}
        last = df.iloc[-1]
        price = None
        for c in ("close", "Close", "priceClose", "matchPrice", "closePrice", "Price"):
            if c in df.columns and pd.notnull(last.get(c)):
                try:
                    price = float(last.get(c))
                    break
                except Exception:
                    pass
        tsv = pd.to_datetime(last.get("tradingDate") or last.get("date") or datetime.now(), errors="coerce")
        if price is None:
            return {"symbol": symbol, "price": None, "time": None, "err": "no price daily"}
        if pd.isnull(tsv):
            tsv = datetime.now()
        return {"symbol": symbol, "price": price, "time": tsv.strftime("%Y-%m-%d %H:%M"),
                "note": "Daily close"}
    except Exception as e:
        return {"symbol": symbol, "price": None, "time": None, "err": f"daily err: {e}"}

# --- get_latest_price with multi-source fallback ---
@st.cache_data(ttl=50)
def get_latest_price(symbol: str) -> dict:
    """Lấy giá mới nhất với fallback qua nhiều nguồn."""
    symbol = symbol.strip().upper()
    
    # Thử vnstock trước
    if VNSTOCK_OK:
        try:
            df = stock_intraday_data(symbol, page_size=1)
            if df is not None and not df.empty and 'price' in df.columns:
                latest = df.iloc[0].to_dict()
                price = latest['price']
                # Chia cho 1000 để chuyển đổi từ 98500.0 thành 98.5
                if price and price > 1000:  # Chỉ chia nếu giá lớn hơn 1000
                    price = price / 1000
                return {
                    "symbol": symbol,
                    "price": price,
                    "time": latest.get('time', datetime.now().strftime("%Y-%m-%d %H:%M")),
                    "note": "vnstock intraday"
                }
        except Exception:
            pass
            
    # Thử VNDirect snapshot
    try:
        vnd = fetch_price_vndirect_snapshot(symbol)
        if vnd and vnd.get("price") is not None:
            price = vnd["price"]
            if price and price > 1000:  # Chỉ chia nếu giá lớn hơn 1000
                vnd["price"] = price / 1000
            return vnd
    except Exception:
        pass
        
    # Cuối cùng thử daily close
    try:
        daily = fetch_price_daily_close(symbol)
        if daily and daily.get("price") is not None:
            price = daily["price"]
            if price and price > 1000:  # Chỉ chia nếu giá lớn hơn 1000
                daily["price"] = price / 1000
            return daily
    except Exception:
        pass
        
    return {"symbol": symbol, "price": None, "time": None, "err": "all providers failed"}

# --- email ---
def log_email(subject: str, to_email: str, status: str, details: str = ""):
    """Log email sending activity to a CSV file"""
    log_file = Path(EMAIL_LOG_FILE)
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'to': to_email,
        'subject': subject,
        'status': status,
        'details': details
    }
    
    # Create file with header if it doesn't exist
    if not log_file.exists():
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writeheader()
    
    # Append log entry
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        writer.writerow(log_entry)

def send_email(subject: str, body: str, to_email: str):
    """Send email with STARTTLS (587) or SSL (465) using certifi CA bundle.
    Set SMTP_DEBUG=1 in .env to enable debug output.
    """
    if not (SMTP_SERVER and SMTP_PORT and SMTP_USER and SMTP_PASS and (to_email or ALERT_TO)):
        error_msg = "Thiếu cấu hình SMTP hoặc email nhận (ALERT_TO) trong .env"
        log_email(subject, to_email, "failed", error_msg)
        raise RuntimeError(error_msg)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = to_email or ALERT_TO
    msg.set_content(body)

    ctx = ssl.create_default_context(cafile=certifi.where())

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            if os.getenv("SMTP_DEBUG"):
                server.set_debuglevel(1)
            server.starttls(context=ctx)
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
            log_email(subject, to_email, "sent")
            return True
    except Exception as e:
        error_msg = str(e)
        log_email(subject, to_email, "failed", error_msg)
        print(f"Error sending email: {error_msg}")
        raise

def get_stock_price(symbol: str) -> float:
    """Get current price for a stock symbol using vnstock"""
    try:
        from vnstock import company_overview
        data = company_overview(symbol)
        if not data.empty and 'lastPrice' in data.columns:
            return float(data['lastPrice'].iloc[0])
        return 0.0
    except Exception as e:
        st.error(f"Lỗi khi lấy giá {symbol}: {str(e)}")
        return 0.0

# --- alert evaluation ---
def evaluate_alerts(price_map: dict, alerts: list):
    now = datetime.now()
    triggered, pending = [], []
    for a in alerts:
        if not a.get("enabled", True):
            pending.append(a); continue
        sym = a["symbol"].upper()
        px  = price_map.get(sym)
        if px is None:
            pending.append(a); continue
        # Get the operator and handle backward compatibility
        op = a.get("op", ">=").strip()  # Default to '>=' if not specified
        
        # Debug logging
        debug_info = f"Checking alert: {a.get('symbol')} {op} {a.get('target')}, Current price: {px}"
        print(debug_info)  # For debugging
        
        # Evaluate the condition
        if op in (">=", "="):
            ok = px >= a["target"]
        elif op == "<=":
            ok = px <= a["target"]
        else:
            print(f"Unknown operator: {op}, defaulting to >=")
            ok = px >= a["target"]
        last_sent_at = a.get("last_sent_at")
        recently_sent = False
        if last_sent_at:
            try:
                dt = datetime.strptime(last_sent_at, "%Y-%m-%d %H:%M")
                recently_sent = (now - dt) < timedelta(minutes=60)
            except Exception:
                pass
        if ok and not recently_sent:
            triggered.append(a)
        else:
            pending.append(a)
    return triggered, pending

# --- Load data ---
watchlist = load_json(WATCHLIST_FILE, [])
watchlist, prices_updated = update_watchlist_prices(watchlist)
if prices_updated:
    save_json(WATCHLIST_FILE, watchlist)

alerts_raw = load_json(ALERTS_FILE, [])
alerts = normalize_alerts(alerts_raw)
save_json(ALERTS_FILE, alerts)  # migrate back

# --- UI ---
st.set_page_config(page_title="VN Watchlist & Alerts", page_icon="📈", layout="wide")
st.title("📈 VN Watchlist & Email Alerts (1 phút cập nhật)")
st.caption("Nguồn: vnstock 0.1.1 + VNDirect snapshot + Daily fallback. Có thể thay sang WebSocket VNDirect/TPBS khi cần.")

if HAS_AUTORF:
    st_autorefresh(interval=60_000, limit=None, key="auto_refresh_60s")
else:
    st.info("Gợi ý: cài `streamlit-autorefresh` để tự refresh mỗi 60s (tuỳ chọn).")

# Sidebar: Watchlist
st.sidebar.header("📊 Danh sách mã")
with st.sidebar.form("add_symbol_form", clear_on_submit=True):
    new_symbol = st.text_input("Thêm mã (ví dụ: FPT, VCB, VNM):").strip().upper()
    if st.form_submit_button("➕ Thêm"):
        if new_symbol and not any((item.get('symbol') if isinstance(item, dict) else item) == new_symbol for item in watchlist):
            watchlist.append({
                "symbol": new_symbol.upper(),
                "price_history": {},
                "enabled": True,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            save_json(WATCHLIST_FILE, watchlist)
            st.toast(f"✅ Đã thêm {new_symbol}", icon="✅")
            st.rerun()
        elif new_symbol:
            st.toast(f"ℹ️ {new_symbol} đã có trong Watchlist", icon="ℹ️")

# Sidebar: Alerts
with st.sidebar:
    st.markdown("### 🔔 Thêm Alert Mới")
    with st.form("add_alert_form", clear_on_submit=True):
        # Stock symbol selection
        alert_options = [item['symbol'] if isinstance(item, dict) else item for item in watchlist]
        
        # Get current prices for all stocks
        price_map = {}
        for item in watchlist:
            symbol = item['symbol'] if isinstance(item, dict) else item
            price = get_stock_price(symbol)
            if price is not None:
                price_map[symbol] = price
        
        # Symbol selection with better styling
        sym_for_alert = st.selectbox(
            "Mã cổ phiếu",
            ["-- thêm mã trước --"] + [item['symbol'] if isinstance(item, dict) else item for item in watchlist],
            key="alert_symbol"
        )
        
        # Add some vertical space
        st.write("")
        
        # Condition selector with radio buttons
        op = st.radio(
            "Điều kiện",
            ["≥ (Lớn hơn hoặc bằng)", "≤ (Nhỏ hơn hoặc bằng)"],
            index=0,
            key="alert_condition",
            horizontal=True
        )
        op = ">=" if "Lớn" in op else "<="
        
        # Get current price for the selected symbol
        default_price = 0.0
        if sym_for_alert and sym_for_alert != "-- thêm mã trước --":
            default_price = price_map.get(sym_for_alert, 0.0)
        
        # Target price input with step of 1
        target = st.number_input(
            "Giá mục tiêu",
            min_value=0.0,
            step=1.0,
            value=default_price,
            format="%.0f",
            key="alert_target"
        )
        
        # Add some vertical space
        st.write("")
        
        # Email input with better styling and full width
        email_container = st.container()
        with email_container:
            email_to = st.text_input(
                "Email nhận thông báo",
                value="",
                placeholder="Nhập email, phân cách bằng dấu phẩy",
                help="Để trống để sử dụng email mặc định từ file .env",
                label_visibility="visible"
            )
        
        # Add some vertical space before submit button
        st.write("")
        
        # Submit button inside the form
        submitted = st.form_submit_button("➕ Tạo Alert")
        
        if submitted:
            error_messages = []
            
            # Validate watchlist
            if not watchlist:
                error_messages.append("Vui lòng thêm mã cổ phiếu vào danh sách trước")
            
            # Validate symbol
            if not sym_for_alert or sym_for_alert == "-- thêm mã trước --":
                error_messages.append("Vui lòng chọn mã cổ phiếu")
            elif sym_for_alert not in [item['symbol'] if isinstance(item, dict) else item for item in watchlist]:
                error_messages.append(f"Mã {sym_for_alert} không tồn tại trong danh sách")
            
            # Validate target price
            if target <= 0:
                error_messages.append("Giá mục tiêu phải lớn hơn 0")
            
            # Process and validate emails
            emails = []
            if email_to.strip():
                # Split by comma or semicolon, strip whitespace, and filter out empty strings
                emails = [e.strip() for e in email_to.replace(';', ',').split(',') if e.strip()]
                for email in emails:
                    if "@" not in email:
                        error_messages.append(f"Email không hợp lệ: {email}")
            
            # If no email provided, use ALERT_TO
            if not emails and not ALERT_TO:
                error_messages.append("Vui lòng nhập ít nhất một email hoặc cấu hình ALERT_TO trong .env")
            
            # If no errors, create the alert
            if not error_messages:
                try:
                    symbol_upper = sym_for_alert.upper()
                    alert = {
                        "id": f"{symbol_upper}-{op}-{target}-{int(time.time())}",
                        "symbol": symbol_upper,
                        "op": op,
                        "target": float(target),
                        "email": email_to.strip() or ALERT_TO,
                        "enabled": True,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    alerts.append(alert)
                    save_json(ALERTS_FILE, alerts)
                    st.toast(f"✅ Đã tạo alert: {alert['symbol']} {op} {target:,.0f}", icon="✅")
                    st.rerun()  # Refresh to show the new alert
                except Exception as e:
                    st.error(f"Lỗi khi tạo alert: {str(e)}")
            else:
                # Show all validation errors
                for error in error_messages:
                    st.error(error)
                st.toast("❌ Không thể tạo alert. Vui lòng kiểm tra thông tin.", icon="❌")

# Initialize session state for refresh tracking
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
    st.session_state.auto_refresh = True

# Refresh control
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("🔄 Làm mới dữ liệu"):
        st.session_state.last_refresh = time.time()
        st.rerun()

# Calculate time until next refresh
current_time = time.time()
time_since_refresh = int(current_time - st.session_state.last_refresh)
time_until_refresh = max(0, 60 - (time_since_refresh % 60))

with col2:
    st.caption(f"⏳ Tự động làm mới sau: {time_until_refresh}s")
    st.progress(time_until_refresh / 60, text="")

# Auto-refresh logic
if st.session_state.auto_refresh and time_since_refresh >= 60:
    st.session_state.last_refresh = current_time
    st.rerun()

# Main table
if not watchlist:
    st.info("Thêm vài mã ở thanh bên trái để bắt đầu.")
else:
    data_rows = []
    price_map = {}
    
    # Prepare data for display
    for item in watchlist:
        if isinstance(item, str):
            symbol = item
            price_history = {}
        else:
            symbol = item.get("symbol", "")
            price_history = item.get("price_history", {})
            
        info = get_latest_price(symbol)
        current_price = info.get("price")
        price_map[symbol] = current_price
        
        # Get price change if we have history
        price_change = None
        if price_history:
            # Get the most recent previous price (excluding today)
            today = datetime.now().strftime("%Y-%m-%d")
            prev_days = [d for d in sorted(price_history.keys()) if d != today]
            if prev_days:
                last_day = prev_days[-1]
                prev_price = price_history[last_day]
                if prev_price and current_price:
                    price_change = ((current_price - prev_price) / prev_price) * 100
        
        note = info.get("err") or info.get("note") or ""
        
        # Format display - show price directly from JSON
        if current_price is not None:
            price_display = f"{current_price:.1f}"  # Show price with 1 decimal place
        else:
            price_display = "N/A"
        
        # Add price change if available
        if price_change is not None:
            change_icon = "🔼" if price_change >= 0 else "🔽"
            price_display += f" ({change_icon} {abs(price_change):.2f}%)"
        
        # Add price history as a tooltip
        history_text = "Lịch sử giá:\n"
        for date, price in sorted(price_history.items(), reverse=True):
            history_text += f"{date}: {price:.1f}\n"
        
        # Store the original price for display
        display_price = current_price
        
        # Get existing note if it exists
        existing_note = ''
        if isinstance(item, dict):
            existing_note = item.get('note', '')
        
        data_rows.append({
            "Mã": symbol,
            "Giá": current_price,  # Store original price for calculations
            "Giá hiển thị": f"{display_price:.1f}" if display_price is not None else 'N/A',  # Formatted display price
            "Thay đổi": price_change,
            "Thời gian": info.get("time") or datetime.now().strftime("%H:%M:%S"),
            "Nguồn": note,
            "Ghi chú": existing_note,  # Add notes field
            "_history": history_text.strip()
        })
    
    # Display the table with tooltips
    if data_rows:
        # Debug: Print data types
        for i, row in enumerate(data_rows):
            print(f"Row {i} data types:")
            for k, v in row.items():
                print(f"  {k}: {type(v)} - {v}")
        
        # Create a clean DataFrame with proper data types
        df_rows = []
        for row in data_rows:
            try:
                # Safely format price change
                price_change = row.get('Thay đổi')
                change_display = ''
                if pd.notnull(price_change):
                    change_icon = '🔼' if float(price_change) >= 0 else '🔽'
                    change_display = f"{change_icon} {abs(float(price_change)):.2f}%"
                
                # Create row with safe string conversion
                df_rows.append({
                    'Mã CK': str(row.get('Mã', '')),
                    'Giá hiện tại': str(row.get('Giá hiển thị', 'N/A')),
                    'Thay đổi': change_display,
                    'Cập nhật': str(row.get('Thời gian', '')),
                    'Nguồn': str(row.get('Nguồn', '')),
                    'Ghi chú': str(row.get('Ghi chú', '')),
                    '_history': str(row.get('_history', ''))
                })
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
                
        df = pd.DataFrame(df_rows)
        
        # Add checkboxes for deletion
        df['Chọn để xóa'] = False
        
        # Display the table with checkboxes and notes
        edited_df = st.data_editor(
            df[['Mã CK', 'Giá hiện tại', 'Cập nhật', 'Nguồn', 'Ghi chú', 'Chọn để xóa']],
            column_config={
                'Mã CK': 'Mã CK',
                'Giá hiện tại': 'Giá hiện tại',
                'Cập nhật': 'Cập nhật',
                'Nguồn': st.column_config.TextColumn('Nguồn', help=df['_history'].values.tolist()),
                'Ghi chú': st.column_config.TextColumn(
                    'Ghi chú',
                    help='Nhập ghi chú cho mã cổ phiếu',
                    default='',
                    max_chars=100
                ),
                'Chọn để xóa': st.column_config.CheckboxColumn(
                    'Chọn để xóa',
                    help='Chọn để xóa khỏi danh sách theo dõi',
                    default=False
                )
            },
            hide_index=True,
            use_container_width=True,
            key='watchlist_editor',
            num_rows='dynamic'
        )
        
        # Add delete button
        if st.button('Xóa các mã đã chọn khỏi danh sách', type='primary'):
            if not edited_df.empty and 'Chọn để xóa' in edited_df.columns:
                rows_to_delete = edited_df[edited_df['Chọn để xóa'] == True]
                if not rows_to_delete.empty:
                    for idx, row in rows_to_delete.iterrows():
                        symbol = row['Mã CK']
                        watchlist = [item for item in watchlist if (item if isinstance(item, str) else item.get('symbol', '')) != symbol]
                        st.toast(f'✅ Đã xóa mã {symbol} khỏi danh sách theo dõi')
                    save_json(WATCHLIST_FILE, watchlist)
                    st.rerun()
        
        # Save changes if any notes were updated
        if not edited_df.empty and 'Ghi chú' in edited_df.columns:
            notes_updated = False
            for idx, row in edited_df.iterrows():
                symbol = row['Mã CK']
                new_note = row.get('Ghi chú', '')
                # Find the item in watchlist and update its note
                for i, item in enumerate(watchlist):
                    item_symbol = item if isinstance(item, str) else item.get('symbol', '')
                    if item_symbol == symbol:
                        if isinstance(watchlist[i], dict):
                            if watchlist[i].get('note') != new_note:
                                watchlist[i]['note'] = new_note
                                notes_updated = True
                        else:
                            watchlist[i] = {'symbol': symbol, 'note': new_note}
                            notes_updated = True
            if notes_updated:
                save_json(WATCHLIST_FILE, watchlist)
                st.toast("✅ Đã cập nhật ghi chú")
                st.rerun()
    
    # Store price_map for alert evaluation
    price_map = {row["Mã"]: row["Giá"] for row in data_rows if row["Giá"] is not None}

# Alerts table & actions
st.subheader("Danh sách Alerts")
if alerts:
    # Create a copy of alerts for display with simplified symbol
    display_alerts = []
    for a in alerts:
        da = a.copy()
        da["Mã CK"] = a["symbol"]
        # Map operators to display symbols
        op_display = {
            ">=": ">=",
            "<=": "≤",
            "=": "="
        }
        da["enabled"] = a.get("enabled", True)  # Add enabled status
        da["Điều kiện"] = op_display.get(a.get("op", ">="), ">=")
        da["Giá mục tiêu"] = f"{a['target']:,.0f}"
        da["Email"] = a.get("email", "")  # Show full email address
        last_sent = a.get("last_sent_at", "")
        da["Lần gửi cuối"] = last_sent.split(" ")[0] if last_sent else "Chưa gửi"
        da["delete"] = False  # Checkbox for deletion
        display_alerts.append(da)
    
    # Create DataFrame for display
    df = pd.DataFrame(display_alerts)
    
    # Ensure all required columns exist
    required_columns = ['enabled', 'Mã CK', 'Điều kiện', 'Giá mục tiêu', 'Email', 'Lần gửi cuối', 'delete']
    for col in required_columns:
        if col not in df.columns:
            df[col] = False  # Default for boolean columns
    
    # Display the table with checkboxes
    edited_df = st.data_editor(
        df[['enabled', 'Mã CK', 'Điều kiện', 'Giá mục tiêu', 'Email', 'Lần gửi cuối', 'delete']],
        column_config={
            "enabled": st.column_config.CheckboxColumn("Bật/Tắt", default=True),
            "Mã CK": "Mã CK",
            "Điều kiện": "Điều kiện",
            "Giá mục tiêu": "Giá mục tiêu",
            "Email": "Email",
            "Lần gửi cuối": "Lần gửi cuối",
            "delete": st.column_config.CheckboxColumn("Chọn để xóa", default=False)
        },
        hide_index=True,
        use_container_width=True,
        key="alerts_editor"
    )
    
    # Add delete button
    if st.button("Xóa các alert đã chọn", type="primary"):
        if not edited_df.empty and 'delete' in edited_df.columns:
            # Get indices of rows to delete
            indices_to_delete = edited_df[edited_df['delete']].index.tolist()
            
            # Remove alerts that are marked for deletion
            updated_alerts = [a for i, a in enumerate(alerts) if i not in indices_to_delete]
            
            # Save the updated alerts
            if len(updated_alerts) < len(alerts):
                save_json(ALERTS_FILE, updated_alerts)
                st.success(f"Đã xóa {len(alerts) - len(updated_alerts)} alert")
                st.rerun()
            else:
                st.warning("Vui lòng chọn ít nhất một alert để xóa")
    
    # Handle toggle changes
    toggle_changed = False
    for idx, row in edited_df.iterrows():
        if idx < len(alerts):  # Make sure we don't go out of bounds
            alert = alerts[idx]
            new_state = row["enabled"]
            if alert and alert.get("enabled") != new_state:
                alert["enabled"] = new_state
                toggle_changed = True
    
    # Save changes and show toast if any toggle was changed
    if toggle_changed:
        save_json(ALERTS_FILE, alerts)
        st.toast("✅ Đã cập nhật trạng thái alert")
        st.rerun()  # Single rerun after all changes are processed
    
else:
    st.info("Chưa có alert nào. Tạo alert ở thanh bên trái.")

# Evaluate & send alerts
if watchlist and alerts:
    try:
        price_map
    except NameError:
        price_map = {sym: get_latest_price(sym).get("price") for sym in watchlist}

    triggered, pending = evaluate_alerts(price_map, alerts)
    if triggered:
        st.warning(f"🔔 Có {len(triggered)} alert khớp điều kiện. Đang gửi email…")
        
        # Create a copy of the alerts to modify
        updated_alerts = alerts.copy()
        
        # Prepare combined email content
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # For single alert, include symbol in subject
        if len(triggered) == 1:
            symbol = triggered[0]['symbol'].upper()
            condition = '≥' if triggered[0]['op'] == '>=' else '≤'
            target_price = f"{triggered[0]['target']:,.0f}"
            subject = f"[ALERTS] {symbol} {condition} {target_price}"
        else:
            # For multiple alerts, show count and common condition if same
            condition = '≥' if len(set(a['op'] for a in triggered)) == 1 and all(a['op'] == '>=' for a in triggered) else '≤' if len(set(a['op'] for a in triggered)) == 1 and all(a['op'] == '<=' for a in triggered) else ''
            target_price = f"{triggered[0]['target']:,.0f}" if len(set(a['target'] for a in triggered)) == 1 else ''
            subject = f"[ALERTS] {len(triggered)} mã đạt điều kiện {condition} {target_price}"
        body = f"Danh sách các mã đạt điều kiện vào lúc {current_time}:\n\n"
        
        # Update the alerts with last_sent_at and keep them in the system
        for a in triggered:
            sym = a["symbol"].upper()
            px = price_map.get(sym)
            body += f"• Mã: {sym}\n"
            body += f"  Điều kiện: Giá {a['op']} {a['target']:,.0f}\n"
            body += f"  Giá hiện tại: {px:,.0f}\n\n"
            
            # Find and update the alert in the original alerts list
            for idx, alert in enumerate(updated_alerts):
                if alert.get('id') == a.get('id'):
                    updated_alerts[idx]["last_sent_at"] = current_time
                    break
        
        try:
            # Get email recipients from alert or fallback to ALERT_TO
            email_recipients = []
            if triggered:
                # Get emails from the first triggered alert (they should be the same for all)
                email_str = triggered[0].get('email', '')
                email_recipients = [e.strip() for e in email_str.split(',') if e.strip()]
            
            # If no emails found in alert, use ALERT_TO
            if not email_recipients and ALERT_TO:
                email_recipients = [ALERT_TO]
            
            # Send to all recipients
            for email in email_recipients:
                send_email(subject, body, email)
                
            # Save all alerts (both triggered and pending) with updated timestamps
            save_json(ALERTS_FILE, updated_alerts)
            st.toast(f"Đã gửi email tổng hợp {len(triggered)} alert", icon="✉️")
            # Force a rerun to update the UI with the latest alerts
            st.rerun()
        except Exception as e:
            st.error(f"Gửi email thất bại: {e}")
    else:
        st.success("✅ Chưa có alert nào khớp lúc này.")

def delete_email_logs(entries_to_keep):
    """Save only the specified entries to the log file"""
    log_file = Path(EMAIL_LOG_FILE)
    fieldnames = ['timestamp', 'to', 'subject', 'status', 'details']
    
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if entries_to_keep is not None:  # If None, it means delete all
            writer.writerows(entries_to_keep)

# --- Display Email History ---
st.markdown("## 📧 Lịch sử gửi email")
try:
    if not Path(EMAIL_LOG_FILE).exists():
        st.info("Chưa có lịch sử gửi email nào.")
    else:
        # Read the log file
        with open(EMAIL_LOG_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            log_entries = list(reader)
        
        if not log_entries:
            st.info("Chưa có lịch sử gửi email nào.")
        else:
            # Add search and filter options in a clean layout
            col1, col2 = st.columns([2, 3])
            with col1:
                search_term = st.text_input("🔍 Tìm kiếm (tiêu đề hoặc email)", 
                                         help="Nhập từ khóa để lọc lịch sử")
            with col2:
                status_filter = st.radio(
                    "Trạng thái:",
                    ["Tất cả", "✅ Thành công", "❌ Thất bại"],
                    horizontal=True,
                    index=0
                )
            
            st.markdown("---")
            filtered_count = 0
            
            # Filter and display emails
            for entry in reversed(log_entries):  # Show newest first
                # Apply filters
                subject = entry.get('subject', '').lower()
                to_email = entry.get('to', '').lower()
                status = entry.get('status', '').lower()
                
                if search_term and search_term.lower() not in subject + to_email:
                    continue
                    
                if status_filter == "✅ Thành công" and status != 'sent':
                    continue
                if status_filter == "❌ Thất bại" and status != 'failed':
                    continue
                
                # Format entry
                timestamp = entry.get('timestamp', '')
                try:
                    dt = pd.to_datetime(timestamp)
                    formatted_time = dt.strftime('%d/%m/%Y %H:%M')
                except:
                    formatted_time = timestamp
                
                # Status display
                status_display = "✅ Gửi thành công" if status == 'sent' else "❌ Gửi thất bại"
                status_color = "#4CAF50" if status == 'sent' else "#f44336"
                
                # Create a unique key for each email's delete button
                delete_key = f"delete_{entry.get('timestamp', '')}_{entry.get('subject', '')}"
                
                # Display as a card with delete button
                col1, col2 = st.columns([6, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style='
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 12px 16px;
                        margin: 8px 0;
                        background-color: white;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    '>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                            <h4 style='margin: 0; color: #1a237e;'>{entry.get('subject', 'Không có tiêu đề')}</h4>
                            <span style='color: {status_color}; font-weight: 500; font-size: 0.9rem;'>{status_display}</span>
                        </div>
                        <div style='color: #555; font-size: 0.9rem; margin-bottom: 4px;'>
                            <span>👤 {entry.get('to', '')}</span>
                            <span style='margin: 0 8px; color: #ddd;'>•</span>
                            <span>🕒 {formatted_time}</span>
                        </div>
                        {f"<div style='color: #666; font-size: 0.85rem; padding: 6px 0; border-top: 1px dashed #eee; margin-top: 8px;'>{entry.get('details', '')}</div>" if entry.get('details') else ''}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("🗑️", key=f"del_{delete_key}", help="Xóa email này"):
                        # Create a new list without the deleted email
                        updated_entries = [e for e in log_entries 
                                        if not (e.get('timestamp') == entry.get('timestamp') 
                                              and e.get('subject') == entry.get('subject'))]
                        
                        # Save the updated entries back to the file
                        with open(EMAIL_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=['timestamp', 'to', 'subject', 'status', 'details'])
                            writer.writeheader()
                            writer.writerows(updated_entries)
                        
                        st.success(f"Đã xóa email: {entry.get('subject', '')}")
                        time.sleep(0.5)  # Small delay to show the success message
                        st.rerun()  # Refresh the page to show the updated list
                    
                    filtered_count += 1
            
            # Show empty state if no results
            if filtered_count == 0:
                st.warning("Không tìm thấy email nào phù hợp với bộ lọc.")
            
            # Management controls
            st.markdown("---")
            
            # Action buttons in columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Download button for the full log
                csv_data = pd.DataFrame(log_entries).to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="💾 Tải xuống toàn bộ lịch sử (CSV)",
                    data=csv_data,
                    file_name=f'email_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            
            with col2:
                # Delete all button with confirmation
                if st.button("🗑️ Xóa tất cả lịch sử", use_container_width=True, type="secondary"):
                    if st.session_state.get('confirm_delete_all'):
                        with open(EMAIL_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=['timestamp', 'to', 'subject', 'status', 'details'])
                            writer.writeheader()
                        st.success("Đã xóa toàn bộ lịch sử email!")
                        time.sleep(0.5)
                        st.session_state.pop('confirm_delete_all', None)
                        st.rerun()
                    else:
                        st.session_state['confirm_delete_all'] = True
                        st.warning("Nhấn lại để xác nhận xóa toàn bộ lịch sử")
            
            # Show confirmation message if needed
            if st.session_state.get('confirm_delete_all'):
                st.warning("⚠️ Cảnh báo: Hành động này sẽ xóa vĩnh viễn toàn bộ lịch sử email. Nhấn nút 'Xóa tất cả lịch sử' lần nữa để xác nhận.")
            
            # Add a note about the date format
            st.caption(f"Hiển thị {filtered_count}/{len(log_entries)} email • Định dạng thời gian: NĂM-THÁNG-NGÀY GIỜ:PHÚT")
except Exception as e:
    st.error(f"Không thể tải lịch sử email: {str(e)}")
    if 'log_entries' in locals():
        st.json(log_entries[:2])  # Show sample for debugging
