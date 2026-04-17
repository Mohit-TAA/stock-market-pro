"""
freemium.py – Freemium logic and email capture for STOCK MARKET Pro.
Uses Web3Forms to securely store user name and email.
"""

import streamlit as st
import requests
import uuid
import re
from datetime import datetime
from pathlib import Path
import csv

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
FREE_TICKER_LIMIT = 3
KO_FI_URL = "https://ko-fi.com/s/d2b839b388"   # CHANGE THIS TO YOUR ACTUAL URL

# -----------------------------------------------------------------------------
# Session State Initialisation
# -----------------------------------------------------------------------------
def init_freemium_session_state():
    """Initialise all freemium-related session state variables."""
    defaults = {
        "analyzed_tickers": set(),          # distinct symbols analysed
        "email_submitted": False,           # unlocks further analyses
        "name": "",
        "email": "",
        "user_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.user_id is None:
        try:
            from streamlit_ws_localstorage import injectWebsocketCode
            HOST_PORT = "wsauthserver.supergroup.ai"
            conn = injectWebsocketCode(hostPort=HOST_PORT, uid=str(uuid.uuid1()))
            uid = conn.getLocalStorageVal(key="stock_app_user_id")
            if not uid:
                uid = str(uuid.uuid4())
                conn.setLocalStorageVal(key="stock_app_user_id", val=uid)
            st.session_state.user_id = uid
        except Exception:
            st.session_state.user_id = str(uuid.uuid4())

# -----------------------------------------------------------------------------
# Helper: Email Validation
# -----------------------------------------------------------------------------
def is_valid_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

# -----------------------------------------------------------------------------
# Web3Forms Storage (with detailed error reporting)
# -----------------------------------------------------------------------------
def store_user_data_web3forms(name: str, email: str, user_id: str, source: str = "mandatory_popup") -> tuple[bool, str]:
    """Send name and email to Web3Forms endpoint. Returns (success, message)."""
    try:
        access_key = st.secrets.get("WEB3FORMS_ACCESS_KEY")
        if not access_key:
            return False, "Missing Web3Forms access key in secrets."

        url = "https://api.web3forms.com/submit"
        data = {
            "access_key": access_key,
            "subject": f"New Signup: {name}",
            "from_name": "STOCK MARKET Pro",
            "name": name,
            "email": email,
            "message": f"User ID: {user_id}\nSource: {source}\nTimestamp: {datetime.now().isoformat()}",
            "botcheck": ""
        }
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return True, "Submission successful."
            else:
                return False, f"Web3Forms error: {result.get('message', 'Unknown error')}"
        else:
            return False, f"HTTP error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Network/exception: {str(e)}"

def store_user_data_csv_fallback(name: str, email: str, user_id: str) -> tuple[bool, str]:
    """Fallback: store user data in local CSV file (data/users.csv)."""
    try:
        Path("data").mkdir(exist_ok=True)
        csv_path = "data/users.csv"
        file_exists = Path(csv_path).exists()
        with open(csv_path, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["user_id", "name", "email", "timestamp"])
            writer.writerow([user_id, name, email, datetime.now().isoformat()])
        return True, "Saved locally (CSV fallback)."
    except Exception as e:
        return False, f"CSV fallback failed: {str(e)}"

def store_user_data(name: str, email: str, user_id: str, source: str = "mandatory_popup") -> tuple[bool, str]:
    """Primary: Web3Forms, fallback: CSV."""
    success, msg = store_user_data_web3forms(name, email, user_id, source)
    if success:
        return True, msg
    # Try fallback
    fb_success, fb_msg = store_user_data_csv_fallback(name, email, user_id)
    if fb_success:
        return True, f"{msg} → {fb_msg}"
    else:
        return False, f"{msg} → {fb_msg}"

# -----------------------------------------------------------------------------
# Freemium Limit Checks
# -----------------------------------------------------------------------------
def can_analyze(symbol: str) -> bool:
    symbol = symbol.upper()
    if st.session_state.email_submitted:
        return True
    if symbol in st.session_state.analyzed_tickers:
        return True
    return len(st.session_state.analyzed_tickers) < FREE_TICKER_LIMIT

def record_analysis(symbol: str):
    symbol = symbol.upper()
    st.session_state.analyzed_tickers.add(symbol)

# -----------------------------------------------------------------------------
# Popups
# -----------------------------------------------------------------------------
@st.dialog("📥 Unlock Unlimited Access", width="medium")
def mandatory_email_popup():
    st.markdown("""
    ### 🎁 You've reached the free limit!
    To analyse a 4th stock, please enter your name and email below.  
    We'll send you the full source code so you can run the app locally with **no limits**.
    """)

    with st.form("email_capture_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Email Address")
        submitted = st.form_submit_button("🔓 Unlock & Download Code", type="primary")

        if submitted:
            if not name or not email:
                st.error("Both name and email are required.")
            elif not is_valid_email(email):
                st.error("Please enter a valid email address.")
            else:
                with st.spinner("Submitting..."):
                    success, message = store_user_data(name, email, st.session_state.user_id, source="mandatory_4th")
                if success:
                    st.session_state.name = name
                    st.session_state.email = email
                    st.session_state.email_submitted = True
                    st.success(f"✅ Thank you! {message}")
                    st.markdown("---")
                    st.link_button("📥 DOWNLOAD THE CODES", KO_FI_URL)
                    st.caption("*Pay what you feel this is worth — $1.11 is a great start.*")
                    st.stop()
                else:
                    st.error(f"❌ Submission failed: {message}")

@st.dialog("📥 Download Full Source Code", width="medium")
def download_reminder_popup():
    st.markdown("## 🎁 Download the Complete Code")
    st.markdown("Run the app locally with **no restrictions**.")
    st.link_button("📥 DOWNLOAD THE CODES", KO_FI_URL, type="primary")
    st.caption("*Pay what you feel this is worth — $1.11 is a great start.*")
    if st.button("Close"):
        st.rerun()
