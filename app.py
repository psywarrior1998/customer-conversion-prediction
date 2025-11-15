import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import warnings
from huggingface_hub import hf_hub_download

# --- 0. Global Configuration & Warning Suppression ---
st.set_page_config(page_title="ShopSense AI | Behavioral Predictor", page_icon="🛍️", layout="wide")
warnings.filterwarnings("ignore") # Suppress sklearn version warnings for cleaner demo

# Constants
REPO_ID = "psyrishi/marketing-conversion-predictor"
FILENAME = "conversion_prediction_model.pkl"
OPTIMAL_THRESHOLD = 0.2136

# --- 1. Data & Model Logic ---
@st.cache_resource
def load_model():
    """Downloads and loads the model with error handling."""
    import sklearn.compose._column_transformer as ct
    
    # Backward compatibility patch
    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list): pass
        ct._RemainderColsList = _RemainderColsList

    try:
        with st.spinner(f"🧠 Awakening AI Brain..."):
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Model Error: {e}")
        return None

def apply_feature_engineering(df):
    """Replicates training transformations."""
    df = df.copy()
    df['EngagementScore'] = df['TimeOnSite'] * df['PagesPerVisit']
    df['WebsiteVisits_safe'] = df['WebsiteVisits'].replace(0, 1)
    df['CostPerVisit'] = df['AdSpend'] / df['WebsiteVisits_safe']
    df.drop('WebsiteVisits_safe', axis=1, inplace=True)
    
    df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 30, 50, 70], labels=['Young', 'Adult', 'Senior'], right=True, include_lowest=True)
    df['IncomeTier'] = pd.cut(df['Income'], bins=[19999, 50000, 90000, 150000], labels=['Low', 'Medium', 'High'], right=True, include_lowest=True)
    return df

def predict_conversion(model, data):
    """Generates prediction probability."""
    try:
        processed_data = apply_feature_engineering(data)
        feature_order = ['Age', 'Gender', 'Income', 'CampaignChannel', 'CampaignType', 'AdSpend',
                         'ClickThroughRate', 'ConversionRate', 'WebsiteVisits', 'PagesPerVisit',
                         'TimeOnSite', 'SocialShares', 'EmailOpens', 'EmailClicks',
                         'PreviousPurchases', 'LoyaltyPoints', 'EngagementScore', 'CostPerVisit',
                         'AgeGroup', 'IncomeTier']

        proba = model.predict_proba(processed_data[feature_order])[:, 1]
        prediction = (proba >= OPTIMAL_THRESHOLD).astype(int)[0]
        return prediction, proba[0]
    except Exception as e:
        st.error(f"Prediction Calculation Error: {e}")
        return 0, 0.0

# --- 2. Mock Database ---
PRODUCTS = [
    {"id": 1, "name": "Noise-Cancel Buds", "price": 149, "img": "🎧", "cat": "Tech"},
    {"id": 2, "name": "ErgoSmart Watch", "price": 399, "img": "⌚", "cat": "Tech"},
    {"id": 3, "name": "Runner Pro Sneakers", "price": 120, "img": "👟", "cat": "Fashion"},
    {"id": 4, "name": "Leather Satchel", "price": 250, "img": "🎒", "cat": "Fashion"},
    {"id": 5, "name": "4K Action Cam", "price": 299, "img": "📷", "cat": "Tech"},
    {"id": 6, "name": "Organic Tee", "price": 35, "img": "👕", "cat": "Fashion"},
]

# --- 3. Session State Management ---
defaults = {
    'page_views': 0, 'cart': [], 'start_time': None, 
    'session_active': False, 'persona': {}, 'social_clicks': 0
}
for key, val in defaults.items():
    if key not in st.session_state: st.session_state[key] = val

# --- 4. Sidebar & Navigation ---
model = load_model()

with st.sidebar:
    st.title("🛍️ ShopSense AI")
    app_mode = st.radio("Mode:", ["🛒 Live Store Simulator", "⚙️ Manual Calculator"], index=0)
    st.divider()
    
    # Live Tracker Panel
    if app_mode == "🛒 Live Store Simulator" and st.session_state.session_active:
        st.subheader("🕵️ Live Tracking")
        
        # Timer
        elapsed = time.time() - st.session_state.start_time
        mins, secs = divmod(int(elapsed), 60)
        st.metric("⏱️ Time on Site", f"{mins:02d}:{secs:02d}")
        
        # Stats
        c1, c2 = st.columns(2)
        c1.metric("Page Views", st.session_state.page_views)
        cart_total = sum(p['price'] for p in st.session_state.cart)
        c2.metric("Cart Total", f"${cart_total}")
        
        # Cart Display
        if st.session_state.cart:
            st.caption("🛒 Cart Items:")
            for item in st.session_state.cart:
                st.text(f"- {item['name']} (${item['price']})")
        
        st.divider()
        if st.button("🛑 Quit Session"):
            st.session_state.session_active = False
            st.rerun()

# =========================================================
# INTERFACE 1: LIVE STORE SIMULATOR
# =========================================================
if app_mode == "🛒 Live Store Simulator":
    
    # --- A. SETUP SCREEN (Persona Selection) ---
    if not st.session_state.session_active:
        st.title("🛒 Customer Simulation Lab")
        st.markdown("### Who is visiting the store today?")
        
        # The 4 Options
        persona_opts = [
            "1️⃣ The Window Shopper (Low Intent)",
            "2️⃣ The Focused Buyer (Mid Intent)",
            "3️⃣ The Whale / Big Spender (High Intent)",
            "4️⃣ ✨ Custom / Build Your Own"
        ]
        
        selection = st.radio("Select a Persona:", persona_opts, index=1, horizontal=False)
        st.divider()

        # Logic for Personas
        if selection == persona_opts[0]: # Window Shopper
            p_age, p_income, p_gender, p_loyalty = 22, 25000, "Female", 0
            st.info(f"**Profile:** Student, Budget Conscious. Unlikely to convert without heavy incentives.")
            
        elif selection == persona_opts[1]: # Focused Buyer
            p_age, p_income, p_gender, p_loyalty = 35, 65000, "Male", 500
            st.success(f"**Profile:** Average professional. Good income. Loyal customer. Conversion likely with engagement.")
            
        elif selection == persona_opts[2]: # Whale
            p_age, p_income, p_gender, p_loyalty = 50, 180000, "Female", 5000
            st.warning(f"**Profile:** High net worth. Historical big spender. Very high conversion probability.")
            
        else: # Custom
            st.subheader("🛠️ Build Custom Profile")
            c1, c2, c3 = st.columns(3)
            p_age = c1.number_input("Age", 18, 90, 28)
            p_gender = c2.selectbox("Gender", ["Male", "Female"])
            p_income = c3.number_input("Income ($)", 0, 1000000, 50000, step=5000)
            p_loyalty = st.slider("Loyalty Points (Past Engagement)", 0, 10000, 100)

        # "Enter Store" Button
        if st.button("🚀 Start Simulation", type="primary", use_container_width=True):
            st.session_state.session_active = True
            st.session_state.start_time = time.time()
            st.session_state.page_views = 1 # Landing page count
            st.session_state.cart = []
            st.session_state.social_clicks = 0
            
            # Save Persona
            st.session_state.persona = {
                'Age': p_age, 'Gender': p_gender, 'Income': p_income, 'LoyaltyPoints': p_loyalty,
                # Default marketing context
                'CampaignChannel': 'Direct', 'CampaignType': 'Organic', 'AdSpend': 0,
                'ClickThroughRate': 0, 'ConversionRate': 0, 'EmailOpens': 0, 'EmailClicks': 0, 
                'PreviousPurchases': 1 if p_loyalty > 0 else 0
            }
            st.rerun()

    # --- B. ACTIVE STORE SCREEN ---
    else:
        # Simulated Navbar
        col1, col2 = st.columns([4, 1])
        with col1: st.title("🛍️ ShopSense Demo Store")
        with col2: st.info(f"👤 {st.session_state.persona['Gender']}, {st.session_state.persona['Age']}yo")

        # Store Tabs
        tab1, tab2, tab3 = st.tabs(["🏠 Home Feed", "📦 Catalog", "💳 Checkout"])

        # Tab 1: Home
        with tab1:
            st.image("https://placehold.co/1200x400?text=Spring+Collection+Live", use_column_width=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔥 View Hot Deals"):
                    st.session_state.page_views += 1
                    st.toast("Browsing Deals (+1 Page View)")
            with c2:
                if st.button("📢 Share Store on Social"):
                    st.session_state.social_clicks += 1
                    st.toast("Social Share Recorded! (+Engagement)")

        # Tab 2: Catalog
        with tab2:
            st.caption("Click 'Add' to put items in cart. This increases your purchase intent score.")
            rows = [st.columns(3), st.columns(3)]
            for i, p in enumerate(PRODUCTS):
                col = rows[i // 3][i % 3]
                with col:
                    with st.container(border=True):
                        st.markdown(f"## {p['img']}")
                        st.write(f"**{p['name']}**")
                        st.write(f"${p['price']}")
                        if st.button(f"Add to Cart", key=f"add_{i}"):
                            st.session_state.cart.append(p)
                            st.session_state.page_views += 1
                            st.toast(f"Added {p['name']}!")
                            time.sleep(0.2)
                            st.rerun() # Force refresh sidebar

        # Tab 3: Prediction/Checkout
        with tab3:
            st.write("### 🛒 Ready to Checkout?")
            st.write(f"**Items in Cart:** {len(st.session_state.cart)}")
            st.write(f"**Total Value:** ${sum(p['price'] for p in st.session_state.cart)}")
            
            st.divider()
            
            if st.button("✨ Analyze Behavior & Predict", type="primary", use_container_width=True):
                # 1. Gather Live Metrics
                duration_mins = (time.time() - st.session_state.start_time) / 60
                if duration_mins < 0.01: duration_mins = 0.01 # Avoid zero errors
                
                final_pages = st.session_state.page_views
                
                # 2. Build Final Input Data
                # Merge Persona with Live Behavior
                data = st.session_state.persona.copy()
                data.update({
                    'TimeOnSite': duration_mins,
                    'PagesPerVisit': final_pages,
                    'WebsiteVisits': final_pages,
                    'SocialShares': st.session_state.social_clicks + len(st.session_state.cart) # Cart adds to engagement score
                })
                
                # 3. Predict
                if model:
                    pred, prob = predict_conversion(model, pd.DataFrame([data]))
                    
                    st.divider()
                    st.subheader("🧠 AI Analysis Report")
                    
                    # Display User Stats
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Income Tier", f"${data['Income']:,}")
                    m2.metric("Time Spent", f"{duration_mins:.2f} min")
                    m3.metric("Pages Viewed", final_pages)
                    m4.metric("Engagement Score", f"{(duration_mins * final_pages):.1f}")

                    # Display Prediction
                    if pred == 1:
                        st.success(f"## ✅ CONVERSION LIKELY (Probability: {prob:.1%})")
                        st.balloons()
                        st.markdown("""
                        **Why?** This user fits the high-intent demographic and demonstrated strong engagement behavior.
                        \n**Recommended Action:** Auto-apply a 10% coupon to close the sale immediately.
                        """)
                    else:
                        st.error(f"## ❄️ CONVERSION UNLIKELY (Probability: {prob:.1%})")
                        st.markdown("""
                        **Why?** Despite browsing, the engagement time or demographic fit is currently too low.
                        \n**Recommended Action:** Add to retargeting email list. Do not offer aggressive discounts yet.
                        """)

# =========================================================
# INTERFACE 2: MANUAL CALCULATOR (Legacy)
# =========================================================
elif app_mode == "⚙️ Manual Calculator":
    st.title("⚙️ Admin Prediction Tool")
    st.write("Test specific data points manually.")
    
    with st.form("admin_form"):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 18, 90, 30)
        income = c2.number_input("Income", 0, 1000000, 60000)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        
        st.divider()
        c3, c4 = st.columns(2)
        time_site = c3.number_input("Time on Site (min)", 0.0, 120.0, 5.0)
        pages = c4.number_input("Pages Per Visit", 1.0, 50.0, 3.0)
        
        if st.form_submit_button("Calculate Probability"):
            # Construct minimal valid dataframe
            df = pd.DataFrame([{
                'Age': age, 'Gender': gender, 'Income': income, 
                'TimeOnSite': time_site, 'PagesPerVisit': pages,
                'WebsiteVisits': pages, 'SocialShares': 0,
                'CampaignChannel': 'Direct', 'CampaignType': 'None',
                'AdSpend': 0, 'ClickThroughRate': 0, 'ConversionRate': 0,
                'EmailOpens': 0, 'EmailClicks': 0, 'PreviousPurchases': 0, 'LoyaltyPoints': 0
            }])
            
            if model:
                pred, prob = predict_conversion(model, df)
                st.metric("Conversion Probability", f"{prob:.1%}", delta="Convert" if pred else "No Convert")