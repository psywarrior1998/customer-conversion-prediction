import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import warnings
import sklearn
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download

# --- 0. GLOBAL CONFIG ---
st.set_page_config(
    page_title="ShopSense AI | Enterprise",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore")

# Constants
REPO_ID = "psyrishi/marketing-conversion-predictor"
FILENAME = "conversion_prediction_model.pkl"
OPTIMAL_THRESHOLD = 0.2136

# Mock Product Database
PRODUCTS = [
    {"id": 1, "name": "Noise-Cancel Buds", "price": 149, "cat": "Tech", "rating": 4.8, "stock": 12, "img": "🎧", "desc": "Immersive sound."},
    {"id": 2, "name": "ErgoSmart Watch", "price": 399, "cat": "Tech", "rating": 4.5, "stock": 5, "img": "⌚", "desc": "Health tracking."},
    {"id": 3, "name": "Runner Pro X1", "price": 120, "cat": "Fashion", "rating": 4.2, "stock": 50, "img": "👟", "desc": "Speed & comfort."},
    {"id": 4, "name": "Italian Leather Bag", "price": 250, "cat": "Fashion", "rating": 4.9, "stock": 2, "img": "🎒", "desc": "Handcrafted."},
    {"id": 5, "name": "4K Action Cam", "price": 299, "cat": "Tech", "rating": 4.6, "stock": 8, "img": "📷", "desc": "Waterproof 4K."},
    {"id": 6, "name": "Organic Cotton Tee", "price": 35, "cat": "Fashion", "rating": 4.1, "stock": 100, "img": "👕", "desc": "Sustainable fit."},
]

# --- 1. CORE AI ENGINE ---
@st.cache_resource
def load_model():
    """Loads the AI model with version compatibility handling."""
    import sklearn.compose._column_transformer as ct
    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list): pass
        ct._RemainderColsList = _RemainderColsList

    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"⚠️ AI Engine Failure: {e}")
        return None

def run_inference(model, input_df):
    """
    Centralized inference function.
    Takes a raw dataframe, applies feature engineering, and returns prediction.
    """
    df = input_df.copy()
    
    # 1. Feature Engineering
    df['EngagementScore'] = df['TimeOnSite'] * df['PagesPerVisit']
    
    # Safety check for division
    if 'WebsiteVisits' in df.columns:
        df['WebsiteVisits_safe'] = df['WebsiteVisits'].replace(0, 1)
        df['CostPerVisit'] = df['AdSpend'] / df['WebsiteVisits_safe']
        df.drop('WebsiteVisits_safe', axis=1, inplace=True)
    else:
        df['CostPerVisit'] = 0
    
    # Binning (Strictly matching training buckets)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 30, 50, 70], labels=['Young', 'Adult', 'Senior'], right=True, include_lowest=True)
    df['IncomeTier'] = pd.cut(df['Income'], bins=[19999, 50000, 90000, 150000], labels=['Low', 'Medium', 'High'], right=True, include_lowest=True)
    
    # 2. Select & Order Features
    feature_order = ['Age', 'Gender', 'Income', 'CampaignChannel', 'CampaignType', 'AdSpend',
                     'ClickThroughRate', 'ConversionRate', 'WebsiteVisits', 'PagesPerVisit',
                     'TimeOnSite', 'SocialShares', 'EmailOpens', 'EmailClicks',
                     'PreviousPurchases', 'LoyaltyPoints', 'EngagementScore', 'CostPerVisit',
                     'AgeGroup', 'IncomeTier']
    
    try:
        # Ensure all columns exist (fill missing with defaults if necessary)
        for col in feature_order:
            if col not in df.columns: df[col] = 0
            
        proba = model.predict_proba(df[feature_order])[:, 1][0]
        pred = (proba >= OPTIMAL_THRESHOLD).astype(int)
        return pred, proba
    except Exception as e:
        st.error(f"Inference Error: {e}")
        return 0, 0.0

# --- 2. SESSION STATE ---
def init_session():
    defaults = {
        'active': False, 
        'start_time': None, 
        'pages': 0, 
        'cart': [], 
        'social': 0, 
        'speed': 1.0,
        'inventory': PRODUCTS.copy(), # Using copy for session isolation
        'history_engagement': [0],
        'persona': {}
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_session()
model = load_model()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🛍️ ShopSense AI")
    mode = st.radio("Interface Mode:", ["🛒 Live Store Simulator", "📊 Manual Predictor"], label_visibility="collapsed")
    st.markdown("---")
    
    # SIMULATOR CONTROLS
    if mode == "🛒 Live Store Simulator":
        st.subheader("⏱️ Temporal Engine")
        speed_map = {
            "Real-time (1x)": 1.0,
            "Fast (10x)": 10.0,
            "Hyper (60x - 1s=1m)": 60.0,
            "Warp (600x)": 600.0
        }
        speed_sel = st.select_slider("Simulation Speed", options=list(speed_map.keys()), value="Hyper (60x - 1s=1m)")
        st.session_state.speed = speed_map[speed_sel]
        
        if st.session_state.active:
            st.divider()
            st.subheader("📊 Live Telemetry")
            
            # Time Calc
            real_elapsed = time.time() - st.session_state.start_time
            sim_seconds = real_elapsed * st.session_state.speed
            sim_time_str = str(timedelta(seconds=int(sim_seconds)))
            
            # Visuals
            st.markdown(f"<h2 style='text-align:center; color:#4CAF50; margin:0'>{sim_time_str}</h2>", unsafe_allow_html=True)
            st.caption("Simulated Session Duration")
            
            engagement = (sim_seconds/60) * st.session_state.pages
            st.progress(min(int(engagement), 100), text=f"Engagement Heat: {engagement:.1f}")
            
            c1, c2 = st.columns(2)
            c1.metric("Page Views", st.session_state.pages)
            cart_val = sum(p['price'] for p in st.session_state.cart)
            c2.metric("Cart", f"${cart_val}", delta=f"{len(st.session_state.cart)} items")

            st.divider()
            if st.button("🛑 End Session", use_container_width=True):
                st.session_state.active = False
                st.rerun()

# =========================================================
# MODE 1: LIVE STORE SIMULATOR
# =========================================================
if mode == "🛒 Live Store Simulator":

    # A. CONFIGURATION SCREEN
    if not st.session_state.active:
        st.title("🛒 Customer Simulation Lab")
        st.markdown("Configure a visitor profile to see how the AI predicts their conversion probability based on real-time behavior.")
        
        with st.container(border=True):
            c1, c2 = st.columns([1, 1.5])
            
            with c1:
                st.subheader("👤 Visitor Persona")
                profile = st.radio("Select Profile:", 
                    ["Window Shopper", "Focused Buyer", "High-Value Whale", "Custom"],
                    captions=["Low Income, Browsing", "Mid Income, Specific Intent", "High Income, Ready to Buy", "Manual Config"]
                )
                
            with c2:
                st.subheader("⚙️ Demographics")
                # Defaults
                if profile == "Window Shopper":
                    age, inc, gen, loy = 22, 25000, "Female", 0
                elif profile == "Focused Buyer":
                    age, inc, gen, loy = 34, 65000, "Male", 400
                elif profile == "High-Value Whale":
                    age, inc, gen, loy = 52, 180000, "Female", 5000
                else:
                    age, inc, gen, loy = 30, 50000, "Female", 100

                # Inputs (Disabled unless Custom)
                disable_inputs = (profile != "Custom")
                
                col_a, col_b = st.columns(2)
                p_age = col_a.number_input("Age", 18, 90, age, disabled=disable_inputs)
                p_gen = col_b.selectbox("Gender", ["Male", "Female"], index=0 if gen=="Male" else 1, disabled=disable_inputs)
                
                col_c, col_d = st.columns(2)
                p_inc = col_c.number_input("Income ($)", 15000, 1000000, inc, step=1000, disabled=disable_inputs)
                p_loy = col_d.number_input("Loyalty Pts", 0, 10000, loy, disabled=disable_inputs)

        if st.button("🏁 Begin Simulation", type="primary", use_container_width=True):
            st.session_state.active = True
            st.session_state.start_time = time.time()
            st.session_state.pages = 1
            st.session_state.cart = []
            st.session_state.social = 0
            st.session_state.inventory = [p.copy() for p in PRODUCTS] # Reset stock
            
            st.session_state.persona = {
                'Age': p_age, 'Gender': p_gen, 'Income': p_inc, 'LoyaltyPoints': p_loy,
                # Hidden Context
                'CampaignChannel': 'Direct', 'CampaignType': 'Organic', 'AdSpend': 0,
                'ClickThroughRate': 0.0, 'ConversionRate': 0.0, 'EmailOpens': 0, 'EmailClicks': 0, 
                'PreviousPurchases': 1 if p_loy > 0 else 0
            }
            st.rerun()

    # B. STORE INTERFACE
    else:
        p = st.session_state.persona
        col1, col2 = st.columns([3, 1])
        with col1: st.markdown(f"### 🛍️ ShopSense <span style='color:gray'>| {p['Gender']}, {p['Age']} | ${p['Income']:,}</span>", unsafe_allow_html=True)
        with col2: 
            if st.button("🛒 View Cart", use_container_width=True):
                st.toast(f"Items: {len(st.session_state.cart)}")

        tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📦 Catalog", "❤️ Social", "💳 Checkout"])
        
        # 1. Home
        with tab1:
            st.image("https://placehold.co/1200x300/232F3E/FFF?text=Spring+Collection", use_container_width=True)
            c1, c2, c3 = st.columns(3)
            if c1.button("🔥 Flash Deals", use_container_width=True):
                st.session_state.pages += 1; st.toast("Browsing Deals")
            if c2.button("💎 New Arrivals", use_container_width=True):
                st.session_state.pages += 1; st.toast("Browsing New Arrivals")
            if c3.button("📰 Blog", use_container_width=True):
                st.session_state.pages += 1; st.toast("Reading Blog")

        # 2. Catalog
        with tab2:
            cat_filter = st.radio("Category:", ["All", "Tech", "Fashion"], horizontal=True)
            items = st.session_state.inventory if cat_filter == "All" else [x for x in st.session_state.inventory if x['cat'] == cat_filter]
            
            cols = st.columns(3)
            for i, item in enumerate(items):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"### {item['img']} {item['name']}")
                        st.caption(item['desc'])
                        
                        r1, r2 = st.columns([1, 1])
                        r1.markdown(f"**${item['price']}**")
                        
                        # Stock Logic
                        disabled = False
                        if item['stock'] == 0:
                            r2.error("Sold Out")
                            disabled = True
                        elif item['stock'] < 5:
                            r2.warning(f"Only {item['stock']} left!")
                        else:
                            r2.success("In Stock")
                            
                        if st.button("Add to Cart", key=f"add_{item['id']}", disabled=disabled, use_container_width=True):
                            item['stock'] -= 1
                            st.session_state.cart.append(item)
                            st.session_state.pages += 1
                            st.toast(f"Added {item['name']}!")
                            time.sleep(0.05)
                            st.rerun()

        # 3. Social
        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                st.image("https://placehold.co/600x400?text=Instagram", use_container_width=True)
                if st.button("❤️ Like", use_container_width=True):
                    st.session_state.social += 1; st.toast("Liked!")
            with c2:
                st.image("https://placehold.co/600x400?text=Review", use_container_width=True)
                if st.button("💬 Comment", use_container_width=True):
                    st.session_state.pages += 1; st.toast("Opening comments")

        # 4. Checkout
        with tab4:
            st.write("## 💳 Checkout")
            
            # Cart Display
            if st.session_state.cart:
                cart_df = pd.DataFrame(st.session_state.cart)
                st.dataframe(cart_df[['name', 'price']], use_container_width=True)
                st.markdown(f"### Total: **${cart_df['price'].sum()}**")
            else:
                st.info("Cart is empty.")

            st.divider()
            
            if st.button("🔮 Predict Purchase Probability", type="primary", use_container_width=True):
                # Final Calc
                real_dur = time.time() - st.session_state.start_time
                sim_mins = (real_dur * st.session_state.speed) / 60
                if sim_mins < 0.1: sim_mins = 0.1
                
                # Build DF
                data = st.session_state.persona.copy()
                data.update({
                    'TimeOnSite': sim_mins,
                    'PagesPerVisit': st.session_state.pages,
                    'WebsiteVisits': st.session_state.pages, 
                    'SocialShares': st.session_state.social + (len(st.session_state.cart)*2)
                })
                
                # Predict
                if model:
                    pred, prob = run_inference(model, pd.DataFrame([data]))
                    
                    st.success("Analysis Complete!")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Engagement Score", f"{(sim_mins * st.session_state.pages):.1f}")
                    c2.metric("Simulated Time", f"{sim_mins:.1f}m")
                    c3.metric("Cart Items", len(st.session_state.cart))

                    st.markdown("---")
                    fc1, fc2 = st.columns([2, 1])
                    with fc1:
                        if pred == 1:
                            st.markdown("## 🚀 Verdict: **CONVERSION LIKELY**")
                            st.info("**Action:** Do not interrupt. Allow organic checkout.")
                            st.balloons()
                        else:
                            st.markdown("## 🛑 Verdict: **AT RISK**")
                            st.warning("**Action:** Trigger 10% Discount Popup.")
                    with fc2:
                        st.metric("Probability", f"{prob:.1%}")
                        st.progress(prob)

# =========================================================
# MODE 2: MANUAL PREDICTOR
# =========================================================
elif mode == "📊 Manual Predictor":
    st.title("📊 Full Control Predictor")
    st.markdown("Manually input every parameter to test the model's sensitivity to specific variables.")
    
    if model:
        with st.form("manual_form"):
            st.subheader("1. User Demographics")
            col1, col2, col3 = st.columns(3)
            age = col1.number_input("Age", 18, 90, 30)
            gender = col2.selectbox("Gender", ["Male", "Female"])
            income = col3.number_input("Income ($)", 0, 1000000, 60000)
            loyalty = col1.number_input("Loyalty Points", 0, 10000, 0)
            prev_purch = col2.number_input("Previous Purchases", 0, 100, 0)
            
            st.subheader("2. Web Engagement")
            col4, col5, col6 = st.columns(3)
            time_site = col4.number_input("Time On Site (min)", 0.0, 500.0, 5.0)
            pages_visit = col5.number_input("Pages Per Visit", 0.0, 100.0, 3.0)
            total_visits = col6.number_input("Total Website Visits", 0, 200, 5)
            social = col4.number_input("Social Shares", 0, 100, 0)
            
            st.subheader("3. Marketing Context")
            col7, col8 = st.columns(2)
            channel = col7.selectbox("Campaign Channel", ["Email", "Social Media", "PPC", "Referral", "Display"])
            ctype = col8.selectbox("Campaign Type", ["Awareness", "Conversion", "Retention", "Consideration"])
            ad_spend = col7.number_input("Ad Spend ($)", 0.0, 10000.0, 1000.0)
            ctr = col8.number_input("Click Through Rate", 0.0, 1.0, 0.05)
            conv_rate = col7.number_input("Campaign Conversion Rate", 0.0, 1.0, 0.05)
            
            st.subheader("4. Email Metrics")
            col9, col10 = st.columns(2)
            e_opens = col9.number_input("Email Opens", 0, 50, 0)
            e_clicks = col10.number_input("Email Clicks", 0, 50, 0)
            
            if st.form_submit_button("Run Prediction", type="primary"):
                # Construct full dataframe
                df = pd.DataFrame([{
                    'Age': age, 'Gender': gender, 'Income': income, 
                    'CampaignChannel': channel, 'CampaignType': ctype,
                    'AdSpend': ad_spend, 'ClickThroughRate': ctr, 'ConversionRate': conv_rate,
                    'WebsiteVisits': total_visits, 'PagesPerVisit': pages_visit, 'TimeOnSite': time_site,
                    'SocialShares': social, 'EmailOpens': e_opens, 'EmailClicks': e_clicks,
                    'PreviousPurchases': prev_purch, 'LoyaltyPoints': loyalty
                }])
                
                pred, prob = run_inference(model, df)
                
                st.divider()
                c_res, c_prob = st.columns([2, 1])
                with c_res:
                    if pred == 1:
                        st.success("### Prediction: CONVERT")
                        st.write("The model predicts this user **WILL** make a purchase.")
                    else:
                        st.error("### Prediction: NO CONVERT")
                        st.write("The model predicts this user **WILL NOT** make a purchase.")
                with c_prob:
                    st.metric("Probability Score", f"{prob:.1%}")