import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import warnings
from datetime import datetime
from huggingface_hub import hf_hub_download

# --- 0. GLOBAL CONFIG & STYLES ---
st.set_page_config(page_title="ShopSense AI | Behavioral Intelligence", page_icon="🧠", layout="wide")
warnings.filterwarnings("ignore") # Keep UI clean

# Constants
REPO_ID = "psyrishi/marketing-conversion-predictor"
FILENAME = "conversion_prediction_model.pkl"
OPTIMAL_THRESHOLD = 0.2136

# Mock Database
PRODUCTS = [
    {"id": 1, "name": "Noise-Cancel Buds", "price": 149, "cat": "Tech", "img": "🎧", "desc": "Immersive sound, 24h battery."},
    {"id": 2, "name": "ErgoSmart Watch", "price": 399, "cat": "Tech", "img": "⌚", "desc": "Track health, calls, and more."},
    {"id": 3, "name": "Runner Pro Sneakers", "price": 120, "cat": "Fashion", "img": "👟", "desc": "Aerodynamic design for speed."},
    {"id": 4, "name": "Leather Satchel", "price": 250, "cat": "Fashion", "img": "🎒", "desc": "Handcrafted Italian leather."},
    {"id": 5, "name": "4K Action Cam", "price": 299, "cat": "Tech", "img": "📷", "desc": "Waterproof, stabilization enabled."},
    {"id": 6, "name": "Organic Tee", "price": 35, "cat": "Fashion", "img": "👕", "desc": "100% sustainable cotton."},
]

# --- 1. BACKEND LOGIC ---
@st.cache_resource
def load_model():
    """Loads model with robust error handling for version mismatches."""
    import sklearn.compose._column_transformer as ct
    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list): pass
        ct._RemainderColsList = _RemainderColsList

    try:
        with st.spinner("🔌 Connecting to AI Inference Engine..."):
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            return joblib.load(model_path)
    except Exception as e:
        st.error(f"⚠️ Core AI Error: {e}\n\n*Action: Retrain model.pkl to fix scikit-learn version conflict.*")
        return None

def prepare_input_data(persona, engagement_stats):
    """Merges static persona data with dynamic session data."""
    data = persona.copy()
    
    # Dynamic Calculation
    data['TimeOnSite'] = engagement_stats['time_mins']
    data['PagesPerVisit'] = engagement_stats['pages']
    data['WebsiteVisits'] = engagement_stats['pages'] # Proxy for session depth
    # Cart items count as "Social Shares" proxy for high-intent interaction in this demo model
    data['SocialShares'] = engagement_stats['social_actions'] + engagement_stats['cart_count']
    
    # Feature Engineering Pipeline (Replicated from Training)
    df = pd.DataFrame([data])
    df['EngagementScore'] = df['TimeOnSite'] * df['PagesPerVisit']
    df['WebsiteVisits_safe'] = df['WebsiteVisits'].replace(0, 1)
    df['CostPerVisit'] = df['AdSpend'] / df['WebsiteVisits_safe']
    df.drop('WebsiteVisits_safe', axis=1, inplace=True)
    
    df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 30, 50, 70], labels=['Young', 'Adult', 'Senior'], right=True, include_lowest=True)
    df['IncomeTier'] = pd.cut(df['Income'], bins=[19999, 50000, 90000, 150000], labels=['Low', 'Medium', 'High'], right=True, include_lowest=True)
    
    return df

def get_prediction(model, df):
    """Returns prediction class and probability."""
    try:
        feature_order = ['Age', 'Gender', 'Income', 'CampaignChannel', 'CampaignType', 'AdSpend',
                         'ClickThroughRate', 'ConversionRate', 'WebsiteVisits', 'PagesPerVisit',
                         'TimeOnSite', 'SocialShares', 'EmailOpens', 'EmailClicks',
                         'PreviousPurchases', 'LoyaltyPoints', 'EngagementScore', 'CostPerVisit',
                         'AgeGroup', 'IncomeTier']
        
        proba = model.predict_proba(df[feature_order])[:, 1][0]
        pred = (proba >= OPTIMAL_THRESHOLD).astype(int)
        return pred, proba
    except Exception as e:
        st.error(f"Inference Failed: {e}")
        return 0, 0.0

# --- 2. SESSION STATE ---
def init_session():
    defaults = {
        'page_views': 0, 'cart': [], 'start_time': None, 'active': False, 
        'persona': {}, 'social_clicks': 0, 'log': [], 'time_warp': 1.0
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def log_action(action, icon="🔹"):
    """Adds an event to the session timeline."""
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.log.insert(0, f"`{ts}` {icon} {action}")

init_session()
model = load_model()

# --- 3. SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.title("🛍️ ShopSense AI")
    st.caption("Real-time Behavioral Prediction Engine")
    
    mode = st.radio("System Mode:", ["🛒 Store Simulator", "⚙️ Model Admin"], label_visibility="collapsed")
    st.divider()

    # LIVE SESSION TRACKER
    if mode == "🛒 Store Simulator" and st.session_state.active:
        st.subheader("🕵️ Session Telemetry")
        
        # Time Warp Logic
        real_sec = time.time() - st.session_state.start_time
        sim_min = (real_sec * st.session_state.time_warp) / 60
        
        c1, c2 = st.columns(2)
        c1.metric("Simulated Time", f"{sim_min:.1f}m")
        c2.metric("Page Views", st.session_state.page_views)
        
        cart_val = sum(p['price'] for p in st.session_state.cart)
        st.metric("🛒 Cart Value", f"${cart_val}", delta=f"{len(st.session_state.cart)} items")

        # Journey Log
        with st.expander("📜 User Journey Log", expanded=True):
            for entry in st.session_state.log[:8]: # Show last 8 actions
                st.markdown(entry)
            if len(st.session_state.log) > 8:
                st.caption("... earlier events hidden")

        st.divider()
        if st.button("🛑 End Simulation", use_container_width=True):
            st.session_state.active = False
            st.rerun()

# =========================================================
# VIEW 1: STORE SIMULATOR (The "Product")
# =========================================================
if mode == "🛒 Store Simulator":

    # --- PHASE A: CONFIGURATION ---
    if not st.session_state.active:
        st.title("🛒 Customer Experience Lab")
        st.markdown("Configure the visitor profile to test how the AI reacts to different demographics and behaviors.")
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("1. Visitor Persona")
            p_type = st.selectbox("Select Profile:", 
                ["👱‍♀️ The Window Shopper (Low Intent)", 
                 "👨‍💼 The Focused Buyer (Mid Intent)", 
                 "👩‍💻 The Tech Whale (High Intent)", 
                 "✨ Custom Profile"])
            
            if "Window Shopper" in p_type:
                p_data = {'Age': 22, 'Income': 25000, 'Gender': 'Female', 'Loyalty': 0}
            elif "Focused Buyer" in p_type:
                p_data = {'Age': 35, 'Income': 75000, 'Gender': 'Male', 'Loyalty': 500}
            elif "Tech Whale" in p_type:
                p_data = {'Age': 45, 'Income': 180000, 'Gender': 'Female', 'Loyalty': 5000}
            else:
                s1, s2 = st.columns(2)
                p_age = s1.number_input("Age", 18, 80, 30)
                p_gen = s2.selectbox("Gender", ["Male", "Female"])
                p_inc = st.number_input("Income ($)", 15000, 500000, 60000, step=5000)
                p_loy = st.slider("Loyalty Pts", 0, 10000, 0)
                p_data = {'Age': p_age, 'Income': p_inc, 'Gender': p_gen, 'Loyalty': p_loy}

            # Add default hidden marketing fields
            p_data.update({
                'CampaignChannel': 'Direct', 'CampaignType': 'Organic', 'AdSpend': 0,
                'ClickThroughRate': 0.0, 'ConversionRate': 0.0, 'EmailOpens': 0, 'EmailClicks': 0,
                'PreviousPurchases': 1 if p_data['Loyalty'] > 0 else 0
            })
        
        with c2:
            st.subheader("2. Simulation Settings")
            st.info("💡 **Time Warp:** Speed up the clock to simulate a 20-minute browse in just 20 seconds.")
            speed = st.select_slider("Time Speed:", options=[1, 10, 60, 600], value=60, format_func=lambda x: f"{x}x Speed")
            
            st.write("##")
            if st.button("🚀 Launch Store Simulator", type="primary", use_container_width=True):
                st.session_state.active = True
                st.session_state.start_time = time.time()
                st.session_state.time_warp = speed
                st.session_state.page_views = 1
                st.session_state.cart = []
                st.session_state.log = []
                st.session_state.social_clicks = 0
                st.session_state.persona = p_data
                log_msg = f"Session Started: {p_data['Gender']}, {p_data['Age']}yo, ${p_data['Income']:,}"
                log_action(log_msg, "🟢")
                st.rerun()

    # --- PHASE B: THE STORE INTERFACE ---
    else:
        # HUD (Head-Up Display)
        st.caption(f"LOGGED IN AS: **{st.session_state.persona['Gender']}, {st.session_state.persona['Age']}**")
        
        # Store Navigation
        tab_home, tab_shop, tab_social, tab_pay = st.tabs(["🏠 Home", "🛍️ Shop", "❤️ Social", "💳 Checkout"])

        # 1. HOME TAB
        with tab_home:
            st.image("https://placehold.co/1200x350/232F3E/FFF?text=Spring+Collection+Live", use_container_width=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🔥 View Flash Deals"):
                    st.session_state.page_views += 1
                    log_action("Browsed Flash Deals", "🔥")
                    st.toast("Browsing Deals...")
            with c2:
                if st.button("📖 Read Our Story"):
                    st.session_state.page_views += 1
                    log_action("Read About Us", "📖")
            with c3:
                if st.button("📞 Contact Support"):
                    st.session_state.page_views += 1
                    log_action("Visited Support Page", "📞")

        # 2. SHOP TAB
        with tab_shop:
            cat_filter = st.radio("Filter:", ["All", "Tech", "Fashion"], horizontal=True, label_visibility="collapsed")
            
            # Filter Logic
            display_items = PRODUCTS if cat_filter == "All" else [p for p in PRODUCTS if p['cat'] == cat_filter]
            
            # Grid Layout
            cols = st.columns(3)
            for i, p in enumerate(display_items):
                with cols[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"### {p['img']}")
                        st.write(f"**{p['name']}**")
                        st.caption(p['desc'])
                        st.write(f"**${p['price']}**")
                        
                        b1, b2 = st.columns(2)
                        if b1.button("View", key=f"v_{p['id']}"):
                            st.session_state.page_views += 1
                            log_action(f"Viewed {p['name']}", "👀")
                            st.toast(f"Viewing {p['name']}")
                            
                        if b2.button("Add", key=f"a_{p['id']}", type="primary"):
                            st.session_state.cart.append(p)
                            st.session_state.page_views += 1
                            log_action(f"Added {p['name']} to Cart", "🛒")
                            st.toast("Added to Cart!")
                            time.sleep(0.1) # UI sync
                            st.rerun()

        # 3. SOCIAL TAB
        with tab_social:
            st.info("Interacting here increases the 'SocialShares' signal to the model.")
            c1, c2 = st.columns(2)
            with c1:
                st.image("https://placehold.co/500x300?text=Instagram+Viral", use_container_width=True)
                if st.button("❤️ Like on Instagram"):
                    st.session_state.social_clicks += 1
                    log_action("Liked Instagram Post", "❤️")
            with c2:
                st.image("https://placehold.co/500x300?text=TikTok+Trend", use_container_width=True)
                if st.button("↗️ Share to Friends"):
                    st.session_state.social_clicks += 2
                    log_action("Shared Product Link", "🔗")
        
        # 4. CHECKOUT TAB
        with tab_pay:
            st.markdown("### 🧾 Checkout Counter")
            
            if not st.session_state.cart:
                st.warning("Your cart is empty. Add items to generate a meaningful prediction.")
            else:
                # Live Stats
                real_sec = time.time() - st.session_state.start_time
                sim_min = (real_sec * st.session_state.time_warp) / 60
                if sim_min < 0.1: sim_min = 0.1 # Avoid zero errors
                
                stats = {
                    'time_mins': sim_min,
                    'pages': st.session_state.page_views,
                    'cart_count': len(st.session_state.cart),
                    'social_actions': st.session_state.social_clicks
                }
                
                # Trigger Prediction
                if st.button("✨ Analyze Behavior & Predict Conversion", type="primary", use_container_width=True):
                    
                    if model:
                        df_input = prepare_input_data(st.session_state.persona, stats)
                        pred, prob = get_prediction(model, df_input)
                        
                        st.divider()
                        st.subheader("🧠 AI Analysis Result")
                        
                        res_col1, res_col2 = st.columns([1.5, 1])
                        
                        with res_col1:
                            # Key Drivers Analysis (Heuristic for Demo)
                            st.markdown("#### 🔑 Key Drivers")
                            
                            # Income Driver
                            inc = st.session_state.persona['Income']
                            if inc > 80000: st.success("✅ **High Income:** Strong purchasing power.")
                            elif inc < 30000: st.error("❌ **Low Income:** Budget constraint likely.")
                            else: st.info("ℹ️ **Middle Income:** Neutral factor.")
                            
                            # Engagement Driver
                            eng_score = df_input['EngagementScore'].iloc[0]
                            if eng_score > 30: st.success(f"✅ **Deep Engagement:** Score {eng_score:.1f} (High Interest)")
                            else: st.warning(f"⚠️ **Low Engagement:** Score {eng_score:.1f} (Browsing)")
                            
                            # Cart Driver
                            if len(st.session_state.cart) > 2: st.success("✅ **High Intent:** Multiple items in cart.")

                        with res_col2:
                            # The Score
                            st.metric("Conversion Probability", f"{prob:.1%}")
                            if pred == 1:
                                st.success("## 🟢 LIKELY")
                                st.balloons()
                            else:
                                st.error("## 🔴 UNLIKELY")
                                st.progress(prob, "Probability")

# =========================================================
# VIEW 2: MODEL ADMIN (Debugging Tool)
# =========================================================
elif mode == "⚙️ Model Admin":
    st.title("⚙️ Model Stress Test")
    st.write("Directly query the model with edge cases.")
    
    with st.form("debug_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 100, 30)
        income = c2.number_input("Income", 0, 1000000, 50000)
        time_site = c3.number_input("Time (min)", 0.1, 500.0, 5.0)
        
        if st.form_submit_button("Run Inference"):
            # Minimal data construction
            p = {'Age': age, 'Income': income, 'Gender': 'Male', 'LoyaltyPoints': 100,
                 'CampaignChannel': 'Direct', 'CampaignType': 'None', 'AdSpend': 0,
                 'ClickThroughRate': 0, 'ConversionRate': 0, 'EmailOpens': 0, 'EmailClicks': 0, 
                 'PreviousPurchases': 0}
            s = {'time_mins': time_site, 'pages': 5, 'cart_count': 1, 'social_actions': 0}
            
            df = prepare_input_data(p, s)
            pred, prob = get_prediction(model, df)
            st.code(f"Prediction: {pred}\nProbability: {prob:.4f}")