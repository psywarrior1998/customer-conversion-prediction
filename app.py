import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from huggingface_hub import hf_hub_download

# --- Configuration ---
REPO_ID = "psyrishi/marketing-conversion-predictor"
FILENAME = "conversion_prediction_model.pkl"
OPTIMAL_THRESHOLD = 0.2136

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Downloads the model from Hugging Face Hub and loads it."""
    import joblib
    import sklearn.compose._column_transformer as ct
    
    # Patch: create dummy class if missing (fixes potential serialization issues)
    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Dummy placeholder for backward compatibility"""
            pass
        ct._RemainderColsList = _RemainderColsList

    try:
        with st.spinner(f"Loading model from Hugging Face: {REPO_ID}..."):
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model.\nDetails: {e}")
        return None

# --- Feature Engineering ---
def apply_feature_engineering(df):
    """Applies necessary feature engineering steps."""
    # Calculate derived features
    df['EngagementScore'] = df['TimeOnSite'] * df['PagesPerVisit']
    df['WebsiteVisits_safe'] = df['WebsiteVisits'].replace(0, 1)
    df['CostPerVisit'] = df['AdSpend'] / df['WebsiteVisits_safe']
    df.drop('WebsiteVisits_safe', axis=1, inplace=True)
    
    # Bins must match training exactly
    df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 30, 50, 70], labels=['Young', 'Adult', 'Senior'], right=True, include_lowest=True)
    df['IncomeTier'] = pd.cut(df['Income'], bins=[19999, 50000, 90000, 150000], labels=['Low', 'Medium', 'High'], right=True, include_lowest=True)
    return df

# --- Prediction Function ---
def predict_conversion(model, data):
    """Generates prediction using the optimal threshold."""
    processed_data = apply_feature_engineering(data)
    
    # Ensure column order matches training
    feature_order = ['Age', 'Gender', 'Income', 'CampaignChannel', 'CampaignType', 'AdSpend',
                     'ClickThroughRate', 'ConversionRate', 'WebsiteVisits', 'PagesPerVisit',
                     'TimeOnSite', 'SocialShares', 'EmailOpens', 'EmailClicks',
                     'PreviousPurchases', 'LoyaltyPoints', 'EngagementScore', 'CostPerVisit',
                     'AgeGroup', 'IncomeTier']

    # Predict
    proba = model.predict_proba(processed_data[feature_order])[:, 1]
    prediction = (proba >= OPTIMAL_THRESHOLD).astype(int)[0]
    return prediction, proba[0]

# --- Main UI Layout ---
st.set_page_config(page_title="Conversion Predictor", page_icon="🎯", layout="wide")

# Initialize Session State
if 'page_views' not in st.session_state:
    st.session_state.page_views = 0
if 'social_shares_count' not in st.session_state:
    st.session_state.social_shares_count = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'demo_active' not in st.session_state:
    st.session_state.demo_active = False

# Load model once
model = load_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Interface:", ["📊 Manual Prediction", "🛍️ Live Store Demo"])

# ---------------------------------------------------------
# MODE 1: MANUAL PREDICTION (Original Functionality)
# ---------------------------------------------------------
if app_mode == "📊 Manual Prediction":
    st.title("🎯 Marketing Conversion Predictor")
    st.write("Enter customer details below to predict the likelihood of conversion.")

    if model:
        with st.form("user_input_form"):
            st.subheader("Customer Details")
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=70, value=35)
                gender = st.selectbox("Gender", ["Male", "Female"])
                income = st.number_input("Income ($)", min_value=0, value=65000, step=1000)
                loyalty_points = st.number_input("Loyalty Points", min_value=0, value=1500)
            with c2:
                previous_purchases = st.number_input("Previous Purchases", min_value=0, value=3)
                website_visits = st.number_input("Website Visits", min_value=0, value=12)
                time_on_site = st.number_input("Time On Site (mins)", min_value=0.0, value=8.5, step=0.5)
                pages_per_visit = st.number_input("Pages Per Visit", min_value=0.0, value=4.2, step=0.1)

            st.subheader("Campaign Metrics")
            c3, c4 = st.columns(2)
            with c3:
                campaign_channel = st.selectbox("Channel", ["Email", "Social Media", "PPC", "Referral", "Display"])
                campaign_type = st.selectbox("Type", ["Awareness", "Conversion", "Retention", "Consideration"])
                ad_spend = st.number_input("Ad Spend ($)", min_value=0.0, value=2500.0, step=50.0)
            with c4:
                click_through_rate = st.number_input("Click Through Rate (CTR)", min_value=0.0, max_value=1.0, value=0.12, step=0.01, format="%.2f")
                conversion_rate = st.number_input("Conversion Rate", min_value=0.0, max_value=1.0, value=0.08, step=0.01, format="%.2f")
                social_shares = st.number_input("Social Shares", min_value=0, value=15)
                email_opens = st.number_input("Email Opens", min_value=0, value=4)
                email_clicks = st.number_input("Email Clicks", min_value=0, value=2)

            submit_button = st.form_submit_button("Predict Conversion Status", type="primary")

        if submit_button:
            # Create dataframe from inputs
            input_data = pd.DataFrame({
                'Age': [age], 'Gender': [gender], 'Income': [income], 'CampaignChannel': [campaign_channel],
                'CampaignType': [campaign_type], 'AdSpend': [ad_spend], 'ClickThroughRate': [click_through_rate],
                'ConversionRate': [conversion_rate], 'WebsiteVisits': [website_visits], 'PagesPerVisit': [pages_per_visit],
                'TimeOnSite': [time_on_site], 'SocialShares': [social_shares], 'EmailOpens': [email_opens],
                'EmailClicks': [email_clicks], 'PreviousPurchases': [previous_purchases], 'LoyaltyPoints': [loyalty_points]
            })

            pred, proba = predict_conversion(model, input_data)

            st.divider()
            if pred == 1:
                st.success(f"### ✅ Prediction: CONVERT (Likely to buy)")
                st.metric("Conversion Probability", f"{proba:.1%}")
            else:
                st.error(f"### ❌ Prediction: NO CONVERT (Unlikely to buy)")
                st.metric("Conversion Probability", f"{proba:.1%}")

# ---------------------------------------------------------
# MODE 2: LIVE STORE DEMO (New Functionality)
# ---------------------------------------------------------
elif app_mode == "🛍️ Live Store Demo":
    st.title("🛍️ Live Store Simulator")
    st.markdown("""
    **How it works:**
    1.  **Select a Persona:** Choose a user type to auto-fill demographic data.
    2.  **Start Session:** Click 'Start' to begin tracking.
    3.  **Browse:** Interact with the store buttons (View Products, Share, etc.). The app tracks your **Time on Site** and **Pages Visited** in the background.
    4.  **End Session:** Click 'End & Predict' to see if your behavior leads to a conversion!
    """)

    # 1. Persona Selection (Static Data)
    st.sidebar.header("1. User Persona")
    persona = st.sidebar.selectbox("Simulate User Type:", ["Average Shopper", "High-Value Lead", "Window Shopper"])
    
    # Set defaults based on persona
    if persona == "Average Shopper":
        def_age, def_income, def_loyalty = 25, 45000, 200
    elif persona == "High-Value Lead":
        def_age, def_income, def_loyalty = 45, 120000, 2500
    else:
        def_age, def_income, def_loyalty = 19, 20000, 0

    # 2. Session Control
    if not st.session_state.demo_active:
        if st.button("▶️ Start Live Session", type="primary"):
            st.session_state.demo_active = True
            st.session_state.start_time = time.time()
            st.session_state.page_views = 1 # Count landing page
            st.session_state.social_shares_count = 0
            st.rerun()
    
    else:
        # --- The Simulated Store Interface ---
        st.divider()
        st.subheader("🏪 Storefront")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("https://placehold.co/300x200?text=Home+Page", caption="Home Page")
            if st.button("🏠 Go to Home"):
                st.session_state.page_views += 1
                st.toast("Navigated to Home (+1 Page View)")
        
        with col2:
            st.image("https://placehold.co/300x200?text=Products", caption="Product Catalog")
            if st.button("👟 Browse Products"):
                st.session_state.page_views += 1
                st.toast("Viewing Products (+1 Page View)")
        
        with col3:
            st.image("https://placehold.co/300x200?text=Social", caption="Social Media Campaign")
            if st.button("📢 Share to Social"):
                st.session_state.social_shares_count += 1
                st.toast("Shared on Social Media! (+1 Share)")

        # Live Counter Display
        st.divider()
        elapsed = time.time() - st.session_state.start_time
        st.metric("⏱️ Live Session Duration", f"{elapsed:.0f} sec")
        
        # --- END SESSION LOGIC ---
        if st.button("🛑 End Session & Predict", type="primary"):
            if model:
                # 1. Calculate Final Stats
                final_time_mins = (time.time() - st.session_state.start_time) / 60
                # Ensure at least some time is registered
                if final_time_mins < 0.01: final_time_mins = 0.01
                
                final_pages = st.session_state.page_views
                final_shares = st.session_state.social_shares_count
                
                # 2. Construct Dataframe with Persona + Live Data
                # We fill unrelated marketing columns with standard averages to isolate behavior impact
                input_data = pd.DataFrame({
                    'Age': [def_age], 
                    'Gender': ['Male'], # Defaulting for demo simplicity
                    'Income': [def_income], 
                    'CampaignChannel': ['Social Media'], 
                    'CampaignType': ['Conversion'], 
                    'AdSpend': [1000.0], 
                    'ClickThroughRate': [0.1], 
                    'ConversionRate': [0.05], 
                    'WebsiteVisits': [final_pages], # Using current pages as total visits
                    'EmailOpens': [2], 
                    'EmailClicks': [1], 
                    'PreviousPurchases': [1], 
                    'LoyaltyPoints': [def_loyalty],
                    
                    # THE DYNAMIC VARIABLES FROM LIVE SESSION
                    'TimeOnSite': [final_time_mins],
                    'PagesPerVisit': [final_pages],
                    'SocialShares': [final_shares]
                })
                
                # 3. Get Prediction
                pred, proba = predict_conversion(model, input_data)
                
                # 4. Display Results
                st.divider()
                st.subheader("🧠 Prediction Results")
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Time on Site", f"{final_time_mins:.2f} min")
                m2.metric("Pages Visited", final_pages)
                m3.metric("Social Shares", final_shares)

                if pred == 1:
                    st.success(f"## 🎉 Prediction: CONVERT (Prob: {proba:.1%})")
                    st.write(f"**Insight:** Based on the persona (Income: ${def_income}) and behavior, this user is highly likely to purchase.")
                else:
                    st.error(f"## ❄️ Prediction: NO CONVERT (Prob: {proba:.1%})")
                    st.write("**Insight:** This user needs more engagement or time on site to convert.")
            
            # Reset state for next time
            st.session_state.demo_active = False
            st.session_state.page_views = 0
            st.session_state.social_shares_count = 0
            if st.button("Reset Simulator"):
                st.rerun()