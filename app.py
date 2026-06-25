import streamlit as st
import pandas as pd
import joblib
 
# =====================================================================
# PART 1 — Load the trained model and column list
# =====================================================================
model = joblib.load('rf_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
 
# =====================================================================
# PART 2 — Page title
# =====================================================================
st.title("🏦 Bank Term Deposit Predictor")
st.write("Enter customer and economic details to predict subscription likelihood.")
 
# =====================================================================
# PART 3 — Economic Climate selector
# This bundles all 6 macro features together so they stay realistic.
# Values come from actual data averages (subscribers vs non-subscribers).
# =====================================================================
st.header("📊 Economic Climate")
 
economy = st.selectbox(
    "Overall Economic Condition",
    options=['Weak Economy', 'Moderate Economy', 'Strong Economy'],
    help="Weak = low rates/shrinking jobs (people seek safe deposits). "
         "Strong = high rates/growing jobs."
)
 
# Map each climate to its real macro values (from your data analysis)
economic_presets = {
    'Weak Economy': {
        'euribor3m': 2.123, 'emp.var.rate': -1.233, 'nr.employed': 5095.116,
        'cons.price.idx': 93.354, 'cons.conf.idx': -39.790, 'economic_index': -0.899
    },
    'Moderate Economy': {
        'euribor3m': 4.857, 'emp.var.rate': 1.100, 'nr.employed': 5191.000,
        'cons.price.idx': 93.749, 'cons.conf.idx': -41.800, 'economic_index': 0.564
    },
    'Strong Economy': {
        'euribor3m': 3.811, 'emp.var.rate': 0.249, 'nr.employed': 5176.167,
        'cons.price.idx': 93.604, 'cons.conf.idx': -40.593, 'economic_index': 0.114
    }
}
macro = economic_presets[economy]
 
# =====================================================================
# PART 4 — Customer details
# =====================================================================
st.header("👤 Customer Details")
 
age = st.slider("Age", min_value=18, max_value=100, value=40)
 
job_group = st.selectbox("Job Category",
                         options=['high (student/retired)',
                                  'medium (admin/management)',
                                  'low (blue-collar/services)'])
 
education = st.selectbox("Education Level",
                         options=['illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
                                  'high.school', 'professional.course',
                                  'university.degree', 'unknown'])
 
default = st.selectbox("Has Credit in Default?",
                       options=['no', 'unknown', 'yes'])
 
# =====================================================================
# PART 5 — Campaign / contact details
# =====================================================================
st.header("📞 Campaign Details")
 
contact = st.selectbox("Contact Type", options=['cellular', 'telephone'])
 
month = st.selectbox("Last Contact Month",
                     options=['mar', 'apr', 'may', 'jun', 'jul', 'aug',
                              'sep', 'oct', 'nov', 'dec'])
 
poutcome = st.selectbox("Previous Campaign Outcome",
                        options=['failure', 'nonexistent', 'success'])
 
contacted = st.radio("Contacted in a previous campaign?", options=['No', 'Yes'])
 
campaign_binned = st.selectbox("Number of Contacts This Campaign",
                               options=['1', '2', '3-5', '>5'])
 
# =====================================================================
# PART 6 — Build the full feature row
# Start with defaults, then overwrite with user inputs
# =====================================================================
 
# Encoding maps
education_map = {'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3,
                 'high.school': 4, 'professional.course': 5,
                 'university.degree': 6, 'unknown': 7}
default_map = {'no': 0, 'unknown': 1, 'yes': 2}
campaign_map = {'1': 0, '2': 1, '3-5': 2, '>5': 3}
 
input_data = {
    'age': age,
    'education': education_map[education],
    'default': default_map[default],
    'housing': 2,                        # default mode (yes)
    'loan': 0,                           # default mode (no)
    'contact': 1 if contact == 'cellular' else 0,
    'previous': 0,                       # default median
    'emp.var.rate': macro['emp.var.rate'],      # from economy preset
    'cons.price.idx': macro['cons.price.idx'],  # from economy preset
    'cons.conf.idx': macro['cons.conf.idx'],    # from economy preset
    'euribor3m': macro['euribor3m'],            # from economy preset
    'nr.employed': macro['nr.employed'],        # from economy preset
    'contacted_before': 1 if contacted == 'Yes' else 0,
    'campaign_binned': campaign_map[campaign_binned],
    'age_binned': 0 if age < 25 else (2 if age >= 65 else 1),
    'economic_index': macro['economic_index'],  # from economy preset
 
    # Month one-hot (apr is baseline)
    'month_aug': 0, 'month_dec': 0, 'month_jul': 0, 'month_jun': 0,
    'month_mar': 0, 'month_may': 0, 'month_nov': 0, 'month_oct': 0,
    'month_sep': 0,
 
    # Day of week one-hot (fri baseline) — kept at default
    'day_of_week_mon': 0, 'day_of_week_thu': 0,
    'day_of_week_tue': 0, 'day_of_week_wed': 0,
 
    # Marital one-hot (divorced baseline) — default married
    'marital_married': 1, 'marital_single': 0,
 
    # Poutcome one-hot (failure baseline)
    'poutcome_nonexistent': 0, 'poutcome_success': 0,
 
    # Job group one-hot (high baseline)
    'job_grouped_low': 0, 'job_grouped_medium': 0,
}
 
# Overwrite one-hot columns based on user choices
if month != 'apr':
    input_data[f'month_{month}'] = 1
 
if poutcome == 'nonexistent':
    input_data['poutcome_nonexistent'] = 1
elif poutcome == 'success':
    input_data['poutcome_success'] = 1
 
if 'low' in job_group:
    input_data['job_grouped_low'] = 1
elif 'medium' in job_group:
    input_data['job_grouped_medium'] = 1
 
# =====================================================================
# PART 7 — Arrange in correct column order
# =====================================================================
input_df = pd.DataFrame([input_data])[feature_columns]
 
# =====================================================================
# PART 8 — Predict button + result
# =====================================================================
st.header("🎯 Prediction")
 
if st.button("Predict"):
    probability = model.predict_proba(input_df)[0][1]
 
    st.subheader(f"Subscription Probability: {probability:.1%}")
 
    if probability >= 0.30:
        st.success("✅ Likely to subscribe — good candidate to call!")
    else:
        st.warning("❌ Unlikely to subscribe — lower priority.")
 
    st.progress(float(probability))
 
    st.caption("Threshold of 30% used (optimized for catching subscribers). "
               "Note: economic climate is the strongest driver of predictions.")
