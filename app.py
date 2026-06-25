import streamlit as st
import pandas as pd
import joblib

# =====================================================================
# PART 1 — Load the trained model and the column list
# =====================================================================
# These files were saved from the notebook using joblib.
# The model makes predictions; the column list ensures correct order.
model = joblib.load('rf_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# =====================================================================
# PART 2 — Page title and description
# =====================================================================
st.title("🏦 Bank Term Deposit Predictor")
st.write("Enter customer details to predict subscription likelihood.")

# =====================================================================
# PART 3 — User input widgets
# Each st.something() creates an interactive element on the page.
# =====================================================================

# Slider for age — returns the number the user picks
age = st.slider("Age", min_value=18, max_value=100, value=40)

# Dropdown for job group — returns the selected string
job_group = st.selectbox("Job Category",
                         options=['high (student/retired)',
                                  'medium (admin/management)',
                                  'low (blue-collar/services)'])

# Dropdown for contact type
contact = st.selectbox("Contact Type", options=['cellular', 'telephone'])

# Dropdown for month
month = st.selectbox("Last Contact Month",
                     options=['mar', 'apr', 'may', 'jun', 'jul', 'aug',
                              'sep', 'oct', 'nov', 'dec'])

# Dropdown for previous campaign outcome
poutcome = st.selectbox("Previous Campaign Outcome",
                        options=['failure', 'nonexistent', 'success'])

# Slider for euribor rate (key economic indicator)
euribor = st.slider("Euribor 3-Month Rate (%)",
                    min_value=0.5, max_value=5.5, value=4.0, step=0.1)

# Yes/No for contacted before
contacted = st.radio("Contacted in a previous campaign?", options=['No', 'Yes'])

# =====================================================================
# PART 4 — Build the full 35-feature input row
# We start with DEFAULT values for everything, then overwrite
# the ones the user actually provided.
# =====================================================================

# Start with a dictionary of sensible defaults (from training medians/modes)
input_data = {
    'age': age,                          # from user
    'education': 6,                      # mode (university degree)
    'default': 0,                        # mode (no)
    'housing': 2,                        # mode (yes)
    'loan': 0,                           # mode (no)
    'contact': 1 if contact == 'cellular' else 0,   # from user
    'previous': 0,                       # median
    'emp.var.rate': 1.1,                 # median
    'cons.price.idx': 93.749,            # median
    'cons.conf.idx': -41.8,              # median
    'euribor3m': euribor,                # from user
    'nr.employed': 5191.0,               # median
    'contacted_before': 1 if contacted == 'Yes' else 0,  # from user
    'campaign_binned': 0,                # mode
    'age_binned': 0 if age < 25 else (2 if age >= 65 else 1),  # derived from age
    'economic_index': 0.564,             # median

    # Month one-hot — set all to 0 first, then flip the chosen one
    'month_aug': 0, 'month_dec': 0, 'month_jul': 0, 'month_jun': 0,
    'month_mar': 0, 'month_may': 0, 'month_nov': 0, 'month_oct': 0,
    'month_sep': 0,

    # Day of week one-hot — default to baseline (fri = all zeros)
    'day_of_week_mon': 0, 'day_of_week_thu': 0,
    'day_of_week_tue': 0, 'day_of_week_wed': 0,

    # Marital one-hot — default to baseline (divorced = all zeros), assume married
    'marital_married': 1, 'marital_single': 0,

    # Poutcome one-hot — set based on user choice
    'poutcome_nonexistent': 0, 'poutcome_success': 0,

    # Job group one-hot — set based on user choice
    'job_grouped_low': 0, 'job_grouped_medium': 0,
}

# --- Now overwrite the one-hot columns based on user choices ---

# Month: flip the chosen month to 1 (unless it's 'apr', the baseline)
if month != 'apr':
    input_data[f'month_{month}'] = 1

# Poutcome: flip the chosen one (unless 'failure', the baseline)
if poutcome == 'nonexistent':
    input_data['poutcome_nonexistent'] = 1
elif poutcome == 'success':
    input_data['poutcome_success'] = 1

# Job group: flip the chosen one (unless 'high', the baseline)
if 'low' in job_group:
    input_data['job_grouped_low'] = 1
elif 'medium' in job_group:
    input_data['job_grouped_medium'] = 1

# =====================================================================
# PART 5 — Convert to DataFrame in the correct column order
# The model expects columns in the EXACT order it was trained on.
# =====================================================================
input_df = pd.DataFrame([input_data])[feature_columns]

# =====================================================================
# PART 6 — Predict button + result
# =====================================================================
if st.button("Predict"):
    # Get probability of subscribing (class 1)
    probability = model.predict_proba(input_df)[0][1]

    # Show result with a percentage
    st.subheader(f"Subscription Probability: {probability:.1%}")

    # Color-coded message based on threshold
    if probability >= 0.30:
        st.success("✅ Likely to subscribe — good candidate to call!")
    else:
        st.warning("❌ Unlikely to subscribe — lower priority.")

    # Show a progress bar as a visual gauge
    st.progress(float(probability))
