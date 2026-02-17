import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="AI Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# ------------------------------------------------
# DARK PROFESSIONAL CSS
# ------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0B0F19;
}
.main {
    background-color: #0B0F19;
}
h1 {
    text-align: center;
    color: #00FF9F;
    font-size: 45px;
}
h3 {
    color: #A0AEC0;
    text-align: center;
}
.sidebar .sidebar-content {
    background-color: #111827;
}
.stButton>button {
    background-color: #00FF9F;
    color: black;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    height: 55px;
    width: 100%;
    box-shadow: 0px 0px 15px #00FF9F;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: black;
    color: #00FF9F;
    border: 2px solid #00FF9F;
    box-shadow: 0px 0px 25px #00FF9F;
}
.metric-box {
    background-color: #111827;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px #1f2937;
}
.footer {
    text-align: center;
    padding-top: 30px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE SECTION
# ------------------------------------------------
st.markdown("<h1>🏦 AI Loan Approval System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Smart Decision Tree Powered Banking Intelligence</h3>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------
# SAMPLE DATA (Training Model)
# ------------------------------------------------
data = {
    "Age": [25, 40, 35, 50, 28, 45],
    "Income": [30000, 80000, 50000, 90000, 40000, 75000],
    "Credit_Score": [650, 720, 680, 750, 660, 710],
    "Loan_Amount": [200000, 500000, 300000, 600000, 250000, 450000],
    "Employment_Type": ["Salaried", "Self-Employed", "Salaried", "Business", "Salaried", "Self-Employed"],
    "Dependents": [1, 2, 3, 2, 0, 1],
    "Loan_Status": [0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

le = LabelEncoder()
df["Employment_Type"] = le.fit_transform(df["Employment_Type"])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ------------------------------------------------
# SIDEBAR INPUT PANEL
# ------------------------------------------------
st.sidebar.markdown("## 📋 Applicant Information")

age = st.sidebar.slider("Age", 18, 60, 30)
income = st.sidebar.number_input("Income (₹)", 20000, 200000, 50000)
credit = st.sidebar.slider("Credit Score", 300, 900, 700)
loan_amount = st.sidebar.number_input("Loan Amount (₹)", 100000, 1000000, 300000)
employment = st.sidebar.selectbox(
    "Employment Type",
    ["Business", "Salaried", "Self-Employed"]
)
dependents = st.sidebar.slider("Dependents", 0, 5, 1)

employment_encoded = le.transform([employment])[0]

# ------------------------------------------------
# MAIN DASHBOARD
# ------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-box">
    <h3>📊 Model Accuracy</h3>
    <h2 style="color:#00FF9F;">{round(accuracy*100,2)}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
    <h3>💰 Requested Loan</h3>
    <h2 style="color:#00FF9F;">₹ {loan_amount}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
    <h3>📈 Credit Score</h3>
    <h2 style="color:#00FF9F;">{credit}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------
if st.button("🚀 Analyze Loan Application"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "Credit_Score": [credit],
        "Loan_Amount": [loan_amount],
        "Employment_Type": [employment_encoded],
        "Dependents": [dependents]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Congratulations! Loan Approved")
        st.balloons()
    else:
        st.error("❌ Sorry, Loan Rejected. Please Improve Credit Profile")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("""
<div class="footer">
AI Banking System • Powered by Decision Tree • Built with Streamlit
</div>
""", unsafe_allow_html=True)
