import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(layout="wide")
st.title("📊 SpikeSense - Real-time Player Churn & Spike Detection")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("sample_churn_3000.csv")

# ✅ Added check to ensure 'event_date' exists
if 'event_date' not in df.columns:
    st.error("The dataset must contain a column named 'event_date'. Please upload a valid file.")
    st.stop()

# Ensure date column is datetime
df["event_date"] = pd.to_datetime(df["event_date"])

# Display DataFrame
st.subheader("📋 Sample Data")
st.dataframe(df.head())

# Visualize churn over time using a line graph
st.subheader("📈 Churn Trend Over Time")
churn_over_time = df.groupby("event_date")["is_churn"].mean()
fig, ax = plt.subplots()
ax.plot(churn_over_time.index, churn_over_time.values, color="red", linewidth=2)
ax.set_title("Daily Churn Rate Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Churn Rate")
st.pyplot(fig)

# ML Model
st.subheader("🤖 Churn Detection Model Training")
X = df[["play_time", "matches_played", "avg_score"]]
y = df["is_churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

st.success(f"Model Accuracy: {accuracy:.2f}")

# Classification report
st.text("Classification Report:")
y_pred = model.predict(X_test)
st.code(classification_report(y_test, y_pred))

# 🎯 Model Evaluation
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
st.subheader("📋 Classification Report")
st.dataframe(pd.DataFrame(report).transpose())

# 🔍 Feature Importance
st.subheader("🌟 Feature Importance")
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots()
# ✅ Fixed seaborn deprecation warning by adding hue & legend
sns.barplot(x="Importance", y="Feature", data=importance_df, hue="Feature", palette="viridis", ax=ax2, legend=False)
ax2.set_title("Feature Importance from Random Forest")
st.pyplot(fig2)

# 📅 Highlight Top Churn Spike Dates
st.subheader("🚨 Top 3 Churn Spike Dates")
churn_over_time_sorted = churn_over_time.sort_values(ascending=False).head(3)
st.table(churn_over_time_sorted)

# 🧠 Sidebar Branding
with st.sidebar:
    st.markdown("### 🚀 SpikeSense Dashboard")
    st.markdown("Built for AI Hackathon 2025 by Team Spike Sense")

# 📌 Footer Note
st.markdown("---")
st.markdown("<center><small>Developed with ❤️ for AI Hackathon Finals</small></center>", unsafe_allow_html=True)
