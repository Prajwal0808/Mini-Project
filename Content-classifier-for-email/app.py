import streamlit as st
import pickle

# ==============================
# 💾 Load Model and Vectorizer
# ==============================
svm_model = pickle.load(open("svm_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ==============================
# 🎨 Page Configuration
# ==============================
st.set_page_config(
    page_title="Content Classifier for Email",
    page_icon="📩",
    layout="centered",
)

# ==============================
# 🌌 Custom Dark Gradient & Styling
# ==============================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
.stTextArea textarea {
    border-radius: 10px;
    border: 2px solid #00bfff;
    background-color: #1e1e1e;
    color: white !important;
    font-size: 1.1em;
    caret-color: white !important; /* ✅ makes cursor visible */
}
.stButton > button {
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    color: white;
    border-radius: 10px;
    font-size: 1.1em;
    transition: 0.3s;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0052cc, #0088cc);
}
.success-box {
    background-color: rgba(40, 167, 69, 0.2);
    color: #a8ffb5;
    border: 1px solid #28a745;
    border-radius: 10px;
    padding: 15px;
    font-size: 1.1em;
    text-align: center;
}
.error-box {
    background-color: rgba(220, 53, 69, 0.2);
    color: #ffb3b3;
    border: 1px solid #dc3545;
    border-radius: 10px;
    padding: 15px;
    font-size: 1.1em;
    text-align: center;
}
hr {
    border: 1px solid #00c6ff;
}
a {
    color: #00c6ff !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ==============================
# 🧠 App Header
# ==============================
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>📬 Content-Classifier-For-Email</h1>
        <p style='font-size:18px; color:#dcdcdc;'>Powered by <b>Support Vector Machine (SVM)</b> 🧠<br>
        Enter a message below to check if it's <b>Spam</b> or <b>Ham</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================
# ✉️ Input Area
# ==============================
user_input = st.text_area(
    "💬 Enter your message here:",
    height=120,
    placeholder="Type your message...",
)

# ==============================
# 🚀 Prediction Section
# ==============================
if st.button("🔍 Check Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = svm_model.predict(input_vector)[0]

        if prediction == 0:
            st.markdown(
                "<div class='success-box'>✅ This message is classified as <b>HAM</b> (Not Spam).</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='error-box'>🚫 This message is classified as <b>SPAM / यह एक धोखा है</b>.</div>",
                unsafe_allow_html=True,
            )

# ==============================
# 📘 Footer
# ==============================
st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 15px; color:#cccccc;'>
        Devloped ❤️ by <b>Prajwal S, Wassy editz, Prajwal k</b> using Streamlit & SVM<br>
    </div>
    """,
    unsafe_allow_html=True,
)
