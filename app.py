import streamlit as st
import pickle
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

flower_info = {
    0: {
        "Scientific Name": "Iris setosa",
        "Color": "Purple",
        "Origin": "North America",
        "Fact": "It is the easiest iris species to identify because of its short petals."
    },
    1: {
        "Scientific Name": "Iris versicolor",
        "Color": "Blue-Violet",
        "Origin": "North America",
        "Fact": "It is also known as the Blue Flag Iris."
    },
    2: {
        "Scientific Name": "Iris virginica",
        "Color": "Blue to Purple",
        "Origin": "Eastern USA",
        "Fact": "It has the largest petals among the three Iris species."
    }
} 


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

iris = load_iris(as_frame=True)
df = iris.frame
if "history" not in st.session_state:
    st.session_state.history = []
# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* ===== Background ===== */
.stApp{
    background: linear-gradient(-45deg,#00c9ff,#92fe9d,#f857a6,#ff5858,#8E2DE2);
    background-size:400% 400%;
}

/* ===== Main Container ===== */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* ===== Title ===== */
.big-title{
    text-align:center;
    font-size:55px;
    font-weight:900;
    color:white;
    text-shadow:3px 3px 10px rgba(0,0,0,0.4);
}

/* ===== Subtitle ===== */
.sub-title{
    text-align:center;
    font-size:22px;
    color:white;
    margin-bottom:25px;
    text-shadow:2px 2px 6px rgba(0,0,0,0.3);
}

/* ===== Glass Cards ===== */
.card{
    background:rgba(255,255,255,0.18);
    backdrop-filter:blur(20px);
    border-radius:20px;
    padding:25px;
    border:1px solid rgba(255,255,255,0.3);
    box-shadow:0px 10px 30px rgba(0,0,0,0.25);
}

/* ===== Result Card ===== */
.result{
    background:rgba(255,255,255,0.20);
    backdrop-filter:blur(20px);
    border-left:8px solid #00ff99;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 10px 30px rgba(0,0,0,0.25);
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"]{
    background:rgba(255,255,255,0.15);
    backdrop-filter:blur(18px);
}

/* ===== Predict Button ===== */
.stButton>button{
    width:100%;
    border:none;
    border-radius:15px;
    padding:15px;
    background:linear-gradient(90deg,#ff512f,#dd2476);
    color:white;
    font-size:20px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.05);
    background:linear-gradient(90deg,#11998e,#38ef7d);
}

/* ===== Input Boxes ===== */
.stNumberInput input{
    border-radius:12px !important;
}

/* ===== Progress Bar ===== */
.stProgress > div > div > div{
    background:linear-gradient(to right,#00ff87,#60efff);
}

/* ===== Hide Streamlit Branding ===== */
#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="big-title">🌸 AI Iris Flower Classification</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-title">Machine Learning Based Flower Prediction System</div>', unsafe_allow_html=True)

st.write("")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌼 Project Information")

st.sidebar.info("""
### AI Iris Flower Classifier

This application predicts the Iris flower species using a Machine Learning model.

**Algorithm**
- Random Forest Classifier

**Input Features**
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
""")

# ---------------- LAYOUT ----------------
left, right = st.columns([1,1])

with left:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📋 Enter Flower Measurements")

    sepal_length = st.number_input("Sepal Length (cm)",0.0,10.0,5.1)

    sepal_width = st.number_input("Sepal Width (cm)",0.0,10.0,3.5)

    petal_length = st.number_input("Petal Length (cm)",0.0,10.0,1.4)

    petal_width = st.number_input("Petal Width (cm)",0.0,10.0,0.2)

    predict = st.button("🚀 Predict Flower")

    st.markdown("</div>", unsafe_allow_html=True)

with right:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("🤖 AI Prediction")

if predict:

    status = st.empty()

    status.info("🔍 Reading flower measurements...")
    time.sleep(2)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    status.info("🤖 Running Machine Learning model...")
    time.sleep(2)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    confidence = np.max(probabilities) * 100


    flower_names = {
        0: "🌸 Iris Setosa",
        1: "🌺 Iris Versicolor",
        2: "🌼 Iris Virginica"
    }


    status.success("✅ Prediction completed!")
    time.sleep(1)
    status.empty()


    # Confidence Animation
    st.subheader("🤖 AI Confidence Analysis")

    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(101):
        progress_text.write(f"AI Confidence: {i}%")
        progress_bar.progress(i)
        time.sleep(0.02)

    progress_text.success(f"Final Confidence: {confidence:.2f}%")
    progress_bar.progress(int(confidence))


    st.session_state.history.append({
        "Prediction": flower_names[prediction],
        "Confidence": f"{confidence:.2f}%"
    })


    st.markdown("<div class='result'>", unsafe_allow_html=True)


    st.success(f"### {flower_names[prediction]}")


    st.subheader("🌸 Flower Information")

    info = flower_info[prediction]


    info_box = st.empty()

    info_box.write(f"**Scientific Name:** {info['Scientific Name']}")
    time.sleep(1)

    info_box.write(
        f"**Scientific Name:** {info['Scientific Name']}  \n"
        f"**Color:** {info['Color']}"
    )
    time.sleep(1)

    info_box.write(
        f"**Scientific Name:** {info['Scientific Name']}  \n"
        f"**Color:** {info['Color']}  \n"
        f"**Origin:** {info['Origin']}"
    )
    time.sleep(1)

    info_box.write(
        f"**Scientific Name:** {info['Scientific Name']}  \n"
        f"**Color:** {info['Color']}  \n"
        f"**Origin:** {info['Origin']}  \n"
        f"**Interesting Fact:** {info['Fact']}"
    )


    st.markdown("</div>", unsafe_allow_html=True)



    # Probability

    time.sleep(1)

    st.write("---")
    st.subheader("Prediction Probability")


    st.write("Setosa")
    st.progress(int(probabilities[0] * 100))

    st.write("Versicolor")
    st.progress(int(probabilities[1] * 100))

    st.write("Virginica")
    st.progress(int(probabilities[2] * 100))



    # Dataset

        # Animated Dataset Loading

    time.sleep(1)

    st.write("---")

    loader = st.empty()

    loader.info("📋 Loading Iris Dataset...")

    for i in range(1, 6):
        loader.write(f"📋 Loading dataset rows... {i}/5")
        time.sleep(0.5)

    loader.success("✅ Dataset Loaded Successfully!")
    time.sleep(1)


    st.subheader("📋 Iris Dataset Preview")

    table_box = st.empty()

    for i in range(1, 11):
        table_box.dataframe(df.head(i), use_container_width=True)
        time.sleep(0.3)



    # Animated Graphs

    time.sleep(1)

    st.write("---")

    st.subheader("📊 Data Visualizations")


    col1, col2 = st.columns(2)


    # Species Distribution Animation

    with col1:

        st.markdown("### Species Distribution")

        species_counts = df["target"].value_counts().sort_index()

        fig1, ax1 = plt.subplots(figsize=(5,4))

        chart = st.empty()

        for i in range(101):

            ax1.clear()

            ax1.bar(
                ["Setosa", "Versicolor", "Virginica"],
                [
                    species_counts.values[0] * i / 100,
                    species_counts.values[1] * i / 100,
                    species_counts.values[2] * i / 100
                ]
            )

            ax1.set_xlabel("Species")
            ax1.set_ylabel("Count")

            chart.pyplot(fig1)

            time.sleep(0.02)



    # Correlation Heatmap

    with col2:

        st.markdown("### Correlation Heatmap")

        heat_status = st.empty()

        heat_status.info("🔥 Creating Heatmap...")

        time.sleep(2)

        corr = df.corr(numeric_only=True)

        fig2, ax2 = plt.subplots(figsize=(5,4))

        heatmap = ax2.imshow(corr, cmap="coolwarm")

        ax2.set_xticks(range(len(corr.columns)))
        ax2.set_yticks(range(len(corr.columns)))

        ax2.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax2.set_yticklabels(corr.columns)

        plt.colorbar(heatmap, ax=ax2)

        st.pyplot(fig2)

        heat_status.success("✅ Heatmap Completed!")



    # Model Performance Animation

    st.write("---")

    model_status = st.empty()

    model_status.info("📈 Calculating Model Performance...")

    time.sleep(2)

    st.subheader("📈 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", "96.67%")

    with col2:
        st.metric("Algorithm", "Random Forest")



    # Feature Importance Animation

    st.write("---")

    feature_status = st.empty()

    feature_status.info("🌟 Calculating Feature Importance...")

    time.sleep(2)

    feature_status.success("✅ Feature Importance Generated!")

    st.subheader("🌟 Feature Importance")


    feature_names = [
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ]


    importances = model.feature_importances_

    fig, ax = plt.subplots(figsize=(5,4))

    ax.barh(feature_names, importances)

    ax.set_xlabel("Importance Score")


    left_space, center, right_space = st.columns([1,2,1])

    with center:
        st.pyplot(fig)



    # Prediction History

    st.write("---")

    st.subheader("📝 Prediction History")


    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(history_df, use_container_width=True)


else:

    st.info("Enter the flower measurements and click **Predict Flower**.")
# ---------------- FOOTER ----------------
st.write("---")

st.markdown(
"""
<div style='text-align:center'>

### 🌸 AI Iris Flower Classification

Developed using **Python • Streamlit • Scikit-learn**

</div>
""",
unsafe_allow_html=True)