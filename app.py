
import torch
import streamlit as st
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PlantGuard AI | Smart Plant Health",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "plant_disease_model_v21.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    class_names = checkpoint["class_names"]

    model = mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, class_names


model, class_names = load_model()


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #f1f8f3 0%,
        #e4f3e8 50%,
        #f9fcf9 100%
    );
    color: #173b29;
}

[data-testid="stSidebar"] {
    background: #123524;
}

[data-testid="stSidebar"] * {
    color: #f2fff5;
}

h1, h2, h3 {
    color: #16452d;
}

.hero {
    padding: 40px;
    border-radius: 24px;
    background: linear-gradient(
        120deg,
        #14532d,
        #16803c
    );
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 12px 30px rgba(16, 80, 40, 0.15);
}

.hero h1 {
    color: white;
    font-size: 44px;
    margin-bottom: 12px;
}

.hero p {
    color: #e1f7e8;
    font-size: 18px;
}

.section-card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(20, 80, 40, 0.07);
    margin-bottom: 18px;
}

.result-card {
    background: linear-gradient(
        120deg,
        #d8f5df,
        #f4fff6
    );
    border-left: 6px solid #15803d;
    padding: 24px;
    border-radius: 16px;
    margin-top: 15px;
    margin-bottom: 20px;
}

.result-card h2 {
    margin-bottom: 8px;
}

.stButton > button {
    background: linear-gradient(
        90deg,
        #15803d,
        #16a34a
    );
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 28px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #166534;
    color: white;
}

[data-testid="stFileUploader"] {
    border-radius: 16px;
}

[data-testid="stMetric"] {
    background: white;
    padding: 16px;
    border-radius: 14px;
    border: 1px solid #d6eadc;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🌿 PlantGuard AI")

    st.caption("Your AI-powered plant health companion")

    st.divider()

    st.markdown("### 🧭 Navigation")

    page = st.radio(
        "Explore",
        [
            "🏠 Home",
            "🔬 Disease Detection",
            "📊 Model Insights",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 🧠 Model Information")

    st.write("**Architecture:** MobileNetV2")
    st.write("**Classes:** 38")
    st.write("**Version:** 2.1")

    st.divider()

    st.caption(
        "Built with PyTorch and Streamlit."
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>🌿 PlantGuard AI</h1>
        <p>
            Smarter plant care starts with a single leaf.
            Explore AI-powered plant disease classification
            through an interactive experience.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🌱 Discover PlantGuard AI")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>📸 Upload</h3>
            <p>Choose a clear photograph of a plant leaf.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>🧠 Analyze</h3>
            <p>Let our MobileNetV2 model process the image.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="section-card">
            <h3>📊 Explore</h3>
            <p>View predicted categories and model scores.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "👈 Select Disease Detection from the sidebar "
        "to analyze your first leaf."
    )


# --------------------------------------------------
# DISEASE DETECTION PAGE
# --------------------------------------------------

elif page == "🔬 Disease Detection":

    st.title("🔬 Plant Disease Detection")

    st.write(
        "Upload a clear leaf photograph to explore "
        "the model's predicted categories."
    )

    left, right = st.columns([1, 1], gap="large")

    with left:

        st.subheader("📸 Upload Leaf Image")

        uploaded_file = st.file_uploader(
            "Choose a JPG or PNG image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption="Your uploaded leaf",
                use_container_width=True
            )

            analyze = st.button(
                "🔍 Analyze Leaf",
                type="primary",
                use_container_width=True
            )

        else:
            analyze = False

            st.info(
                "Upload an image to begin analysis."
            )

    with right:

        st.subheader("🧠 AI Analysis")

        if analyze:

            with st.spinner("Analyzing your leaf..."):

                input_tensor = preprocess(
                    image
                ).unsqueeze(0).to(DEVICE)

                with torch.inference_mode():

                    logits = model(input_tensor)

                    probabilities = torch.softmax(
                        logits,
                        dim=1
                    )[0]

                scores, indices = torch.topk(
                    probabilities,
                    k=min(5, len(class_names))
                )

            top_class = class_names[
                indices[0].item()
            ]

            top_score = scores[0].item()

            friendly_name = top_class.replace(
                "___", " — "
            ).replace("_", " ")

            st.markdown(
                f"""
                <div class="result-card">
                    <h3>🌱 Top Predicted Category</h3>
                    <h2>{friendly_name}</h2>
                    <p>
                        Model score:
                        <strong>{top_score:.1%}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if top_score < 0.50:

                st.warning(
                    "The model has a low top prediction score. "
                    "Try a clearer image. This is not a "
                    "confirmed diagnosis."
                )

            else:

                st.info(
                    "This result is an experimental prediction, "
                    "not a confirmed plant disease diagnosis."
                )

            st.markdown("### 📊 Top 5 Predictions")

            for score, idx in zip(
                scores,
                indices
            ):

                label = class_names[
                    idx.item()
                ]

                label = label.replace(
                    "___", " — "
                ).replace("_", " ")

                value = score.item()

                st.write(
                    f"**{label}** — {value:.1%}"
                )

                st.progress(value)

            st.caption(
                "Scores are softmax outputs, not calibrated "
                "probabilities of correctness."
            )

        else:

            st.markdown("""
            <div class="section-card">
                <h3>🌿 Ready to analyze</h3>
                <p>
                    Upload an image and click Analyze Leaf
                    to view AI predictions here.
                </p>
            </div>
            """, unsafe_allow_html=True)


# --------------------------------------------------
# MODEL INSIGHTS PAGE
# --------------------------------------------------

elif page == "📊 Model Insights":

    st.title("📊 Model Insights")

    st.write(
        "Explore the performance of PlantGuard AI "
        "Version 2.1 on validation datasets."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "PlantVillage Validation",
        "86.56%"
    )

    col2.metric(
        "PlantDoc Validation",
        "41.56%"
    )

    col3.metric(
        "PlantDoc Macro F1",
        "0.375"
    )

    st.markdown("### 🧪 Model Development")

    st.markdown("""
    | Version | Main improvement |
    |---|---|
    | Version 1 | Initial PlantVillage training |
    | Version 2 | Fine-tuning with PlantDoc images |
    | Version 2.1 | Balanced sampling and fine-tuning |
    """)

    st.warning(
        "Validation results are based on limited datasets. "
        "Performance on new photographs may differ."
    )


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "ℹ️ About":

    st.title("ℹ️ About PlantGuard AI")

    st.markdown("""
    **PlantGuard AI** is an experimental deep learning
    application for plant leaf image classification.

    ### 🧠 Technology

    - Python
    - PyTorch
    - MobileNetV2
    - Streamlit

    ### 🌿 Training Data

    The model was developed using subsets of
    PlantVillage and PlantDoc.

    ### 🎯 Project Goal

    Explore how transfer learning and domain adaptation
    affect plant disease classification across different
    image datasets.
    """)

    st.warning(
        "PlantGuard AI is an educational project. "
        "It should not be used as the sole basis "
        "for plant treatment decisions."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "🌿 PlantGuard AI | Experimental AI-powered "
    "Plant Disease Classification"
)
