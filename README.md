# 🌿 PlantGuard AI

🚀 **[Try PlantGuard AI](https://plantguard-ai-26.streamlit.app/)**

Upload a plant leaf image and explore predictions from a
MobileNetV2-based plant disease classification model.
## AI-Powered Plant Disease Classification

PlantGuard AI is an end-to-end deep learning project that classifies
plant leaf images into 38 categories using a fine-tuned MobileNetV2 model.

The project combines computer vision, transfer learning, model evaluation,
and an interactive Streamlit web application.

**Disclaimer:** PlantGuard AI is an experimental educational project.
Its predictions are not confirmed diagnoses.

---

## 🌱 Features

- Upload plant leaf images in JPG, JPEG, or PNG format.
- Classify images into 38 plant disease and healthy-leaf categories.
- Display the top five predicted categories.
- Visualize prediction scores.
- Explore model performance through an interactive dashboard.
- Use a responsive, nature-inspired Streamlit interface.

## 🧠 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| PyTorch | Deep learning |
| Torchvision | MobileNetV2 and image preprocessing |
| Streamlit | Web application |
| Pillow | Image processing |
| Scikit-learn | Model evaluation |

## 📂 Project Structure

PlantGuard-AI/
    app.py
    plant_disease_model_v21.pth
    requirements.txt
    .gitignore
    README.md

## 📊 Datasets

### PlantVillage

A subset containing 1,900 images across 38 categories was used
for initial model development.

### PlantDoc

Real-world plant leaf photographs were used to evaluate and improve
the model's performance beyond controlled PlantVillage images.

The model was fine-tuned using compatible labeled examples from
PlantDoc alongside PlantVillage training images.

## 🔬 Model Development

### Version 1 — Initial Model

A MobileNetV2 classifier was trained using PlantVillage images.

### Version 2 — Cross-Dataset Fine-Tuning

The model was fine-tuned using a combination of PlantVillage and
PlantDoc images.

### Version 2.1 — Balanced Fine-Tuning

Moderate class balancing was introduced using a weighted random
sampler, followed by additional fine-tuning at a reduced learning rate.

## 📈 Validation Results

| Metric | Version 2 | Version 2.1 |
|---|---:|---:|
| PlantVillage validation accuracy | 86.56% | 86.56% |
| PlantDoc validation accuracy | 37.86% | 41.56% |
| PlantDoc validation macro F1 | 0.346 | 0.375 |
| Combined validation score | 62.21% | 64.06% |

The combined validation score is the unweighted average of
PlantVillage and PlantDoc validation accuracies.

## ⚠️ Limitations

- Performance varies across disease categories.
- Some visually similar diseases are frequently confused.
- PlantDoc validation accuracy remains limited.
- The datasets contain relatively few examples per class.
- PlantDoc test results were explored during development and
  are not treated as an untouched final benchmark.
- Softmax scores are not calibrated probabilities of correctness.

## ⚙️ Installation

1. Clone this GitHub repository.
2. Open the project directory.
3. Install the required libraries:

    pip install -r requirements.txt

4. Run the Streamlit application:

    streamlit run app.py



## 🔮 Future Improvements

- Collect more diverse real-world plant images.
- Improve classification of visually similar diseases.
- Evaluate the model on a new, untouched external dataset.
- Explore additional augmentation techniques.
- Improve model calibration.

## 📌 Project Status

Version 2.1 — Functional Streamlit application.

Developed as a hands-on project to explore deep learning,
computer vision, and real-world model generalization.
