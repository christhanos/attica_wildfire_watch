import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# Import your custom model architecture
from model import WildfireClassifier

# Set Page Config (This makes it look professional)
st.set_page_config(
    page_title="Attica Wildfire Watch",
    page_icon="🔥",
    layout="centered"
)

# --- 1. Header & Description ---
st.title("🔥 Attica Wildfire Watch")
st.subheader("Satellite-Based AI Detection System")
st.markdown("""
This application uses a Deep Learning (ResNet18) model trained on thousands of satellite images to detect wildfire smoke plumes.
**Accuracy on unseen data: 96.21%**
""")

# --- 2. Load the Model Brain ---
@st.cache_resource # This prevents the app from re-loading the model every time you click a button
def load_trained_model():
    model = WildfireClassifier()
    # Load the weights into the architecture
    model.load_state_dict(torch.load("wildfire_model.pth", map_location=torch.device('cpu')))
    model.eval() # Set to evaluation mode
    return model

model = load_trained_model()

# --- 3. Image Preprocessing Pipeline ---
# This MUST match the transformations we used in train.py and test.py
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. The Sidebar (Optional Info) ---
st.sidebar.info("Upload a satellite image (JPG/PNG) to check for wildfire activity in the Attica region.")
st.sidebar.write("Developed for University Project")

# --- 5. Main Interaction: File Uploader ---
uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Satellite View", use_container_width=True)
    
    st.write("---")
    st.write("🔍 **Analyzing Pixels...**")

    # Prepare image for the model
    input_tensor = preprocess(image).unsqueeze(0) # Add a 'batch' dimension (1, 3, 224, 224)

    # Perform Prediction
    with torch.no_grad():
        output = model(input_tensor)
        # Apply Sigmoid to get probability
        probability = torch.sigmoid(output).item()

    # --- 6. The Verdict ---
    if probability >= 0.5:
        # It's a fire!
        confidence = probability * 100
        st.error(f"🚨 **DANGER DETECTED**")
        st.metric(label="Wildfire Confidence", value=f"{confidence:.2f}%")
        st.warning("Potential wildfire smoke plume identified in the image.")
    else:
        # It's safe!
        confidence = (1 - probability) * 100
        st.success(f"✅ **NO WILDFIRE DETECTED**")
        st.metric(label="Safe Confidence", value=f"{confidence:.2f}%")
        st.info("The area appears to be clear of smoke activity.")

# Footer
st.markdown("---")
st.caption("Disclaimer: This is an AI prototype for educational purposes and should not be used as a primary emergency system.")