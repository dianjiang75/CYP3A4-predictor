# app.py

# --------------------------------------------------------------------------
# Step 1: Import all necessary libraries
# --------------------------------------------------------------------------
import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


# --------------------------------------------------------------------------
# Step 2: Copy the feature generation function from your notebook
# --------------------------------------------------------------------------
def generate_features(smiles_string):
    """
    Generates a Morgan fingerprint from a single SMILES string.
    Returns None if the SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    # Use the same parameters as in your notebook
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fingerprint).reshape(1, -1)


# --------------------------------------------------------------------------
# Step 3: Load the trained model
# --------------------------------------------------------------------------
# Use a caching decorator to load the model only once
@st.cache_resource
def load_model():
    """Loads the saved model file."""
    try:
        model = joblib.load('cyp3a4_model.joblib')
        return model
    except FileNotFoundError:
        return None


model = load_model()

# --------------------------------------------------------------------------
# Step 4: Build the Streamlit User Interface
# --------------------------------------------------------------------------
st.title('CYP3A4 Substrate Prediction Tool 🧪')
st.write("Enter a compound's SMILES string to predict if it is a CYP3A4 substrate.")

# User input text box
smiles_input = st.text_input('SMILES string input:', 'CCO')  # 'CCO' is an example for ethanol

# Prediction button
if st.button('Predict'):
    if model is not None and smiles_input:
        # ------------------------------------------------------------------
        # Step 5: The Prediction Logic
        # ------------------------------------------------------------------
        # Generate features for the input SMILES
        features = generate_features(smiles_input)

        if features is not None:
            # Predict the probability
            probability = model.predict_proba(features)[0, 1]

            # Apply your chosen threshold of 0.25
            chosen_threshold = 0.25

            # Display the verdict
            if probability >= chosen_threshold:
                st.error(f'Prediction: SUBSTRATE (Probability: {probability:.2f})')
                st.warning('This compound should be flagged for experimental testing.')
            else:
                st.success(f'Prediction: NOT a substrate (Probability: {probability:.2f})')
        else:
            st.error("Invalid SMILES string. Please check the input.")
    elif model is None:
        st.error("Model file not found. Make sure 'cyp3a4_model.joblib' is in the same folder.")