# sonar_mine_prediction_app.py

import streamlit as st
import numpy as np
import pickle

def load_models():
    try:
        lr_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
        rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
        return lr_model, rf_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def main():
    st.title("🔍 Sonar Signal - Mine or Rock Detector")
    st.markdown("This model predicts whether a sonar signal is bouncing off a mine or a rock.")
    st.markdown("Enter 60 values representing sonar signal strength (0.0 to 1.0):")
    
    # Load models
    lr_model, rf_model = load_models()
    
    if lr_model is None or rf_model is None:
        st.warning("Could not load models. Please check your model files.")
        return
    
    # Input method selection
    input_method = st.radio("Choose input method:", 
                           ["Manual Input", "Paste CSV Values", "Use Sample Data"])
    
    input_data = []
    
    if input_method == "Manual Input":
        # Create columns for better input layout
        col1, col2, col3 = st.columns(3)
        
        # User input fields
        for i in range(60):
            if i < 20:
                with col1:
                    value = st.number_input(f"Feature {i+1}", value=0.0, min_value=0.0, max_value=1.0, step=0.01, format="%.4f", key=f"feat_{i}")
                    input_data.append(value)
            elif i < 40:
                with col2:
                    value = st.number_input(f"Feature {i+1}", value=0.0, min_value=0.0, max_value=1.0, step=0.01, format="%.4f", key=f"feat_{i}")
                    input_data.append(value)
            else:
                with col3:
                    value = st.number_input(f"Feature {i+1}", value=0.0, min_value=0.0, max_value=1.0, step=0.01, format="%.4f", key=f"feat_{i}")
                    input_data.append(value)
    
    elif input_method == "Paste CSV Values":
        csv_input = st.text_area("Paste 60 comma-separated values (0.0 to 1.0):", 
                               placeholder="0.0164, 0.0627, 0.0738, ...", height=150)
        if csv_input:
            try:
                # Process the input string to extract values
                values = csv_input.replace("\n", "").replace(" ", "").split(",")
                # Convert to float and validate
                values = [float(v) for v in values if v]
                if len(values) != 60:
                    st.error(f"Expected 60 values, but got {len(values)}. Please check your input.")
                else:
                    input_data = values
                    st.success(f"Successfully parsed 60 values.")
            except Exception as e:
                st.error(f"Error parsing input: {e}")
    
    else:  # Use Sample Data
        # Sample test data
        input_data = [0.0164, 0.0627, 0.0738, 0.0608, 0.0233, 0.1048, 0.1338, 0.0644,
                    0.1522, 0.0780, 0.1791, 0.2681, 0.1788, 0.1039, 0.1980, 0.3234,
                    0.3748, 0.2586, 0.3680, 0.3508, 0.5606, 0.5231, 0.5469, 0.6954,
                    0.6352, 0.6757, 0.8499, 0.8025, 0.6563, 0.8591, 0.6655, 0.5369,
                    0.3118, 0.3763, 0.2801, 0.0875, 0.3319, 0.4237, 0.1801, 0.3743,
                    0.4627, 0.1614, 0.2494, 0.3202, 0.2265, 0.1146, 0.0476, 0.0943,
                    0.0824, 0.0171, 0.0244, 0.0258, 0.0143, 0.0226, 0.0187, 0.0185,
                    0.0110, 0.0094, 0.0078, 0.0112]
        st.info("Using sample test data (known to be a mine)")
        
        # Display sample data in readable format
        with st.expander("View sample data"):
            st.write(np.array(input_data).reshape(10, 6))

    # Predict button
    if st.button("Predict"):
        if len(input_data) != 60:
            st.error("Please provide all 60 feature values before prediction.")
            return
            
        with st.spinner("Analyzing sonar signal..."):
            try:
                input_np_array = np.asarray(input_data).reshape(1, -1)
                
                # Make predictions
                prediction_lr = lr_model.predict(input_np_array)
                result_lr = "💣 Mine" if prediction_lr[0] == 1 else "⛰️ Rock"
                
                prediction_rf = rf_model.predict(input_np_array)
                result_rf = "💣 Mine" if prediction_rf[0] == 1 else "⛰️ Rock"
                
                # Show prediction probabilities
                proba_lr = lr_model.predict_proba(input_np_array)[0]
                proba_rf = rf_model.predict_proba(input_np_array)[0]
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Logistic Regression Prediction", result_lr)
                    st.write(f"**Confidence:** Rock: {proba_lr[0]:.2%}, Mine: {proba_lr[1]:.2%}")
                    
                with col2:
                    st.metric("Random Forest Prediction", result_rf)
                    st.write(f"**Confidence:** Rock: {proba_rf[0]:.2%}, Mine: {proba_rf[1]:.2%}")
                
                # Final verdict based on both models
                if prediction_lr[0] == prediction_rf[0]:
                    verdict = "💣 Mine" if prediction_lr[0] == 1 else "⛰️ Rock"
                    st.success(f"Both models agree: This is a {verdict}")
                else:
                    st.warning("Models disagree. Consider the confidence levels.")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.write("Please check your input data and try again.")

if __name__ == "__main__":
    main()