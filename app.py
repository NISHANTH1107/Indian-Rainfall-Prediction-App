import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.title('India Rainfall Prediction ☔')

# Load the list of states
df = pd.read_csv("rainfall in india 1901-2015.csv")
states = sorted(df.SUBDIVISION.unique())

def load_model(state):
    model_filename = f"models/{state.replace(' ', '_').lower()}_model.pkl"
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Create the interface
st.write('Select a state and enter rainfall data for the last 3 months to predict next month\'s rainfall')

# State selection
selected_state = st.selectbox('Select State:', states)

# Input for rainfall data
col1, col2, col3 = st.columns(3)

with col1:
    month1 = st.number_input('Month 1 Rainfall (mm)', 
                            min_value=0.0, 
                            max_value=1000.0, 
                            value=0.0)

with col2:
    month2 = st.number_input('Month 2 Rainfall (mm)', 
                            min_value=0.0, 
                            max_value=1000.0, 
                            value=0.0)

with col3:
    month3 = st.number_input('Month 3 Rainfall (mm)', 
                            min_value=0.0, 
                            max_value=1000.0, 
                            value=0.0)

if st.button('Predict Rainfall'):
    try:
        # Load the model
        model = load_model(selected_state)
        
        # Make prediction
        input_data = np.array([[month1, month2, month3]])
        prediction = model.predict(input_data)
        
        # Display result
        st.success(f'Predicted rainfall for next month in {selected_state}: {prediction[0]:.2f} mm')
        st.info('📌 Note:The prediction is based on the rainfall data between 1901-2015.')
        
        # Create a simple visualization
        import plotly.graph_objects as go
        
        months_data = [month1, month2, month3, prediction[0]]
        months_labels = ['Month 1', 'Month 2', 'Month 3', 'Predicted']
        
        fig = go.Figure(data=[
            go.Bar(name='Rainfall', x=months_labels, y=months_data)
        ])
        
        fig.update_layout(
            title='Rainfall Pattern',
            yaxis_title='Rainfall (mm)',
            xaxis_title='Month'
        )
        
        st.plotly_chart(fig)
        
    except FileNotFoundError:
        st.error('Model not found. Please ensure you have trained the models first.')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')

# Add some helpful information
st.markdown("""
### How to use:
1. Select your state from the dropdown
2. Enter the rainfall data for the past three months
3. Click 'Predict Rainfall' to see the prediction

The prediction is based on historical rainfall patterns from 1901-2015.
""")