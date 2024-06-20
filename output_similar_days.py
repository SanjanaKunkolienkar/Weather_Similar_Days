import pandas as pd
import streamlit as st

def get_data():
    similar_days_distances = pd.read_csv('distances1950_2021.csv')
    similar_days_indices = pd.read_csv('similar_days_indices_1950_2021.csv')
    return similar_days_distances, similar_days_indices

try:
    st.write('Welcome to the Similar Days App!')
    df, indices = get_data()
    min_value = df['Date'].unique()[0]
    mac_value = df['Date'].unique()[-1]
    date = st.select_slider(
        'Select a date: ',
        options=df['Date'].unique())
    st.write('Currently selected date:', date)

    if not date:
        st.error('Please select a date.')
    else:
        st.write('Good Job!')
except Exception as e:
    st.error(f" ***  An error occurred: {e} ***")

#