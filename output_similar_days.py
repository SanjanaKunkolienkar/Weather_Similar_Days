import pandas as pd
import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

st.set_page_config(
    page_title="Texas Weather Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")
cwd = os.getcwd() # streamlit only uses 'relative to root' paths

def get_data(filemetric):
    if filemetric == 'All 5 measurements':
        similar_days_distances = pd.read_csv(os.path.join(cwd, 'Weather_Data_1950_2021/distances1950_2021.csv'))
        similar_days_indices = pd.read_csv(os.path.join(cwd, 'Weather_Data_1950_2021/similar_days_indices_1950_2021.csv'))
    else:
        similar_days_distances = pd.read_csv(os.path.join(cwd, 'Weather_Data_1950_2021/distances1950_2021_by_{}.csv'.format(filemetric)))
        similar_days_indices = pd.read_csv(os.path.join(cwd, 'Weather_Data_1950_2021/similar_days_indices_1950_2021_by_{}.csv'.format(filemetric)))
    return similar_days_distances, similar_days_indices

def display_similar_days(df, indices, date, col):
    # get the three most similar days
    ind_of_date_selected = indices.loc[indices['Date'] == date].iloc[0][1]
    # st.write("Index of selected Date: ", ind_of_date_selected)
    ind_of_similar_days = indices.loc[indices['Date'] == date].values[0][1:5]
    # st.write('Index of similar days: ', ind_of_similar_days)

    # get dates with indices ind_of_similar_days
    similar_days = indices.loc[indices['0'].isin(ind_of_similar_days)]['Date']

    with st.sidebar:
        st.write('The three most similar days to the selected date are:')
        st.write(similar_days[1:5])

    return similar_days

def get_data_from_aux():
    aux_file_path = os.path.join(cwd, 'Weather Aux By Years/mean_weather_data_1950_2021.csv')
    mean_weather_data = pd.read_csv(aux_file_path, index_col=0)

    mean_weather_data['DateTime'] = pd.to_datetime(mean_weather_data['DateTime'])
    # st.write(mean_weather_data.head())
    return mean_weather_data

# def select_metrics_and_labels(option):

def get_weather_for_similar_days(mean_weather_data, similar_days, my_bar, metrics, y_labels, colors, col, texas_gen):

    allopts = ['Temperature', 'Dew Point', 'Wind Speed', 'Cloud Cover']

    # split DateTime into Date and Hour for mean_weather_data
    mean_weather_data[['Date', 'Hour']] = mean_weather_data['DateTime'].astype(str).str.split(' ', expand=True)

    # st.write('Mean Weather Data:')
    # st.write(mean_weather_data.head())
    # st.write('Similar Days:')
    # st.write(similar_days.head())
    # filter mean_weather_data for similar days
    data_to_plot = mean_weather_data[mean_weather_data['Date'].isin(similar_days)]

    # st.write('Data to plot:')
    # st.write(data_to_plot.head())
    # get hour
    data_to_plot['Hour'] = data_to_plot['DateTime'].dt.hour  # Extract hour for plotting

    my_bar.progress(25, text='Extracting information...')

    with col[0]:
        st.subheader('Weather Profiles for Similar Days')
        # Create subplots dynamically
        fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=False,
                            subplot_titles=[f"{metric} Profiles" for metric in metrics])

        # Plot each metric for similar days
        for row, (metric, y_label) in enumerate(zip(metrics, y_labels), start=1):
            for i, day in enumerate(similar_days):
                day_data = data_to_plot[data_to_plot['Date'] == day]
                fig.add_trace(
                    go.Scatter(
                        x=day_data['Hour'],
                        y=day_data[metric],
                        mode='lines',
                        name=f"{day_data['Date'].iloc[0]}",
                        line=dict(color=colors[i % len(colors)]),
                        marker=dict(color=colors[i % len(colors)], size=8),
                        showlegend=(row == 1)  # Only show legend for the first subplot
                    ),
                    row=row, col=1
                )

        # Update layout dynamically
        fig.update_layout(
            title="Weather Profiles for Similar Days",
            height=300 * len(metrics),  # Adjust height based on the number of subplots
            width=1000,
            template="plotly_white"
        )

        # Set y-axis titles dynamically
        for i, y_label in enumerate(y_labels, start=1):
            fig.update_yaxes(title_text=y_label, row=i, col=1)

        # Only set x-axis title for the last subplot
        fig.update_xaxes(title_text="Hour of the Day", row=len(metrics), col=1)

        # Streamlit app
        st.plotly_chart(fig)

    my_bar.progress(100, text='Weather data loaded successfully!')

    if option != 'All 5 measurements':
        with col[1]:
            st.subheader(' ')
            allplotopts = list(set(allopts) - set(metrics))
            metrics = []
            metrics = allplotopts
            y_labels = allplotopts
            # Create subplots dynamically
            figa = make_subplots(rows=len(metrics), cols=1, shared_xaxes=False,
                                subplot_titles=[f"{metric} Profiles" for metric in metrics])

            # Plot each metric for similar days
            for row, (metric, y_label) in enumerate(zip(metrics, y_labels), start=1):
                for i, day in enumerate(similar_days):
                    day_data = data_to_plot[data_to_plot['Date'] == day]
                    figa.add_trace(
                        go.Scatter(
                            x=day_data['Hour'],
                            y=day_data[metric],
                            mode='lines',
                            name=f"{day_data['Date'].iloc[0]}",
                            line=dict(color=colors[i % len(colors)]),
                            marker=dict(color=colors[i % len(colors)], size=8),
                            showlegend=(row == 1)  # Only show legend for the first subplot
                        ),
                        row=row, col=1
                    )

            # Update layout dynamically
            figa.update_layout(
                title="Remaining weather profiles",
                height=300 * len(metrics),  # Adjust height based on the number of subplots
                width=1000,
                template="plotly_white"
            )

            # Set y-axis titles dynamically
            for i, y_label in enumerate(y_labels, start=1):
                figa.update_yaxes(title_text=y_label, row=i, col=1)

            # Only set x-axis title for the last subplot
            figa.update_xaxes(title_text="Hour of the Day", row=len(metrics), col=1)

            # Streamlit app
            st.plotly_chart(figa)


    # for each day in similar days, get the generation data: solar and wind
    gen = {}
    figaa = make_subplots(rows=len(metrics), cols=1, shared_xaxes=False,
                          subplot_titles=[f"{metric} Profiles" for metric in metrics])

    col2 = st.columns((2, 2, 2, 2), gap='small')
    i=0
    with col[1]:
        for _, d in enumerate(similar_days):
            with col2[i]:
                texas_gen['Date'] = pd.to_datetime(texas_gen['Date'])
                solar_gen = texas_gen.loc[texas_gen['Date'] == d, '48 Gen MW Solar'].sum()
                wind_gen = texas_gen.loc[texas_gen['Date'] == d, '48 Gen MW Wind'].sum()
                gen[d] = {'Solar Generation': solar_gen, 'Wind Generation': wind_gen}
                # st.write(f"Generation data for {d}: Solar Generation: {solar_gen} MW, Wind Generation: {wind_gen} MW")

                figaa.add_trace(
                    go.Pie(
                        labels=list(gen[d].keys()),
                        values=list(gen[d].values()),
                        name=d
                    )
                )

                # Update layout for better visualization
                figaa.update_layout(
                    title_text=d,
                    annotations=[dict(text='Categories', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )

                # Streamlit app
                st.plotly_chart(figaa)
                i+=1

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

try:
    with st.sidebar:
        option = st.selectbox(
            "Find similar days based on: ",
            ('All 5 measurements', 'temperature', 'windspeed', 'cloudcover'))

        metrics = []
        y_labels = []
        colors = ['orange', 'blue', 'green', 'red']

        if option == 'All 5 measurements':
            metrics = ['Temperature', 'Dew Point', 'Cloud Cover']
            y_labels = metrics
        elif option == 'temperature': #by temperature and dew point
            metrics = ['Temperature', 'Dew Point']
            y_labels = metrics
        elif option == 'windspeed': #wind speed (not 100m)
            metrics = ['Wind Speed']
            y_labels = metrics
        elif option == 'cloudcover':
            metrics = ['Cloud Cover']
            y_labels = metrics

        df, indices = get_data(option)
        min_value = df['Date'].unique()[0]
        mac_value = df['Date'].unique()[-1]
        date = st.select_slider(
            'Select a date: ',
            options=df['Date'].unique())

        if not date:
            st.error('Please select a date.')
        else:
            st.write('Currently selected date:', date)

        my_bar = st.progress(0, text='Getting weather data...')

    col = st.columns((5, 5), gap='small')

    sim_days = display_similar_days(df, indices, date, col)
    mean_weather_data = get_data_from_aux()

    texas_gen = pd.read_csv(os.path.join(cwd, 'Weather Aux By Years/Texas_EIA2024Q1.csv'))
    texas_gen[['48 Gen MW Wind', '48 Gen MW Solar']] = texas_gen[['48 Gen MW Wind', '48 Gen MW Solar']].apply(pd.to_numeric, errors='coerce')
    get_weather_for_similar_days(mean_weather_data, sim_days, my_bar, metrics, y_labels, colors, col, texas_gen)


except Exception as e:
    st.error(f" ***  An error occurred: {e} ***")

