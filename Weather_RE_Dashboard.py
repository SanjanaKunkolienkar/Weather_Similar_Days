import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from plotly.subplots import make_subplots
from main import get_generator_data
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity


cwd = os.getcwd()
def get_data(metric, filename):
    """ Get the similar days data from the CSV files.
    Args:
        metric: Metric to get similar days for
        filename: Name of the file to read from
    Returns:
        similar_days_distances: DataFrame containing the distances of the similar days
        similar_days_indices: DataFrame containing the indices of the similar days
    """

    similar_days_distances = pd.read_csv(os.path.join(cwd, 'Weather_Data_1950_2021/distances_{}.csv'.format(filename)))
    similar_days_indices = pd.read_csv(os.path.join(cwd, 'Weather_Data_1950_2021/similar_days_indices_{}.csv'.format(filename)))
    return similar_days_distances, similar_days_indices

def get_from_grid_search(date):
    """ Get the similar days data from the grid search.
    Args:
        date: Selected date
    Returns:
        similar_days: DataFrame containing the similar days

        !!! Working on getting this from the excel sheet instead !!!
    """
    #sim_days = maindays.maindays(date)
    sim_days = pd.Dataframe(['1950-01-01', '1950-01-02', '1950-01-03', '1950-01-04', '1950-01-05'], col=['Date'])
    return sim_days

def display_similar_days(df, indices, date, grid_search):
    """ Display the three most similar days to the selected date.
    Args:
        ""  Similar days DataFrame
        indices: DataFrame containing the indices of the similar days
        date: Selected date
        grid_search: Boolean indicating whether the data is from grid search or not
        Returns:
            ""  DataFrame containing the three most similar days
    """
    if not grid_search:
        # get the three most similar days
        ind_of_date_selected = indices.loc[indices['Date'] == date].iloc[0][1]
        # st.write("Index of selected Date: ", ind_of_date_selected)
        ind_of_similar_days = indices.loc[indices['Date'] == date].values[0][1:10]
        # st.write('Index of similar days: ', ind_of_similar_days)

        similar_days = indices.loc[indices['0'].isin(ind_of_similar_days)]['Date']

        # get dates that are NOT the selected date
        ind_of_similar_days = [ind for ind in ind_of_similar_days if ind != ind_of_date_selected]
        similar_days2 = indices.loc[indices['0'].isin(ind_of_similar_days)]['Date']

        similar_days2.reset_index(drop=True, inplace=True)
    else:
        similar_days = get_from_grid_search(date)

    # Display the three most similar days in the sidebar
    with st.sidebar:
        st.write('The three most similar days to the selected date are:')
        st.write(f"1. {similar_days2.iloc[0]}")
        st.write(f"2. {similar_days2.iloc[1]}")
        st.write(f"3. {similar_days2.iloc[2]}")

    return similar_days

def get_data_from_aux():
    """ Get the generator data from the AUX file.
        Renames the columns to be more descriptive.
        Splits Date Time to two columns.
    Returns:
        mean_weather_data : DataFrame containing the mean weather data
    """

    csv_file_path = os.path.join(cwd, 'Weather_Data_1950_2021/mean_weather_data_1940_2023.csv')
    mean_weather_data = pd.read_csv(csv_file_path)

    # convert Date and Hour columns to datetime
    mean_weather_data[['Date', 'Time']] = mean_weather_data['UTCISO8601'].str.replace('Z', '').str.split('T', expand=True)
    mean_weather_data['DateTime'] = mean_weather_data['Date'] + ' ' + mean_weather_data['Time']

    mean_weather_data['DateTime'] = pd.to_datetime(mean_weather_data['DateTime'])
    # convert date time to subtract 5 hours
    mean_weather_data['Texas_DateTime'] = mean_weather_data['DateTime'] - pd.Timedelta(hours=5, minutes=30)
    # st.write(mean_weather_data.head())
    # Rename columns
    new_column_names = {
        'DewPointF': 'Dew Point',
        'tempF': 'Temperature',
        'GlobalHorizontalIrradianceWM2': 'Global Horizontal Irradiance',
        'CloudCoverPerc': 'Cloud Cover',
        'DirectHorizontalIrradianceWM2': 'Direct Horizontal Irradiance',
        'WindSpeedmph': 'Wind Speed',
        'WindSpeed100mph': 'Wind Speed at 100m'
    }

    mean_weather_data = mean_weather_data.rename(columns=new_column_names)

    return mean_weather_data

def get_weather_for_similar_days(mean_weather_data, similar_days, metrics, y_labels, colors, col, texas_gen, selected_day, option):

    """
    Display the weather profiles for the similar days.
    Args:
        mean_weather_data: DataFrame containing the mean weather data
        similar_days: DataFrame containing the similar days
        metrics: List of metrics to plot
        y_labels: List of y-axis labels
        colors: List of colors for the plots
        col: Streamlit column object
        texas_gen: DataFrame containing the Texas generation data
        selected_day: Selected day
        option: Option selected by the user
    """

    # a list of all possible metrics
    allopts = ['Temperature', 'Dew Point', 'Wind Speed', 'Cloud Cover', 'Global Horizontal Irradiance', 'Direct Horizontal Irradiance', 'Wind Speed at 100m']
    # split DateTime into Date and Hour for mean_weather_data
    mean_weather_data[['Date', 'Hour']] = mean_weather_data['Texas_DateTime'].astype(str).str.split(' ', expand=True)

    # extract data to plot for the four days from mean_weather_data
    data_to_plot = mean_weather_data[mean_weather_data['Date'].isin(similar_days)]
    data_to_plot['Hour'] = data_to_plot['Texas_DateTime'].dt.hour  # Extract hour for plotting

    # calculate mape between the user selected day and the three other days
    mape = pd.DataFrame() # dataframe to store the MAPE values
    temp_din = data_to_plot[data_to_plot['Date'] == selected_day]
    temp_din.reset_index(inplace=True)
    for din in similar_days:
        predicted = mean_weather_data[mean_weather_data['Date'] == din]
        predicted.reset_index(inplace=True)
        if din != selected_day:
            for metric in metrics:
                mape.loc[metric, din] = calculate_mape(temp_din[metric], predicted[metric])

    # Display MAPE in the side bar
    with st.sidebar:
        st.write('Mean Absolute Percentage Error for each similar day:')
        st.write(mape)

    # Display the weather profiles for the similar days: shows the plots for selected metrics and selected days
    with col[0]:
        st.subheader('Weather profiles based on similarity of selected metrics')
        # Create subplots dynamically
        fig = make_subplots(rows=1, cols=len(metrics), shared_xaxes=False,
                            subplot_titles=[f"<span style='font-size:20px'>{metric} Profiles</span>" for metric in metrics])

        # Plot the selected metrics for the four days using a line plot
        # This code will automatically adjust the plots based on the number of plots
        for row, (metric, y_label) in enumerate(zip(metrics, y_labels), start=1):
            for i, day in enumerate(similar_days):
                day_data = data_to_plot[data_to_plot['Date'] == day]
                print(day_data)
                fig.add_trace(
                    go.Scatter(
                        x=day_data['Hour'],
                        y=day_data[metric],
                        mode='lines',
                        name=f"{day_data['Date'].iloc[0]}",
                        line=dict(color=colors[i % len(colors)]),
                        marker=dict(color=colors[i % len(colors)], size=8),
                        showlegend=(row == 1),  # Only show legend for the first subplot
                    ),
                    row=1, col=row
                )

        # make plots fancy with more details (vanity)
        # Set y-axis titles dynamically
        for i, y_label in enumerate(y_labels, start=1):
            fig.update_yaxes(title_text=y_label, row=1, col=i, title_font=dict(size=25),  # Font size for x-axis title
        tickfont=dict(size=25))

        for coll in range(1, len(metrics) + 1):
            fig.update_xaxes(title_text="Hour of the Day", row=1, col=coll,title_font=dict(size=25),  # Font size for x-axis title
        tickfont=dict(size=25))
        # Update layout dynamically
        fig.update_layout(
            height=300,
            width=580 * len(metrics),  # Adjust height based on the number of subplots
            template="plotly_white",
            margin=dict(t=50, b=0, l=0, r=0),
            legend_font=dict(size=25)
        )

        # Display these plots on the Streamlit app
        st.plotly_chart(fig)

    # Display the weather profiles for the similar days: shows the plots for remaining metrics and selected days
    with col[2]:
        # changes heading/title based on the option selected. If the user has selected all measurements, it will display additional weather profiles
        if option != 'All measurements':
            st.subheader('Remaining Weather Profiles')
        elif option == 'All measurements':
            st.subheader('Additional Weather Profiles')
        allplotopts = list(set(allopts) - set(metrics))
        metrics_hp = metrics # metrics for heatmap
        metrics = allplotopts
        y_labels = allplotopts

        # Create subplots dynamically
        figa = make_subplots(rows=len(metrics), cols=1, shared_xaxes=False,
                             subplot_titles=[f"<span style='font-size:20px'>{metric} Profiles</span>" for metric in metrics])

        # Plot each metric for similar days
        for cola, (metric, y_label) in enumerate(zip(metrics, y_labels), start=1):
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
                        showlegend=(cola == 1)  # Only show legend for the first subplot
                    ),
                    row=cola, col=1
                )

        # Set y-axis titles dynamically
        for cola, y_label in enumerate(y_labels, start=1):
            figa.update_yaxes(title_text=y_label, row=cola, col=1, title_font=dict(size=25),  # Font size for x-axis title
        tickfont=dict(size=25))

        # Only set x-axis title for the subplots
        for cola in range(1, len(metrics) + 1):
            figa.update_xaxes(title_text="Hour of the Day", row=cola, col=1,title_font=dict(size=25),  # Font size for x-axis title
        tickfont=dict(size=25))

        # Update layout dynamically
        figa.update_layout(
            height=350 * len(metrics),  # Adjust height as needed
            width=400,  # Adjust width based on the number of subplots
            showlegend=False,
            xaxis_title_font=dict(size=25),  # Adjust x-axis title font size
            yaxis_title_font=dict(size=25),  # Adjust y-axis title font size
            xaxis_tickfont=dict(size=25),  # Adjust x-axis tick labels font size
            yaxis_tickfont=dict(size=25),  # Adjust y-axis tick labels font size
            legend_font=dict(size=25)
        )

        # Display these plots on the Streamlit app
        st.plotly_chart(figa)

    ### Renewable Generation Plots ###
    # for each day in similar days, get the generation data: solar and wind
    gen = {}
    with col[0]:
        st.markdown('---')
        st.subheader('Comparing Renewable Generation')
        col2 = st.columns((1, 1, 1, 1), gap='small')

        ### Renewable Generation Plots ### Pie Charts ###
        i = 0
        j = 0
        for _, d in enumerate(similar_days):
            # assigns positions for the plots with the main point being it ensures selected date is shown first
            if d == selected_day:
                i = 0 # so that selected day is plotted in first col always
                title = d + ' (Selected)'
            else:
                j+=1
                i = j
                title = d

            print(i)
            with col2[i]:
                texas_gen['Date'] = pd.to_datetime(texas_gen['Date'])
                texas_gen['Hour'] = pd.to_datetime(texas_gen['Time']).dt.hour
                solar_gen = texas_gen.loc[texas_gen['Date'] == d, 'Solar Gen. (MW)'].sum()
                wind_gen = texas_gen.loc[texas_gen['Date'] == d, 'Wind Gen. (MW)'].sum()
                gen[d] = {'Solar Generation': solar_gen, 'Wind Generation': wind_gen}

                # Plots pie charts for solar and wind generation
                figaa_pie = go.Figure()
                figaa_pie.add_trace(
                    go.Pie(
                        labels=list(gen[d].keys()),
                        values=list(gen[d].values()),
                        marker_colors =['EDD83D', '6AB547'], #FEA82F
                        showlegend=(i == 0)
                    ),
                )

                figaa_pie.update_layout(
                    height=150,
                    width=300,
                    title=dict(text=title, font=dict(size=30)),
                    margin=dict(t=50, b=0, l=0, r=0),
                    legend_font=dict(size=20)
                )

                # Streamlit app
                st.plotly_chart(figaa_pie)

        ### Renewable Generation Plots ### Bar Plots ###
        i=0 #reset
        j=0
        val = 0
        for _, day in enumerate(similar_days):
            temp_df = texas_gen[texas_gen['Date'] == day]
            maxi = temp_df[['Solar Gen. (MW)', 'Wind Gen. (MW)']].max().max()
            if val < maxi:
                val = maxi
        col3 = st.columns((1, 1, 1, 1), gap='small')
        for r, day in enumerate(similar_days):
            d = day
            if d == selected_day:
                i=0 # so that selected day is plotted in first col always
                title = d + ' (Selected Day)'
            else:
                j=j+1
                i=j
                title = d
            with col3[i]:
                temp_df = texas_gen[texas_gen['Date'] == d]
                figaa_bar = go.Figure()
                figaa_bar.add_trace(go.Bar(
                    x=temp_df['Hour'],
                    y=temp_df['Solar Gen. (MW)'],
                    name='Solar',
                    marker_color='#EDD83D',
                    showlegend=False
                ))
                figaa_bar.add_trace(go.Bar(
                    x=temp_df['Hour'],
                    y=temp_df['Wind Gen. (MW)'],
                    name='Wind',
                    marker_color='#6AB547',
                    showlegend=False
                ))
                figaa_bar.update_layout(
                    height=300,
                    width=400,
                    barmode='group',
                    yaxis_range=[0, val],
                    xaxis_tickfont=dict(size=25),  # Adjust x-axis tick labels font size
                    yaxis_tickfont=dict(size=25),
                )

                figaa_bar.update_yaxes(title_text="MW Gen.", title_font=dict(size=25),
                                       # Font size for x-axis title
                                       tickfont=dict(size=25))

                figaa_bar.update_xaxes(title_text="Hour of the Day", title_font=dict(size=25),
                                       # Font size for x-axis title
                                       tickfont=dict(size=25))
                st.plotly_chart(figaa_bar)

        ### Renewable Generation Plots ### Heat Maps ###
        colscatter = st.columns((1, 1), gap='small')
        # Heat map 1 : This if for the entire day with respect to the selected day for all metrics + generation
        # Range here is from -1 to 1, because we want to see how different the weather metrics and generation is from the selected day
        with colscatter[0]:
            # Initialize an empty DataFrame for storing aggregated data
            df_heatmap = pd.DataFrame()
            df_24hr_similar = pd.DataFrame()
            for i, day in enumerate(similar_days):
                # Filter data for the specific day
                temp_df = texas_gen[texas_gen['Date'] == day]
                temp_df['Hour'] = pd.to_datetime(temp_df['Time']).dt.hour

                # Merge with the data_to_plot to add the metrics
                temp_df = temp_df.merge(data_to_plot[data_to_plot['Date'] == day], on='Hour')
                temp_df.drop(columns=['UTCISO8601', 'Time_x', 'Time_y', 'DateTime', 'Texas_DateTime',
                                      'Date_y', 'WindDirection'], inplace=True)
                df_24hr_similar = pd.concat([df_24hr_similar, temp_df])
                temp_df.drop(columns=['Hour', 'Date_x'], inplace=True)

                # Aggregate the metrics by averaging over the 24 hours
                aggregated_metrics = temp_df.mean()
                aggregated_metrics['Day'] = str(day).split(' ')[0]  # Add the day label

                # Append the aggregated data to the heatmap DataFrame
                df_heatmap = df_heatmap._append(aggregated_metrics, ignore_index=True)

            # only include the chosen metrics
            df_heatmap = df_heatmap[metrics_hp + ['Day', 'Wind Gen. (MW)', 'Solar Gen. (MW)']]
            df_heatmap.rename(columns={'Wind Gen. (MW)': 'Wind Generation', 'Solar Gen. (MW)': 'Solar Generation'}, inplace=True)
            # Set the 'Day' as the index for the heatmap
            df_heatmap.set_index('Day', inplace=True)
            # Normalize with reference to the selected day
            selected_day_data = df_heatmap.loc[selected_day]
            df_heatmap_normalized = (df_heatmap - selected_day_data) / selected_day_data
            # delete the selected day from the heatmap
            df_heatmap_normalized.drop(selected_day, inplace=True)

            # Create the heatmap using Plotly, excluding the selected day in the display
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=df_heatmap_normalized.T.values,
                x=df_heatmap_normalized.index,
                y=df_heatmap_normalized.columns,
                colorscale='PuOr',  # Red color gradient
                zmin = -1,  # Set minimum value of the color scale
                zmax = 1  # Set maximum value of the color scale
            ))

            fig_heatmap.update_layout(
                height=650,
                title=dict(text="Daily Similarity Heatmap of Metric & Generation: Comparisons to Selected Day", font=dict(size=25)),
                xaxis=dict(
                    title='Day',
                    tickmode='array',
                    type='category',
                    tickvals=df_heatmap_normalized.index,
                    ticktext=df_heatmap_normalized.index,
                    categoryorder='category ascending'
                ),
                yaxis=dict(
                    tickmode='array',
                    type='category',
                    tickvals=df_heatmap_normalized.columns,
                    ticktext=df_heatmap_normalized.columns
                ),
                xaxis_tickfont=dict(size=25),  # Adjust x-axis tick labels font size
                yaxis_tickfont=dict(size=25),
                font=dict(size=25)
            )

            # Display the heatmap to streamlit app
            st.plotly_chart(fig_heatmap)

        # Heat map 2 : This is for the hourly similarity of metrics and generation with respect to the selected day
        # Range here is from 0 to 1, because we want to see how different the days are from the selected day
        with colscatter[1]:
            # Filter the base day data
            base_day_data = df_24hr_similar[df_24hr_similar['Date_x'] == selected_day]
            other_days_data = df_24hr_similar[df_24hr_similar['Date_x'] != selected_day]

            # Initialize a DataFrame to store similarities
            similarity_scores = pd.DataFrame(index=df_24hr_similar['Hour'].unique(),
                                             columns=[day for day in similar_days if day != selected_day])

            # Calculate cosine similarity for each day against the base day
            for hour in df_24hr_similar['Hour'].unique():
                base_hour_data = base_day_data[base_day_data['Hour'] == hour].drop(columns=['Date_x', 'Hour'])
                for day in similar_days:
                    if day == selected_day:
                        continue
                    hour_data = other_days_data[
                        (other_days_data['Date_x'] == day) & (other_days_data['Hour'] == hour)].drop(
                        columns=['Date_x', 'Hour'])
                    # Compute cosine similarity
                    sim_score = cosine_similarity(base_hour_data, hour_data)[0][0]
                    similarity_scores.at[hour, day] = sim_score

            # Create the heatmap using Plotly
            fig_cosine_sim = go.Figure(
                data=go.Heatmap(
                    z=similarity_scores.values,
                    x=similarity_scores.columns,
                    y=similarity_scores.index,
                    colorscale=[(0, 'white'), (1, '#3C074F')],  # Gradient
                    zmin=0,  # Set minimum value of the color scale
                    zmax=1  # Set maximum value of the color scale
                ),
            )

            # Update the layout
            fig_cosine_sim.update_layout(
                title=dict(
                        text="Hourly Similarity Heatmap of Metrics & Generation: Comparison to Selected Day",
                        font=dict(size=25)
                ),
                height=650,
                xaxis=dict(
                    title='Day',
                    tickmode='array',
                    type='category',
                    tickvals=similarity_scores.columns,
                    ticktext=similarity_scores.columns,
                    categoryorder='category ascending',
                ),
                yaxis=dict(
                    title='Hour of Day',
                    tickmode='array',
                    type='category',
                    tickvals=similarity_scores.index,
                    ticktext=similarity_scores.index,
                    categoryorder='array',
                    dtick=1,
                ),
                xaxis_tickfont=dict(size=25),  # Adjust x-axis tick labels font size
                yaxis_tickfont=dict(size=25),
                xaxis_title_font=dict(size=25),  # Adjust x-axis title font size
                yaxis_title_font=dict(size=25),
                font=dict(size=25)
            )

            # Display the heatmap to steamlit app
            st.plotly_chart(fig_cosine_sim)

        ### Box plot for the entire year ###
        # generate a box plot to show the distribution of the metrics across the similar days based on mean weather data for entire year
        # Create a list to hold the box plots for each metric
        box_plots = []
        yearly_data = mean_weather_data.drop(columns=['Date', 'Hour', 'Texas_DateTime', 'DateTime', 'UTCISO8601', 'Time', 'WindDirection'])
        # normalize the data
        yearly_data = (yearly_data - yearly_data.mean()) / yearly_data.std()
        yearly_data_gen = texas_gen.drop(columns=['Date', 'Time', 'Hour'])
        yearly_data_gen = (yearly_data_gen - yearly_data_gen.mean()) / yearly_data_gen.std()

        # Loop through each column in the DataFrame to create a box plot for each metric
        for column in yearly_data.columns:
            box_plot = go.Box(
                y=yearly_data[column],
                name=column,
                width=0.5,
            )
            box_plots.append(box_plot)

        for column in yearly_data_gen.columns:
            box_plot = go.Box(
                y=yearly_data_gen[column],
                name=column,
                width=0.5,
            )
            box_plots.append(box_plot)

        # Create the figure
        fig_box_plot = go.Figure(box_plots)

        # Update the layout to add title, labels, etc.
        fig_box_plot.update_layout(
            title=dict(text="Values Relative to All 80 years", font=dict(size=25)),
            xaxis_title="Metrics",
            yaxis_title="Normalized Values",
            showlegend=True,
            boxmode='group',  # For grouping the boxes together
            xaxis_tickfont=dict(size=25),  # Adjust x-axis tick labels font size
            yaxis_tickfont=dict(size=25),
            xaxis_title_font=dict(size=25),  # Adjust x-axis title font size
            yaxis_title_font=dict(size=25),
            font=dict(size=25)
        )

        # Show where the days are with respect to the box plot of 80 years
        # add dots for the similar days
        for i, day in enumerate(similar_days):
            fig_box_plot.add_trace(go.Scatter(x=allopts,
                                              y=yearly_data.loc[mean_weather_data['Date'] == day, allopts].mean(),
                                              mode='markers',
                                              marker=dict(color=colors[i % len(colors)], size=8),
                                              name=day,
                                              showlegend=True
                                              )
                                  )
            fig_box_plot.add_trace(go.Scatter(x=['Wind Gen. (MW)', 'Solar Gen. (MW)'],
                                              y=yearly_data_gen.loc[texas_gen['Date'] == day, ['Wind Gen. (MW)', 'Solar Gen. (MW)']].mean(),
                                              mode='markers',
                                              marker=dict(color=colors[i % len(colors)], size=8),
                                              name=day,
                                              showlegend=False
                                              )
                                   )

        fig_box_plot.update_layout(
            height=600,
            font=dict(
                size=30
            )
        )
        # Show the plot
        st.plotly_chart(fig_box_plot)

def calculate_mape(y_true, y_pred):
    """ Calculate the Mean Absolute Percentage Error (MAPE) between two arrays."""
    return (abs((y_true - y_pred) / y_true).mean()) * 100
def main():
    st.set_page_config(layout="wide")
    st.title("Historical Weather and Renewable Energy Insights Panel")

    # Sidebar for settings
    with st.sidebar:
        st.subheader('Substation View')
        # Placeholder for a network map
        gen_df = get_generator_data()
        colors = ['#EDD83D', '#6AB547']  # Define your two specific colors
        labels = ['Solar', 'Wind']  # Corresponding labels for the legend

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lon=gen_df['longitude'],
            lat=gen_df['latitude'],
            text=gen_df['BusNum'],
            marker=dict(
                size=5,
                color=gen_df['color'],  # Assuming 'color' column contains actual color values like '#EDD83D', '#6AB547'
            ),
            showlegend=False  # Hide legend entry for this trace
        ))

        for color, label in zip(colors, labels):
            fig.add_trace(go.Scattergeo(
                lon=[None],  # Longitude data from DataFrame
                lat=[None],  # Latitude data from DataFrame
                marker=dict(
                    size=5,  # Uniform size for all markers
                    color=color,  # Use the 'color' column for marker colors
                ),
                name=label
            ))
        fig.update_layout(
            geo=dict(
                scope='usa',
                lonaxis_range=[-106.65, -93.51],  # Longitude range for Texas
                lataxis_range=[25.84, 36.5],  # Latitude range for Texas
                showland=True,
                landcolor="rgb(255, 255, 255)",
                center=dict(lat=31.0886, lon=-99.9018)),
            height=300,
            width=400,
            margin=dict(t=50, b=0, l=0, r=0),
            showlegend=False
        )

        st.plotly_chart(fig)
    st.sidebar.header('Select Date and Metrics')
    # selected_date = st.sidebar.date_input("Select a date", min_value=datetime(1950, 1, 1), max_value=datetime(2023, 12, 31))
    metric_options = ['All measurements', 'Temperature and Dew Point', 'Wind Speed', 'GHI, DHI and Cloud Cover']
    selected_metric = st.sidebar.selectbox("Select a metric", options=metric_options)

    if selected_metric == 'All measurements':
        filename = 'all'
        metrics = ['Temperature', 'Dew Point', 'Cloud Cover']
        y_labels = metrics
    elif selected_metric == 'Temperature and Dew Point':  # by temperature and dew point
        filename = 'temp_and_dewpoint'
        metrics = ['Temperature', 'Dew Point']
        y_labels = metrics
    elif selected_metric == 'Wind Speed':  # not including wind direction
        filename = 'windspeed'
        metrics = ['Wind Speed', 'Wind Speed at 100m']
        y_labels = metrics
    elif selected_metric == 'GHI, DHI and Cloud Cover':
        filename = 'sun'
        metrics = ['Cloud Cover', 'Global Horizontal Irradiance', 'Direct Horizontal Irradiance']
        y_labels = metrics

    df, indices = get_data(selected_metric, filename)

    min_value = pd.to_datetime(df['Date'].unique()[0])
    max_value = pd.to_datetime(df['Date'].unique()[-1])
    date = str(st.sidebar.date_input("Select a date", min_value=min_value, max_value=max_value))

    st.session_state.submit_button = st.sidebar.button("Find Similar Days")
    if st.session_state.submit_button:
        sim_days = display_similar_days(df, indices, date, False)
        # sim_days = get_from_grid_search(date)
        day_to_analyse = [din for _, din in enumerate(sim_days)]
        mean_weather_data = get_data_from_aux()

        texas_gen = pd.read_csv(os.path.join(cwd, 'Weather_Data_1950_2021/Texas_EIA2024Q1.csv'))
        texas_gen.rename(columns={'48 Gen MW Wind': 'Wind Gen. (MW)', '48 Gen MW Solar': 'Solar Gen. (MW)'}, inplace=True)
        texas_gen[['Wind Gen. (MW)', 'Solar Gen. (MW)']] = texas_gen[['Wind Gen. (MW)', 'Solar Gen. (MW)']].apply(
            pd.to_numeric, errors='coerce')

        colors = ['orange', 'blue', 'green', 'red']
        col = st.columns([19.8, 0.2, 6.6])
        get_weather_for_similar_days(mean_weather_data, sim_days, metrics, y_labels, colors, col, texas_gen, date, selected_metric)

        with col[1]:
            st.html(
                '''
                    <div class="divider-vertical-line"></div>
                    <style>
                        .divider-vertical-line {
                            border-left: 10px solid rgba(49, 51, 63, 0.2);
                            height: 1020px;
                            margin: auto;
                        }
                    </style>
                '''
            )


if __name__ == "__main__":
    main()
