# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 16:17:58 2024

@author: kvyja
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns

def process_data(df):
    """Clean and filter input DataFrame for analysis.

   Parameters:
   - df (pd.DataFrame): Raw data to be processed.

   Returns:
   - filtered_data (pd.DataFrame): Data for Population growth and 
                                               CO2 emissions in 1998.
   - filtered_data1 (pd.DataFrame): Data for Population growth and 
                                               CO2 emissions in 2018.
   """
    # Drop last 5 rows
    df = df.iloc[:-5]
    
    # Replace '..' with NaN
    df.replace('..', np.nan, inplace=True)
    
    # Extract data for Population growth in 1998 & 2018
    data1 = df[df['Indicator Name'] == 
               'Population growth (annual %)'][['Country Name', '1998']]\
        .rename(columns={'1998': 'Population growth (annual %)'})
    
    data2 = df[df['Indicator Name'] == 
               'Population growth (annual %)'][['Country Name', '2018']]\
        .rename(columns={'2018': 'Population growth (annual %)'})
    
    # Extract data for CO2 emissions in 1998 & 2018
    data3 = df[df['Indicator Name'] == 
             'CO2 emissions (kg per PPP $ of GDP)'][['Country Name', '1998']]\
        .rename(columns={'1998': 'CO2 emissions (kg per PPP $ of GDP)'})
            
    data4 = df[df['Indicator Name'] == 
             'CO2 emissions (kg per PPP $ of GDP)'][['Country Name', '2018']]\
        .rename(columns={'2018': 'CO2 emissions (kg per PPP $ of GDP)'})
    
    # Merge data for 1998 and 2018 on 'Country Name'
    merged_data = pd.merge(data1, data3, on='Country Name', 
                           how='outer').reset_index(drop=True)
    merged_data1 = pd.merge(data2, data4, on='Country Name', 
                            how='outer').reset_index(drop=True)
    
    # Drop rows with any NaN values
    filtered_data = merged_data.dropna(how='any').reset_index(drop=True)
    filtered_data1 = merged_data1.dropna(how='any').reset_index(drop=True)
    
    return filtered_data, filtered_data1

# Read the DataFrame
df = pd.read_csv("API_19_DS2_en_csv_v2_6300757.csv", skiprows=3)

# Process the data using the function
filtered_data, filtered_data1 = process_data(df)

def plot_combined_elbow(data1, data2, title):
    """Generate an elbow plot for two datasets to help determine the optimal 
    number of clusters (k).

   Parameters:
   - data1 (pd.DataFrame): First dataset for clustering analysis.
   - data2 (pd.DataFrame): Second dataset for clustering analysis.
   - title (str): Title of the plot."""
   
    distortions1, distortions2 = [], []

    for i in range(1, 11):
        kmeans1 = KMeans(n_clusters=i, random_state=42)
        kmeans1.fit(data1)
        distortions1.append(kmeans1.inertia_)

        kmeans2 = KMeans(n_clusters=i, random_state=42)
        kmeans2.fit(data2)
        distortions2.append(kmeans2.inertia_)

    plt.plot(range(1, 11), distortions1, marker='o', label='filtered_data')
    plt.plot(range(1, 11), distortions2, marker='o', label='filtered_data1')

    plt.title(f'Elbow Plot for {title}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion (Inertia)')
    plt.legend()
    plt.show()

# Elbow plot for both data frames
plot_combined_elbow(
    filtered_data[['Population growth (annual %)', 
                   'CO2 emissions (kg per PPP $ of GDP)']],
    filtered_data1[['Population growth (annual %)', 
                                 'CO2 emissions (kg per PPP $ of GDP)']],
                              'Comparison of filtered_data and filtered_data1')

# Function to perform KMeans clustering, add cluster labels, and return cluster centers
def perform_kmeans(data, num_clusters):
    """Apply KMeans clustering to a given dataset.

    Parameters:
    - data (pd.DataFrame): Input dataset containing features for clustering.
    - num_clusters (int): Number of clusters to form.

    Returns:
    - data (pd.DataFrame): Original dataset with an additional 'Cluster' column.
    - centers (numpy.ndarray): Coordinates of cluster centers."""
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[
    ['Population growth (annual %)', 'CO2 emissions (kg per PPP $ of GDP)']])
    centers = kmeans.cluster_centers_
    return data, centers

# Process the data
filtered_data, filtered_data1 = process_data(df)

# Apply KMeans clustering to filtered_data with, for example, 4 clusters
num_clusters_filtered_data = 4
filtered_data_clustered, centers_filtered_data = perform_kmeans(filtered_data, 
                                                    num_clusters_filtered_data)

# Apply KMeans clustering to filtered_data1 with, for example, 4 clusters
num_clusters_filtered_data1 = 4
filtered_data1_clustered, centers_filtered_data1 = perform_kmeans(
                                filtered_data1, num_clusters_filtered_data1)

# Plotting for filtered_data_clustered
plt.figure(figsize=(6, 5))
sns.scatterplot(x='Population growth (annual %)',
                y='CO2 emissions (kg per PPP $ of GDP)',hue='Cluster',
                palette='Set2',data=filtered_data_clustered)

# Plotting cluster centers for filtered_data1 in the scatter plot.
plt.scatter(centers_filtered_data[:, 0], centers_filtered_data[:, 1],
                marker='x', s=50, color='black', label='Cluster Centers')

# Set the title of the plot to indicate the year of the data.
plt.title('Clustering of Population growth & CO2 emissions (1998)')
# Display the legend to identify different clusters.
plt.legend()
# Adjust layout for better appearance.
plt.tight_layout()
# Save the plot as an image file.
plt.savefig('clustering.png')

# Plotting for filtered_data1_clustered
plt.figure(figsize=(6, 5))
sns.scatterplot(x='Population growth (annual %)',
                y='CO2 emissions (kg per PPP $ of GDP)',hue='Cluster',
                palette='Set2',data=filtered_data1_clustered)

# Plotting cluster centers for filtered_data1 in the scatter plot.
plt.scatter(centers_filtered_data1[:, 0], centers_filtered_data1[:, 1], 
            marker='x', s=50, color='black', label='Cluster Centers')

# Set the title of the plot to indicate the year of the data.
plt.title('Clustering of Population growth & CO2 emissions (2018)')
# Display the legend to identify different clusters.
plt.legend()
# Adjust layout for better appearance.
plt.tight_layout()
# Save the plot as an image file.
plt.savefig('clustering1.png')

def plot_forecast_by_country(df, selected_countries, indicator_name, degree=3):
    """
  Plot forecast for a specific indicator across selected countries.

  Parameters:
  - df (pd.DataFrame): The input DataFrame containing historical data.
  - selected_countries (list): List of countries to be included in the plot.
  - indicator_name (str): The name of the indicator to be plotted.
  - degree (int): Degree of the polynomial regression model (default is 3)."""
  
    # Filter the data
    data_selected = df[df['Country Name'].isin(selected_countries) &
              (df['Indicator Name'] == indicator_name)].reset_index(drop=True)

    # Melt the DataFrame
    data_forecast = data_selected.melt(id_vars=['Country Name', 'Indicator Name'
                                         ],var_name='Year', value_name='Value')

    # Filter out non-numeric values in the 'Year' column
    data_forecast = data_forecast[data_forecast['Year'].str.isnumeric()]

    # Convert 'Year' to integers
    data_forecast['Year'] = data_forecast['Year'].astype(int)

    # Handle NaN values by filling with the mean value
    data_forecast['Value'].fillna(data_forecast['Value'].mean(), inplace=True)

    # Filter data for the years between 1990 and 2020
    data_forecast = data_forecast[(data_forecast['Year'] >= 1998) &
                                  (data_forecast['Year'] <= 2018)]

    # Extend the range of years to include 2025
    all_years_extended = list(range(1998, 2026))

    # Creating line plots for each country with a grid and unique style.
    for country in selected_countries:
        plt.figure(figsize=(6, 3))

        # Plot actual data with a specific color
        sns.lineplot(x='Year', y='Value',
                     data=data_forecast[data_forecast['Country Name'] == country],
                     marker='o', markersize=5,linestyle='-', 
                     label=f'Actual Data for {country}', color='orchid')

        # Prepare data for the current country.
        country_data = data_forecast[data_forecast['Country Name'] == country]
        X_country = country_data[['Year']]
        y_country = country_data['Value']

        # Fit polynomial regression model with the specified degree.
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X_country)

        model = LinearRegression()
        model.fit(X_poly, y_country)

        # Predict values for all years (1990 to 2025)
        X_pred = poly_features.transform(pd.DataFrame(all_years_extended, 
                                                      columns=['Year']))
        forecast_values = model.predict(X_pred)

        # Plot the fitted curve with a different color
        sns.lineplot(x=all_years_extended, y=forecast_values,
                     label=f'Fitted Curve for {country}', linestyle='-', 
                                                         color='royalblue')

        # Plot forecast for 2025
        prediction_2025 = forecast_values[-1]
        plt.plot(2025, prediction_2025, marker='*', markersize=8,
                 label=f'Prediction for 2025: {prediction_2025:.2f}',
                 color='black')

        plt.title(f'{indicator_name} Forecast for {country}', fontsize=12)
        plt.xlabel('Year', fontsize=10)

        # Set y-axis label dynamically based on the indicator
        plt.ylabel('(kg per PPP $ of GDP)', fontsize=10)

        # Set x-axis limits and ticks
        plt.xlim(1998, 2030)
        plt.xticks(range(1998, 2031, 5))  # Adjust the step as needed

        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)

        # Set legend font size.
        plt.legend(fontsize=7)
        # Create a filename for the saved plot.
        filename = f"{indicator_name}_Forecast_{country.replace(' ', '_')}.png"
        # Save the plot with a tight bounding box.
        plt.savefig(filename, bbox_inches='tight')

'''Plot forecasts for CO2 emissions in selected countries, 
showing fitted curves and predictions for 2025.'''

selected_countries = ['India', 'Pakistan', 'Bhutan']
indicator_name = 'CO2 emissions (kg per PPP $ of GDP)'
plot_forecast_by_country(df, selected_countries, indicator_name)