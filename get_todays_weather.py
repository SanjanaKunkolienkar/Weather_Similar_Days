import cdsapi

# Initialize the API client
c = cdsapi.Client()

# Define the API request for ERA5 data
c.retrieve(
    'reanalysis-era5-single-levels', {
        'product_type': 'reanalysis',
        'format': 'netcdf',  # You can choose other formats like GRIB
        'variable': [
            '2m_temperature', 'dew_point_temperature', '10m_u_component_of_wind',
            '10m_v_component["WindSpeed100Avg", "WindSpeed100Min", "WindSpeed100Max"]', 'wind_speed', 'wind_direction',
            'total_cloud_cover', 'surface_solar_radiation_downwards',
        ],
        'year': '2024',
        'month': '06',
        'day': '15',
        'time': [
            '00:00', '06:00', '12:00', '18:00'
        ],
        'area': [
            36.5, -106.5, 25.5, -93.5,  # North, West, South, East coordinates for Texas
        ],
    },
    'download.nc')  # Filename

print("Data download is complete.")
