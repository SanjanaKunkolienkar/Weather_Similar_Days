import cdsapi
from datetime import date, timedelta

class Request:
    def __init__(self, __date: date) -> None:
        self.date: date = __date - timedelta(10)                             # Latest data is 5 days ago
        self.quarter: int = ((self.date.month - 1) // 3) + 1                # What quarter is the data from, e.g. Q1, Q2, etc.
        self.file: str = f"Texas{self.date.year}_Q{self.quarter}"    # File name. e.g. NorthAmerica2024_Q1

req = Request(date.today())
print(f"Downloading data for {req.date.strftime('%d %B %Y')}")
# Initialize the API client
CDS = cdsapi.Client()
CDS.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '2m_dewpoint_temperature', '2m_temperature',
                '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_wind',
                '10m_v_component_of_wind', 'total_cloud_cover', 'high_cloud_cover', 'low_cloud_cover',
                'medium_cloud_cover', 'surface_solar_radiation_downwards',
                'total_sky_direct_solar_radiation_at_surface', 'geopotential'
            ],
            'area': [
                '36', '-108', '24', '-90'
            ],
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00',
                '05:00', '06:00', '07:00', '08:00', '09:00',
                '10:00', '11:00', '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00', '18:00', '19:00',
                '20:00', '21:00', '22:00', '23:00',
            ],
            'day': [f'{req.date.day:02d}'],
            'month': [f'{req.date.month:02d}'],
            'year': f'{req.date.year}',
        },
        f"{req.file}.nc"
    )