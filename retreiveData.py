import requests
import pandas as pd
from io import StringIO
import os

#cat pass: personalproject2

API1 = 'PVAX3PXCWLX8PHPZF5XZVRNKS'
API2 = '9NZLU5D5GHCP4JB5MGQEDLDT5'
API3 = 'RBBGF5FGWUEM7WA3MNT2C4VDQ'
API4 = 'U86KS9Q5SFJ2YUQ3E8FZWXVQ3'

# Define variables for the API URL
base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Zurich/"
start_date = "1998-06-02"
end_date = "2001-01-01"
unit_group = "metric"
include = "days"
api_key = API4
content_type = "csv"
file_path = os.path.join('data', f"{start_date}_{end_date}.csv")


# Construct the API URL with variables
api_url = f"{base_url}{start_date}/{end_date}?unitGroup={unit_group}&include={include}&key={api_key}&contentType={content_type}"

# Send a GET request to the API
response = requests.get(api_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Read the CSV data into a Pandas DataFrame
    df = pd.read_csv(StringIO(response.text))
    
    # Print the first few rows of the DataFrame to verify
    print(df.head())
    
    # Now you can work with the 'df' DataFrame as needed
    
    # For example, you can save it to a CSV file
    df.to_csv(file_path, index=False)
else:
    print("Failed to retrieve data from the API. Status code:", response.status_code)

