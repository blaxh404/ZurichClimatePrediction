import os
import pandas as pd

# Define a function to map dates to seasons
def get_season(date):
    month = date.month
    day = date.day
    
    if (month == 3 and day >= 20) or (month == 4 or month == 5) or (month == 6 and day < 21):
        return 1  # Spring
    elif (month == 6 and day >= 21) or (month == 7 or month == 8) or (month == 9 and day < 23):
        return 2  # Summer
    elif (month == 9 and day >= 23) or (month == 10 or month == 11) or (month == 12 and day < 21):
        return 3  # Fall
    else:
        return 4  # Winter

# Define the folder path where the CSV files are located
folder_path = 'data'

# Initialize an empty DataFrame to store the merged data
merged_df = pd.DataFrame()

# Loop through all files in the folder with a .csv extension
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # Create the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Drop the "name" column
        df.drop(columns=['name'], inplace=True)
        
        # Split the "datetime" column into year, month, and day
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        
        # Add a "season" column based on the date
        df['season'] = df['datetime'].apply(get_season)
        
        # Merge the data into the 'merged_df' DataFrame
        merged_df = pd.concat([merged_df, df], ignore_index=True)
        

merged_df.sort_values(by='datetime', inplace=True)

# Do some processing/cleaning on the merged dataset

merged_df['rainTomorrow']= merged_df['precip'].apply(lambda x: 1 if x > 0 else 0)
merged_df['rainTomorrow'] = merged_df['rainTomorrow'].shift(-1)

merged_df = merged_df[['year', 'month', 'day', 'season', 'tempmax', 'tempmin', 'temp', 'humidity','windspeed', 'winddir','cloudcover', 'sealevelpressure','rainTomorrow', 'precip', 'precipcover']]

## check how many values this removes. Since its very small compared to all the entries we don't need to re-adjust the rainTomorrow column
# I need to fix that caus now it removes quite a bit of Nans
merged_df.dropna(inplace=True)

        

# Save the merged and transformed data as a Pickle file
output_file_path = 'datasets/Zurich_weather_cleaned.pkl'
merged_df.to_pickle(output_file_path)

print(f"Merged and transformed data saved as {output_file_path}")
