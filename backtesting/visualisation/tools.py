import pandas as pd
import json

def convert_CSV_df(csv_path, skip_rows):
  df = pd.read_csv(
    csv_path,
    skiprows=skip_rows,
  )

  # Reset column names
  df.columns = ["datetime", "close", "high", "low", "open", "volume", "log_return", "volatility"]

  # Keep only the required columns
  df = df[["datetime", "close", "high", "low", "open", "volume"]]

  # Convert datetime column to actual datetime objects
  df["datetime"] = pd.to_datetime(df["datetime"])
  
  return df

def reformatDates(df):

  df['datetime'] = pd.to_datetime(df['datetime'])

  return df

def convert_JSON_df(JSON_path):
  # Load the JSON data from the file
  with open(JSON_path, 'r') as f:
    data = json.load(f)

  # If the data is a single dictionary, wrap it in a list
  if isinstance(data, dict):
    data = [data]
  
  # Convert the JSON data to a pandas DataFrame
  df = pd.DataFrame(data)

  # Ensure the datetime column is converted to pandas datetime type for easier manipulation
  # df['datetime'] = pd.to_datetime(df['datetime'])

  # If you want to set datetime as the index (optional), you can do that
  # df.set_index('datetime', inplace=True)

  return df