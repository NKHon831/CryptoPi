import json
import pandas as pd

def convert_JSON_list(JSON_path):
  list = []

  with open(JSON_path, 'r') as file:
    data = json.load(file)

  for x in data:
    dict = {}
    
    for key, value in x.items():
      dict[key] = value
    
    list.append(dict)
    
  return list

def process_benchmark_data(raw_df: pd.DataFrame) -> pd.Series:
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = ['_'.join(col).strip() for col in raw_df.columns.values]

    datetime_col = [col for col in raw_df.columns if 'datetime' in col.lower()][0]
    price_col = [col for col in raw_df.columns if 'benchmark' in col.lower()][0]

    raw_df[datetime_col] = pd.to_datetime(raw_df[datetime_col])
    raw_df.set_index(datetime_col, inplace=True)

    raw_df[price_col] = pd.to_numeric(raw_df[price_col], errors='coerce')

    raw_df.dropna(subset=[price_col], inplace=True)

    return raw_df[price_col]