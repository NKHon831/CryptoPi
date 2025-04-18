import pandas as pd

def mergePerformanceData(performanceData, anotherPerformanceData):
  merged = []

  for strategy_name, data in performanceData.items():
    merged.append({strategy_name: data})

  for strategy_name, data in anotherPerformanceData.items():
    existing_names = [list(d.keys())[0] for d in merged]
    new_name = strategy_name
    if new_name in existing_names:
      suffix = 1
      while f"{new_name}_{suffix}" in existing_names:
        suffix += 1
      new_name = f"{new_name}_{suffix}"
    merged.append({new_name: data})

  return merged

def mergePerformanceMetric(performanceMetric, anotherPerformanceMetric):
  merged = pd.concat([performanceMetric, anotherPerformanceMetric], ignore_index=True).fillna(0)

  return merged