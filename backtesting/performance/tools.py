import json

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