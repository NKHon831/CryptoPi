import re
import html
from nltk.tokenize import word_tokenize
import emoji
from dateutil import parser
from datetime import timezone

def reformatDate(data):
  """
  Ensure all dates are in the same ISO 8601 format, UTC (YYYY-MM-DDTHH:MM:SSZ)
  """
  formatted_data = []
  for item in data:
    try:
      dt = parser.parse(item["publishedAt"])  # parse any date format
      dt_utc = dt.astimezone(timezone.utc)    # convert to UTC
      item["publishedAt"] = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
      print(f"Date parse error: {item['publishedAt']} -> {e}")
      continue  # skip malformed date entries or handle as needed
    formatted_data.append(item)
  
  return formatted_data

def cleanNewsText(data):
  """
  Extra cleaning steps specific to news titles
  """
  cleaned_data = []
  for item in data:
    if (item['text']):
      text = item['text']

      # Remove publisher tags or anything in brackets
      text = re.sub(r'\[.*?\]', '', text)

      # Remove trailing ellipses
      text = re.sub(r'\.\.\.$', '', text)

      # Normalize quotation marks and dashes
      text = text.replace('“', '"').replace('”', '"').replace('–', '-').replace('—', '-')

      cleaned_data.append({
        'crypto': item['crypto'],
        'text': text.strip(),
        'publishedAt': item['publishedAt']
      })
  
  return cleanAllText(cleaned_data)

def cleanSMPostsText(data):
  """
  Extra cleaning steps specific to social media posts
  """
  cleaned_data = []
  for item in data:
    text = item['text']

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags (keep the word, remove '#')
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')  # requires `emoji` library

    # Remove cashtags (optional: keep tickers like $BTC)
    # text = re.sub(r'\$\w+', '', text)  # Uncomment to remove cashtags

    # Remove "RT" or reply markers
    text = re.sub(r'^RT\s+', '', text)
    text = re.sub(r'^@\w+:', '', text)

    cleaned_data.append({
      'crypto': item['crypto'],
      'text': text.strip(),
      'publishedAt': item['publishedAt']
    })
  
  # Apply base cleaner after specific clean
  return cleanAllText(cleaned_data)

def cleanAllText(data):
  """
  Input: list of dictionaries with keys 'crypto', 'text' and 'publishedAt'
  Output: list of dictionaries with cleaned text, ready to fit into model and 'publishedAt'
  """
  cleaned_data = []

  for item in data:
    text = item['text']

    # Lowercasing
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Remove HTML tags/entities
    text = html.unescape(text)
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters & punctuations (preserve $ and %)
    text = re.sub(r'[^\w\s$%]', '', text)

    # Normalize white spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization (optional for modeling)
    tokens = word_tokenize(text)
    text = ' '.join(tokens)

    cleaned_data.append({
      'crypto': item['crypto'],
      'text': text,
      'publishedAt': item['publishedAt']
    })

  return cleaned_data

def mergeCleanedData(data1, data2):
  """
  Input: list of dictionaries with keys 'crypto', 'text' and 'publishedAt'
  Output: list of dictionaries with cleaned text, ready to fit into model and 'publishedAt'
  """
  return data1 + data2