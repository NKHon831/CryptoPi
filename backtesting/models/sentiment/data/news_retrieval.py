from config import Config
import requests

class NewsAPI:
    # https://newsapi.org/docs/endpoints/everything
    def __init__(self):
        config = Config()
        self.api_key = config.NEWSAPI_API_KEY
        self.url = "https://newsapi.org/v2/everything"

    # timeframe: YYYY-MM-DDTHH:MM:SS
    def get_news(self, query, fromTimeFrame, toTimeFrame, sortBy="popularity", language="en"):
        data = []
        page = 1
        per_page = 100

        while True:
            params = {
                "apiKey": self.api_key,
                "q": query,
                "searchIn": "title", # options: search in title, description, or/and content
                "language": language,
                "from": fromTimeFrame,
                "to": toTimeFrame,
                "sortBy": sortBy,
                "page": page,
                "pageSize": per_page
            }

            response = requests.get(self.url, params=params)

            # Check if the request was successful
            if response.status_code == 200:
                news_data = response.json()
                articles = news_data.get("articles", [])

                for article in articles:
                    article_data = {
                      "crypto": "bitcoin",
                      "text": article["title"],
                      "publishedAt": article["publishedAt"],
                    }
                    data.append(article_data)

                # Check if we have retrieved all articles
                total_results = news_data.get("totalResults", 0)
                if len(data) >= total_results:
                    break  # Break if we've fetched all available articles
                else:
                    page += 1  # Go to the next page
            else:
                raise Exception(f"Error: {response.status_code}, {response.text}")
            
        print(f"Retrieved {len(data)} articles from NewsAPI.")

        return data  # Return all collected articles
    
import json
import os
if __name__ == "__main__":
    # Example usage
    news_api = NewsAPI()

    query = "bitcoin OR BTC"
    fromTimeFrame = "2025-04-15T00:00:00"
    toTimeFrame = "2025-04-15T23:59:59"

    data = news_api.get_news(query, fromTimeFrame, toTimeFrame)

    print("Retrieved %s articles." % len(data))
    print(data)

    # Save to __pycache__ directory
    pycache_dir = os.path.join(os.path.dirname(__file__), "__pycache__")
    os.makedirs(pycache_dir, exist_ok=True)

    output_file = os.path.join(pycache_dir, "news_data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved data to {output_file}")