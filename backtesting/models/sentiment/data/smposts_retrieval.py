import tweepy
from config import Config

# class TweepyAPI:
#   # https://developer.x.com/en/docs/x-api/v1/tweets/search/api-reference/get-search-tweets
#   def __init__(self):
#     self.api_key = Config.TWITTER_API_KEY
#     self.api_secret_key = Config.TWITTER_API_SECRET_KEY
#     self.access_token = Config.TWITTER_ACCESS_TOKEN
#     self.access_token_secret = Config.TWITTER_ACCESS_TOKEN_SECRET
#     # self.bearer_token = Config.TWITTER_BEARER_TOKEN

#     # Authenticate to Twitter
#     auth = tweepy.OAuth1UserHandler(
#       self.api_key,
#       self.api_secret_key,
#       self.access_token,
#       self.access_token_secret
#       )
#     self.api = tweepy.API(auth, wait_on_rate_limit=True)

#   # Free plan limit: 500 tweets per 15 minutes window
#   def get_tweets(self, query, lang="en", result_type="mixed", max_tweets=100):
#     data = []

#     # Create a Tweepy cursor to paginate through results
#     tweets = tweepy.Cursor(
#         self.api.search_tweets,
#         q=query,
#         lang=lang,
#         result_type=result_type,
#         tweet_mode="extended"
#     ).items(max_tweets)

#     # for tweet in tweets:
#     #   tweet_data = {
#     #     "text": tweet.full_text,
#     #     "publishedAt": tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"),
#     #   }
#     #   data.append(tweet_data)

#     return data
  
# import json
# import os
# if __name__ == "__main__":
#     # Example usage
#     tweepy_api = TweepyAPI()

#     query = ""

#     data = tweepy_api.get_tweets(query)

#     print("Retrieved %s tweets." % len(data))
#     print(data)

#     # Save to __pycache__ directory
#     pycache_dir = os.path.join(os.path.dirname(__file__), "__pycache__")
#     os.makedirs(pycache_dir, exist_ok=True)

#     output_file = os.path.join(pycache_dir, "socialmedia_data.json")
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)

#     print(f"Saved data to {output_file}")

class TweepyAPI:
  # https://docs.tweepy.org/en/v4.0.1/client.html
  def __init__(self):
    config = Config()
    self.api_key = config.TWITTER_API_KEY
    self.api_secret_key = config.TWITTER_API_SECRET_KEY
    self.access_token = config.TWITTER_ACCESS_TOKEN
    self.access_token_secret = config.TWITTER_ACCESS_TOKEN_SECRET
    self.bearer_token = config.TWITTER_BEARER_TOKEN
    self.client = tweepy.Client(
      bearer_token=self.bearer_token,
      consumer_key=self.api_key,
      consumer_secret=self.api_secret_key,
      access_token=self.access_token,
      access_token_secret=self.access_token_secret,
      # wait_on_rate_limit=True # default is False
    )

  def get_tweets(self, query, fromTimeFrame, toTimeFrame, max_tweets=100, lang="en"):
    data = []
    tweets_fetched = 0

    paginator = tweepy.Paginator(
      self.client.search_recent_tweets,
      query=query + f" lang:{lang} -is:retweet",  # filter retweets and language
      start_time=fromTimeFrame,
      end_time=toTimeFrame,
      tweet_fields=["created_at", "text"],
      max_results=100  # max allowed per call
    )

    try:
      for response in paginator:
        if response.data:
          for tweet in response.data:
            tweet_data = {
              "crypto": "bitcoin",
              "text": tweet.text,
              "publishedAt": tweet.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            data.append(tweet_data)
            tweets_fetched += 1
            if tweets_fetched >= max_tweets:
              print(f"Reached max tweets limit: {max_tweets}.")
              return data
    except tweepy.TooManyRequests:
      print("Rate limit exceeded, returning tweets collected so far. Please wait and try again.")
    
    print(f"Retrieved {len(data)} tweets from Twitter API.")

    return data
    # for response in paginator:
    #   if response.data:
    #     for tweet in response.data:
    #       tweet_data = {
    #         "crypto": "bitcoin",
    #         "text": tweet.text,
    #         "publishedAt": tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"),
    #       }
    #       data.append(tweet_data)
    #       tweets_fetched += 1
    #       if tweets_fetched >= max_tweets:
    #         return data

    # return data
  
import json
import os
if __name__ == "__main__":
  # Example usage
  tweepy_api = TweepyAPI()

  query = "bitcoin OR BTC"
  fromTimeFrame = "2025-04-15T00:00:00Z"
  toTimeFrame = "2025-04-15T23:59:59Z"
  data = tweepy_api.get_tweets(query,fromTimeFrame,toTimeFrame)

  print("Retrieved %s tweets." % len(data))
  print(data)

  # Save to __pycache__ directory
  pycache_dir = os.path.join(os.path.dirname(__file__), "__pycache__")
  os.makedirs(pycache_dir, exist_ok=True)

  output_file = os.path.join(pycache_dir, "socialmedia_data.json")
  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

  print(f"Saved data to {output_file}")