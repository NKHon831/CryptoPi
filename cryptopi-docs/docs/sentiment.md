## Sentiment Analysis Model

### Retrieving data

#### News Articles

##### `NewsAPI()` Class

The `NewsAPI` class retrieves news articles based on a query from NewsAPI. This API allows you to access news articles based on specified criteria such as keywords, dates, and sorting options. For more information, refer to the **NewsAPI** - [NewsAPI Official documentation](https://newsapi.org/docs/endpoints/everything).

###### Methods

`get_news(query,fromTimeFrame, toTimeFrame, sortBy="popularity", language="en")`

Parameters:

- query (str): The search query (e.g., "bitcoin").
- fromTimeFrame (str): The start time for the search in YYYY-MM-DDTHH:MM:SS format.
- toTimeFrame (str): The end time for the search in YYYY-MM-DDTHH:MM:SS format.
- sortBy (str): Sort the results by popularity or other metrics (default is "popularity").
- language (str): The language of the articles (default is "en").

Returns a list of dictionaries containing information about the articles (e.g., crypto, text, and published date).

#### Social Media Posts

##### `TweepyAPI()` Class

The TweepyAPI class retrieves tweets from Twitter API v2 via Tweepy. It allows you to fetch posts containing specific keywords or hashtags from Twitter. The class supports filtering by language and time frame, as well as excluding retweets for cleaner data. For more details, check the **Tweepy: Twitter API v2** - [Tweepy Official documentation](https://docs.tweepy.org/en/v4.0.1/client.html).

###### Methods

`get_tweets(query, fromTimeFrame, toTimeFrame, max_tweets=100, lang="en")`

Parameters:

- query (str): The search query (e.g., "bitcoin OR BTC"). You can also include operators like lang:{language} to filter by language.
- fromTimeFrame (str): The start time for the search in YYYY-MM-DDTHH:MM:SS format.
- toTimeFrame (str): The end time for the search in YYYY-MM-DDTHH:MM:SS format.
- max_tweets (int): The maximum number of tweets to retrieve (default is 100).
- lang (str): The language of the tweets (default is "en").

This method retrieves tweets from Twitter within the specified query, time frame, and language. It supports pagination for fetching a large number of tweets and excludes retweets.

### Preprocessing data

###### Methods

| Method                         | Description                                                                                               |
| ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| reformatDate(data)             | Ensures all dates are in ISO 8601 format and UTC (YYYY-MM-DDTHH:MM:SSZ).                                  |
| cleanNewsText(data)            | Performs extra cleaning specific to news article titles (e.g., removes publisher tags, ellipses).         |
| cleanSMPostsText(data)         | Performs extra cleaning specific to social media posts (e.g., removes mentions, hashtags, emojis).        |
| cleanAllText(data)             | General text cleaning for all data (e.g., lowercasing, removing URLs, HTML tags, and special characters). |
| mergeCleanedData(data1, data2) | Merges two datasets into one.                                                                             |

### Analyse sentiment

Loads a pretrained model from HuggingFace and perform sentiment analysis. For more information, refer to [CryptoBert](https://huggingface.co/ElKulako/cryptobert).

#### Method

`classify_sentiment(texts)`

Parameter:

- texts: Processed texts that is ready to fit into model for sentiment analysis

Returns the sentiment label and score of the input text.
