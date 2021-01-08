# Tweepy: An easy-to-use Python library for accessing the Twitter API.
___

## 1. Install library

```
conda install -c conda-forge tweepy
```

## 2. Set up API

It is a good practice to store the keys in a file, for example, a .json.

```Python
# API Twitter credentials
# ------------------------------------------------------------------------------

# Open .json file containing credentials/tokens as a dictionary
with open("twitter_api_keys.json") as file:
    api_credentials = json.load(file)
    
# Assign each value of the dictionary to a new variable
consumer_key = api_credentials['consumer_key']
consumer_secret = api_credentials['consumer_secret']
access_token = api_credentials['access_token']
access_token_secret = api_credentials['access_token_secret']
```

```Python
# API set up
# ------------------------------------------------------------------------------

# Create an instance with consumer key and secret, and pass the tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key_token, access_secret_token)
    
# Construct the API instance
api = tweepy.API(auth)
```

## 3. Extracting information from Twitter

### 3.1. Download Tweets

```Python
# Tweets extractor
# ------------------------------------------------------------------------------

# Instantiate an extractor object linked to the API instance
extractor = api_setup()

# Introduce the target Twitter account
target = 'lexfridman'

# Tweets list
tweets = extractor.user_timeline(screen_name=target, count=20)
```