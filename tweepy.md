# Tweepy: An easy-to-use Python library for accessing the Twitter API.
___

## 1. Install library

```
conda install -c conda-forge tweepy
```

## 2. Set up API

It is a good practice to store the keys in a file, for example, a .json. This avoid to include the keys in the main code, and therefore, in the repo.

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

# Create an auth instance with key and secret consumer, and pass the tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
    
# Instance the API authorization
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
```

## 3. Extracting information from Twitter

### 3.1. Post a tweet from your account

```Python
# Post a tweet from Python
# ------------------------------------------------------------------------------
api.update_status("Look, I'm tweeting from #Python")
```

### 3.2. Download info from a twitter account

```Python
# General information
# ------------------------------------------------------------------------------
# Introduce the target Twitter account
target = 'lexfridman'

# User info
data = api.get_user(target)
```

```Python
# Followers
# ------------------------------------------------------------------------------
# Introduce the target Twitter account and number of items to download
target = 'lexfridman'
n_items = 100

# User info
data = tweepy.Cursor(api.followers, screen_name=target).items(n_items)
```

### 3.3. Download tweets from a twitter account

```Python
# Tweets extractor (limited)
# ------------------------------------------------------------------------------

# Introduce the target Twitter account and number of items to download
target = 'lexfridman'
n_items = 200               # limited to 200 tweets

# Tweets list
tweets = api.user_timeline(screen_name=target, count=n_items)

# Export the list of tweets to a dataframe
data = pd.DataFrame(
    data=[tweet.text for tweet in tweets],
    columns=['tweet']
)
```

```Python
# Tweet extractor
# ------------------------------------------------------------------------------

# Introduce the target Twitter account and number of items to download
target = 'lexfridman'
n_items = 300

# Tweets list (iterator)
tweets = tweepy.Cursor(
    api.user_timeline,
    screen_name=target,
    tweet_mode='extended').items(n_items)
```

