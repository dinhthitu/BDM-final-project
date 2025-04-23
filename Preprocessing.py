import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import TextBlob
import json
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(country_code):
    # Load data
    df = pd.read_csv(f'{country_code}_youtube_trending_data.csv')

    # Drop unnecessary columns
    columns_to_drop = [
        'thumbnail_link', 'comments_disabled', 'ratings_disabled',
        'description', 'channelId', 'channelTitle'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Drop duplicates
    df = df.drop_duplicates(subset=['video_id'], keep='first')

    # Fill missing values
    numeric_cols = ['view_count', 'likes', 'dislikes', 'comment_count']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df['tags'] = df['tags'].fillna('')

    # Convert datetime
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['trending_date'] = pd.to_datetime(df['trending_date'])

    # Calculate hours_to_trend (handle invalid values)
    df['hours_to_trend'] = (df['trending_date'] - df['publishedAt']).dt.total_seconds() / 3600
    df = df[df['hours_to_trend'] >= 0]  # Remove invalid records

    # Remove outliers
    Q99 = df['hours_to_trend'].quantile(0.99)
    df = df[df['hours_to_trend'] <= Q99]

    # Map categoryId to category name
    with open(f'{country_code}_category_id_to_name.json', 'r') as f:
        category_map = json.load(f)
    category_id_to_name = {int(key): value for key, value in category_map.items()}
    df['category_name'] = df['categoryId'].map(category_id_to_name)

    # Text preprocessing
    stop_words = set(stopwords.words('english'))

    def clean_tags(tags):
        if tags == '[none]': 
            return []
        return [tag.strip().lower() for tag in tags.split('|') if tag.strip()]

    df['tags'] = df['tags'].apply(clean_tags)
    df['num_tags'] = df['tags'].apply(len)

    def clean_title(title):
        tokens = title.lower().split()
        cleaned = [token for token in tokens 
                   if token.isalpha() and token not in stop_words]
        return ' '.join(cleaned)

    df['title_cleaned'] = df['title'].apply(clean_title)
    df['title_length'] = df['title_cleaned'].str.len()
    df['title_sentiment'] = df['title_cleaned'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Time-based features
    df['publish_hour'] = df['publishedAt'].dt.hour
    df['publish_dayofweek'] = df['publishedAt'].dt.dayofweek

    # Calculate engagement and velocity features
    df['engagement_rate'] = (df['likes'] + df['dislikes'] + df['comment_count']) / (df['view_count'] + 1e-6)
    df['like_dislike_ratio'] = df['likes'] / (df['dislikes'] + 1e-6)
    df['comment_view_ratio'] = df['comment_count'] / (df['view_count'] + 1e-6)
    df['dislikes_per_comment'] = df['dislikes'] / (df['comment_count'] + 1e-6)
    df['days_since_publication'] = (df['trending_date'] - df['publishedAt']).dt.days
    df['likes_per_day'] = df['likes'] / (df['days_since_publication'] + 1e-6)
    df['comments_per_day'] = df['comment_count'] / (df['days_since_publication'] + 1e-6)
    df['view_velocity'] = np.log(df['view_count'] + 1) / (df['days_since_publication'] + 1)

    # Loại bỏ outliers cho likes_per_day và comments_per_day
    Q99_likes_per_day = df['likes_per_day'].quantile(0.99)
    df = df[df['likes_per_day'] <= Q99_likes_per_day]

    Q99_comments_per_day = df['comments_per_day'].quantile(0.99)
    df = df[df['comments_per_day'] <= Q99_comments_per_day]

    # Final features
    final_features = [
        'view_count', 'likes', 'dislikes', 'comment_count', 
        'title_length', 'title_sentiment', 'num_tags', 
        'publish_hour', 'publish_dayofweek',
        'trending_date', 'tags',
        'categoryId', 'hours_to_trend',
        'engagement_rate', 'like_dislike_ratio', 'comment_view_ratio',
        'dislikes_per_comment', 'days_since_publication', 'likes_per_day',
        'comments_per_day', 'view_velocity'
    ]
    df_final = df[final_features]

    # Analyze correlations
    correlation_matrix = df_final[['view_count', 'likes', 'dislikes', 'comment_count', 'days_since_publication', 'view_velocity']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Matrix of Features ({country_code})')
    plt.savefig(f'{country_code}_output/{country_code}_correlation_matrix.png')
    plt.close()

    # Analyze title length and sentiment impact on view_velocity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_final, x='title_length', y='view_velocity', hue='title_sentiment', size='title_sentiment')
    plt.title(f'Title Length and Sentiment vs View Velocity ({country_code})')
    plt.savefig(f'{country_code}_output/{country_code}_title_analysis.png')
    plt.close()

    # Save processed data
    df_final.to_csv(f'{country_code}_processed_youtube_trending.csv', index=True)
    print(f"✅ Preprocessing completed for {country_code}! Shape: {df_final.shape}")

    return df_final

if __name__ == "__main__":
    countries = ['US', 'IN', 'BR', 'GB', 'KR']
    for country in countries:
        preprocess_data(country)