import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import TextBlob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

def preprocess_data(country_code):
    """
    Preprocess YouTube trending data for a given country, including feature engineering and EDA.

    Args:
        country_code (str): Country code (e.g., 'US', 'KR').

    Returns:
        pd.DataFrame: Processed DataFrame with engineered features.
    """
    # Ensure output directory exists
    output_dir = f'{country_code}_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # Remove outliers for likes_per_day and comments_per_day
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

    # Extended EDA: Correlation Matrix
    correlation_matrix = df_final[['view_count', 'likes', 'dislikes', 'comment_count', 'engagement_rate',
                                   'like_dislike_ratio', 'comment_view_ratio', 'dislikes_per_comment',
                                   'days_since_publication', 'likes_per_day', 'comments_per_day',
                                   'title_length', 'title_sentiment', 'view_velocity']].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Matrix of Features ({country_code})')
    plt.savefig(f'{output_dir}/{country_code}_correlation_matrix.png')
    plt.close()

    # EDA: Distribution of view_velocity by category
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_final, x='categoryId', y='view_velocity')
    plt.title(f'View Velocity Distribution by Category ({country_code})')
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/{country_code}_view_velocity_by_category.png')
    plt.close()

    # EDA: Title Length and Sentiment vs View Velocity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_final, x='title_length', y='view_velocity', hue='title_sentiment', size='title_sentiment')
    plt.title(f'Title Length and Sentiment vs View Velocity ({country_code})')
    plt.savefig(f'{output_dir}/{country_code}_title_analysis.png')
    plt.close()

    # EDA: Top words in titles
    title_words = [word for title in df['title_cleaned'] for word in title.split()]
    word_counts = Counter(title_words).most_common(20)
    word_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=word_df, x='Count', y='Word')
    plt.title(f'Top 20 Words in Trending Video Titles ({country_code})')
    plt.savefig(f'{output_dir}/{country_code}_top_title_words.png')
    plt.close()
    word_df.to_csv(f'{output_dir}/{country_code}_top_title_words.csv', index=False)

    # EDA: Number of tags vs View Velocity
    df_final['num_tags'] = df_final['tags'].apply(len)
    tag_velocity = df_final.groupby('num_tags')['view_velocity'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=tag_velocity, x='num_tags', y='view_velocity')
    plt.title(f'Average View Velocity by Number of Tags ({country_code})')
    plt.savefig(f'{output_dir}/{country_code}_num_tags_vs_velocity.png')
    plt.close()

    # Save processed data
    df_final.to_csv(f'{country_code}_processed_youtube_trending.csv', index=True)
    print(f"✅ Preprocessing completed for {country_code}! Shape: {df_final.shape}")

    return df_final

if __name__ == "__main__":
    countries = ['US', 'IN', 'BR', 'GB', 'KR']
    for country in countries:
        preprocess_data(country)