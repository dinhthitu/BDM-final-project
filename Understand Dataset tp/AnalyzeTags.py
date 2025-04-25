import pandas as pd
import json
from collections import Counter
from tabulate import tabulate
import os

def analyze_tags(country_code):
    """
    Analyze popular tags across categories and their impact on view velocity.

    Args:
        country_code (str): Country code (e.g., 'US', 'KR').

    Returns:
        pd.DataFrame: DataFrame containing tag analysis results.
    """
    # Ensure output directory exists
    output_dir = f'{country_code}_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    df = pd.read_csv(f'{country_code}_youtube_trending_data.csv')

    # Load category mapping
    with open(f'{country_code}_category_id_to_name.json', 'r') as f:
        category_names = json.load(f)

    # Process tags (convert string to list and clean)
    def process_tags(tag_str):
        try:
            if pd.isna(tag_str):
                return []
            tags = [tag.strip().lower() for tag in tag_str.replace('"', '').split('|') if tag.strip()]
            return tags
        except:
            return []

    df['tags_processed'] = df['tags'].apply(process_tags)

    # Analyze tags across all categories
    all_tags = [tag for sublist in df['tags_processed'] for tag in sublist]
    tag_counts = Counter(all_tags)

    # Get top N most popular tags
    top_n = 30
    top_tags = tag_counts.most_common(top_n)

    # Create DataFrame for results
    tags_df = pd.DataFrame(top_tags, columns=['Tag', 'Count'])
    tags_df['Percentage'] = (tags_df['Count'] / len(df) * 100).round(2)

    # Display results
    print(f"=== Top {top_n} Most Popular Tags Across All Categories ({country_code}) ===")
    print(tabulate(tags_df, headers='keys', tablefmt='pretty', showindex=False))

    print(f"\n=== Tag Popularity by Category ({country_code}) ===")
    category_tags_list = []

    for category_id, category_name in category_names.items():
        category_videos = df[df['categoryId'] == int(category_id)]
        cat_tags = [tag for sublist in category_videos['tags_processed'] for tag in sublist]
        cat_tag_counts = Counter(cat_tags).most_common(5)
        for tag, count in cat_tag_counts:
            category_tags_list.append({'Category': category_name, 'Tag': tag, 'Count': count})

    # Convert to DataFrame
    category_tags_df = pd.DataFrame(category_tags_list)

    # Display sample
    print(tabulate(category_tags_df, headers='keys', tablefmt='pretty', showindex=False))

    # Save results
    category_tags_df.to_csv(f'{output_dir}/{country_code}_most_popular_tags_by_category.csv', index=False)
    print(f"\n✅ Saved results to '{output_dir}/{country_code}_most_popular_tags_by_category.csv'")

    return category_tags_df

if __name__ == "__main__":
    countries = ['US', 'IN', 'BR', 'GB', 'KR']
    for country in countries:
        analyze_tags(country)
