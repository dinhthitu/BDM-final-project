import pandas as pd
import json
import matplotlib.pyplot as plt

def analyze_trending_hours(country_code):
    # Load data từ file CSV đã preprocess
    df = pd.read_csv(f'{country_code}_youtube_trending_data.csv')
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['trending_date'] = pd.to_datetime(df['trending_date'])
    df['hours_to_trend'] = (df['trending_date'] - df['publishedAt']).dt.total_seconds() / 3600

    # Load category mapping từ JSON
    with open(f'{country_code}_category_id_to_name.json', 'r') as f:
        category_names = json.load(f)

    # Tính average hours_to_trend theo category
    category_avg = df.groupby('categoryId')['hours_to_trend'].mean().sort_values()

    # Ánh xạ categoryId sang tên category
    category_avg = category_avg.rename(index=category_names)

    # In kết quả
    print(f"=== Average Hours to Trend by Category ({country_code}) ===")
    print(category_avg.to_string(float_format="%.2f"))
    category_avg.to_csv(f'{country_code}_output/{country_code}_avg_hours_to_trend_by_category.csv', float_format="%.2f")
    print(f"\n✅ Saved CSV to '{country_code}_avg_hours_to_trend_by_category.csv'")

    # Visualize
    plt.figure(figsize=(10, 6))
    category_avg.plot(kind='barh', color='skyblue')
    plt.title(f'Average Hours to Become Trending by Category ({country_code})', fontsize=14)
    plt.xlabel('Average Hours', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Lưu hình ảnh
    plt.savefig(f'{country_code}_output/{country_code}_avg_hours_to_trend_by_category.png')
    plt.close()
    print(f"\n✅ Saved visualization to '{country_code}_avg_hours_to_trend_by_category.png'")

if __name__ == "__main__":
    countries = ['US', 'IN', 'BR', 'GB', 'KR']
    for country in countries:
        analyze_trending_hours(country)