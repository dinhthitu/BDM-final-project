import pandas as pd
import json
from tabulate import tabulate

def analyze_categories(country_code):
    # Load data
    df = pd.read_csv(f'{country_code}_youtube_trending_data.csv')

    # Load category mapping
    with open(f'{country_code}_category_id_to_name.json', 'r') as f:
        category_names = json.load(f)

    # Group theo categoryId và tính toán
    stats = df.groupby('categoryId').agg({
        'view_count': 'mean',
        'likes': 'mean',
        'dislikes': 'mean',
        'comment_count': 'mean',
        'video_id': 'count'
    }).rename(columns={
        'view_count': 'Avg Views',
        'likes': 'Avg Likes',
        'dislikes': 'Avg Dislikes',
        'comment_count': 'Avg Comments',
        'video_id': 'Video Count'
    }).reset_index()

    # Thêm tên danh mục (Category Name)
    stats['Category'] = stats['categoryId'].astype(str).map(category_names)

    # Thêm phần trăm
    stats['Percentage'] = (stats['Video Count'] / len(df) * 100).round(2)

    # Định dạng cột
    stats = stats[['Category', 'Video Count', 'Percentage', 'Avg Views', 'Avg Likes', 'Avg Dislikes', 'Avg Comments']]
    stats = stats.round(2)

    # Hiển thị bảng
    print(f"=== All Categories with Engagement Stats ({country_code}) ===")
    print(tabulate(stats, headers='keys', showindex=True, tablefmt='pretty'))

    # Lưu ra file
    stats.to_csv(f'{country_code}_output/{country_code}_category_stats.csv', index=False)
    print(f"\n✅ Saved updated results with engagement stats to '{country_code}_category_stats.csv'")

if __name__ == "__main__":
    countries = ['US', 'IN', 'BR', 'GB', 'KR']
    for country in countries:
        analyze_categories(country)