import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tabulate import tabulate

def analyze_trending_patterns(country_code):
    """
    Analyze trending patterns by day of the week and optimal posting hours per category.

    Args:
        country_code (str): Country code (e.g., 'US', 'KR').

    Returns:
        pd.DataFrame: DataFrame with trending pattern statistics.
    """
    # Ensure output directory exists
    output_dir = f'{country_code}_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    df = pd.read_csv(f'{country_code}_processed_youtube_trending.csv')
    df['trending_date'] = pd.to_datetime(df['trending_date'])

    # Add day_of_week if not present
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['trending_date'].dt.day_name()

    # Load category mapping
    with open(f'{country_code}_category_id_to_name.json', 'r') as f:
        category_map = json.load(f)

    # Create mapping from categoryId to category name
    category_id_to_name = {int(key): value for key, value in category_map.items()}
    df['category_name'] = df['categoryId'].map(category_id_to_name)

    # Create visualization (heatmap)
    plt.figure(figsize=(12, 8))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = (df.groupby(['day_of_week', 'category_name'])
                    .size()
                    .unstack()
                    .fillna(0)
                    .reindex(day_order))
    sns.heatmap(heatmap_data.T,
                cmap='YlGnBu',
                annot=True,
                fmt='.0f',
                cbar_kws={'label': 'Số lượng video'})
    plt.title(f'Video Trending by Day and Category ({country_code})', pad=20, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{country_code}_trending_analysis.png', dpi=150)
    plt.close()

    # Analyze optimal posting hour by category
    peak_hours_by_category = df.groupby(['category_name', 'publish_hour'])['view_velocity'].mean().reset_index()
    peak_hours = peak_hours_by_category.loc[peak_hours_by_category.groupby('category_name')['view_velocity'].idxmax()]
    peak_hours.to_csv(f'{output_dir}/{country_code}_peak_hours_by_category.csv', index=False)

    # Export detailed results
    results = []
    for cat in df['category_name'].unique():
        cat_data = df[df['category_name'] == cat]
        if len(cat_data) == 0:
            print(f"⚠️ No data for category {cat} in country {country_code}")
            continue
        peak_day = cat_data['day_of_week'].value_counts().idxmax()
        avg_hour = peak_hours[peak_hours['category_name'] == cat]['publish_hour'].values[0]
        avg_time_to_trend = cat_data['hours_to_trend'].median()
        results.append({
            'Danh mục': cat,
            'Ngày đỉnh': peak_day,
            'Giờ đăng tối ưu': f"{avg_hour:.1f}",
            'Thời gian lên trending (giờ)': f"{avg_time_to_trend:.1f}",
            'Số video': len(cat_data)
        })

    if not results:
        print(f"❌ No data to analyze for country {country_code}")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    print(f"\n=== Trending Stats by Category ({country_code}) ===")
    print(tabulate(results_df, headers='keys', tablefmt='pretty', showindex=False))
    results_df.to_csv(f'{output_dir}/{country_code}_trending_stats_by_category.csv', index=False)

    print(f"✅ Analysis completed for {country_code}!")
    print(f"- Chart: {output_dir}/{country_code}_trending_analysis.png")
    print(f"- Data: {output_dir}/{country_code}_trending_stats_by_category.csv")

    return results_df.sort_values('Số video', ascending=False)

if __name__ == "__main__":
    countries = ['GB', 'US', 'BR', 'CA', 'MX']
    for country in countries:
        analyze_trending_patterns(country)