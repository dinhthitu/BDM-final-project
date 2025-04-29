import os
from Preprocessing import preprocess_data
from AnalyzeCategories import analyze_categories
from AnalyzeTags import analyze_tags
from AnalyzeTrendingHours import analyze_trending_hours
from AnalyzeTrendingPatterns import analyze_trending_patterns
from TrainModel import train_model
from GenerateReport import generate_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    countries = ['GB', 'US', 'BR', 'CA', 'MX']
    
    # Step 1: Preprocess data
    print("=== Preprocessing Data ===")
    for country in countries:
        preprocess_data(country)
    
    # Step 2: Analyze categories
    print("\n=== Analyzing Categories ===")
    for country in countries:
        analyze_categories(country)
    
    # Step 3: Analyze tags
    print("\n=== Analyzing Tags ===")
    for country in countries:
        analyze_tags(country)
    
    # Step 4: Analyze trending hours
    print("\n=== Analyzing Trending Hours ===")
    for country in countries:
        analyze_trending_hours(country)
    
    # Step 5: Analyze trending patterns
    print("\n=== Analyzing Trending Patterns ===")
    for country in countries:
        analyze_trending_patterns(country)
    
    # Step 6: Train models
    print("\n=== Training Models ===")
    for country in countries:
        train_model(country)
    
    # Step 7: Generate reports
    print("\n=== Generating Reports ===")
    for country in countries:
        generate_report(country)
    
    # Step 8: Compare across countries
    print("\n=== Comparing Across Countries ===")
    compare_countries(countries)

def compare_countries(countries):
    """Compare category statistics across countries."""
    all_stats = []
    for country in countries:
        try:
            stats = pd.read_csv(f'{country}_output/{country}_category_stats.csv')
            stats['Country'] = country
            all_stats.append(stats)
        except Exception as e:
            print(f"⚠️ Failed to load stats for {country}: {str(e)}")
    
    if not all_stats:
        print("❌ No data to compare across countries")
        return
    
    combined_stats = pd.concat(all_stats)
    combined_stats.to_csv('combined_category_stats.csv', index=False)
    
    # Visualize comparison
    plt.figure(figsize=(14, 8))
    sns.catplot(
        data=combined_stats,
        x='Category',
        y='Percentage',
        hue='Country',
        kind='bar',
        height=8,
        aspect=2
    )
    plt.xticks(rotation=45)
    plt.title('Category Distribution Across Countries')
    plt.tight_layout()
    plt.savefig('country_comparison.png')
    plt.close()
    print("✅ Saved country comparison to 'country_comparison.png'")

if __name__ == "__main__":
    main()
