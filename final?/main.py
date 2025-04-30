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
    

if __name__ == "__main__":
    main()
