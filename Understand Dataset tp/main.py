import logging
from Preprocessing import preprocess_data
from AnalyzeTags import analyze_tags
from AnalyzeTrendingHours import analyze_trending_hours
from AnalyzeCategories import analyze_categories
from AnalyzeTrendingPatterns import analyze_trending_patterns
from TrainModel import train_model
from Recommend import YouTubeRecommendationSystem
from GenerateReport import generate_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename='pipeline.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_pipeline(country_code):
    """
    Run the entire analysis pipeline for a specific country.

    Args:
        country_code (str): Country code (e.g., 'US', 'KR').
    """
    logging.info(f"Starting pipeline for {country_code}")

    # Step 1: Preprocessing
    logging.info("Preprocessing data...")
    preprocess_data(country_code)

    # Step 2: Analyze tags
    logging.info("Analyzing tags...")
    analyze_tags(country_code)

    # Step 3: Analyze trending hours
    logging.info("Analyzing trending hours...")
    analyze_trending_hours(country_code)

    # Step 4: Analyze categories
    logging.info("Analyzing categories...")
    analyze_categories(country_code)

    # Step 5: Analyze trending patterns
    logging.info("Analyzing trending patterns...")
    analyze_trending_patterns(country_code)

    # Step 6: Train model
    logging.info("Training model...")
    train_model(country_code)

    # Step 7: Generate recommendations
    logging.info("Generating recommendations...")
    recommender = YouTubeRecommendationSystem(country_code)
    try:
        recommender.load_model_and_shap()
        for cat_id in recommender.category_names.keys():
            recommendations = recommender.get_recommendations(cat_id)
            logging.info(f"Recommendations for category {cat_id} in {country_code}:\n{recommendations}")
    except Exception as e:
        logging.error(f"Error generating recommendations for {country_code}: {str(e)}")

    # Step 8: Generate report
    logging.info("Generating report...")
    generate_report(country_code)

    logging.info(f"Pipeline completed for {country_code}")

if __name__ == "__main__":
    countries = ['US', 'IN', 'BR', 'GB', 'KR']
    for country in countries:
        print(f"Running pipeline for {country}...")
        run_pipeline(country)
        print(f"Pipeline completed for {country}. Check logs in pipeline.log.")