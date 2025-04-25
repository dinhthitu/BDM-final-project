import pandas as pd
import numpy as np
import shap
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

class YouTubeRecommendationSystem:
    def __init__(self, country_code):
        """
        Initialize the YouTube Recommendation System for a specific country.

        Args:
            country_code (str): Country code (e.g., 'US', 'KR').
        """
        self.country_code = country_code
        sns.set(style='darkgrid', palette='deep', font='DejaVu Sans', font_scale=1.2)
        self.setup_features()
        self.load_data()
        self.X_test = None
        self.shap_values = None
        self.model = None

    def load_data(self):
        """
        Load preprocessed data and related statistics.
        """
        try:
            self.df = pd.read_csv(f'{self.country_code}_preprocessed_youtube_trending_data.csv')
            missing_features = [f for f in self.features if f not in self.df.columns]
            if missing_features:
                raise ValueError(f"Missing columns in self.df: {missing_features}")

            with open(f'{self.country_code}_category_id_to_name.json', 'r') as f:
                self.category_names = json.load(f)

            self.category_stats = pd.read_csv(f'{self.country_code}_output/{self.country_code}_category_stats.csv')
            self.trending_stats = pd.read_csv(f'{self.country_code}_output/{self.country_code}_trending_stats_by_category.csv')
            self.tag_stats = pd.read_csv(f'{self.country_code}_output/{self.country_code}_most_popular_tags_by_category.csv')
            self.hours_to_trend = pd.read_csv(f'{self.country_code}_output/{self.country_code}_avg_hours_to_trend_by_category.csv')
            self.peak_hours = pd.read_csv(f'{self.country_code}_output/{self.country_code}_peak_hours_by_category.csv')
            self.top_title_words = pd.read_csv(f'{self.country_code}_output/{self.country_code}_top_title_words.csv')

        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def setup_features(self):
        """
        Define the features used for modeling.
        """
        self.features = [
            'view_count', 'likes', 'dislikes', 'comment_count',
            'engagement_rate', 'like_dislike_ratio', 'comment_view_ratio',
            'dislikes_per_comment', 'days_since_publication', 'likes_per_day',
            'comments_per_day', 'title_length', 'title_sentiment'
        ]

    def load_model_and_shap(self):
        """
        Load the best model and corresponding SHAP values.
        """
        try:
            with open(f'{self.country_code}_output/{self.country_code}_best_model.json', 'r') as f:
                model_info = json.load(f)
            best_model_name = model_info['best_model']

            self.X_test = pd.read_csv(f'{self.country_code}_output/{self.country_code}_X_test.csv', index_col=0)
            available_features = [f for f in self.features if f in self.X_test.columns]
            if len(available_features) < len(self.features):
                missing = [f for f in self.features if f not in self.X_test.columns]
                raise ValueError(f"Missing columns in X_test: {missing}")
            self.X_test = self.X_test[available_features]

            if best_model_name == "XGBoost":
                self.model = joblib.load(f'{self.country_code}_output/{self.country_code}_model_xgb.joblib')
                self.shap_values = joblib.load(f'{self.country_code}_output/{self.country_code}_shap_values_xgb.joblib')
            else:
                self.model = joblib.load(f'{self.country_code}_output/{self.country_code}_model_lgb.joblib')
                self.shap_values = joblib.load(f'{self.country_code}_output/{self.country_code}_shap_values_lgb.joblib')

        except Exception as e:
            raise ValueError(f"Error loading model or SHAP: {str(e)}")

    def get_category_insights(self, category_id):
        """
        Gather insights for a specific category.

        Args:
            category_id (str): Category ID.

        Returns:
            dict: Insights including general stats, trending patterns, tags, and time to trend.
        """
        category_id_str = str(category_id)
        category_id_int = int(category_id)
        category_name = self.category_names.get(category_id_str, f"Category {category_id_str}")

        insights = {
            'name': category_name,
            'general_stats': {},
            'trending_patterns': {},
            'tags': [],
            'time_to_trend': {},
            'optimal_hour': None
        }

        try:
            if not self.category_stats.empty:
                stats = self.category_stats[self.category_stats['Category'] == category_name]
                if not stats.empty:
                    insights['general_stats'] = {
                        'percentage': float(stats['Percentage'].values[0]),
                        'avg_views': float(stats['Avg Views'].values[0]) if 'Avg Views' in stats else 0,
                        'avg_likes': float(stats['Avg Likes'].values[0]) if 'Avg Likes' in stats else 0
                    }

            if not self.trending_stats.empty:
                patterns = self.trending_stats[self.trending_stats['Danh mục'] == category_name]
                if not patterns.empty:
                    insights['trending_patterns'] = {
                        'peak_day': patterns.iloc[0].get('Ngày đỉnh', 'N/A'),
                        'peak_hour': patterns.iloc[0].get('Giờ đăng tối ưu', 'N/A'),
                        'avg_time': float(patterns.iloc[0].get('Thời gian lên trending (giờ)', 0))
                    }

            if not self.peak_hours.empty:
                optimal_hour = self.peak_hours[self.peak_hours['category_name'] == category_name]
                if not optimal_hour.empty:
                    insights['optimal_hour'] = float(optimal_hour['publish_hour'].values[0])

            if not self.tag_stats.empty:
                tags_rows = self.tag_stats[self.tag_stats['Category'] == category_name]
                if not tags_rows.empty:
                    valid_tags = [tag for tag in tags_rows['Tag'].head(5).tolist() if tag != '[none]']
                    insights['tags'] = valid_tags

            if not self.hours_to_trend.empty:
                time_stats = self.hours_to_trend[self.hours_to_trend['categoryId'] == category_id_int]
                if not time_stats.empty:
                    insights['time_to_trend'] = {
                        'avg': float(time_stats['hours_to_trend'].mean()),
                        'median': float(time_stats['hours_to_trend'].median()),
                        'std': float(time_stats['hours_to_trend'].std()) if len(time_stats['hours_to_trend']) > 1 else None,
                        'sample_count': len(time_stats['hours_to_trend'])
                    }

        except Exception as e:
            pass

        return insights

    def generate_shap_recommendations(self, category_data, category_id):
        """
        Generate recommendations based on SHAP analysis for a specific category.

        Args:
            category_data (pd.DataFrame): Data for the specific category.
            category_id (str): Category ID.

        Returns:
            tuple: List of recommendations and top features DataFrame.
        """
        recommendations = []
        try:
            test_indices = self.X_test.index.intersection(category_data.index)
            if len(test_indices) == 0:
                return ["No sufficient data for SHAP analysis."], pd.DataFrame()

            test_positions = [list(self.X_test.index).index(idx) for idx in test_indices]
            missing_features = [f for f in self.features if f not in category_data.columns]
            if missing_features:
                raise ValueError(f"Missing columns in category_data: {missing_features}")

            if isinstance(self.shap_values, shap.Explanation):
                shap_values_array = self.shap_values.values
                shap_feature_names = self.shap_values.feature_names
                if shap_feature_names and shap_feature_names != self.features:
                    raise ValueError("SHAP feature names do not match model features.")
            else:
                shap_values_array = self.shap_values

            category_shap = shap_values_array[test_positions]
            shap_df = pd.DataFrame(category_shap, columns=self.features)

            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'mean_abs_shap': np.abs(category_shap).mean(axis=0),
                'mean_shap': category_shap.mean(axis=0),
                'positive_impact': (category_shap > 0).mean(axis=0),
                'negative_impact': (category_shap < 0).mean(axis=0),
                'p25': [category_data[f].quantile(0.25) for f in self.features],
                'p75': [category_data[f].quantile(0.75) for f in self.features]
            }).sort_values('mean_abs_shap', ascending=False)

            total_shap = feature_importance['mean_abs_shap'].sum()
            if total_shap == 0:
                raise ValueError("Total SHAP contribution is zero.")
            feature_importance['percent_contribution'] = feature_importance['mean_abs_shap'] / total_shap * 100

            feature_importance['priority'] = feature_importance['percent_contribution'].apply(
                lambda x: 'High' if x > 30 else 'Medium' if x > 10 else 'Low'
            )

            top_features = feature_importance.head(5)

            # Create SHAP heatmap
            try:
                plt.figure(figsize=(12, 8))
                heatmap_data = feature_importance[['feature', 'percent_contribution', 'positive_impact', 'negative_impact']].set_index('feature')
                heatmap_data.columns = ['% Contribution', 'Positive Impact', 'Negative Impact']
                sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'Impact Intensity'})
                plt.title(f'Feature Impact on View Velocity\nCategory: {self.category_names.get(str(category_id), "Category " + str(category_id))} ({self.country_code})', fontsize=14)
                plt.xlabel('Metrics', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.tight_layout()
                plot_path = f'{self.country_code}_output/{self.country_code}_shap_impact_category_{category_id}.png'
                plt.savefig(plot_path, dpi=300)
                plt.close()
            except Exception as e:
                recommendations.append(f"Could not create SHAP heatmap: {str(e)}")
                plot_path = None

            # SHAP summary
            summary = (
                f"📊 SHAP Analysis Summary\n"
                f"- SHAP analysis measures the impact of each factor on video view velocity.\n"
                f"- Positive factors help videos trend faster, while negative ones may hinder.\n"
                f"- Top 5 most important factors:\n"
            )
            for i, row in top_features.iterrows():
                summary += f"  • {row['feature']}: {row['percent_contribution']:.1f}% (Priority: {row['priority']})\n"
            recommendations.append(summary)

            # Create summary table
            table = (
                f"\n📋 SHAP Summary Table\n"
                f"+{'-'*25}+{'-'*12}+{'-'*10}+\n"
                f"| {'Factor':<23} | {'Contribution':<10} | {'Priority':<8} |\n"
                f"+{'-'*25}+{'-'*12}+{'-'*10}+\n"
            )
            for i, row in top_features.iterrows():
                table += f"| {row['feature']:<23} | {row['percent_contribution']:<10.1f}% | {row['priority']:<8} |\n"
            table += f"+{'-'*25}+{'-'*12}+{'-'*10}+\n"
            recommendations.append(table)

            if plot_path:
                recommendations.append(
                    f"📈 SHAP Impact Heatmap\n"
                    f"- View heatmap at: {plot_path}\n"
                    f"- The heatmap shows the impact intensity of each factor."
                )

            # Detailed recommendations
            recommendations.append("\n🔍 Detailed Recommendations")
            for i, row in top_features.iterrows():
                feature = row['feature']
                percent_contribution = row['percent_contribution']
                pos_impact = row['positive_impact']
                neg_impact = row['negative_impact']
                p25 = row['p25']
                p75 = row['p75']
                priority = row['priority']

                if feature == 'view_count':
                    rec = (f"• Views ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Target: {p75:,.0f} views within 24h\n"
                           f"  - Reason: Helps trending in {pos_impact:.0%} of videos, but hinders in {neg_impact:.0%} if growth is slow.")
                elif feature == 'likes':
                    rec = (f"• Likes ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Target: {p75:,.0f} likes within 6h\n"
                           f"  - Reason: Increasing like/view ratio (> {(p75/category_data['view_count'].median()):.2%}) boosts trending.")
                elif feature == 'dislikes':
                    rec = (f"• Dislikes ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Keep: Below {p25:,.0f} dislikes within 24h\n"
                           f"  - Reason: High dislike/view ratio hinders trending in {neg_impact:.0%} of videos.")
                elif feature == 'comment_count':
                    days_median = category_data['days_since_publication'].median()
                    if days_median == 0 or pd.isna(days_median):
                        comments_per_day = "N/A (insufficient time data)"
                    else:
                        comments_per_day = f"{p75/days_median:.0f}/day"
                    rec = (f"• Comments ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Target: {p75:,.0f} comments within 12h (~{comments_per_day})\n"
                           f"  - Reason: Higher engagement boosts trending in {pos_impact:.0%} of videos.")
                elif feature == 'engagement_rate':
                    rec = (f"• Engagement Rate ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Target: >{p75:.2%}\n"
                           f"  - Reason: High engagement promotes trending in {pos_impact:.0%} of videos.")
                elif feature == 'like_dislike_ratio':
                    rec = (f"• Like/Dislike Ratio ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Target: >{p75:.1f}\n"
                           f"  - Reason: High ratio helps trending in {pos_impact:.0%} of videos.")
                elif feature == 'comment_view_ratio':
                    rec = (f"• Comment/View Ratio ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Target: >{p75:.4f} (1 comment per {(1/p75):.0f} views)\n"
                           f"  - Reason: Higher interaction helps trending in {pos_impact:.0%} of videos.")
                elif feature == 'dislikes_per_comment':
                    rec = (f"• Dislike/Comment Ratio ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Keep: <{p25:.2f}\n"
                           f"  - Reason: High ratio indicates controversial content, hindering {neg_impact:.0%} of videos.")
                elif feature == 'days_since_publication':
                    rec = (f"• Time Since Publication ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Target: Trend within {p25:.0f} days\n"
                           f"  - Reason: Longer times hinder trending in {neg_impact:.0%} of videos due to loss of freshness.")
                elif feature == 'likes_per_day':
                    rec = (f"• Likes/Day ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Target: {p75:,.0f}/day\n"
                           f"  - Reason: High rate helps trending in {pos_impact:.0%} of videos.")
                elif feature == 'comments_per_day':
                    rec = (f"• Comments/Day ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Target: {p75:,.0f}/day\n"
                           f"  - Reason: Increases interaction, helping trending in {pos_impact:.0%} of videos.")
                elif feature == 'title_length':
                    rec = (f"• Title Length ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Keep: Between {p25:.0f}-{p75:.0f} characters\n"
                           f"  - Reason: Optimal length boosts trending in {pos_impact:.0%} of videos.")
                elif feature == 'title_sentiment':
                    rec = (f"• Title Sentiment ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Target: Sentiment > {p75:.2f} (positive)\n"
                           f"  - Reason: Positive titles promote trending in {pos_impact:.0%} of videos.")
                recommendations.append(rec)

            return recommendations, feature_importance.head(5)

        except Exception as e:
            return [f"Could not perform SHAP analysis: {str(e)}"], pd.DataFrame()

    def generate_eda_recommendations(self, insights, top_features, category_data):
        """
        Generate content optimization recommendations based on EDA.

        Args:
            insights (dict): Insights for the category.
            top_features (pd.DataFrame): Top features from SHAP analysis.
            category_data (pd.DataFrame): Data for the specific category.

        Returns:
            list: List of EDA-based recommendations.
        """
        recommendations = []

        if insights['trending_patterns'] and insights['optimal_hour'] is not None:
            patterns = insights['trending_patterns']
            rec = (f"• Post videos on {patterns['peak_day']} at {insights['optimal_hour']:.0f}h "
                   f"to trend faster (around {patterns['avg_time']:.1f} hours). "
                   f"This reduces publication time, a key factor ({top_features.iloc[0]['percent_contribution']:.1f}%).")
            recommendations.append(rec)

        if insights['tags']:
            rec = f"• Use popular tags: {', '.join(insights['tags'])} to increase visibility."
            recommendations.append(rec)

        # Recommend optimal number of tags (assuming 5 is optimal based on prior analysis)
        optimal_num_tags = 5
        rec = f"• Use around {optimal_num_tags} tags to maximize trending potential."
        recommendations.append(rec)

        if insights['time_to_trend']:
            time_stats = insights['time_to_trend']
            if time_stats.get('sample_count', 0) > 1 and time_stats.get('std') is not None and not np.isnan(time_stats['std']) and time_stats['std'] > 0:
                std_text = f"±{time_stats['std']:.1f}"
                rec = (f"• Aim to trend within {time_stats['median']:.1f} hours "
                       f"(variation {std_text} hours) to optimize view velocity.")
            else:
                rec = (f"• Aim to trend within {time_stats['median']:.1f} hours "
                       f"to optimize view velocity (limited data, based on median).")
            recommendations.append(rec)

        # Recommend title keywords
        top_words_list = self.top_title_words['Word'].head(5).tolist()
        rec = f"• Use popular keywords in titles: {', '.join(top_words_list)} to attract viewers."
        recommendations.append(rec)

        return recommendations if recommendations else ["No optimization tips from data analysis."]

    def get_recommendations(self, category_id):
        """
        Generate comprehensive recommendations for a specific category.

        Args:
            category_id (str): Category ID.

        Returns:
            str: Formatted recommendations.
        """
        try:
            category_id = str(category_id)
            if category_id not in self.category_names:
                return f"Category {category_id} does not exist."

            insights = self.get_category_insights(category_id)
            category_data = self.df[self.df['categoryId'].astype(str) == category_id]

            missing_features = [f for f in self.features if f not in category_data.columns]
            if missing_features:
                shap_recs = [f"Cannot perform SHAP analysis: Missing columns in category_data: {missing_features}"]
                top_features = pd.DataFrame()
            else:
                shap_recs, top_features = self.generate_shap_recommendations(category_data, category_id)

            eda_recs = self.generate_eda_recommendations(insights, top_features, category_data)

            # Priority actions
            priority_actions = []
            if not top_features.empty:
                for i, row in top_features.head(2).iterrows():
                    if row['feature'] == 'days_since_publication':
                        priority_actions.append(f"• Post and promote heavily within the first {row['p25']:.0f} days to trend quickly.")
                    elif row['feature'] == 'comment_count':
                        priority_actions.append(f"• Encourage {row['p75']:,.0f} comments within 12h by prompting engagement.")
                    elif row['feature'] == 'view_count':
                        priority_actions.append(f"• Achieve {row['p75']:,.0f} views within 24h through ads and sharing.")
                    elif row['feature'] == 'likes_per_day':
                        priority_actions.append(f"• Achieve {row['p75']:,.0f} likes per day with compelling content.")
                    elif row['feature'] == 'comments_per_day':
                        priority_actions.append(f"• Achieve {row['p75']:,.0f} comments per day through questions or discussions.")
                    elif row['feature'] == 'title_length':
                        priority_actions.append(f"• Keep title length between {row['p25']:.0f}-{row['p75']:.0f} characters to optimize trending.")
                    elif row['feature'] == 'title_sentiment':
                        priority_actions.append(f"• Ensure titles have positive sentiment (> {row['p75']:.2f}) to boost trending chances.")

            output = [
                f"📊 Recommendations for Category {insights['name']} ({self.country_code})",
                "="*50,
            ]
            if priority_actions:
                output.append("\n🚀 Priority Actions")
                output.extend(priority_actions)
            output.extend([
                "\n🔍 Key Factor Analysis (SHAP)",
                *shap_recs,
                "\n💡 Content Optimization Tips",
                *eda_recs
            ])

            return "\n".join(output)

        except Exception as e:
            return f"Error generating recommendations: {str(e)}"

if __name__ == "__main__":
    countries = ['US', 'IN', 'BR', 'GB', 'KR']
    print("🎯 YouTube Content Recommendation System")
    print("--------------------------------------")
    print("Available countries: US, IN, BR, GB, KR")

    while True:
        country_input = input("\nEnter your country (or 'exit' to quit): ").strip().upper()
        if country_input.lower() == 'exit':
            break
        if country_input not in countries:
            print("⚠️ Invalid country")
            continue

        recommender = YouTubeRecommendationSystem(country_input)
        try:
            recommender.load_model_and_shap()
        except Exception as e:
            print(f"Could not load model or SHAP: {str(e)}")
            continue

        print(f"\nAvailable categories ({country_input}):")
        for cat_id, name in recommender.category_names.items():
            print(f"- {cat_id}: {name}")

        while True:
            user_input = input("\nEnter category ID or name (or 'back' to select another country, 'exit' to quit): ").strip()
            if user_input.lower() == 'exit':
                exit()
            elif user_input.lower() == 'back':
                break

            category_id = None
            if user_input.isdigit():
                if user_input in recommender.category_names:
                    category_id = user_input
            else:
                for cid, name in recommender.category_names.items():
                    if user_input.lower() in name.lower():
                        category_id = cid
                        break

            if not category_id:
                print("⚠️ Category not found")
                continue

            recommendations = recommender.get_recommendations(category_id)
            print("\n" + recommendations)