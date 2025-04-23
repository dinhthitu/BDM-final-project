import pandas as pd
import numpy as np
import shap
import json
import joblib
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class YouTubeRecommendationSystem:
    def __init__(self, country_code):
        self.country_code = country_code
        sns.set(style='darkgrid', palette='deep', font='DejaVu Sans', font_scale=1.2)
        self.setup_features()
        self.load_data()
        self.X_test = None
        self.shap_values = None
        self.model = None
    
    def load_data(self):
        try:
            # Đọc file từ thư mục con tương ứng với quốc gia
            self.df = pd.read_csv(f'{self.country_code}_preprocessed_youtube_trending_data.csv')
            
            missing_features = [f for f in self.features if f not in self.df.columns]
            if missing_features:
                raise ValueError(f"Thiếu các cột trong self.df: {missing_features}")
            
            with open(f'{self.country_code}_category_id_to_name.json', 'r') as f:
                self.category_names = json.load(f)
            
            self.category_stats = pd.read_csv(f'{self.country_code}_output/{self.country_code}_category_stats.csv')
            self.trending_stats = pd.read_csv(f'{self.country_code}_output/{self.country_code}_trending_stats_by_category.csv')
            self.tag_stats = pd.read_csv(f'{self.country_code}_output/{self.country_code}_most_popular_tags_by_category.csv')
            self.hours_to_trend = pd.read_csv(f'{self.country_code}_output/{self.country_code}_avg_hours_to_trend_by_category.csv')
            
        except Exception as e:
            raise ValueError(f"Lỗi khi tải dữ liệu: {str(e)}")

    def setup_features(self):
        self.features = [
            'view_count', 'likes', 'dislikes', 'comment_count',
            'engagement_rate', 'like_dislike_ratio', 'comment_view_ratio',
            'dislikes_per_comment', 'days_since_publication', 'likes_per_day',
            'comments_per_day', 'title_length', 'title_sentiment'
        ]
    
    def load_model_and_shap(self):
        try:
            # Đọc file JSON để biết mô hình tốt nhất
            with open(f'{self.country_code}_output/{self.country_code}_best_model.json', 'r') as f:
                model_info = json.load(f)
            best_model_name = model_info['best_model']

            # Đọc X_test từ thư mục con
            self.X_test = pd.read_csv(f'{self.country_code}_output/{self.country_code}_X_test.csv', index_col=0)
            
            available_features = [f for f in self.features if f in self.X_test.columns]
            if len(available_features) < len(self.features):
                missing = [f for f in self.features if f not in self.X_test.columns]
                raise ValueError(f"Thiếu các cột trong X_test: {missing}")
            self.X_test = self.X_test[available_features]
            
            # Tải mô hình và SHAP values dựa trên best_model_name
            if best_model_name == "XGBoost":
                self.model = joblib.load(f'{self.country_code}_output/{self.country_code}_model_xgb.joblib')
                self.shap_values = joblib.load(f'{self.country_code}_output/{self.country_code}_shap_values_xgb.joblib')
            else:  # LightGBM
                self.model = joblib.load(f'{self.country_code}_output/{self.country_code}_model_lgb.joblib')
                self.shap_values = joblib.load(f'{self.country_code}_output/{self.country_code}_shap_values_lgb.joblib')
            
        except Exception as e:
            raise ValueError(f"Lỗi khi tải mô hình hoặc SHAP: {str(e)}")
    
    def get_category_insights(self, category_id):
        category_id_str = str(category_id)
        category_id_int = int(category_id)
        category_name = self.category_names.get(category_id_str, f"Category {category_id_str}")

        insights = {
            'name': category_name,
            'general_stats': {},
            'trending_patterns': {},
            'tags': [],
            'time_to_trend': {}
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
                        'peak_hour': patterns.iloc[0].get('Giờ đăng trung bình', 'N/A'),
                        'avg_time': float(patterns.iloc[0].get('Thời gian lên trending (giờ)', 0))
                    }

            if not self.tag_stats.empty:
                tags_rows = self.tag_stats[self.tag_stats['Category'] == category_name]
                if not tags_rows.empty:
                    valid_tags = [tag for tag in tags_rows['Tag'].head(5).tolist() if tag != '[none]']
                    insights['tags'] = valid_tags

            if not self.hours_to_trend.empty:
                time_stats = self.hours_to_trend[self.hours_to_trend['categoryId'] == category_id_int]
                if not time_stats.empty:
                    print(f"Debug: Số lượng mẫu cho categoryId={category_id_int} ({self.country_code}): {len(time_stats)}")
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
        recommendations = []
        
        try:
            test_indices = self.X_test.index.intersection(category_data.index)
            if len(test_indices) == 0:
                return ["Không có đủ dữ liệu để phân tích SHAP."], pd.DataFrame()
            
            test_positions = [list(self.X_test.index).index(idx) for idx in test_indices]
            
            missing_features = [f for f in self.features if f not in category_data.columns]
            if missing_features:
                raise ValueError(f"Thiếu các cột trong category_data: {missing_features}")
            
            missing_test_features = [f for f in self.features if f not in self.X_test.columns]
            if missing_test_features:
                raise ValueError(f"Thiếu các cột trong X_test: {missing_test_features}")
            
            if category_data[self.features].isna().any().any():
                raise ValueError("Dữ liệu chứa giá trị NaN.")
            
            if isinstance(self.shap_values, shap.Explanation):
                shap_values_array = self.shap_values.values
                shap_feature_names = self.shap_values.feature_names
                if shap_feature_names and shap_feature_names != self.features:
                    raise ValueError(f"Tên đặc trưng SHAP không khớp với đặc trưng mô hình.")
            else:
                shap_values_array = self.shap_values
            
            if shap_values_array.shape[1] != len(self.features):
                raise ValueError(f"Kích thước SHAP values không khớp với số đặc trưng.")
            
            category_shap = shap_values_array[test_positions]
            
            shap_df = pd.DataFrame(category_shap, columns=self.features)
            
            if shap_df.isna().any().any():
                raise ValueError("SHAP values chứa giá trị NaN.")
            
            # Tính feature_importance với p25, p75
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
                raise ValueError("Tổng đóng góp SHAP bằng 0.")
            feature_importance['percent_contribution'] = feature_importance['mean_abs_shap'] / total_shap * 100
            
            # Xác định ưu tiên
            feature_importance['priority'] = feature_importance['percent_contribution'].apply(
                lambda x: 'Cao' if x > 30 else 'Trung bình' if x > 10 else 'Thấp'
            )
            
            top_features = feature_importance.head(5)
            
            # Tạo heatmap SHAP
            try:
                plt.figure(figsize=(12, 8))
                heatmap_data = feature_importance[['feature', 'percent_contribution', 'positive_impact', 'negative_impact']].set_index('feature')
                heatmap_data.columns = ['% Contribution', 'Positive Impact', 'Negative Impact']
                sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'Cường độ ảnh hưởng'})
                plt.title(f'Ảnh hưởng của các đặc trưng đến tốc độ lượt xem\nDanh mục: {self.category_names.get(str(category_id), "Category " + str(category_id))} ({self.country_code})', fontsize=14)
                plt.xlabel('Metrics', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.tight_layout()
                # Lưu heatmap vào thư mục output của quốc gia
                plot_path = f'{self.country_code}_output/{self.country_code}_shap_impact_category_{category_id}.png'
                plt.savefig(plot_path, dpi=300)
                plt.close()
            except Exception as e:
                recommendations.append(f"Không thể tạo heatmap SHAP: {str(e)}")
                plot_path = None
            
            # Tóm tắt SHAP
            summary = (
                f"📊 Tóm tắt phân tích SHAP\n"
                f"- Phân tích SHAP đo lường mức độ mỗi yếu tố ảnh hưởng đến tốc độ lượt xem của video.\n"
                f"- Yếu tố tích cực giúp video trending nhanh, yếu tố tiêu cực có thể cản trở.\n"
                f"- 5 yếu tố quan trọng nhất:\n"
            )
            for i, row in top_features.iterrows():
                summary += f"  • {row['feature']}: {row['percent_contribution']:.1f}% (Ưu tiên: {row['priority']})\n"
            recommendations.append(summary)
            
            # Tạo bảng tóm tắt
            table = (
                f"\n📋 Bảng tóm tắt SHAP\n"
                f"+{'-'*25}+{'-'*12}+{'-'*10}+\n"
                f"| {'Yếu tố':<23} | {'Đóng góp':<10} | {'Ưu tiên':<8} |\n"
                f"+{'-'*25}+{'-'*12}+{'-'*10}+\n"
            )
            for i, row in top_features.iterrows():
                table += f"| {row['feature']:<23} | {row['percent_contribution']:<10.1f}% | {row['priority']:<8} |\n"
            table += f"+{'-'*25}+{'-'*12}+{'-'*10}+\n"
            recommendations.append(table)
            
            # Thêm thông báo về heatmap
            if plot_path:
                recommendations.append(
                    f"📈 Heatmap ảnh hưởng SHAP\n"
                    f"- Xem heatmap tại: {plot_path}\n"
                    f"- Heatmap hiển thị mức độ ảnh hưởng của các yếu tố. Cột '% Contribution' cho thấy tỷ lệ đóng góp của từng yếu tố."
                )
            
            # Khuyến nghị chi tiết
            recommendations.append("\n🔍 Khuyến nghị chi tiết")
            for i, row in top_features.iterrows():
                feature = row['feature']
                percent_contribution = row['percent_contribution']
                pos_impact = row['positive_impact']
                neg_impact = row['negative_impact']
                p25 = row['p25']
                p75 = row['p75']
                priority = row['priority']
                
                if feature == 'view_count':
                    rec = (f"• Lượt xem ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Phải đạt: {p75:,.0f} lượt trong 24h\n"
                           f"  - Lý do: Giúp trending nhanh trong {pos_impact:.0%} video, nhưng cản trở trong {neg_impact:.0%} nếu tăng chậm.")
                elif feature == 'likes':
                    rec = (f"• Lượt thích ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Phải đạt: {p75:,.0f} lượt trong 6h\n"
                           f"  - Lý do: Tăng tỷ lệ like/view (> {(p75/category_data['view_count'].median()):.2%}) giúp trending nhanh.")
                elif feature == 'dislikes':
                    rec = (f"• Lượt không thích ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Phải giữ: Dưới {p25:,.0f} lượt trong 24h\n"
                           f"  - Lý do: Tỷ lệ dislike/view cao cản trở trending trong {neg_impact:.0%} video.")
                elif feature == 'comment_count':
                    rec = (f"• Bình luận ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Phải đạt: {p75:,.0f} bình luận trong 12h (~{p75/category_data['days_since_publication'].median():.0f}/ngày)\n"
                           f"  - Lý do: Tăng tương tác giúp trending trong {pos_impact:.0%} video.")
                elif feature == 'engagement_rate':
                    rec = (f"• Tỷ lệ tương tác ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Phải đạt: >{p75:.2%}\n"
                           f"  - Lý do: Tương tác cao thúc đẩy trending trong {pos_impact:.0%} video.")
                elif feature == 'like_dislike_ratio':
                    rec = (f"• Tỷ lệ like/dislike ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Phải đạt: >{p75:.1f}\n"
                           f"  - Lý do: Tỷ lệ cao giúp trending trong {pos_impact:.0%} video.")
                elif feature == 'comment_view_ratio':
                    rec = (f"• Tỷ lệ bình luận/view ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Phải đạt: >{p75:.4f} (1 bình luận/{(1/p75):.0f} lượt xem)\n"
                           f"  - Lý do: Tăng tương tác giúp trending trong {pos_impact:.0%} video.")
                elif feature == 'dislikes_per_comment':
                    rec = (f"• Tỷ lệ dislike/bình luận ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Phải giữ: <{p25:.2f}\n"
                           f"  - Lý do: Tỷ lệ cao cho thấy nội dung gây tranh cãi, cản trở trong {neg_impact:.0%} video.")
                elif feature == 'days_since_publication':
                    rec = (f"• Thời gian từ khi đăng ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Phải đạt: Trending trong {p25:.0f} ngày\n"
                           f"  - Lý do: Thời gian dài cản trở trending trong {neg_impact:.0%} video vì mất tính mới mẻ.")
                elif feature == 'likes_per_day':
                    rec = (f"• Lượt thích/ngày ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Phải đạt: {p75:,.0f}/ngày\n"
                           f"  - Lý do: Tốc độ cao giúp trending trong {pos_impact:.0%} video.")
                elif feature == 'comments_per_day':
                    rec = (f"• Bình luận/ngày ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Phải đạt: {p75:,.0f}/ngày\n"
                           f"  - Lý do: Tăng tương tác giúp trending trong {pos_impact:.0%} video.")
                elif feature == 'title_length':
                    rec = (f"• Độ dài tiêu đề ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Nên giữ: Khoảng {p25:.0f}-{p75:.0f} ký tự\n"
                           f"  - Lý do: Độ dài phù hợp giúp tăng tốc độ trending trong {pos_impact:.0%} video.")
                elif feature == 'title_sentiment':
                    rec = (f"• Cảm xúc tiêu đề ({percent_contribution:.1f}%, {priority}):\n"
                           f"  - Nên giữ: Sentiment > {p75:.2f} (tích cực)\n"
                           f"  - Lý do: Tiêu đề tích cực thúc đẩy trending trong {pos_impact:.0%} video.")
                recommendations.append(rec)
            
            return recommendations, feature_importance.head(5)
        
        except Exception as e:
            recommendations = [f"Không thể phân tích SHAP: {str(e)}"]
            return recommendations

    def get_top_velocity_factors(self, category_data):
        if 'view_velocity' in category_data.columns:
            corr = category_data.corr()['view_velocity'].abs().sort_values(ascending=False)
            top_factors = corr[1:4].index.tolist()
            return ", ".join(top_factors)
        return "Không có dữ liệu view_velocity"
    
    def generate_eda_recommendations(self, insights, top_features, category_data):
        recommendations = []
        
        if insights['trending_patterns']:
            patterns = insights['trending_patterns']
            rec = (f"• Đăng video vào {patterns['peak_day']} lúc {patterns['peak_hour']} giờ "
                   f"để lên trending nhanh (khoảng {patterns['avg_time']:.1f} giờ). "
                   f"Điều này giúp giảm thời gian đăng, yếu tố quan trọng nhất ({top_features.iloc[0]['percent_contribution']:.1f}%).")
            recommendations.append(rec)
        
        if insights['tags']:
            rec = f"• Sử dụng các tag phổ biến: {', '.join(insights['tags'])} để tăng khả năng hiển thị."
            recommendations.append(rec)
        
        if insights['time_to_trend']:
            time_stats = insights['time_to_trend']
            if time_stats.get('sample_count', 0) > 1 and time_stats.get('std') is not None and not np.isnan(time_stats['std']) and time_stats['std'] > 0:
                std_text = f"±{time_stats['std']:.1f}"
                rec = (f"• Nhắm đến thời gian trending dưới {time_stats['median']:.1f} giờ "
                       f"(dao động {std_text} giờ) để tối ưu hóa tốc độ lượt xem.")
            else:
                rec = (f"• Nhắm đến thời gian trending dưới {time_stats['median']:.1f} giờ "
                       f"để tối ưu hóa tốc độ lượt xem (dữ liệu hạn chế, chỉ dựa trên giá trị trung bình).")
            recommendations.append(rec)
        
        return recommendations if recommendations else ["Không có mẹo tối ưu từ phân tích dữ liệu."]
    
    def get_recommendations(self, category_id):
        try:
            category_id = str(category_id)
            
            if category_id not in self.category_names:
                return f"Danh mục {category_id} không tồn tại."
            
            insights = self.get_category_insights(category_id)
            category_data = self.df[self.df['categoryId'].astype(str) == category_id]
            
            missing_features = [f for f in self.features if f not in category_data.columns]
            if missing_features:
                shap_recs = [f"Không thể phân tích SHAP: Thiếu các cột trong category_data: {missing_features}"]
                top_features = pd.DataFrame()
            else:
                shap_recs, top_features = self.generate_shap_recommendations(category_data, category_id)
            
            eda_recs = self.generate_eda_recommendations(insights, top_features, category_data)
            
            # Tạo hành động ưu tiên
            priority_actions = []
            if not top_features.empty:
                for i, row in top_features.head(2).iterrows():
                    if row['feature'] == 'days_since_publication':
                        priority_actions.append(f"• Đăng video và quảng bá mạnh trong {row['p25']:.0f} ngày đầu để trending nhanh.")
                    elif row['feature'] == 'comment_count':
                        priority_actions.append(f"• Khuyến khích {row['p75']:,.0f} bình luận trong 12h bằng cách kêu gọi tương tác.")
                    elif row['feature'] == 'view_count':
                        priority_actions.append(f"• Đạt {row['p75']:,.0f} lượt xem trong 24h qua quảng cáo và chia sẻ.")
                    elif row['feature'] == 'likes_per_day':
                        priority_actions.append(f"• Đạt {row['p75']:,.0f} lượt thích mỗi ngày bằng nội dung hấp dẫn.")
                    elif row['feature'] == 'comments_per_day':
                        priority_actions.append(f"• Đạt {row['p75']:,.0f} bình luận mỗi ngày qua các câu hỏi hoặc thảo luận.")
                    elif row['feature'] == 'title_length':
                        priority_actions.append(f"• Giữ độ dài tiêu đề trong khoảng {row['p25']:.0f}-{row['p75']:.0f} ký tự để tối ưu hóa tốc độ trending.")
                    elif row['feature'] == 'title_sentiment':
                        priority_actions.append(f"• Đảm bảo tiêu đề có cảm xúc tích cực (sentiment > {row['p75']:.2f}) để tăng khả năng trending.")
            
            output = [
                f"📊 Khuyến nghị cho danh mục {insights['name']} ({self.country_code})",
                "="*50,
            ]
            if priority_actions:
                output.append("\n🚀 Hành động ưu tiên")
                output.extend(priority_actions)
            output.extend([
                "\n🔍 Phân tích yếu tố quan trọng (SHAP)",
                *shap_recs,
                "\n💡 Mẹo tối ưu hóa nội dung",
                *eda_recs
            ])
            
            return "\n".join(output)
        
        except Exception as e:
            return f"Lỗi khi tạo khuyến nghị: {str(e)}"

if __name__ == "__main__":
    countries = ['US', 'IN', 'BR', 'GB', 'KR']
    print("🎯 Hệ thống khuyến nghị nội dung YouTube")
    print("--------------------------------------")
    print("Các quốc gia có sẵn: US, IN, BR, GB, KR")
    
    while True:
        country_input = input("\nNhập quốc gia bạn đang sống (nhập 'exit' để thoát): ").strip().upper()
        
        if country_input.lower() == 'exit':
            break
        
        if country_input not in countries:
            print("⚠️ Quốc gia không hợp lệ")
            continue
        
        recommender = YouTubeRecommendationSystem(country_input)
        
        try:
            recommender.load_model_and_shap()
        except Exception as e:
            print(f"Không thể tải mô hình hoặc SHAP: {str(e)}")
            continue
        
        print(f"\nDanh mục có sẵn ({country_input}):")
        for cat_id, name in recommender.category_names.items():
            print(f"- {cat_id}: {name}")
        
        while True:
            user_input = input("\nNhập ID hoặc tên danh mục (nhập 'back' để chọn quốc gia khác, 'exit' để thoát): ").strip()
            
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
                print("⚠️ Danh mục không tìm thấy")
                continue
            
            recommendations = recommender.get_recommendations(category_id)
            print("\n" + recommendations)