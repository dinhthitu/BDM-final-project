import pandas as pd
from fpdf import FPDF, XPos, YPos
import os
from PIL import Image
import logging
from Recommend import YouTubeRecommendationSystem
import warnings
warnings.filterwarnings('ignore', category=Warning)

def setup_logging(output_dir):
    """Set up logging to a file in the output directory."""
    log_file = os.path.join(output_dir, 'report_generation.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def clean_text(text, max_length=100):
    """Clean and truncate text for PDF output with enhanced error handling."""
    if pd.isna(text) or not text:
        return "N/A"
    text = str(text)
    # Replace known OCR errors and normalize
    replacements = {
        'silhouette': 'since',
        'fortnighte': 'fortnite',
        'days silhouette publication': 'days_since_publication'
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    # Remove non-printable characters and normalize
    text = ''.join(c for c in text if c.isprintable() and ord(c) < 128)
    text = ' '.join(text.split())
    if len(text) > max_length:
        text = text[:max_length-3] + '...'
    return text if text else "Unknown"

class PDF(FPDF):
    def __init__(self, country_code):
        super().__init__()
        self.country_code = country_code
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(left=15, top=15, right=15)
        font_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        font_path = os.path.join(font_dir, 'DejaVuSans.ttf')
        font_bold_path = os.path.join(font_dir, 'DejaVuSans-Bold.ttf')
        if not os.path.exists(font_path) or not os.path.exists(font_bold_path):
            raise FileNotFoundError(f"Font files not found in {font_dir}")
        self.add_font('DejaVu', '', font_path)
        self.add_font('DejaVu', 'B', font_bold_path)
        self.set_font('DejaVu', '', 10)

    def header(self):
        self.set_font('DejaVu', 'B', 16)
        self.cell(0, 10, f'YouTube Trending Analysis Report ({self.country_code})', 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(8)

    def chapter_title(self, title):
        self.set_font('DejaVu', 'B', 14)
        self.cell(0, 10, title, 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)

    def wrap_text(self, text, max_width, font_size):
        """Wrap text to fit within the specified width."""
        self.set_font('DejaVu', '', font_size)
        words = text.split(' ')
        lines = []
        current_line = ''
        for word in words:
            if self.get_string_width(current_line + word) < max_width:
                current_line += word + ' '
            else:
                lines.append(current_line.strip())
                current_line = word + ' '
        if current_line:
            lines.append(current_line.strip())
        return lines or ['']

    def add_section(self, title, content=None, image_path=None, table_data=None, bullet_list=None, sub_bullet_lists=None):
        if title:
            self.add_page()
            self.chapter_title(title)
        
        available_width = self.w - self.l_margin - self.r_margin
        
        if content:
            content = clean_text(content)
            self.set_font('DejaVu', '', 10)
            wrapped_content = self.wrap_text(content, available_width, 10)
            for line in wrapped_content:
                self.multi_cell(0, 7, line)
            self.ln(4)
        
        if bullet_list:
            self.set_font('DejaVu', '', 10)
            for item in bullet_list:
                item = clean_text(item)
                wrapped_lines = self.wrap_text(f"• {item}", available_width - 10, 10)
                for line in wrapped_lines:
                    self.set_x(20)
                    self.multi_cell(0, 7, line)
                if sub_bullet_lists and item in sub_bullet_lists:
                    for sub_item in sub_bullet_lists[item]:
                        sub_item = clean_text(sub_item)
                        wrapped_sub_lines = self.wrap_text(f"  - {sub_item}", available_width - 15, 10)
                        for sub_line in wrapped_sub_lines:
                            self.set_x(25)
                            self.multi_cell(0, 7, sub_line)
            self.ln(4)
        
        if image_path and os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                    if img_width == 0 or img_height == 0:
                        raise ValueError("Image has invalid dimensions")
                    aspect_ratio = img_width / img_height
                    new_width = min(180, available_width * 0.9)
                    new_height = new_width / aspect_ratio
                    max_height = (self.h - 2 * self.t_margin) * 0.66
                    if new_height > max_height:
                        new_height = max_height
                        new_width = new_height * aspect_ratio
                    min_width, min_height = 50, 50
                    if new_width < min_width:
                        new_width = min_width
                        new_height = new_width / aspect_ratio
                    if new_height < min_height:
                        new_height = min_height
                        new_width = new_height * aspect_ratio
                
                x_pos = (self.w - new_width) / 2
                if self.get_y() + new_height + 15 > self.h - self.b_margin:
                    self.add_page()
                self.image(image_path, x=x_pos, w=new_width)
                self.ln(8)
            except Exception as e:
                logging.error(f"Failed to load image {image_path}: {str(e)}")
                self.multi_cell(0, 7, f"Could not load image: {clean_text(str(e))}")
                self.ln(4)
        
        if table_data is not None:
            if table_data.columns.tolist() == ['Category', 'Percentage', 'Avg Views', 'Avg Likes']:
                self.add_category_table(table_data)
            else:
                self.add_table(table_data)

    def add_category_table(self, data):
        """Render a table for category stats without Top Tag."""
        if data.empty:
            self.set_font('DejaVu', '', 9)
            self.multi_cell(0, 7, "No data available for table.")
            self.ln(4)
            return
        
        available_width = self.w - self.l_margin - self.r_margin
        col_widths = [60, 30, 50, 40]  # Category, Percentage, Avg Views, Avg Likes
        total_width = sum(col_widths)
        if total_width > available_width:
            scale_factor = available_width / total_width
            col_widths = [w * scale_factor for w in col_widths]
        
        self.set_font('DejaVu', 'B', 9)
        headers = ['Category', 'Percentage', 'Avg Views', 'Avg Likes']
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 9, header, border=1)
        self.ln()
        
        self.set_font('DejaVu', '', 9)
        for _, row in data.iterrows():
            wrapped_category = self.wrap_text(clean_text(row['Category']), col_widths[0] - 6, 9)
            self.cell(col_widths[0], 7, wrapped_category[0], border=1)
            self.cell(col_widths[1], 7, clean_text(row['Percentage']), border=1)
            self.cell(col_widths[2], 7, clean_text(row['Avg Views']), border=1)
            self.cell(col_widths[3], 7, clean_text(row['Avg Likes']), border=1)
            self.ln()

    def add_table(self, data):
        """Render a table with merged Category and Top Tag cells."""
        if data.empty:
            self.set_font('DejaVu', '', 9)
            self.multi_cell(0, 7, "No data available for table.")
            self.ln(4)
            return
        
        available_width = self.w - self.l_margin - self.r_margin
        col_widths = [60, 30, 60, 20]  # Category, Top Tag, Keyword, Count
        total_width = sum(col_widths)
        if total_width > available_width:
            scale_factor = available_width / total_width
            col_widths = [w * scale_factor for w in col_widths]
        
        self.set_font('DejaVu', 'B', 9)
        for i, header in enumerate(data.columns):
            self.cell(col_widths[i], 9, str(header), border=1)
        self.ln()
        
        try:
            grouped = data.groupby(['Category', 'Top Tag'])
        except Exception as e:
            logging.error(f"Failed to group table data: {str(e)}")
            self.set_font('DejaVu', '', 9)
            self.multi_cell(0, 7, f"Failed to render table: {clean_text(str(e))}")
            self.ln(4)
            return
        
        self.set_font('DejaVu', '', 9)
        for (category, top_tag), group in grouped:
            num_rows = max(len(group), 1)
            row_height = 7
            total_height = row_height * num_rows
            
            start_y = self.get_y()
            if start_y + total_height + 15 > self.h - self.b_margin:
                self.add_page()
                start_y = self.get_y()
            
            wrapped_category = self.wrap_text(clean_text(category), col_widths[0] - 6, 9)
            wrapped_top_tag = self.wrap_text(clean_text(top_tag), col_widths[1] - 6, 9)
            
            try:
                self.multi_cell(col_widths[0], total_height / max(len(wrapped_category), 1), wrapped_category[0], border=1, align='L')
                self.set_xy(self.l_margin + col_widths[0], start_y)
                self.multi_cell(col_widths[1], total_height / max(len(wrapped_top_tag), 1), wrapped_top_tag[0], border=1, align='L')
            except Exception as e:
                logging.error(f"Failed to render Category/Top Tag for {category}: {str(e)}")
                self.set_xy(self.l_margin, start_y)
                self.multi_cell(col_widths[0], row_height, "Error", border=1)
                self.set_xy(self.l_margin + col_widths[0], start_y)
                self.multi_cell(col_widths[1], row_height, "Error", border=1)
            
            if group.empty:
                self.set_xy(self.l_margin + col_widths[0] + col_widths[1], start_y)
                self.cell(col_widths[2], row_height, "N/A", border=1)
                self.cell(col_widths[3], row_height, "N/A", border=1)
            else:
                for idx, row in group.iterrows():
                    self.set_xy(self.l_margin + col_widths[0] + col_widths[1], start_y + (idx - group.index[0]) * row_height)
                    keyword_text = clean_text(row['Keyword'], max_length=50)
                    wrapped_keyword = self.wrap_text(keyword_text, col_widths[2] - 6, 9)
                    self.cell(col_widths[2], row_height, wrapped_keyword[0], border=1)
                    count_text = clean_text(str(row['Count']))
                    self.cell(col_widths[3], row_height, count_text, border=1)
            
            self.set_xy(self.l_margin, start_y + total_height)
        self.ln(4)

def generate_report(country_code):
    """Generate a comprehensive PDF report with enhanced recommendations."""
    output_dir = f'{country_code}_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    setup_logging(output_dir)
    logging.info(f"Starting report generation for {country_code}")
    
    pdf = PDF(country_code)
    
    # Section 1: Introduction
    pdf.add_section(
        "1. Introduction",
        content="This report provides an analysis of YouTube trending videos, including category performance, popular tags, keywords, trending patterns, and content optimization recommendations based on machine learning insights."
    )
    
    # Section 2: Correlation Matrix
    image_path = f'{output_dir}/{country_code}_correlation_matrix.png'
    if os.path.exists(image_path):
        pdf.add_section(
            "2. Feature Correlation",
            content="Correlation between key metrics affecting video trending performance.",
            image_path=image_path
        )
    else:
        logging.warning(f"Correlation matrix image not found: {image_path}")
        pdf.add_section(
            "2. Feature Correlation",
            content="Correlation matrix image unavailable."
        )
    
    # Section 3: Top Categories
    try:
        stats_path = f'{output_dir}/{country_code}_category_stats.csv'
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Category stats file not found: {stats_path}")
        stats = pd.read_csv(stats_path)
        top_categories = stats.sort_values('Percentage', ascending=False).head(5)[
            ['Category', 'Percentage', 'Avg Views', 'Avg Likes']
        ]
        top_categories['Percentage'] = top_categories['Percentage'].apply(lambda x: f"{x:.2f}%")
        top_categories['Avg Views'] = top_categories['Avg Views'].apply(lambda x: f"{int(x):,}")
        top_categories['Avg Likes'] = top_categories['Avg Likes'].apply(lambda x: f"{int(x):,}")
        logging.info(f"Top categories: {top_categories['Category'].tolist()}")
        pdf.add_section(
            "3. Top Categories",
            content="Top 5 categories by video count and their average engagement metrics:",
            table_data=top_categories
        )
    except Exception as e:
        logging.error(f"Failed to load category stats: {str(e)}")
        pdf.add_section(
            "3. Top Categories",
            content=f"Unable to load category stats: {clean_text(str(e))}"
        )
    
    # Section 4: Tags and Keywords
    try:
        tags_path = f'{output_dir}/{country_code}_most_popular_tags_by_category.csv'
        keywords_path = f'{output_dir}/{country_code}_top_title_keywords_by_category.csv'
        if not os.path.exists(tags_path):
            logging.warning(f"Tags file not found: {tags_path}")
            tags_df = pd.DataFrame()
        else:
            tags_df = pd.read_csv(tags_path)
        
        if not os.path.exists(keywords_path):
            logging.warning(f"Keywords file not found: {keywords_path}")
            keywords_df = pd.DataFrame()
        else:
            keywords_df = pd.read_csv(keywords_path)
        
        top_categories_list = stats.sort_values('Percentage', ascending=False)['Category'].head(5).tolist()
        
        table_data = []
        for category in top_categories_list:
            top_tag_row = tags_df[tags_df['Category'] == category][['Tag']].head(1)
            tag = top_tag_row['Tag'].iloc[0] if not top_tag_row.empty else 'N/A'
            if tag == '[none]':
                tag = 'N/A'
            
            category_keywords = keywords_df[keywords_df['Category'] == category].head(5)
            if category_keywords.empty:
                table_data.append({
                    'Category': category,
                    'Top Tag': tag,
                    'Keyword': 'N/A',
                    'Count': 'N/A'
                })
            else:
                for _, row in category_keywords.iterrows():
                    table_data.append({
                        'Category': category,
                        'Top Tag': tag,
                        'Keyword': clean_text(row['Keyword']),
                        'Count': str(row['Count'])
                    })
        
        table_df = pd.DataFrame(table_data)
        pdf.add_section(
            "4. Popular Tags and Keywords",
            content="Top tags and title keywords for the 5 most popular categories.",
            table_data=table_df
        )
    except Exception as e:
        logging.error(f"Failed to load tags/keywords: {str(e)}")
        pdf.add_section(
            "4. Popular Tags and Keywords",
            content=f"Unable to load tag/keyword data: {clean_text(str(e))}"
        )
    
    # Section 5: Trending Patterns
    trending_image = f'{output_dir}/{country_code}_trending_analysis.png'
    if os.path.exists(trending_image):
        pdf.add_section(
            "5. Trending Patterns",
            content="Distribution of trending videos by day and category:",
            image_path=trending_image
        )
    else:
        logging.warning(f"Trending patterns image not found: {trending_image}")
        pdf.add_section(
            "5. Trending Patterns",
            content="Trending patterns image unavailable."
        )
    
    time_image = f'{output_dir}/{country_code}_avg_hours_to_trend_by_category.png'
    if os.path.exists(time_image):
        pdf.add_section(
            "6. Time to Trend",
            content="Average time for videos to become trending by category:",
            image_path=time_image
        )
    else:
        logging.warning(f"Time to trend image not found: {time_image}")
        pdf.add_section(
            "6. Time to Trend",
            content="Time to trend image unavailable."
        )

    # Section 7: Content Recommendations
    try:
        recommender = YouTubeRecommendationSystem(country_code)
        recommender.load_model_and_shap()
        top_categories = stats.sort_values('Percentage', ascending=False)['Category'].head(5).tolist()
        
        # Add the main header for Section 7 first
        pdf.add_section(
            "7. Content Recommendations",
            content="Recommendations for top 5 categories:"
        )
        
        # Now loop through each category and add its recommendations as subsections
        for cat_id, cat_name in sorted(recommender.category_names.items(), key=lambda x: x[1]):
            if cat_name not in top_categories:
                continue
            
            try:
                recs = recommender.get_recommendations(cat_id)
                logging.info(f"Generated recommendations for {cat_name}")
            except Exception as e:
                logging.error(f"Failed to get recommendations for {cat_name}: {str(e)}")
                recs = f"Unable to generate recommendations: {str(e)}"
            
            # Parse recommendation text into sections
            rec_lines = recs.split('\n')
            priority_actions = []
            shap_analysis = []
            optimization_tips = []
            current_section = None
            for line in rec_lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("🚀 Priority Actions"):
                    current_section = "priority"
                elif line.startswith("🔍 Key Factor Analysis (SHAP)"):
                    current_section = "shap"
                elif line.startswith("💡 Content Optimization Tips"):
                    current_section = "optimization"
                elif line.startswith("•") and current_section:
                    cleaned_line = clean_text(line[2:].strip())
                    if current_section == "priority":
                        priority_actions.append(cleaned_line)
                    elif current_section == "shap":
                        shap_analysis.append(cleaned_line)
                    elif current_section == "optimization":
                        optimization_tips.append(cleaned_line)
            
            # Build the recommendation bullet list for this category
            rec_text = f"Category: {cat_name}"
            sub_bullet_lists = {}
            sub_bullet_lists[rec_text] = []
            
            if priority_actions:
                sub_bullet_lists[rec_text].append("Priority Actions:")
                sub_bullet_lists[rec_text].extend(priority_actions[:3])
            if shap_analysis:
                sub_bullet_lists[rec_text].append("Key Factors (SHAP):")
                sub_bullet_lists[rec_text].extend(shap_analysis[:5])
            if optimization_tips:
                sub_bullet_lists[rec_text].append("Optimization Tips:")
                sub_bullet_lists[rec_text].extend(optimization_tips[:5])
            
            # Add SHAP image availability
            shap_image = f'{output_dir}/{country_code}_shap_impact_category_{cat_id}.png'
            sub_bullet_lists[rec_text].append(
                f"SHAP Analysis Plot: {'Available' if os.path.exists(shap_image) else 'Not available'}"
            )
            
            # Add the category recommendation as a subsection under Section 7
            pdf.add_section(
                "",
                bullet_list=[rec_text],
                sub_bullet_lists={rec_text: sub_bullet_lists[rec_text]},
                image_path=shap_image if os.path.exists(shap_image) else None
            )

    except Exception as e:
        logging.error(f"Failed to generate recommendations: {str(e)}")
        pdf.add_section(
            "7. Content Recommendations",
            content=f"Unable to load recommendations: {clean_text(str(e))}"
        )
        
    
    output_path = f'{output_dir}/{country_code}_report.pdf'
    try:
        pdf.output(output_path)
        print(f"✅ Generated report: {output_path}")
        logging.info(f"Successfully generated report for {country_code}")
    except Exception as e:
        logging.error(f"Failed to save PDF: {str(e)}")
        print(f"❌ Failed to generate report: {str(e)}")

if __name__ == "__main__":
    countries = ['GB', 'US', 'BR', 'CA', 'MX']
    for country in countries:
        generate_report(country)