import pandas as pd
from fpdf import FPDF
import os
import glob
import re

class PDF(FPDF):
    def header(self):
        self.set_font('DejaVu', 'B', 16)
        self.cell(0, 10, f'YouTube Trending Analysis Report ({self.country_code})', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('DejaVu', 'B', 14)
        self.cell(0, 10, title, 0, 1)
        self.ln(5)

def clean_text(text):
    """
    Clean text by removing non-ASCII characters and replacing multiple spaces with a single space.
    
    Args:
        text (str): Input text to clean.
    
    Returns:
        str: Cleaned text.
    """
    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any leading/trailing spaces
    return text.strip()

def generate_report(country_code):
    """
    Generate a PDF report summarizing the analysis for a specific country.

    Args:
        country_code (str): Country code (e.g., 'US', 'KR').
    """
    output_dir = f'{country_code}_output'
    pdf = PDF()
    pdf.country_code = country_code  # Add country_code to PDF instance

    # Add a Unicode font that supports Korean and emoji
    pdf.add_font('DejaVu', '', '/Users/phuowgz/BDM-final-project optimized/Understand Dataset/fonts/DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', '/Users/phuowgz/BDM-final-project optimized/Understand Dataset/fonts/DejaVuSans-Bold.ttf', uni=True)

    # Set smaller margins to ensure enough horizontal space
    pdf.set_margins(left=5, top=10, right=5)

    # Reduce font size to ensure enough space
    pdf.set_font('DejaVu', '', 10)

    pdf.add_page()

    # Section 1: Correlation Matrix
    pdf.chapter_title("1. Correlation Matrix")
    pdf.set_font('DejaVu', '', 10)
    pdf.multi_cell(0, 10, "Correlation between key features affecting view velocity.")
    pdf.image(f'{output_dir}/{country_code}_correlation_matrix.png', x=10, w=190)
    pdf.ln(10)

    # Section 2: Top Categories
    pdf.chapter_title("2. Top Categories")
    pdf.set_font('DejaVu', '', 10)
    pdf.multi_cell(0, 10, "Top 5 categories by video count:")
    
    # Load stats and format manually to avoid issues
    stats = pd.read_csv(f'{output_dir}/{country_code}_category_stats.csv')
    stats = stats.head()  # Take top 5 rows

    # Define column widths
    col_widths = [40, 20, 20, 20]  # Widths for Category, Percentage, Avg Views, Avg Likes
    header = ["Category", "Percentage", "Avg Views", "Avg Likes"]

    # Draw header
    pdf.set_font('DejaVu', 'B', 10)
    for i, (h, w) in enumerate(zip(header, col_widths)):
        pdf.cell(w, 10, h, border=1, ln=0 if i < len(header) - 1 else 1)

    # Draw table rows
    pdf.set_font('DejaVu', '', 10)
    for _, row in stats.iterrows():
        category = clean_text(str(row['Category']))[:18]  # Limit length to fit
        percentage = f"{row['Percentage']:.2f}%"
        avg_views = f"{int(row['Avg Views'])}"  # Remove commas
        avg_likes = f"{int(row['Avg Likes'])}"  # Remove commas

        pdf.cell(col_widths[0], 10, category, border=1, ln=0)
        pdf.cell(col_widths[1], 10, percentage, border=1, ln=0)
        pdf.cell(col_widths[2], 10, avg_views, border=1, ln=0)
        pdf.cell(col_widths[3], 10, avg_likes, border=1, ln=1)

    pdf.ln(10)

    # Section 3: Top Tags
    pdf.chapter_title("3. Popular Tags by Category")
    pdf.set_font('DejaVu', '', 10)
    pdf.multi_cell(0, 10, "Top tags for first 5 categories:")
    
    # Load tags and format manually
    tags = pd.read_csv(f'{output_dir}/{country_code}_most_popular_tags_by_category.csv')
    tags = tags.head()  # Take top 5 rows

    # Define column widths for tags table
    col_widths = [40, 30]  # Widths for Category, Tag
    header = ["Category", "Tag"]

    # Draw header
    pdf.set_font('DejaVu', 'B', 10)
    for i, (h, w) in enumerate(zip(header, col_widths)):
        pdf.cell(w, 10, h, border=1, ln=0 if i < len(header) - 1 else 1)

    # Draw table rows
    pdf.set_font('DejaVu', '', 10)
    for _, row in tags.iterrows():
        category = clean_text(str(row['Category']))[:18]
        tag = clean_text(str(row['Tag']))[:15]

        pdf.cell(col_widths[0], 10, category, border=1, ln=0)
        pdf.cell(col_widths[1], 10, tag, border=1, ln=1)

    pdf.ln(10)

    # Section 4: Trending Patterns
    pdf.chapter_title("4. Trending Patterns by Day")
    pdf.set_font('DejaVu', '', 10)
    pdf.multi_cell(0, 10, "Heatmap of trending videos by day and category:")
    pdf.image(f'{output_dir}/{country_code}_trending_analysis.png', x=10, w=190)
    pdf.ln(10)

    # Section 5: Feature Importance
    pdf.chapter_title("5. Feature Importance Comparison")
    pdf.set_font('DejaVu', '', 10)
    pdf.multi_cell(0, 10, "Comparison of feature importance between XGBoost and LightGBM:")
    pdf.image(f'{output_dir}/{country_code}_feature_importance_comparison.png', x=10, w=190)
    pdf.ln(10)

    # Section 6: SHAP Summary
    pdf.chapter_title("6. SHAP Analysis")
    pdf.set_font('DejaVu', '', 10)
    pdf.multi_cell(0, 10, "SHAP summary plot for all categories:")
    pdf.image(f'{output_dir}/{country_code}_shap_summary.png', x=10, w=190)
    pdf.ln(10)

    # Save PDF
    pdf.output(f'{output_dir}/{country_code}_report.pdf')
    print(f"Generated report: {output_dir}/{country_code}_report.pdf")

if __name__ == "__main__":
    countries = ['US', 'IN', 'BR', 'GB', 'KR']
    for country in countries:
        generate_report(country)