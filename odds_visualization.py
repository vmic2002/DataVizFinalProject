"""
Script to create visualizations analyzing betting odds accuracy in MMA fights.
Creates 7 different visualizations:
1. Odds confidence vs accuracy
2. Favorite vs underdog win rates
3. Upset frequency by odds margin
4. Temporal trends
5. By weight class/gender
6. Confusion matrix / outcome heatmap
7. Odds difference distribution (violin plots)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
DATA_FILE = Path("data/per_fight_data.csv")
PLOTS_DIR = Path("odds_visualizations")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Create output directory
PLOTS_DIR.mkdir(exist_ok=True)


def convert_odds_to_probability(odds):
    """
    Convert American odds to implied probability.
    Negative odds (favorite): |odds| / (|odds| + 100) * 100
    Positive odds (underdog): 100 / (odds + 100) * 100
    """
    if pd.isna(odds) or not np.isfinite(odds):
        return np.nan
    
    if odds < 0:
        return abs(odds) / (abs(odds) + 100) * 100
    else:
        return 100 / (odds + 100) * 100


def determine_favorite(r_odds, b_odds):
    """
    Determine which fighter is the favorite based on odds.
    Returns: 'Red' if R_fighter is favorite, 'Blue' if B_fighter is favorite, None if invalid
    """
    if pd.isna(r_odds) or pd.isna(b_odds) or not np.isfinite(r_odds) or not np.isfinite(b_odds):
        return None
    
    # Favorite has negative odds (or smaller positive odds, but typically negative)
    if r_odds < 0 and b_odds > 0:
        return 'Red'
    elif b_odds < 0 and r_odds > 0:
        return 'Blue'
    elif r_odds < 0 and b_odds < 0:
        # Both negative, more negative = more favorite
        return 'Red' if abs(r_odds) < abs(b_odds) else 'Blue'
    elif r_odds > 0 and b_odds > 0:
        # Both positive, smaller = more favorite
        return 'Red' if r_odds < b_odds else 'Blue'
    else:
        return None


def load_and_prepare_data(data_file):
    """
    Load fight data and prepare it for analysis.
    """
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    print(f"Total fights loaded: {len(df)}")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    
    # Convert odds to numeric
    df['R_odds'] = pd.to_numeric(df['R_odds'], errors='coerce')
    df['B_odds'] = pd.to_numeric(df['B_odds'], errors='coerce')
    
    # Filter out rows with invalid odds or winner
    initial_count = len(df)
    df = df.dropna(subset=['R_odds', 'B_odds', 'Winner'])
    df = df[df['Winner'].isin(['Red', 'Blue'])]  # Remove draws
    print(f"Fights with valid odds and winner: {len(df)} (removed {initial_count - len(df)} invalid rows)")
    
    # Calculate odds difference (absolute)
    df['odds_diff'] = abs(df['R_odds'] - df['B_odds'])
    
    # Determine favorite
    df['favorite'] = df.apply(lambda row: determine_favorite(row['R_odds'], row['B_odds']), axis=1)
    
    # Determine if favorite won
    df['favorite_won'] = df['favorite'] == df['Winner']
    
    # Calculate implied probabilities
    df['R_prob'] = df['R_odds'].apply(convert_odds_to_probability)
    df['B_prob'] = df['B_odds'].apply(convert_odds_to_probability)
    
    # Determine predicted winner (higher probability)
    df['predicted_winner'] = df.apply(
        lambda row: 'Red' if row['R_prob'] > row['B_prob'] else 'Blue', axis=1
    )
    
    # Check if prediction was correct
    df['prediction_correct'] = df['predicted_winner'] == df['Winner']
    
    # Filter out rows where we couldn't determine favorite
    df = df[df['favorite'].notna()]
    
    print(f"Final dataset size: {len(df)} fights")
    
    return df


def plot_1_confidence_vs_accuracy(df):
    """
    1. Odds confidence vs accuracy
    Scatter/heatmap showing how odds difference (confidence) relates to prediction accuracy.
    """
    print("\nCreating plot 1: Odds confidence vs accuracy...")
    
    # Bin the odds differences for better visualization
    df['odds_diff_bin'] = pd.cut(df['odds_diff'], bins=20, precision=0)
    
    # Calculate accuracy for each bin
    bin_stats = df.groupby('odds_diff_bin', observed=True).agg({
        'favorite_won': ['mean', 'count']
    }).reset_index()
    bin_stats.columns = ['odds_diff_bin', 'accuracy', 'count']
    bin_stats['odds_diff_mid'] = bin_stats['odds_diff_bin'].apply(lambda x: x.mid)
    
    # Filter out bins with too few samples
    bin_stats = bin_stats[bin_stats['count'] >= 5]
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Scatter plot with sample size
    scatter = ax1.scatter(bin_stats['odds_diff_mid'], bin_stats['accuracy'], 
                         s=bin_stats['count']*2, alpha=0.6, c=bin_stats['count'], 
                         cmap='viridis', edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% (Random)')
    ax1.set_xlabel('Odds Difference (Confidence Level)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Favorite Win Rate (Accuracy)', fontsize=12, fontweight='bold')
    ax1.set_title('Odds Confidence vs Prediction Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Number of Fights')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '1_confidence_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 1_confidence_vs_accuracy.png")


def plot_2_favorite_vs_underdog_win_rates(df):
    """
    2. Favorite vs underdog win rates
    Bar chart comparing win rates for favorites vs underdogs.
    """
    print("\nCreating plot 2: Favorite vs underdog win rates...")
    
    # Calculate win rates
    favorite_wins = df[df['favorite'] == df['Winner']]
    underdog_wins = df[df['favorite'] != df['Winner']]
    
    favorite_win_rate = len(favorite_wins) / len(df)
    underdog_win_rate = len(underdog_wins) / len(df)
    
    # Also calculate by odds difference bins
    df['odds_diff_category'] = pd.cut(df['odds_diff'], 
                                       bins=[0, 50, 100, 200, 500, float('inf')],
                                       labels=['Very Close (0-50)', 'Close (50-100)', 
                                              'Moderate (100-200)', 'Large (200-500)', 
                                              'Very Large (500+)'])
    
    category_stats = df.groupby('odds_diff_category', observed=True).agg({
        'favorite_won': ['mean', 'count']
    }).reset_index()
    category_stats.columns = ['category', 'win_rate', 'count']
    category_stats = category_stats[category_stats['count'] >= 10]  # Filter small samples
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overall comparison
    categories = ['Favorites', 'Underdogs']
    win_rates = [favorite_win_rate * 100, underdog_win_rate * 100]
    counts = [len(favorite_wins), len(underdog_wins)]
    
    bars = ax1.bar(categories, win_rates, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (Random)')
    ax1.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Overall Win Rates\nFavorites: {favorite_win_rate*100:.1f}% ({len(favorite_wins)}/{len(df)})\nUnderdogs: {underdog_win_rate*100:.1f}% ({len(underdog_wins)}/{len(df)})', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'n={count}', ha='center', va='bottom', fontweight='bold')
    
    # By confidence level
    if len(category_stats) > 0:
        colors = plt.cm.RdYlGn(category_stats['win_rate'] / 100)
        bars2 = ax2.bar(range(len(category_stats)), category_stats['win_rate'] * 100, 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xticks(range(len(category_stats)))
        ax2.set_xticklabels(category_stats['category'], rotation=45, ha='right')
        ax2.set_ylabel('Favorite Win Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Favorite Win Rate by Odds Difference Category', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars2, category_stats['count'])):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'n={count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '2_favorite_vs_underdog_win_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 2_favorite_vs_underdog_win_rates.png")


def plot_3_upset_frequency_by_margin(df):
    """
    3. Upset frequency by odds margin
    Histogram/bar chart showing how often upsets occur at different confidence levels.
    """
    print("\nCreating plot 3: Upset frequency by odds margin...")
    
    # Define upset (underdog wins)
    df['upset'] = df['favorite'] != df['Winner']
    
    # Bin by odds difference
    df['odds_diff_bin'] = pd.cut(df['odds_diff'], 
                                  bins=[0, 25, 50, 100, 150, 200, 300, 500, float('inf')],
                                  labels=['0-25', '25-50', '50-100', '100-150', 
                                         '150-200', '200-300', '300-500', '500+'])
    
    upset_stats = df.groupby('odds_diff_bin', observed=True).agg({
        'upset': ['mean', 'sum', 'count']
    }).reset_index()
    upset_stats.columns = ['odds_diff_bin', 'upset_rate', 'upset_count', 'total_count']
    upset_stats = upset_stats[upset_stats['total_count'] >= 5]  # Filter small samples
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Upset rate by margin
    colors = plt.cm.Reds(upset_stats['upset_rate'])
    bars1 = ax1.bar(range(len(upset_stats)), upset_stats['upset_rate'] * 100,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(upset_stats)))
    ax1.set_xticklabels(upset_stats['odds_diff_bin'], rotation=45, ha='right')
    ax1.set_ylabel('Upset Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Odds Difference Range', fontsize=12, fontweight='bold')
    ax1.set_title('Upset Frequency by Odds Margin\n(Underdog Win Rate)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add labels
    for i, (bar, rate, count) in enumerate(zip(bars1, upset_stats['upset_rate'], upset_stats['total_count'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate*100:.1f}%\n(n={count})', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Stacked bar chart: favorites vs upsets
    favorite_wins_by_bin = df.groupby('odds_diff_bin', observed=True).apply(
        lambda x: (x['favorite'] == x['Winner']).sum(), include_groups=False
    ).reset_index(name='favorite_wins')
    upset_wins_by_bin = df.groupby('odds_diff_bin', observed=True).apply(
        lambda x: (x['favorite'] != x['Winner']).sum(), include_groups=False
    ).reset_index(name='upset_wins')
    
    merge_df = favorite_wins_by_bin.merge(upset_wins_by_bin, on='odds_diff_bin')
    merge_df = merge_df[merge_df['favorite_wins'] + merge_df['upset_wins'] >= 5]
    
    x_pos = range(len(merge_df))
    ax2.bar(x_pos, merge_df['favorite_wins'], label='Favorite Wins', 
           color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.bar(x_pos, merge_df['upset_wins'], bottom=merge_df['favorite_wins'], 
           label='Upsets (Underdog Wins)', color='#e74c3c', alpha=0.7, 
           edgecolor='black', linewidth=1.5)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(merge_df['odds_diff_bin'], rotation=45, ha='right')
    ax2.set_ylabel('Number of Fights', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Odds Difference Range', fontsize=12, fontweight='bold')
    ax2.set_title('Fight Outcomes by Odds Margin (Stacked)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '3_upset_frequency_by_margin.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 3_upset_frequency_by_margin.png")


def plot_4_temporal_trends(df):
    """
    4. Temporal trends
    Line chart showing odds accuracy over time.
    """
    print("\nCreating plot 4: Temporal trends...")
    
    # Extract year and month
    df['year'] = df['date'].dt.year
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Calculate monthly statistics
    monthly_stats = df.groupby('year_month').agg({
        'favorite_won': ['mean', 'count'],
        'odds_diff': 'mean'
    }).reset_index()
    monthly_stats.columns = ['year_month', 'accuracy', 'count', 'avg_odds_diff']
    monthly_stats = monthly_stats[monthly_stats['count'] >= 5]  # Filter small samples
    monthly_stats['year_month_str'] = monthly_stats['year_month'].astype(str)
    
    # Calculate yearly statistics
    yearly_stats = df.groupby('year').agg({
        'favorite_won': ['mean', 'count'],
        'odds_diff': 'mean'
    }).reset_index()
    yearly_stats.columns = ['year', 'accuracy', 'count', 'avg_odds_diff']
    yearly_stats = yearly_stats[yearly_stats['count'] >= 20]  # Filter small samples
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Monthly accuracy
    ax1.plot(range(len(monthly_stats)), monthly_stats['accuracy'] * 100, 
            marker='o', linewidth=2, markersize=4, alpha=0.7, color='#3498db')
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (Random)')
    ax1.set_xlabel('Time (Months)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Favorite Win Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Odds Accuracy Over Time (Monthly)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add sample size as secondary axis
    ax1_twin = ax1.twinx()
    ax1_twin.bar(range(len(monthly_stats)), monthly_stats['count'], 
                alpha=0.2, color='gray', label='Fight Count')
    ax1_twin.set_ylabel('Number of Fights', fontsize=10)
    ax1_twin.legend(loc='upper right')
    
    # Yearly accuracy
    ax2.plot(yearly_stats['year'], yearly_stats['accuracy'] * 100, 
            marker='o', linewidth=3, markersize=8, alpha=0.7, color='#2ecc71')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (Random)')
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Favorite Win Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Odds Accuracy Over Time (Yearly)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add count labels
    for year, acc, count in zip(yearly_stats['year'], yearly_stats['accuracy'], yearly_stats['count']):
        ax2.text(year, acc * 100 + 1, f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # Average odds difference over time (monthly)
    ax3.plot(range(len(monthly_stats)), monthly_stats['avg_odds_diff'], 
            marker='o', linewidth=2, markersize=4, alpha=0.7, color='#e74c3c')
    ax3.set_xlabel('Time (Months)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Odds Difference', fontsize=12, fontweight='bold')
    ax3.set_title('Average Odds Confidence Over Time (Monthly)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Yearly average odds difference
    ax4.plot(yearly_stats['year'], yearly_stats['avg_odds_diff'], 
            marker='o', linewidth=3, markersize=8, alpha=0.7, color='#9b59b6')
    ax4.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Average Odds Difference', fontsize=12, fontweight='bold')
    ax4.set_title('Average Odds Confidence Over Time (Yearly)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '4_temporal_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 4_temporal_trends.png")


def plot_5_by_weight_class_gender(df):
    """
    5. By weight class/gender
    Grouped bar charts showing accuracy by weight class and gender.
    """
    print("\nCreating plot 5: By weight class/gender...")
    
    # Filter out missing values
    df_filtered = df[df['weight_class'].notna() & df['gender'].notna()].copy()
    
    # Calculate statistics by weight class
    weight_class_stats = df_filtered.groupby('weight_class').agg({
        'favorite_won': ['mean', 'count']
    }).reset_index()
    weight_class_stats.columns = ['weight_class', 'accuracy', 'count']
    weight_class_stats = weight_class_stats[weight_class_stats['count'] >= 10]
    weight_class_stats = weight_class_stats.sort_values('accuracy', ascending=False)
    
    # Calculate statistics by gender
    gender_stats = df_filtered.groupby('gender').agg({
        'favorite_won': ['mean', 'count']
    }).reset_index()
    gender_stats.columns = ['gender', 'accuracy', 'count']
    
    # Calculate statistics by weight class AND gender
    combined_stats = df_filtered.groupby(['weight_class', 'gender']).agg({
        'favorite_won': ['mean', 'count']
    }).reset_index()
    combined_stats.columns = ['weight_class', 'gender', 'accuracy', 'count']
    combined_stats = combined_stats[combined_stats['count'] >= 5]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # By weight class
    colors_wc = plt.cm.viridis(weight_class_stats['accuracy'])
    bars1 = ax1.barh(range(len(weight_class_stats)), weight_class_stats['accuracy'] * 100,
                    color=colors_wc, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_yticks(range(len(weight_class_stats)))
    ax1.set_yticklabels(weight_class_stats['weight_class'])
    ax1.set_xlabel('Favorite Win Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Odds Accuracy by Weight Class', fontsize=14, fontweight='bold')
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars1, weight_class_stats['count'])):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'n={count}', ha='left', va='center', fontsize=9)
    
    # By gender
    colors_g = ['#3498db', '#e91e63']
    bars2 = ax2.bar(range(len(gender_stats)), gender_stats['accuracy'] * 100,
                   color=colors_g[:len(gender_stats)], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(gender_stats)))
    ax2.set_xticklabels(gender_stats['gender'])
    ax2.set_ylabel('Favorite Win Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Odds Accuracy by Gender', fontsize=14, fontweight='bold')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars2, gender_stats['count'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%\n(n={count})', ha='center', va='bottom', fontweight='bold')
    
    # Combined: Top weight classes by gender
    top_weight_classes = weight_class_stats.head(10)['weight_class'].tolist()
    combined_filtered = combined_stats[combined_stats['weight_class'].isin(top_weight_classes)]
    
    if len(combined_filtered) > 0:
        # Pivot for grouped bar chart
        pivot_data = combined_filtered.pivot_table(
            index='weight_class', columns='gender', values='accuracy', fill_value=np.nan
        )
        
        x = np.arange(len(pivot_data))
        width = 0.35
        
        if 'MALE' in pivot_data.columns:
            ax3.bar(x - width/2, pivot_data['MALE'] * 100, width, 
                   label='MALE', color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
        if 'FEMALE' in pivot_data.columns:
            ax3.bar(x + width/2, pivot_data['FEMALE'] * 100, width,
                   label='FEMALE', color='#e91e63', alpha=0.7, edgecolor='black', linewidth=1)
        
        ax3.set_xlabel('Weight Class', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Favorite Win Rate (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Odds Accuracy by Weight Class and Gender (Top 10)', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(pivot_data.index, rotation=45, ha='right')
        ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Upset rate by weight class
    upset_by_wc = df_filtered.groupby('weight_class').agg({
        'upset': ['mean', 'count']
    }).reset_index()
    upset_by_wc.columns = ['weight_class', 'upset_rate', 'count']
    upset_by_wc = upset_by_wc[upset_by_wc['count'] >= 10]
    upset_by_wc = upset_by_wc.sort_values('upset_rate', ascending=False)
    
    colors_upset = plt.cm.Reds(upset_by_wc['upset_rate'])
    bars4 = ax4.barh(range(len(upset_by_wc)), upset_by_wc['upset_rate'] * 100,
                    color=colors_upset, alpha=0.7, edgecolor='black', linewidth=1)
    ax4.set_yticks(range(len(upset_by_wc)))
    ax4.set_yticklabels(upset_by_wc['weight_class'])
    ax4.set_xlabel('Upset Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Upset Frequency by Weight Class', fontsize=14, fontweight='bold')
    ax4.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars4, upset_by_wc['count'])):
        width = bar.get_width()
        ax4.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'n={count}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '5_by_weight_class_gender.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 5_by_weight_class_gender.png")


def plot_6_confusion_matrix(df):
    """
    6. Confusion matrix / outcome heatmap
    2D heatmap showing predicted vs actual outcomes.
    """
    print("\nCreating plot 6: Confusion matrix / outcome heatmap...")
    
    # Create confusion matrix: predicted winner vs actual winner
    confusion_data = pd.crosstab(df['predicted_winner'], df['Winner'], normalize='index') * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap
    sns.heatmap(confusion_data, annot=True, fmt='.1f', cmap='RdYlGn', 
               vmin=0, vmax=100, cbar_kws={'label': 'Percentage (%)'},
               ax=ax1, square=True, linewidths=1, linecolor='black')
    ax1.set_xlabel('Actual Winner', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Winner (by Odds)', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix: Predicted vs Actual Winner\n(Normalized by Row)', 
                 fontsize=14, fontweight='bold')
    
    # Count matrix (absolute numbers)
    confusion_counts = pd.crosstab(df['predicted_winner'], df['Winner'])
    sns.heatmap(confusion_counts, annot=True, fmt='d', cmap='Blues',
               cbar_kws={'label': 'Number of Fights'},
               ax=ax2, square=True, linewidths=1, linecolor='black')
    ax2.set_xlabel('Actual Winner', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Winner (by Odds)', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix: Predicted vs Actual Winner\n(Absolute Counts)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '6_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 6_confusion_matrix.png")


def plot_7_odds_difference_distribution(df):
    """
    7. Odds difference distribution (violin plots)
    Violin plots showing distribution of odds differences for correct vs incorrect predictions.
    """
    print("\nCreating plot 7: Odds difference distribution (violin plots)...")
    
    # Create categories for better visualization
    df['prediction_status'] = df['prediction_correct'].map({True: 'Correct Prediction', False: 'Incorrect Prediction'})
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Violin plot: odds difference by prediction correctness
    parts = ax1.violinplot([df[df['prediction_correct']]['odds_diff'].values,
                           df[~df['prediction_correct']]['odds_diff'].values],
                          positions=[0, 1], widths=0.6, showmeans=True, showmedians=True)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Correct\nPrediction', 'Incorrect\nPrediction'])
    ax1.set_ylabel('Odds Difference', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Odds Differences\nby Prediction Accuracy', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Color the violins
    for pc, color in zip(parts['bodies'], ['#2ecc71', '#e74c3c']):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Box plot comparison
    box_data = [df[df['prediction_correct']]['odds_diff'].values,
                df[~df['prediction_correct']]['odds_diff'].values]
    bp = ax2.boxplot(box_data, tick_labels=['Correct\nPrediction', 'Incorrect\nPrediction'],
                    patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#e74c3c')
    bp['boxes'][1].set_alpha(0.7)
    ax2.set_ylabel('Odds Difference', fontsize=12, fontweight='bold')
    ax2.set_title('Box Plot: Odds Differences by Prediction Accuracy', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Histogram overlay
    ax3.hist(df[df['prediction_correct']]['odds_diff'], bins=30, alpha=0.6, 
            label='Correct Prediction', color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax3.hist(df[~df['prediction_correct']]['odds_diff'], bins=30, alpha=0.6,
            label='Incorrect Prediction', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Odds Difference', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Histogram: Odds Differences by Prediction Accuracy', 
                 fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '7_odds_difference_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: 7_odds_difference_distribution.png")


def main():
    """
    Main function to run all visualizations.
    """
    print("=" * 60)
    print("MMA Betting Odds Accuracy Analysis")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data(DATA_FILE)
    
    # Create all visualizations
    plot_1_confidence_vs_accuracy(df)
    plot_2_favorite_vs_underdog_win_rates(df)
    plot_3_upset_frequency_by_margin(df)
    plot_4_temporal_trends(df)
    plot_5_by_weight_class_gender(df)
    plot_6_confusion_matrix(df)
    plot_7_odds_difference_distribution(df)
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
