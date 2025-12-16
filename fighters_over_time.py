"""
Script to visualize fighter performance over time (wins, win rate, fights per year).
Creates time series plots for top fighters grouped by gender and weight class.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
DATA_FILE = Path("data/per_fight_data.csv")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Filtering criteria
MIN_YEARS_DATA = 3  # Minimum years of fight data
MIN_TOTAL_FIGHTS = 5  # Minimum total fights
TOP_N_FIGHTERS = 2  # Top N fighters per gender to visualize

# Set style (consistent with visualizations.py)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


def parse_date(date_str):
    """
    Parse date string in format M/D/YYYY to datetime object.
    Returns None if parsing fails.
    """
    if pd.isna(date_str) or not date_str:
        return None
    try:
        # Handle format like "3/14/2020"
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except:
        try:
            # Try alternative format
            return pd.to_datetime(date_str)
        except:
            return None


def load_and_process_fight_data(data_file):
    """
    Load fight data and process into fighter-year statistics.
    
    Returns:
        DataFrame with columns: fighter_name, year, gender, weight_class, 
                               wins, losses, fights, win_rate
    """
    print(f"Loading fight data from {data_file}...")
    df = pd.read_csv(data_file)
    
    print(f"Total fights loaded: {len(df)}")
    
    # Parse dates and extract year
    df['date_parsed'] = df['date'].apply(parse_date)
    df['year'] = df['date_parsed'].dt.year
    
    # Filter out rows with invalid dates
    df = df[df['year'].notna()]
    df['year'] = df['year'].astype(int)
    
    print(f"Fights with valid dates: {len(df)}")
    
    # Process each fight to determine wins/losses for each fighter
    def make_stats():
        return {'wins': 0, 'losses': 0, 'fights': 0}
    
    fighter_stats = defaultdict(make_stats)
    
    for _, row in df.iterrows():
        r_fighter = str(row['R_fighter']).strip() if pd.notna(row['R_fighter']) else None
        b_fighter = str(row['B_fighter']).strip() if pd.notna(row['B_fighter']) else None
        winner = str(row['Winner']).strip() if pd.notna(row['Winner']) else None
        year = row['year']
        gender = str(row['gender']).strip() if pd.notna(row['gender']) else None
        weight_class = str(row['weight_class']).strip() if pd.notna(row['weight_class']) else None
        
        if not r_fighter or not b_fighter or not year or not gender or not weight_class:
            continue
        
        # Skip draws
        if winner and winner.upper() == 'DRAW':
            continue
        
        # Determine winner
        r_won = winner and winner.upper() == 'RED'
        b_won = winner and winner.upper() == 'BLUE'
        
        # Key: (fighter_name, year, gender, weight_class)
        r_key = (r_fighter, year, gender, weight_class)
        b_key = (b_fighter, year, gender, weight_class)
        
        # Update stats for Red fighter
        fighter_stats[r_key]['fights'] += 1
        if r_won:
            fighter_stats[r_key]['wins'] += 1
        elif b_won:
            fighter_stats[r_key]['losses'] += 1
        
        # Update stats for Blue fighter
        fighter_stats[b_key]['fights'] += 1
        if b_won:
            fighter_stats[b_key]['wins'] += 1
        elif r_won:
            fighter_stats[b_key]['losses'] += 1
    
    # Convert to DataFrame
    records = []
    for (fighter_name, year, gender, weight_class), stats in fighter_stats.items():
        wins = stats['wins']
        losses = stats['losses']
        fights = stats['fights']
        win_rate = wins / fights if fights > 0 else 0.0
        
        records.append({
            'fighter_name': fighter_name,
            'year': year,
            'gender': gender,
            'weight_class': weight_class,
            'wins': wins,
            'losses': losses,
            'fights': fights,
            'win_rate': win_rate
        })
    
    fighter_df = pd.DataFrame(records)
    
    print(f"\nProcessed {len(fighter_df)} fighter-year records")
    print(f"Unique fighters: {fighter_df['fighter_name'].nunique()}")
    print(f"Year range: {fighter_df['year'].min()} - {fighter_df['year'].max()}")
    
    return fighter_df


def filter_qualified_fighters(fighter_df):
    """
    Filter fighters based on minimum data requirements.
    Aggregates across all weight classes for each fighter.
    
    Returns:
        DataFrame with only qualified fighters (aggregated by fighter_name and gender)
    """
    print("\nFiltering fighters based on data requirements...")
    
    # Aggregate across all weight classes for each fighter
    fighter_totals = fighter_df.groupby(['fighter_name', 'gender']).agg({
        'fights': 'sum',
        'wins': 'sum',
        'year': ['min', 'max', 'nunique']
    }).reset_index()
    
    fighter_totals.columns = ['fighter_name', 'gender', 
                              'total_fights', 'total_wins',
                              'first_year', 'last_year', 'years_active']
    
    # Filter by minimum requirements
    qualified = fighter_totals[
        (fighter_totals['total_fights'] >= MIN_TOTAL_FIGHTS) &
        (fighter_totals['years_active'] >= MIN_YEARS_DATA)
    ].copy()
    
    print(f"Qualified fighters: {len(qualified)}")
    print(f"  (min {MIN_TOTAL_FIGHTS} fights, min {MIN_YEARS_DATA} years of data)")
    
    return qualified


def get_top_fighters_overall(qualified_fighters, fighter_df):
    """
    Get top N fighters per gender (ignoring weight class).
    
    Returns:
        DataFrame with time series data for top fighters only (aggregated across weight classes)
    """
    print(f"\nSelecting top {TOP_N_FIGHTERS} fighters per gender...")
    
    # Aggregate fighter data across all weight classes
    fighter_df_agg = fighter_df.groupby(['fighter_name', 'gender', 'year']).agg({
        'wins': 'sum',
        'losses': 'sum',
        'fights': 'sum'
    }).reset_index()
    
    # Calculate win rate
    fighter_df_agg['win_rate'] = fighter_df_agg['wins'] / fighter_df_agg['fights']
    fighter_df_agg['win_rate'] = fighter_df_agg['win_rate'].fillna(0.0)
    
    # Sort by total wins (descending) and get top N per gender
    qualified_sorted = qualified_fighters.sort_values('total_wins', ascending=False)
    
    top_fighters = []
    for gender in ['MALE', 'FEMALE']:
        gender_fighters = qualified_sorted[qualified_sorted['gender'] == gender]
        top_n = gender_fighters.head(TOP_N_FIGHTERS)
        top_fighters.extend(top_n[['fighter_name', 'gender']].to_dict('records'))
        print(f"  {gender}: {', '.join(top_n['fighter_name'].tolist())}")
    
    # Create DataFrame of top fighters
    top_fighters_df = pd.DataFrame(top_fighters)
    
    # Filter aggregated fighter_df to only include top fighters
    fighter_df_filtered = fighter_df_agg.merge(
        top_fighters_df,
        on=['fighter_name', 'gender'],
        how='inner'
    )
    
    print(f"\nSelected {len(top_fighters_df)} fighters total ({TOP_N_FIGHTERS} per gender)")
    
    return fighter_df_filtered, top_fighters_df


def create_time_series_plot(fighter_df, gender, feature, output_path):
    """
    Create a time series plot showing feature over time for multiple fighters.
    
    Args:
        fighter_df: DataFrame with fighter-year data (aggregated across weight classes)
        gender: Gender filter ('MALE' or 'FEMALE')
        feature: Feature to plot ('wins', 'win_rate', 'fights')
        output_path: Path to save the plot
    """
    # Filter data
    plot_data = fighter_df[fighter_df['gender'] == gender].copy()
    
    if len(plot_data) == 0:
        print(f"  No data for {gender} - skipping")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get unique fighters and assign colors
    fighters = plot_data['fighter_name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(fighters)))
    
    # Plot line for each fighter
    for i, fighter in enumerate(fighters):
        fighter_data = plot_data[plot_data['fighter_name'] == fighter].sort_values('year')
        
        if len(fighter_data) == 0:
            continue
        
        ax.plot(fighter_data['year'], fighter_data[feature], 
               marker='o', linewidth=2.5, markersize=7, 
               label=fighter, color=colors[i % len(colors)], alpha=0.85)
    
    # Customize plot
    feature_labels = {
        'wins': 'Wins per Year',
        'win_rate': 'Win Rate',
        'fights': 'Fights per Year'
    }
    
    ylabel = feature_labels.get(feature, feature.capitalize())
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    
    gender_label = 'Male' if gender == 'MALE' else 'Female'
    title = f'{ylabel} Over Time: Top {TOP_N_FIGHTERS} {gender_label} Fighters'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=1)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format y-axis for win_rate
    if feature == 'win_rate':
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Main function to create all time series visualizations."""
    print("=" * 60)
    print("Fighter Performance Over Time Visualization")
    print("=" * 60)
    
    # Check if data file exists
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found!")
        return
    
    # Load and process data
    fighter_df = load_and_process_fight_data(DATA_FILE)
    
    # Filter qualified fighters
    qualified_fighters = filter_qualified_fighters(fighter_df)
    
    # Get top fighters overall (ignoring weight class)
    fighter_df_filtered, top_fighters_df = get_top_fighters_overall(qualified_fighters, fighter_df)
    
    print(f"\nCreating visualizations for top {TOP_N_FIGHTERS} fighters per gender...")
    print("-" * 60)
    
    # Features to visualize
    features = ['wins', 'win_rate', 'fights']
    genders = ['MALE', 'FEMALE']
    
    plot_count = 0
    for gender in genders:
        for feature in features:
            # Create safe filename
            safe_gender = gender.lower()
            safe_feature = feature.lower()
            
            filename = f"{safe_feature}_over_time_top{TOP_N_FIGHTERS}_{safe_gender}.png"
            output_path = PLOTS_DIR / filename
            
            create_time_series_plot(fighter_df_filtered, gender, feature, output_path)
            plot_count += 1
    
    print("\n" + "=" * 60)
    print(f"Created {plot_count} visualizations successfully!")
    print("=" * 60)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Total qualified fighters: {len(qualified_fighters)}")
    print(f"  Top fighters selected: {len(top_fighters_df)} ({TOP_N_FIGHTERS} per gender)")
    print(f"  Year range: {fighter_df['year'].min()} - {fighter_df['year'].max()}")


if __name__ == "__main__":
    main()
