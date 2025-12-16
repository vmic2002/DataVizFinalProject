"""
Script to create probability distribution plots comparing male vs female fighters
for various features (height, weight, reach, age).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
DATA_FILE = Path("data/fighter_stats_with_gender.csv")
PLOTS_DIR = Path("plots")
MAX_FIGHTERS = 500  # Maximum number of fighters to include per gender (top by wins)

# Features to plot
FEATURES = ['height', 'weight', 'reach', 'age']

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_and_filter_data(data_file, max_fighters):
    """
    Load the fighter data and filter to top fighters by wins for each gender.
    
    Returns:
        tuple: (male_df, female_df) filtered DataFrames
    """
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    print(f"Total fighters loaded: {len(df)}")
    print(f"  Male: {len(df[df['gender'] == 'MALE'])}")
    print(f"  Female: {len(df[df['gender'] == 'FEMALE'])}")
    
    # Separate by gender
    male_df = df[df['gender'] == 'MALE'].copy()
    female_df = df[df['gender'] == 'FEMALE'].copy()
    
    # Sort by wins (descending) and take top MAX_FIGHTERS for each gender
    male_df = male_df.sort_values('wins', ascending=False).head(max_fighters)
    female_df = female_df.sort_values('wins', ascending=False).head(max_fighters)
    
    print(f"\nAfter filtering (top {max_fighters} by wins):")
    print(f"  Male fighters: {len(male_df)}")
    print(f"  Female fighters: {len(female_df)}")
    
    return male_df, female_df


def create_distribution_plot(male_df, female_df, feature, output_path):
    """
    Create a probability distribution plot comparing male vs female for a given feature.
    
    Args:
        male_df: DataFrame with male fighter data
        female_df: DataFrame with female fighter data
        feature: Name of the feature column to plot
        output_path: Path to save the plot
    """
    # Filter out NaN values for this feature
    male_data = male_df[feature].dropna()
    female_data = female_df[feature].dropna()
    
    if len(male_data) == 0 or len(female_data) == 0:
        print(f"Warning: Not enough data for feature '{feature}'. Skipping...")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create KDE plots for smooth probability distributions
    sns.kdeplot(data=male_data, label='Male', fill=True, alpha=0.5, color='#3498db', linewidth=2, ax=ax)
    sns.kdeplot(data=female_data, label='Female', fill=True, alpha=0.5, color='#e91e63', linewidth=2, ax=ax)
    
    # Customize plot
    ax.set_xlabel(feature.capitalize(), fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
    ax.set_title(f'Probability Distribution of {feature.capitalize()}: Male vs Female Fighters', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add statistics text
    male_mean = male_data.mean()
    female_mean = female_data.mean()
    male_std = male_data.std()
    female_std = female_data.std()
    
    stats_text = (f"Male: μ={male_mean:.2f}, σ={male_std:.2f} (n={len(male_data)})\n"
                  f"Female: μ={female_mean:.2f}, σ={female_std:.2f} (n={len(female_data)})")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    """Main function to create all distribution plots."""
    print("=" * 60)
    print("Fighter Statistics Visualization: Gender Comparison")
    print("=" * 60)
    
    # Check if data file exists
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found!")
        return
    
    # Create plots directory if it doesn't exist
    PLOTS_DIR.mkdir(exist_ok=True)
    print(f"\nPlots will be saved to: {PLOTS_DIR}")
    
    # Load and filter data
    male_df, female_df = load_and_filter_data(DATA_FILE, MAX_FIGHTERS)
    
    # Create plots for each feature
    print(f"\nCreating distribution plots for {len(FEATURES)} features...")
    print("-" * 60)
    
    for feature in FEATURES:
        output_path = PLOTS_DIR / f"{feature}_distribution_gender_comparison.png"
        create_distribution_plot(male_df, female_df, feature, output_path)
    
    print("\n" + "=" * 60)
    print("All plots created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()