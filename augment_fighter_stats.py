"""
Script to augment fighter_stats.csv with gender information from per_fight_data.csv
by matching fighter names.
"""

import csv
from collections import defaultdict
from pathlib import Path

# File paths
DATA_DIR = Path("data")
FIGHTER_STATS_FILE = DATA_DIR / "fighter_stats.csv"
PER_FIGHT_DATA_FILE = DATA_DIR / "per_fight_data.csv"
OUTPUT_FILE = DATA_DIR / "fighter_stats_with_gender.csv"


def build_gender_dict(per_fight_file):
    """
    Build a dictionary mapping fighter names to their gender from per_fight_data.csv.
    
    Returns:
        dict: {fighter_name: gender} where gender is 'MALE' or 'FEMALE'
        dict: {fighter_name: [genders]} for tracking conflicts
    """
    gender_map = {}
    gender_counts = defaultdict(lambda: defaultdict(int))
    
    print(f"Reading gender data from {per_fight_file}...")
    
    with open(per_fight_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get fighters from both R_fighter and B_fighter columns
            r_fighter = row['R_fighter'].strip() if row.get('R_fighter') else None
            b_fighter = row['B_fighter'].strip() if row.get('B_fighter') else None
            gender = row.get('gender', '').strip()
            
            if not gender or gender not in ['MALE', 'FEMALE']:
                continue
            
            # Track gender for each fighter
            if r_fighter:
                gender_counts[r_fighter][gender] += 1
            if b_fighter:
                gender_counts[b_fighter][gender] += 1
    
    # For each fighter, use the most common gender (should be consistent, but handle edge cases)
    conflicts = []
    for fighter_name, gender_freq in gender_counts.items():
        if len(gender_freq) > 1:
            # Conflict: fighter appears with multiple genders
            conflicts.append(fighter_name)
            # Use the most common gender
            most_common_gender = max(gender_freq.items(), key=lambda x: x[1])[0]
            gender_map[fighter_name] = most_common_gender
        else:
            # No conflict
            gender_map[fighter_name] = list(gender_freq.keys())[0]
    
    if conflicts:
        print(f"Warning: Found {len(conflicts)} fighters with conflicting gender assignments:")
        for fighter in conflicts[:10]:  # Show first 10
            print(f"  - {fighter}: {dict(gender_counts[fighter])}")
        if len(conflicts) > 10:
            print(f"  ... and {len(conflicts) - 10} more")
    
    print(f"Extracted gender information for {len(gender_map)} unique fighters")
    return gender_map


def augment_fighter_stats(fighter_stats_file, gender_map, output_file):
    """
    Read fighter_stats.csv, add gender column, and write to output file.
    Only includes rows where gender information was found.
    """
    print(f"\nReading fighter stats from {fighter_stats_file}...")
    
    with open(fighter_stats_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ['gender'] if reader.fieldnames else None
        
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        matched_count = 0
        unmatched_count = 0
        
        for row in reader:
            fighter_name = row['name'].strip()
            
            # Look up gender
            gender = gender_map.get(fighter_name, '')
            
            if gender:
                # Only include rows with gender information
                row['gender'] = gender
                writer.writerow(row)
                matched_count += 1
            else:
                unmatched_count += 1
                if unmatched_count <= 10:  # Log first 10 unmatched
                    print(f"  No gender found for: {fighter_name}")
        
        print(f"\nResults:")
        print(f"  Matched: {matched_count} fighters")
        print(f"  Unmatched: {unmatched_count} fighters")
        print(f"\nOutput written to {output_file}")


def main():
    """Main function to run the augmentation process."""
    print("=" * 60)
    print("Fighter Stats Gender Augmentation")
    print("=" * 60)
    
    # Check if input files exist
    if not PER_FIGHT_DATA_FILE.exists():
        print(f"Error: {PER_FIGHT_DATA_FILE} not found!")
        return
    
    if not FIGHTER_STATS_FILE.exists():
        print(f"Error: {FIGHTER_STATS_FILE} not found!")
        return
    
    # Build gender mapping from per_fight_data
    gender_map = build_gender_dict(PER_FIGHT_DATA_FILE)
    
    # Augment fighter_stats with gender
    augment_fighter_stats(FIGHTER_STATS_FILE, gender_map, OUTPUT_FILE)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()