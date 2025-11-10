# ========================================
# ✅ Kaggle-Only NBA Data Fetcher (GitHub)
# ========================================

import os
import pandas as pd
import zipfile

# ----------------------------------------
# 1. Download and Extract Kaggle Dataset
# ----------------------------------------
print("📥 Downloading dataset from Kaggle...")
os.system("kaggle datasets download -d eoinamoore/historical-nba-data-and-player-box-scores")

with zipfile.ZipFile("historical-nba-data-and-player-box-scores.zip", 'r') as zip_ref:
    zip_ref.extractall("nba_data")
print("✅ Extraction complete!")

# ----------------------------------------
# 2. Filter to 2025–26 Season
# ----------------------------------------
print("🔎 Filtering data for 2025–26...")

games_df = pd.read_csv("nba_data/Games.csv", low_memory=False)
player_stats_df = pd.read_csv("nba_data/PlayerStatistics.csv", low_memory=False)
team_stats_df = pd.read_csv("nba_data/TeamStatistics.csv", low_memory=False)

games_df["gameDate"] = pd.to_datetime(games_df["gameDate"], errors="coerce", utc=True)
start_date = pd.Timestamp("2025-10-21", tz="UTC")
current_season_games = games_df[games_df["gameDate"] >= start_date].copy()

keep_ids = set(current_season_games["gameId"].dropna().astype(str))
player_stats_df["gameId"] = player_stats_df["gameId"].astype(str)
team_stats_df["gameId"] = team_stats_df["gameId"].astype(str)

current_season_players = player_stats_df[player_stats_df["gameId"].isin(keep_ids)].copy()
current_season_teams = team_stats_df[team_stats_df["gameId"].isin(keep_ids)].copy()

# ----------------------------------------
# 3. Save Filtered Files Only
# ----------------------------------------
print("💾 Saving filtered datasets...")
current_season_games.to_csv("Games_filtered.csv", index=False)
current_season_players.to_csv("PlayerStatistics_filtered.csv", index=False)
current_season_teams.to_csv("TeamStatistics_filtered.csv", index=False)
print(f"✅ Saved {len(current_season_games)} games")
print(f"✅ Saved {len(current_season_players)} player stats")
print(f"✅ Saved {len(current_season_teams)} team stats")

# ----------------------------------------
# 4. Cleanup Raw Files
# ----------------------------------------
print("🧹 Cleaning up unneeded large files...")
for f in [
    "historical-nba-data-and-player-box-scores.zip",
    "nba_data/Games.csv",
    "nba_data/PlayerStatistics.csv",
    "nba_data/TeamStatistics.csv"
]:
    try:
        os.remove(f)
    except FileNotFoundError:
        pass
print("✅ Cleanup complete!")
