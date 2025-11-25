# ========================================================
# ✅ Kaggle NBA Data Fetcher – Schema-Proof Version
# ========================================================

import os
import pandas as pd
import zipfile

print("📥 Downloading dataset from Kaggle...")
os.system("kaggle datasets download -d eoinamoore/historical-nba-data-and-player-box-scores")

with zipfile.ZipFile("historical-nba-data-and-player-box-scores.zip", 'r') as zip_ref:
    zip_ref.extractall("nba_data")

print("✅ Extraction complete!")

# ========================================================
# 1. Load CSVs with safe options
# ========================================================
print("📄 Loading CSV files...")

games_df = pd.read_csv("nba_data/Games.csv", low_memory=False)
player_stats_df = pd.read_csv("nba_data/PlayerStatistics.csv", low_memory=False)
team_stats_df = pd.read_csv("nba_data/TeamStatistics.csv", low_memory=False)

print(f"Games CSV columns: {list(games_df.columns)}")
print(f"Player Stats CSV columns: {list(player_stats_df.columns)}")
print(f"Team Stats CSV columns: {list(team_stats_df.columns)}")

# ========================================================
# 2. Ensure `gameDate` column exists (Fix for Kaggle schema change)
# ========================================================

def ensure_game_date(df):
    """
    Adds a 'gameDate' column no matter how Kaggle names the timestamp.
    Keeps your Streamlit code compatible forever.
    """
    possible_cols = [
        "gameDate",
        "gameDateTimeEst",
        "gameDateUTC",
        "GAME_DATE",
        "GameDate",
        "game_date"
    ]

    for col in possible_cols:
        if col in df.columns:
            try:
                df["gameDate"] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.date
                print(f"✔ Extracted gameDate from column: {col}")
                return df
            except Exception:
                pass

    raise KeyError("❌ No valid date column found in the Kaggle dataset.")

games_df = ensure_game_date(games_df)
player_stats_df = ensure_game_date(player_stats_df)
team_stats_df = ensure_game_date(team_stats_df)

# ========================================================
# 3. Filter for 2025–26 season
# ========================================================

print("⏳ Filtering for 2025–26 season...")

start_date = pd.Timestamp("2025-10-21", tz="UTC").date()
current_season_games = games_df[games_df["gameDate"] >= start_date].copy()

keep_ids = set(current_season_games["gameId"].astype(str))

player_stats_df["gameId"] = player_stats_df["gameId"].astype(str)
team_stats_df["gameId"] = team_stats_df["gameId"].astype(str)

current_season_players = player_stats_df[player_stats_df["gameId"].isin(keep_ids)].copy()
current_season_teams = team_stats_df[team_stats_df["gameId"].isin(keep_ids)].copy()

# ========================================================
# 4. Save cleaned / filtered files
# ========================================================

print("💾 Saving filtered datasets...")
current_season_games.to_csv("Games_filtered.csv", index=False)
current_season_players.to_csv("PlayerStatistics_filtered.csv", index=False)
current_season_teams.to_csv("TeamStatistics_filtered.csv", index=False)

print(f"✔ Saved {len(current_season_games)} games")
print(f"✔ Saved {len(current_season_players)} player stats")
print(f"✔ Saved {len(current_season_teams)} team stats")

# ========================================================
# 5. Cleanup raw files
# ========================================================

print("🧹 Cleaning raw downloads...")

for f in [
    "historical-nba-data-and-player-box-scores.zip",
    "nba_data/Games.csv",
    "nba_data/PlayerStatistics.csv",
    "nba_data/TeamStatistics.csv"
]:
    try:
        os.remove(f)
    except:
        pass

print("✨ Cleanup done! Automation completed successfully.")

