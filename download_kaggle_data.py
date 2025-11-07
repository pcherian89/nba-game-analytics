# =============================
# 0. Install dependencies (if running in Colab)
# =============================
try:
    import nba_api
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "nba_api", "kaggle", "pandas"])

import os
import pandas as pd
from nba_api.stats.endpoints import leaguestandings
import zipfile

# =============================
# 1. Get NBA Standings
# =============================
print("📊 Fetching current NBA standings (2025–26)...")

try:
    standings = leaguestandings.LeagueStandings(season='2025-26')
    df = standings.get_data_frames()[0]

    cols_to_keep = {
        "TeamCity": "City",
        "TeamName": "TeamName",
        "Conference": "Conference",
        "WINS": "Wins",
        "LOSSES": "Losses",
        "WinPCT": "Win%",
        "PlayoffRank": "Rank",
        "CurrentStreak": "Streak",
        "HOME": "Home",
        "ROAD": "Road",
        "DiffPointsPG": "PointDiff"
    }

    df = df[list(cols_to_keep.keys())].rename(columns=cols_to_keep)
    df["Team"] = df["City"] + " " + df["TeamName"]

    final_df = df[["Team", "Conference", "Rank", "Wins", "Losses", "Win%", "Streak", "Home", "Road", "PointDiff"]]
    final_df = final_df.sort_values(["Conference", "Rank"]).reset_index(drop=True)
    final_df.to_csv("nba_standings.csv", index=False)
    print("✅ nba_standings.csv saved!")

except Exception as e:
    print(f"⚠️ Could not fetch standings due to: {e}")
    pd.DataFrame().to_csv("nba_standings.csv", index=False)
    print("⚠️ Empty nba_standings.csv created as fallback.")

# =============================
# 2. Download & Extract Kaggle Dataset
# =============================
print("📥 Downloading dataset from Kaggle...")

# Assumes kaggle.json is already authenticated in the environment
os.system("kaggle datasets download -d eoinamoore/historical-nba-data-and-player-box-scores")

with zipfile.ZipFile("historical-nba-data-and-player-box-scores.zip", 'r') as zip_ref:
    zip_ref.extractall("nba_data")

print("✅ Extraction complete!")

# =============================
# 3. Filter to 2025–26 Season
# =============================
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

# =============================
# 4. Save only filtered data to ROOT folder
# =============================
print("💾 Saving filtered datasets...")

current_season_games.to_csv("Games_filtered.csv", index=False)
current_season_players.to_csv("PlayerStatistics_filtered.csv", index=False)
current_season_teams.to_csv("TeamStatistics_filtered.csv", index=False)

print(f"✅ Saved {len(current_season_games)} games")
print(f"✅ Saved {len(current_season_players)} player stats")
print(f"✅ Saved {len(current_season_teams)} team stats")

# =============================
# 5. Cleanup large files to avoid GitHub limit errors
# =============================
print("🧹 Cleaning up unneeded large files...")

os.remove("historical-nba-data-and-player-box-scores.zip")
os.remove("nba_data/Games.csv")
os.remove("nba_data/PlayerStatistics.csv")
os.remove("nba_data/TeamStatistics.csv")

print("✅ Cleanup complete!")
