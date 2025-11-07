import os
import pandas as pd
import zipfile
import subprocess
from nba_api.stats.endpoints import leaguestandings

# ========== STEP 0: Install Requirements ==========
# You can remove this section if already installing in workflow
try:
    import kaggle
except ImportError:
    subprocess.check_call(["pip", "install", "kaggle"])

try:
    import nba_api
except ImportError:
    subprocess.check_call(["pip", "install", "nba_api"])

# ========== STEP 1: Fetch NBA Standings (2025–26) ==========
print("📊 Fetching current NBA standings (2025–26)...")

def save_empty_standings():
    pd.DataFrame(columns=[
        "Team", "Conference", "Rank", "Wins", "Losses", "Win%", "Streak", "Home", "Road", "PointDiff"
    ]).to_csv("nba_standings.csv", index=False)
    print("⚠️ Empty nba_standings.csv created as fallback.")

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
    final_df.to_csv("nba_standings_backup.csv", index=False)
    print("✅ nba_standings.csv saved!")

except Exception as e:
    print(f"⚠️ Could not fetch standings due to: {e}")
    if os.path.exists("nba_standings_backup.csv"):
        backup_df = pd.read_csv("nba_standings_backup.csv")
        backup_df.to_csv("nba_standings.csv", index=False)
        print("♻️ Restored nba_standings.csv from backup.")
    else:
        save_empty_standings()

# ========== STEP 2: Download Kaggle Dataset ==========
print("📥 Downloading dataset from Kaggle...")

dataset_name = "eoinamoore/historical-nba-data-and-player-box-scores"
zip_file = "historical-nba-data-and-player-box-scores.zip"
extract_dir = "nba_data"

# Download
subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name], check=True)

# Extract
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("✅ Extraction complete!")

# ========== STEP 3: Filter Data for 2025–26 ==========
print("🔎 Filtering data for 2025–26...")

games_df = pd.read_csv(f"{extract_dir}/Games.csv", low_memory=False)
player_stats_df = pd.read_csv(f"{extract_dir}/PlayerStatistics.csv", low_memory=False)
team_stats_df = pd.read_csv(f"{extract_dir}/TeamStatistics.csv", low_memory=False)

# Parse dates
games_df["gameDate"] = pd.to_datetime(games_df["gameDate"], errors="coerce", utc=True)

# Filter from start of 2025–26 season
start_date = pd.Timestamp("2025-10-21", tz="UTC")
current_season_games = games_df[games_df["gameDate"] >= start_date].copy()

keep_ids = set(current_season_games["gameId"].dropna().astype(str))
player_stats_df["gameId"] = player_stats_df["gameId"].astype(str)
team_stats_df["gameId"] = team_stats_df["gameId"].astype(str)

current_season_players = player_stats_df[player_stats_df["gameId"].isin(keep_ids)].copy()
current_season_teams = team_stats_df[team_stats_df["gameId"].isin(keep_ids)].copy()

# ========== STEP 4: Save Filtered Files ==========
print("💾 Saving filtered datasets...")

current_season_games.to_csv("Games_filtered.csv", index=False)
current_season_players.to_csv("PlayerStatistics_filtered.csv", index=False)
current_season_teams.to_csv("TeamStatistics_filtered.csv", index=False)

print(f"✅ Saved {len(current_season_games)} games")
print(f"✅ Saved {len(current_season_players)} player stats")
print(f"✅ Saved {len(current_season_teams)} team stats")

