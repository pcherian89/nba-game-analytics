# =============================
# Automated NBA Data + Standings Updater
# =============================
import os
import time
import pandas as pd
import zipfile
from pathlib import Path

# Ensure dependencies exist in GitHub Action environment
try:
    from nba_api.stats.endpoints import leaguestandings
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "nba_api", "kaggle", "pandas"])

# ---------- 1. Fetch NBA Standings with Retry ----------
print("📊 Fetching current NBA standings (2025–26)...")

def fetch_standings(max_retries=5, delay=10):
    from nba_api.stats.endpoints import leaguestandings

    for attempt in range(max_retries):
        try:
            standings = leaguestandings.LeagueStandings(season='2025-26', timeout=60)
            df = standings.get_data_frames()[0]
            print(f"✅ Standings fetched successfully on attempt {attempt + 1}")
            return df
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("❌ All attempts failed. Returning None.")
                return None

df = fetch_standings()

if df is not None and not df.empty:
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
    print("✅ nba_standings.csv saved successfully!")
else:
    # fallback: keep last known file if exists
    if Path("nba_standings.csv").exists():
        print("⚠️ Could not fetch new standings — keeping previous file.")
    else:
        cols = ["Team", "Conference", "Rank", "Wins", "Losses", "Win%", "Streak", "Home", "Road", "PointDiff"]
        pd.DataFrame(columns=cols).to_csv("nba_standings.csv", index=False)
        print("⚠️ Created empty nba_standings.csv with headers as fallback.")

# ---------- 2. Download & Extract Kaggle Dataset ----------
print("📥 Downloading dataset from Kaggle...")
os.system("kaggle datasets download -d eoinamoore/historical-nba-data-and-player-box-scores")

with zipfile.ZipFile("historical-nba-data-and-player-box-scores.zip", 'r') as zip_ref:
    zip_ref.extractall("nba_data")
print("✅ Extraction complete!")

# ---------- 3. Filter to 2025–26 Season ----------
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

# ---------- 4. Save only filtered datasets ----------
print("💾 Saving filtered datasets...")
current_season_games.to_csv("Games_filtered.csv", index=False)
current_season_players.to_csv("PlayerStatistics_filtered.csv", index=False)
current_season_teams.to_csv("TeamStatistics_filtered.csv", index=False)
print(f"✅ Saved {len(current_season_games)} games")
print(f"✅ Saved {len(current_season_players)} player stats")
print(f"✅ Saved {len(current_season_teams)} team stats")

# ---------- 5. Cleanup large files ----------
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

