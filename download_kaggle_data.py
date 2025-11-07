from nba_api.stats.endpoints import leaguestandings
import pandas as pd
import os

# =============================
# 1. Save Standings for 2025-26
# =============================
standings = leaguestandings.LeagueStandings(season='2025-26')
df = standings.get_data_frames()[0]

# Rename and reduce columns
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


# =============================
# 2. Load Full Kaggle Data
# =============================
print("📥 Loading full NBA datasets...")
games_df = pd.read_csv("nba_data/Games.csv", low_memory=False)
player_stats_df = pd.read_csv("nba_data/PlayerStatistics.csv", low_memory=False)
team_stats_df = pd.read_csv("nba_data/TeamStatistics.csv", low_memory=False)

# =============================
# 3. Filter to 2025–26 Season
# =============================
games_df["gameDate"] = pd.to_datetime(games_df["gameDate"], errors="coerce", utc=True)
start_date = pd.Timestamp("2025-10-21", tz="UTC")
current_season_games = games_df[games_df["gameDate"] >= start_date].copy()

keep_ids = set(current_season_games["gameId"].dropna().astype(str))

player_stats_df["gameId"] = player_stats_df["gameId"].astype(str)
team_stats_df["gameId"] = team_stats_df["gameId"].astype(str)

current_season_players = player_stats_df[player_stats_df["gameId"].isin(keep_ids)].copy()
current_season_teams = team_stats_df[team_stats_df["gameId"].isin(keep_ids)].copy()

# =============================
# 4. Save to ROOT folder (for Streamlit)
# =============================
current_season_games.to_csv("Games_filtered.csv", index=False)
current_season_players.to_csv("PlayerStatistics_filtered.csv", index=False)
current_season_teams.to_csv("TeamStatistics_filtered.csv", index=False)

print(f"✅ Saved {len(current_season_games)} games to Games_filtered.csv")
print(f"✅ Saved {len(current_season_players)} player rows to PlayerStatistics_filtered.csv")
print(f"✅ Saved {len(current_season_teams)} team rows to TeamStatistics_filtered.csv")

