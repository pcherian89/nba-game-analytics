# streamlit_app.py
import streamlit as st
import pandas as pd

# === Load filtered data from GitHub ===
@st.cache_data
def load_data():
    base_url = "https://raw.githubusercontent.com/pcherian89/nba-game-analytics/main/"
    games = pd.read_csv(base_url + "Games_filtered.csv", low_memory=False)
    players = pd.read_csv(base_url + "PlayerStatistics_filtered.csv", low_memory=False)
    teams = pd.read_csv(base_url + "TeamStatistics_filtered.csv", low_memory=False)

    games['gameDate'] = pd.to_datetime(games['gameDate'], errors='coerce')
    players['gameDate'] = pd.to_datetime(players['gameDate'], errors='coerce')
    teams['gameDate'] = pd.to_datetime(teams['gameDate'], errors='coerce')

    return games, players, teams

games_df, player_df, team_df = load_data()

# === UI: Matchup Input ===
st.title("🏀 NBA Game Analyzer (2024–25)")
user_input = st.text_input("What game do you want to check? (e.g., 'Knicks vs Pacers')", "")

if "vs" in user_input.lower():
    team1, team2 = [t.strip().lower() for t in user_input.split("vs")]

    # === Filter matching games ===
    matches = games_df[
        ((games_df['hometeamName'].str.lower() == team1) & (games_df['awayteamName'].str.lower() == team2)) |
        ((games_df['hometeamName'].str.lower() == team2) & (games_df['awayteamName'].str.lower() == team1))
    ].copy()

    if not matches.empty:
        matches['label'] = (
            "Game on " + matches['gameDate'].dt.strftime("%Y-%m-%d") + " — " +
            matches['hometeamName'] + " " + matches['homeScore'].astype(str) +
            " vs " +
            matches['awayteamName'] + " " + matches['awayScore'].astype(str)
        )

        selected_label = st.selectbox("✅ Select a game", matches['label'].tolist())
        selected_game = matches[matches['label'] == selected_label].iloc[0]
        selected_gameId = selected_game['gameId']

        # === Filter player stats for that game ===
        game_players = player_df[player_df['gameId'] == selected_gameId].copy()

        # Split by team
        home_team = selected_game['hometeamName']
        away_team = selected_game['awayteamName']
        home_players = game_players[game_players['playerteamName'] == home_team]
        away_players = game_players[game_players['playerteamName'] == away_team]

        # === Display Player Stats (Full View) ===
        player_display_cols = [
            'firstName', 'lastName', 'numMinutes', 'points', 'assists', 'blocks', 'steals',
            'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalsPercentage',
            'threePointersMade', 'threePointersAttempted', 'threePointersPercentage',
            'freeThrowsMade', 'freeThrowsAttempted', 'freeThrowsPercentage',
            'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal',
            'turnovers', 'foulsPersonal', 'plusMinusPoints'
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📊 {home_team} Player Stats")
            st.dataframe(home_players[player_display_cols].reset_index(drop=True))

        with col2:
            st.subheader(f"📊 {away_team} Player Stats")
            st.dataframe(away_players[player_display_cols].reset_index(drop=True))

        # === Display Team Stats (Full View) ===
        team_stats = team_df[team_df['gameId'] == selected_gameId].copy()
        team_display_cols = [
            'teamName', 'teamScore', 'assists', 'blocks', 'steals',
            'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalsPercentage',
            'threePointersMade', 'threePointersAttempted', 'threePointersPercentage',
            'freeThrowsMade', 'freeThrowsAttempted', 'freeThrowsPercentage',
            'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal',
            'turnovers', 'foulsPersonal', 'plusMinusPoints', 'benchPoints',
            'q1Points', 'q2Points', 'q3Points', 'q4Points',
            'biggestLead', 'biggestScoringRun', 'leadChanges',
            'pointsFastBreak', 'pointsFromTurnovers', 'pointsInThePaint', 'pointsSecondChance'
        ]

        st.subheader("🏟️ Full Team Stats")
        st.dataframe(team_stats[team_display_cols].reset_index(drop=True))

    else:
        st.warning("❌ No games found for that matchup.")
else:
    st.info("Type a matchup using the format: `Team1 vs Team2`")
