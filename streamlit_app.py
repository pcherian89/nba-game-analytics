# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI  # ✅ new SDK

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # ✅ secure and Streamlit Cloud-ready


st.set_page_config(page_title="NBA Game Analyzer", layout="wide")

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
user_input = st.text_input("What game do you want to check? (e.g., 'Warriors vs Celtics')", "")

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

        st.subheader(f"📊 {home_team} Player Stats")
        st.dataframe(home_players[player_display_cols].reset_index(drop=True), use_container_width=True)

        st.subheader(f"📊 {away_team} Player Stats")
        st.dataframe(away_players[player_display_cols].reset_index(drop=True), use_container_width=True)

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

        # === Compute Offensive & Defensive Ratings ===
        def estimate_possessions(row):
            return (
                row["fieldGoalsAttempted"] +
                0.44 * row["freeThrowsAttempted"] -
                row["reboundsOffensive"] +
                row["turnovers"]
            )
        
        team_stats["possessions"] = team_stats.apply(estimate_possessions, axis=1)
        
        # Match rows
        team1 = team_stats.iloc[0]
        team2 = team_stats.iloc[1]
        
        team_stats.loc[team_stats.index[0], "OffensiveRating"] = 100 * team1["teamScore"] / team1["possessions"]
        team_stats.loc[team_stats.index[0], "DefensiveRating"] = 100 * team2["teamScore"] / team1["possessions"]
        
        team_stats.loc[team_stats.index[1], "OffensiveRating"] = 100 * team2["teamScore"] / team2["possessions"]
        team_stats.loc[team_stats.index[1], "DefensiveRating"] = 100 * team1["teamScore"] / team2["possessions"]
        
        # === Visualize Ratings ===
        ratings_df = team_stats[["teamName", "OffensiveRating", "DefensiveRating"]].copy()
        ratings_melted = ratings_df.melt(id_vars="teamName", var_name="RatingType", value_name="Value")
        
        fig_ratings = px.bar(
            ratings_melted,
            x="teamName",
            y="Value",
            color="RatingType",
            barmode="group",
            title="Team Offensive vs Defensive Ratings",
            labels={"teamName": "Team", "Value": "Rating", "RatingType": "Metric"},
            color_discrete_map={"OffensiveRating": "green", "DefensiveRating": "red"}
        )
        
        st.plotly_chart(fig_ratings, use_container_width=True)

        
        import plotly.express as px

        # === Combine Home & Away Players ===
        combined_players = pd.concat([home_players, away_players], ignore_index=True)
        
        # === Add Full Name Column ===
        combined_players["fullName"] = combined_players["firstName"] + " " + combined_players["lastName"]
        
        # === Add Rebound Total (if not already in the data) ===
        if "reboundsTotal" not in combined_players.columns:
            combined_players["reboundsTotal"] = (
                combined_players.get("reboundsOffensive", 0) + combined_players.get("reboundsDefensive", 0)
            )
        
        # === User Selects Stat to View ===
        stat_option = st.selectbox("📈 View top players by:", ["points", "assists", "reboundsTotal", "turnovers", "plusMinusPoints"])
        
        # === Filter Top 6 Players for the Selected Stat ===
        top6 = combined_players.sort_values(by=stat_option, ascending=False).head(6)
        
        # === Create Interactive Plotly Bar Chart ===
        fig = px.bar(
            top6,
            x=stat_option,
            y="fullName",
            color="playerteamName",
            orientation="h",
            title=f"Top 6 Players by {stat_option.replace('Points', ' Points').title()}",
            labels={stat_option: stat_option.title(), "fullName": "Player", "playerteamName": "Team"},
            color_discrete_sequence=["dodgerblue", "darkorange"]  # Customize color mapping
        )
        
        # Reverse Y-axis (so highest is on top)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        
        # === Display Chart in Streamlit ===
        st.plotly_chart(fig, use_container_width=True)

        import numpy as np
        
        # === Step 1: Estimate Player Possessions ===
        team_minutes_home = home_players["numMinutes"].fillna(0).sum()
        team_minutes_away = away_players["numMinutes"].fillna(0).sum()
        
        # Avoid division by zero
        if team_minutes_home == 0:
            team_minutes_home = 1
        if team_minutes_away == 0:
            team_minutes_away = 1
        
        # Use possessions from team stats
        poss_home = team_stats.iloc[0]["possessions"]
        poss_away = team_stats.iloc[1]["possessions"]
        
        # Estimate possessions per player
        home_players["estimatedPossessions"] = (
            home_players["numMinutes"].fillna(0) / team_minutes_home
        ) * poss_home
        
        away_players["estimatedPossessions"] = (
            away_players["numMinutes"].fillna(0) / team_minutes_away
        ) * poss_away
        
        # === Step 2: Compute Custom Scores and Ratings ===
        for df in [home_players, away_players]:
            df["offensiveScore"] = (
                df["points"]
                + 1.5 * df["assists"]
                - 2.0 * df["turnovers"]
                + 1.0 * df["reboundsOffensive"] 
            )
        
            df["defensiveScore"] = (
                1.5 * df["steals"]
                + 1.5 * df["blocks"]
                + 1.0 * df["reboundsDefensive"]
                - 0.5 * df["foulsPersonal"]
            )
        
            # Normalize by possessions, then scale to 100 possessions
            df["OffensiveRating"] = 100 * df["offensiveScore"] / df["estimatedPossessions"]
            df["DefensiveRating"] = 100 * df["defensiveScore"] / df["estimatedPossessions"]
        
        # === Step 3: Combine & Clean ===
        combined_players = pd.concat([home_players, away_players], ignore_index=True)
        combined_players["fullName"] = combined_players["firstName"] + " " + combined_players["lastName"]
        
        # Clean NaNs/Infs
        for col in ["OffensiveRating", "DefensiveRating"]:
            combined_players[col] = combined_players[col].replace([np.inf, -np.inf], np.nan)
        
        combined_players = combined_players.dropna(subset=["OffensiveRating", "DefensiveRating"])
        
        # Optional: Filter players who played < 5 mins
        combined_players = combined_players[combined_players["numMinutes"].fillna(0) >= 5]
        
        # === Step 4: Interactive Visualization ===
        st.subheader("📊 Player Impact Ratings Per 100 Possessions")
        
        rating_type = st.radio("Select rating type to display:", ["OffensiveRating", "DefensiveRating"])
        
        combined_sorted = combined_players.sort_values(by=rating_type, ascending=False)
        
        # Dynamic tooltips based on rating type
        if rating_type == "OffensiveRating":
            hover_cols = {
                "numMinutes": True,
                "estimatedPossessions": True,
                "points": True,
                "assists": True,
                "turnovers": True,
                "reboundsOffensive": True,
                "OffensiveRating": True,
                "DefensiveRating": False,
                "fullName": False
            }
        else:
            hover_cols = {
                "numMinutes": True,
                "estimatedPossessions": True,
                "steals": True,
                "blocks": True,
                "reboundsDefensive": True,
                "foulsPersonal": True,
                "DefensiveRating": True,
                "OffensiveRating": False,
                "fullName": False
            }
        
        # Updated chart with smart hover
        fig = px.bar(
            combined_sorted,
            x="fullName",
            y=rating_type,
            color="playerteamName",
            title=f"Player {rating_type} (Per 100 Possessions)",
            labels={"fullName": "Player", "playerteamName": "Team", rating_type: "Rating"},
            color_discrete_sequence=["dodgerblue", "darkorange"],
            hover_data=hover_cols
        )

        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


        st.subheader("🏆 MVP Comparison Table – Top 3 Performers")

        # Top 3 players by Offensive Rating
        top3 = combined_players.sort_values(by="OffensiveRating", ascending=False).head(3)
            
        # Define the stats you want to display
        stat_fields = [
            "points", "assists", "reboundsTotal", "turnovers",
            "OffensiveRating", "DefensiveRating", "plusMinusPoints"
        ]
            
        # Format player names as column headers
        comparison_df = top3[["fullName"] + stat_fields].set_index("fullName").T
            
        # Rename rows to be more readable
        comparison_df.index = [
            "Points", "Assists", "Total Rebounds", "Turnovers",
            "Offensive Rating", "Defensive Rating", "+/- Impact"
        ]
            
        st.dataframe(comparison_df, use_container_width=True)

            
        st.subheader("📊 Compare Any Two Players")

        # Create dropdowns to select players
        # Only include players who had recorded minutes
        valid_players = combined_players[combined_players["numMinutes"].notna() & (combined_players["numMinutes"] > 0)]
        player_names = valid_players["fullName"].unique().tolist()
    
        col1, col2 = st.columns(2)
        with col1:
            player1 = st.selectbox("Select Player 1", player_names, key="p1")
        with col2:
            player2 = st.selectbox("Select Player 2", player_names, key="p2")
            
        # Filter player stats
        p1_stats = valid_players[valid_players["fullName"] == player1].iloc[0]
        p2_stats = valid_players[valid_players["fullName"] == player2].iloc[0]

        # Stats to display
        compare_fields = [
            "numMinutes", "points", "assists", "reboundsOffensive", "reboundsDefensive",
            "reboundsTotal", "steals", "blocks", "turnovers", "foulsPersonal",
            "fieldGoalsMade", "fieldGoalsAttempted", "fieldGoalsPercentage",
            "threePointersMade", "threePointersAttempted", "threePointersPercentage",
            "freeThrowsMade", "freeThrowsAttempted", "freeThrowsPercentage",
            "plusMinusPoints", "OffensiveRating", "DefensiveRating"
        ]
            
        # Prepare comparison table
        comparison_table = pd.DataFrame({
            "Stat": [field for field in compare_fields],
            player1: [p1_stats[field] for field in compare_fields],
            player2: [p2_stats[field] for field in compare_fields]
        })
            
        # Beautify stat names
        rename_map = {
            "numMinutes": "Minutes Played", "points": "Points", "assists": "Assists",
            "reboundsOffensive": "Offensive Rebounds", "reboundsDefensive": "Defensive Rebounds",
            "reboundsTotal": "Total Rebounds", "steals": "Steals", "blocks": "Blocks",
            "turnovers": "Turnovers", "foulsPersonal": "Fouls",
            "fieldGoalsMade": "FG Made", "fieldGoalsAttempted": "FG Attempted", "fieldGoalsPercentage": "FG%",
            "threePointersMade": "3P Made", "threePointersAttempted": "3P Attempted", "threePointersPercentage": "3P%",
            "freeThrowsMade": "FT Made", "freeThrowsAttempted": "FT Attempted", "freeThrowsPercentage": "FT%",
            "plusMinusPoints": "+/-", "OffensiveRating": "Offensive Rating", "DefensiveRating": "Defensive Rating"
        }
        comparison_table["Stat"] = comparison_table["Stat"].replace(rename_map)
            
        # Display
        st.dataframe(comparison_table.set_index("Stat"), use_container_width=True)


        
        # === AI-Generated Summary ===
        st.subheader("🧠 AI Game Summary")
        
        if st.button("Generate AI Summary"):
        
            # Convert DataFrames to markdown
            team_md = team_stats[["teamName", "teamScore", "assists", "turnovers", "reboundsTotal", 
                                  "fieldGoalsPercentage", "threePointersPercentage"]].to_markdown(index=False)
        
            player_md = combined_players[["fullName", "points", "assists", "reboundsOffensive", 
                                          "reboundsDefensive", "turnovers", "plusMinusPoints", 
                                          "OffensiveRating", "DefensiveRating"]].to_markdown(index=False)
        
            prompt = f"""
            You are a professional sports analyst. Analyze the following NBA game using the stats below:
            
            TEAM STATS:
            {team_md}
            
            PLAYER STATS:
            {player_md}
            
            Generate a structured analysis with the following sections:
            
            1. **Game Summary** – Provide a brief overview of the final score, standout players, and momentum shifts.
            2. **Offensive Analysis** – Discuss offensive efficiency, field goal %, 3P%, assists, and player offensive ratings. Highlight who drove scoring and any inefficiencies.
            3. **Defensive Analysis** – Analyze steals, blocks, defensive rebounds, and defensive ratings. Identify defensive anchors or lapses.
            4. **Bench & Support Players** – Review bench contributions, depth scoring, and any surprising impact players.
            5. **Final Verdict** – Conclude why the winning team prevailed and what limited the losing side.
            
            Keep your tone professional but readable, like a sports media recap. Use specific stat references when helpful.
            """

        
            with st.spinner("Analyzing game..."):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=500
                )
        
            st.markdown("### 📝 AI-Generated Game Summary")
            st.write(response.choices[0].message.content)

            
    else:
        st.warning("❌ No games found for that matchup.")
else:
    st.info("Type a matchup using the format: `Team1 vs Team2`")
