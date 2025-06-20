# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import numpy as np


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
st.title("🏀 B-Ball IQ")
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
        
        
        # Create the bar chart
        fig_ratings = px.bar(
            ratings_melted,
            x="teamName",
            y="Value",
            color="RatingType",
            barmode="group",
            title="Team Offensive vs Defensive Ratings",
            labels={"teamName": "Team", "Value": "Rating", "RatingType": "Metric"},
            color_discrete_map={"OffensiveRating": "green", "DefensiveRating": "red"},
            width=700,   # ✅ Narrow chart
            height=400
        )
        
        # Optional: aesthetic layout tweaks
        fig_ratings.update_layout(
            title_font=dict(size=20),
            font=dict(size=14),
            margin=dict(l=40, r=40, t=50, b=40),
            plot_bgcolor="white",
            legend_title_text=""
        )
        
        # Show it directly (left-aligned)
        st.plotly_chart(fig_ratings, use_container_width=False)  # ✅ Don't stretch full width


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

        
        # === Step 1: Compute Custom Scores and Per-Minute Ratings ===
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
        
            # Score per minute (avoid divide-by-zero)
            df["off_per_min"] = df["offensiveScore"] / df["numMinutes"].replace(0, 1)
            df["def_per_min"] = df["defensiveScore"] / df["numMinutes"].replace(0, 1)
        
        # === Step 2: Combine, Normalize, and Clean ===
        combined_players = pd.concat([home_players, away_players], ignore_index=True)
        combined_players["fullName"] = combined_players["firstName"] + " " + combined_players["lastName"]
        
        # Normalize per-minute scores to a 0–10 scale
        off_min, off_max = combined_players["off_per_min"].min(), combined_players["off_per_min"].max()
        def_min, def_max = combined_players["def_per_min"].min(), combined_players["def_per_min"].max()
        off_range = off_max - off_min if off_max - off_min != 0 else 1
        def_range = def_max - def_min if def_max - def_min != 0 else 1
        
        combined_players["OffensiveRating"] = 10 * (combined_players["off_per_min"] - off_min) / off_range
        combined_players["DefensiveRating"] = 10 * (combined_players["def_per_min"] - def_min) / def_range
        
        # Round for better UI
        combined_players["OffensiveRating"] = combined_players["OffensiveRating"].round(2)
        combined_players["DefensiveRating"] = combined_players["DefensiveRating"].round(2)
        
        # Clean up infinite/NaN
        for col in ["OffensiveRating", "DefensiveRating"]:
            combined_players[col] = combined_players[col].replace([np.inf, -np.inf], np.nan)
        
        # Filter: Played at least 10 minutes
        combined_players = combined_players[combined_players["numMinutes"].fillna(0) >= 10].dropna(
            subset=["OffensiveRating", "DefensiveRating"]
        )

        
        # === Step 3: Interactive Visualization ===
        st.subheader("📊 Player Impact Ratings ")
        
        rating_type = st.radio("Select rating type to display:", ["OffensiveRating", "DefensiveRating"])
        
        combined_sorted = combined_players.sort_values(by=rating_type, ascending=False)
        
        # Dynamic hover columns
        if rating_type == "OffensiveRating":
            hover_cols = {
                "numMinutes": True,
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
                "steals": True,
                "blocks": True,
                "reboundsDefensive": True,
                "foulsPersonal": True,
                "DefensiveRating": True,
                "OffensiveRating": False,
                "fullName": False
            }
        
        # Chart
        fig = px.bar(
            combined_sorted,
            x="fullName",
            y=rating_type,
            color="playerteamName",
            title=f"Player {rating_type} (Per Minute)",
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

        import io
        from fpdf import FPDF
        import plotly.graph_objects as go
        
        # === Scouting Card Snapshot ===
        st.subheader("📋 Scouting Card Snapshot")
        
        selected_scout_player = st.selectbox("Select a player to view scouting card:", combined_players["fullName"].unique())
        scout_data = combined_players[combined_players["fullName"] == selected_scout_player].iloc[0]
        
        # === Formatter ===
        def format_value(val, style="int"):
            if pd.isna(val):
                return "-"
            if style == "int":
                return str(int(round(val)))
            elif style == "float":
                return f"{val:.2f}"
            elif style == "pct":
                return f"{val:.1%}"
            else:
                return str(val)
        
        # === Tables ===
        offense_df = pd.DataFrame({
            "Metric": ["Points", "Assists", "Turnovers", "FG%", "3P%", "FT%"],
            "Value": [
                format_value(scout_data["points"], "int"),
                format_value(scout_data["assists"], "int"),
                format_value(scout_data["turnovers"], "int"),
                format_value(scout_data["fieldGoalsPercentage"], "pct"),
                format_value(scout_data["threePointersPercentage"], "pct"),
                format_value(scout_data["freeThrowsPercentage"], "pct"),
            ]
        })
        
        defense_df = pd.DataFrame({
            "Metric": ["Rebounds", "Steals", "Blocks"],
            "Value": [
                format_value(scout_data["reboundsTotal"], "int"),
                format_value(scout_data["steals"], "int"),
                format_value(scout_data["blocks"], "int"),
            ]
        })
        
        summary_df = pd.DataFrame({
            "Metric": ["Minutes Played", "Plus/Minus", "Offensive Rating", "Defensive Rating"],
            "Value": [
                format_value(scout_data["numMinutes"], "float"),
                format_value(scout_data["plusMinusPoints"], "float"),
                format_value(scout_data["OffensiveRating"], "float"),
                format_value(scout_data["DefensiveRating"], "float"),
            ]
        })
        
        # === Display Clean Tables ===
        st.markdown("#### 🔥 Offensive Summary")
        st.table(offense_df.set_index("Metric"))
        
        st.markdown("#### 🧱 Defensive Summary")
        st.table(defense_df.set_index("Metric"))
        
        st.markdown("#### 📈 Player Summary")
        st.table(summary_df.set_index("Metric"))

        from langchain.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain.chains import LLMChain
        
        # === Define prompt template ===
        summary_prompt = ChatPromptTemplate.from_template("""
        You are a basketball performance analyst.
        
        Below are game stats for {player_name}, who played {minutes} minutes in a recent game.
        
        Your task is to write a concise performance summary with the following:
        - Key strengths (e.g., efficient scoring, strong defense, rebounding, etc.)
        - Notable weaknesses (e.g., low shooting %, high turnovers, low impact)
        - Clear suggestions for improvement, if applicable
        
        Important:
        - In this system, higher Offensive and Defensive Ratings indicate better performance.
        - Consider the player's stats relative to their minutes played.
        - Do not assume values are low or high without comparing to playing time or efficiency.
        - Keep the summary in 2–3 clear bullet points, each up to 50 words max.
        
        Stats:
        {stats}
        """)
        
        # === Initialize LLM ===
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        
        # === Prepare input ===
        player_name = scout_data["fullName"]
        minutes = scout_data["numMinutes"]
        stats_text = f"""
        Points: {scout_data['points']}
        Assists: {scout_data['assists']}
        Turnovers: {scout_data['turnovers']}
        FG%: {scout_data['fieldGoalsPercentage']:.1%}
        3P%: {scout_data['threePointersPercentage']:.1%}
        FT%: {scout_data['freeThrowsPercentage']:.1%}
        Rebounds: {scout_data['reboundsTotal']}
        Steals: {scout_data['steals']}
        Blocks: {scout_data['blocks']}
        Plus/Minus: {scout_data['plusMinusPoints']}
        Offensive Rating: {scout_data['OffensiveRating']:.2f}
        Defensive Rating: {scout_data['DefensiveRating']:.2f}
        """
        
        # === Run the agent ===
        summary_output = summary_chain.run({
            "player_name": player_name,
            "minutes": minutes,
            "stats": stats_text
        })
        
        # === Display the scouting summary ===
        st.markdown("### 🧠 Scouting Summary Report")
        st.markdown(summary_output)

            
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

        
        # === AI-Generated Game Summary with Session Persistence ===
        st.subheader("🧠 Game Summary")
        
        # Check if summary already exists for current game
        if "ai_summary" not in st.session_state or st.session_state.get("summary_game_id") != selected_gameId:
        
            # Convert team + player stats to markdown
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
        
            1. **Game Summary** – Brief overview of final score, standout players, momentum shifts.
            2. **Offensive Analysis** – Field goal %, 3P%, assists, offensive ratings, top scorers.
            3. **Defensive Analysis** – Steals, blocks, defensive rebounds, defensive ratings, impact defenders.
            4. **Bench & Support Players** – Contributions from depth players or surprises.
            5. **Final Verdict** – Why the winner prevailed and what limited the losing team.
        
            Keep the tone analytical but readable — like a top-tier sports recap.
            """
        
            # Generate + store summary
            with st.spinner("🧠 Generating game summary..."):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=600
                )
                summary_text = response.choices[0].message.content
                st.session_state.ai_summary = summary_text
                st.session_state.summary_game_id = selected_gameId
        
        # Display stored summary
        # st.markdown("#### 📝 Game Summary")
        st.write(st.session_state.ai_summary)


        st.markdown("### 🤖 Bot Analyst")
        st.markdown("Ask follow-up questions about this game — player roles, tactics, bench impact, or who the MVP was!")
        
        # Create one markdown table of relevant stats
        team_md = team_stats[["teamName", "teamScore", "assists", "turnovers", "reboundsTotal", 
                              "fieldGoalsPercentage", "threePointersPercentage"]].to_markdown(index=False)
        
        player_md = combined_players[["fullName", "playerteamName", "points", "assists", "reboundsTotal", 
                                      "turnovers", "plusMinusPoints", "OffensiveRating", "DefensiveRating"]].to_markdown(index=False)
        
        # Full context
        context = f"""TEAM STATS:\n{team_md}\n\nPLAYER STATS:\n{player_md}"""
        
        # Define role + tone of the analyst
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
        You are a highly skilled basketball analyst working for a professional team. 
        You are reviewing detailed game data to provide sharp, insightful answers.
        
        Game context:
        {context}
        
        Answer the user's question using this data. 
        Always highlight tactical trends, key player impact, and any relevant performance nuance.
        
        Question: {question}
        Answer as an expert analyst:
        """
        )
        
        # Setup LLM
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], temperature=0.4)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        
        # Input and response
        # === Session State Initialization ===
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # === Chat Input & Response Handling ===
        user_question = st.chat_input("Ask your basketball question...")
        
        if user_question:
            with st.spinner("🧠 Analyzing game data..."):
                response = chain.run({"context": context, "question": user_question})
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", response))
        
        # === Display Chat History ===
        if st.session_state.chat_history:
            st.markdown("#### 🤖 Bot Analyst Conversation")
        
            for sender, msg in st.session_state.chat_history:
                if sender == "You":
                    st.markdown(f"🧍‍♂️ **{sender}**: {msg}")
                else:
                    st.markdown(f"🤖 **{sender}**: {msg}")
        
            # Add reset button
            if st.button("🧹 Clear Chat"):
                st.session_state.chat_history = []

            
    else:
        st.warning("❌ No games found for that matchup.")
else:
    st.info("Type a matchup using the format: `Team1 vs Team2`")
