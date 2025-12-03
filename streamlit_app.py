# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import numpy as np
import time
from nba_api.stats.endpoints import leaguestandings

from openai import OpenAI  # ✅ new SDK

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # ✅ secure and Streamlit Cloud-ready


st.set_page_config(page_title="NBA Game Analyzer", layout="wide")

import streamlit as st
import pandas as pd

# === Load filtered data from GitHub ===
@st.cache_data
def load_data():
    base_url = "https://raw.githubusercontent.com/pcherian89/nba-game-analytics/main/"
    
    games = pd.read_csv(base_url + "Games_filtered.csv", low_memory=False)
    players = pd.read_csv(base_url + "PlayerStatistics_filtered.csv", low_memory=False)
    teams = pd.read_csv(base_url + "TeamStatistics_filtered.csv", low_memory=False)
    standings = pd.read_csv(base_url + "nba_standings.csv")

    games['gameDate'] = pd.to_datetime(games['gameDate'], errors='coerce')
    players['gameDate'] = pd.to_datetime(players['gameDate'], errors='coerce')
    teams['gameDate'] = pd.to_datetime(teams['gameDate'], errors='coerce')

    standings['Team'] = standings['Team'].str.replace("*", "", regex=False).str.strip()
    
    return games, players, teams, standings

games_df, player_df, team_df, standings_df = load_data()

# === Add Team Logos ===
logo_base_url = "https://raw.githubusercontent.com/pcherian89/nba-game-analytics/main/"

team_abbrev_map = {
    "Atlanta Hawks": "atl", "Boston Celtics": "bos", "Brooklyn Nets": "bkn", "Charlotte Hornets": "cha",
    "Chicago Bulls": "chi", "Cleveland Cavaliers": "cle", "Dallas Mavericks": "dal", "Denver Nuggets": "den",
    "Detroit Pistons": "det", "Golden State Warriors": "gsw", "Houston Rockets": "hou", "Indiana Pacers": "ind",
    "LA Clippers": "lac", "Los Angeles Lakers": "lal", "Memphis Grizzlies": "mem", "Miami Heat": "mia",
    "Milwaukee Bucks": "mil", "Minnesota Timberwolves": "min", "New Orleans Pelicans": "nop", "New York Knicks": "nyk",
    "Oklahoma City Thunder": "okc", "Orlando Magic": "orl", "Philadelphia 76ers": "phl", "Phoenix Suns": "phx",
    "Portland Trail Blazers": "por", "Sacramento Kings": "sac", "San Antonio Spurs": "sas", "Toronto Raptors": "tor",
    "Utah Jazz": "uth", "Washington Wizards": "was"
}

# === Streamlit Title ===
st.title("🏀 ThynkBall")

# === Standings Table Function ===
def display_standings_table(df, conference_code):
    display_name = "Eastern Conference" if conference_code == "East" else "Western Conference"
    st.markdown(f"<h2 style='text-align: left; font-family: Inter, sans-serif; padding-top: 20px;'>{display_name}</h2>", unsafe_allow_html=True)

    conf_df = df[df["Conference"] == conference_code].copy()
    conf_df["Team"] = conf_df["Team"].str.strip()
    conf_df = conf_df.sort_values(by=["Rank"]).reset_index(drop=True)

    conf_df["Abbrev"] = conf_df["Team"].map(team_abbrev_map)
    conf_df["Logo URL"] = conf_df["Abbrev"].apply(
        lambda abbr: f"{logo_base_url}{abbr}.png" if pd.notnull(abbr) else "")

    table_html = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 40px;
            font-family: Inter, sans-serif;
        }
        thead th {
            background-color: #f0f2f6;
            padding: 10px;
            font-size: 16px;
            text-align: center;
        }
        td {
            padding: 12px;
            font-size: 15px;
            text-align: center;
        }
        td.team-name {
            font-weight: 600;
            text-align: left;
            padding-left: 10px;
        }
        td.logo {
            text-align: center;
        }
        img {
            vertical-align: middle;
        }
    </style>
    <table>
        <thead>
            <tr>
                <th>Logo</th>
                <th style='text-align: left;'>Team</th>
                <th>Rank</th>
                <th>Wins</th>
                <th>Losses</th>
                <th>Win%</th>
                <th>Streak</th>
                <th>Home</th>
                <th>Road</th>
                <th>Point Diff</th>
            </tr>
        </thead>
        <tbody>
    """

    for _, row in conf_df.iterrows():
        logo_img = f"<img src='{row['Logo URL']}' width='40'>" if row["Logo URL"] else ""
        table_html += f"""
        <tr>
            <td class='logo'>{logo_img}</td>
            <td class='team-name'>{row["Team"]}</td>
            <td>{row["Rank"]}</td>
            <td>{row["Wins"]}</td>
            <td>{row["Losses"]}</td>
            <td>{row["Win%"]:.3f}</td>
            <td>{row["Streak"]}</td>
            <td>{row["Home"]}</td>
            <td>{row["Road"]}</td>
            <td>{row["PointDiff"]}</td>
        </tr>
        """

    table_html += "</tbody></table>"
    st.components.v1.html(table_html, height=600, scrolling=True)

# === Display Standings ===
display_standings_table(standings_df, "East")
display_standings_table(standings_df, "West")

# === Add spacing after both standings tables ===
st.markdown("<br>", unsafe_allow_html=True)


import streamlit as st
import pandas as pd

# === Load Data ===
todays_games = pd.read_csv("todays_games.csv")
team_stats = pd.read_csv("TeamStatistics_filtered.csv")

# === Stat Fields and Labels ===
stat_fields = {
    "Score Differential": lambda df: (df["teamScore"] - df["opponentScore"]).mean(),
    "Rebounds Total": lambda df: df["reboundsTotal"].mean(),
    "Assists": lambda df: df["assists"].mean(),
    "Steals": lambda df: df["steals"].mean(),
    "Field Goal %": lambda df: df["fieldGoalsPercentage"].mean(),
    "Three Point %": lambda df: df["threePointersPercentage"].mean(),
    "Turnovers": lambda df: df["turnovers"].mean()
}

# === Robust Parsing for teamCity/teamName ===
def split_city_name(full_name):
    if full_name == "Los Angeles Clippers":
        return "LA", "Clippers"
    parts = full_name.split(" ")
    for i in range(1, len(parts)):
        city = " ".join(parts[:i])
        name = " ".join(parts[i:])
        subset = team_stats[
            (team_stats["teamCity"] == city) & (team_stats["teamName"] == name)
        ]
        if not subset.empty:
            return city, name
    return None, None

# === Rank Coloring Function ===
def rank_color(rank):
    if rank <= 10:
        return "green"
    elif rank <= 20:
        return "orange"
    else:
        return "red"


# ========================================================
# 🚀 NEW: LLM-POWERED TEAM SCOUTING REPORT (FULL FUNCTION)
# ========================================================

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import openai  # for RateLimitError

llm = ChatOpenAI(model="gpt-4o", temperature=0.45, max_retries=3)  # balanced creativity

team_scout_prompt = ChatPromptTemplate.from_template("""

You are an elite NBA scouting analyst generating a professional matchup report.

Two teams are playing:
- Team 1: {team1_name}
- Team 2: {team2_name}

Team 1 Strengths: {team1_strengths}
Team 1 Weaknesses: {team1_weaknesses}

Team 2 Strengths: {team2_strengths}
Team 2 Weaknesses: {team2_weaknesses}

=========================
🚨 VERY IMPORTANT RULES
=========================
- ALWAYS include ALL FOUR section headings below.
- DO NOT remove, rename, or merge any sections.
- DO NOT add new section headings.
- DO NOT repeat raw numeric values.
- Write 3–5 sentences per section.
- Make sure each section is matchup-specific.
- Do NOT output anything before or after the sections.

=========================
📌 FORMAT EXACTLY LIKE THIS:
=========================

#### Offensive Analysis
[Write analysis here]

#### Defensive Analysis
[Write analysis here]

#### Rebounding & Possession Control
[Write analysis here]

#### Summary: Keys to the Matchup
[Write analysis here]

=========================
Now generate the report.
=========================

""")

team_scout_chain = LLMChain(llm=llm, prompt=team_scout_prompt)


# ========= CACHE LLM OUTPUT PER MATCHUP =========
@st.cache_data(show_spinner=False)
def get_team_scout_report_cached(
    team1_name: str,
    team2_name: str,
    strengths1: str,
    weaknesses1: str,
    strengths2: str,
    weaknesses2: str,
) -> str:
    """Call the LLM once per unique matchup and reuse the result."""
    return team_scout_chain.run({
        "team1_name": team1_name,
        "team2_name": team2_name,
        "team1_strengths": strengths1,
        "team1_weaknesses": weaknesses1,
        "team2_strengths": strengths2,
        "team2_weaknesses": weaknesses2,
    })


# ========================================================
# 🎯 FINAL generate_scouting_report FUNCTION (LLM VERSION)
# ========================================================

def generate_scouting_report(team1_name, team2_name, df1, df2):

    # helper: determine advantage from ranks
    def determine_advantage(stat_label):
        val1, _ = league_ranks[stat_label][team1_name]
        val2, _ = league_ranks[stat_label][team2_name]
        is_lower_better = stat_label == "Turnovers"
        if (val1 < val2 and is_lower_better) or (val1 > val2 and not is_lower_better):
            return team1_name
        elif val1 == val2:
            return "Even"
        return team2_name

    # compute strengths/weaknesses
    team1_strengths = [s for s in stat_fields if determine_advantage(s) == team1_name]
    team1_weaknesses = [s for s in stat_fields if determine_advantage(s) == team2_name]
    team2_strengths = [s for s in stat_fields if determine_advantage(s) == team2_name]
    team2_weaknesses = [s for s in stat_fields if determine_advantage(s) == team1_name]

    strengths1_str = ", ".join(team1_strengths) if team1_strengths else "None"
    weaknesses1_str = ", ".join(team1_weaknesses) if team1_weaknesses else "None"
    strengths2_str = ", ".join(team2_strengths) if team2_strengths else "None"
    weaknesses2_str = ", ".join(team2_weaknesses) if team2_weaknesses else "None"

    # === Run LLM (cached) ===
    try:
        scouting_output = get_team_scout_report_cached(
            team1_name,
            team2_name,
            strengths1_str,
            weaknesses1_str,
            strengths2_str,
            weaknesses2_str,
        )
        # === Render ===
        st.markdown(scouting_output)
    except openai.RateLimitError:
        st.warning(
            "⚠️ OpenAI rate limit reached while generating the scouting report. "
            "Please wait a few seconds and rerun, or avoid refreshing too often."
        )

    


# === Pre-calculate League Ranks ===
league_ranks = {}
for stat_label, func in stat_fields.items():
    values = []
    for team in team_stats.groupby(["teamCity", "teamName"]):
        val = func(team[1])
        values.append({
            "team": f"{team[0][0]} {team[0][1]}",
            "value": val
        })
    
    rank_df = pd.DataFrame(values)

    # 👇 Use ascending=True for Turnovers (lower is better), descending for others
    ascending = True if stat_label == "Turnovers" else False

    rank_df["rank"] = rank_df["value"].rank(ascending=ascending, method="min").astype(int)
    rank_map = dict(zip(rank_df["team"], zip(rank_df["value"], rank_df["rank"])))
    league_ranks[stat_label] = rank_map


# === Streamlit Header ===
st.markdown("## Today's NBA Matchups with Key Stats")

# === Display Matchups in Pairs ===
matchups = list(todays_games.iterrows())
for i in range(0, len(matchups), 2):
    cols = st.columns(2)
    
    for idx in range(2):
        if i + idx >= len(matchups):
            continue
        
        row = matchups[i + idx][1]
        home_team = row["Home_Team"]
        away_team = row["Away_Team"]

        home_city, home_name = split_city_name(home_team)
        away_city, away_name = split_city_name(away_team)

        if not home_city or not away_city:
            cols[idx].warning(f"🚫 Stats not available for: {home_team} vs {away_team}")
            continue

        home_df = team_stats[(team_stats["teamCity"] == home_city) & (team_stats["teamName"] == home_name)]
        away_df = team_stats[(team_stats["teamCity"] == away_city) & (team_stats["teamName"] == away_name)]

        if home_df.empty or away_df.empty:
            cols[idx].warning(f"🚫 Stats not available for: {home_team} vs {away_team}")
            continue

        home_full = f"{home_city} {home_name}"
        away_full = f"{away_city} {away_name}"
        home_abbr = team_abbrev_map.get(home_full, "").lower()
        away_abbr = team_abbrev_map.get(away_full, "").lower()

        # === Logos and VS text ===
        logo_col1, logo_col2, logo_col3 = cols[idx].columns([1, 0.3, 1])
        with logo_col1:
            st.image(f"{logo_base_url}{home_abbr}.png", width=100)
        with logo_col2:
            st.markdown(
                "<div style='text-align:center; display:flex; justify-content:center; align-items:center; height:100%;'>"
                "<h3><strong>vs</strong></h3></div>",
                unsafe_allow_html=True
            )
        with logo_col3:
            st.image(f"{logo_base_url}{away_abbr}.png", width=100)

        # === Stat Comparison with Rank Colors ===
        stat_col1, stat_col2 = cols[idx].columns(2)
        for label, func in stat_fields.items():
            home_val = func(home_df)
            away_val = func(away_df)

            home_stat, home_rank = league_ranks[label].get(home_full, (home_val, None))
            away_stat, away_rank = league_ranks[label].get(away_full, (away_val, None))

            home_rank_str = f"<span style='color:{rank_color(home_rank)}; font-weight:bold;'>Rank: {home_rank}</span>"
            away_rank_str = f"<span style='color:{rank_color(away_rank)}; font-weight:bold;'>Rank: {away_rank}</span>"

            with stat_col1:
                stat_col1.markdown(f"**{label}**<br>{home_val:.2f} ({home_rank_str})", unsafe_allow_html=True)
            with stat_col2:
                stat_col2.markdown(f"<br>{away_val:.2f} ({away_rank_str})", unsafe_allow_html=True)
                # === Scouting Report Section ===
        with cols[idx]:
            generate_scouting_report(home_full, away_full, home_df, away_df)


    st.markdown("---")


# === Top Players by Stat Section ===

# Add headshot URLs
player_df["playerImageURL"] = player_df["personId"].apply(
    lambda pid: f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"
)

# --- Stat Categories ---
avg_stats = [
    "points", "assists", "reboundsTotal", "steals", "blocks",
    "turnovers", "plusMinusPoints", "numMinutes"
]

# Totals for shooting calculations
totals_to_sum = [
    "fieldGoalsMade", "fieldGoalsAttempted",
    "threePointersMade", "threePointersAttempted",
    "freeThrowsMade", "freeThrowsAttempted"
]

# Additional totals to display
extra_totals = ["threePointersMade", "freeThrowsMade"]

# === Compute Averages ===
avg_players = (
    player_df.groupby(["firstName", "lastName", "personId", "playerImageURL"])[avg_stats]
    .mean()
    .reset_index()
)

# === Compute Totals ===
totals = (
    player_df.groupby(["firstName", "lastName", "personId"])[totals_to_sum]
    .sum()
    .reset_index()
)

# === Merge avg + totals ===
avg_players = pd.merge(avg_players, totals, on=["firstName", "lastName", "personId"], how="left")

# === Compute shooting percentages based on totals ===
avg_players["fieldGoalsPercentage"] = avg_players["fieldGoalsMade"] / avg_players["fieldGoalsAttempted"]
avg_players["threePointersPercentage"] = avg_players["threePointersMade"] / avg_players["threePointersAttempted"]
avg_players["freeThrowsPercentage"] = avg_players["freeThrowsMade"] / avg_players["freeThrowsAttempted"]


# ✅ Apply filter: Only keep players who attempted > 10 three-pointers
avg_players = avg_players[avg_players["threePointersAttempted"] > 10]

# === Display Labels ===
top_stat_fields = {
    "Points Per Game (PPG)": "points",
    "Rebounds Per Game (RPG)": "reboundsTotal",
    "Assists Per Game (APG)": "assists",
    "Steals Per Game": "steals",
    "Blocks Per Game": "blocks",
    "Turnovers Per Game": "turnovers",
    "Plus Minus": "plusMinusPoints",
    "Field Goal %": "fieldGoalsPercentage",
    "Three Point %": "threePointersPercentage",
    # "Free Throw %": "freeThrowsPercentage",
    "Total 3-Pointers Made": "threePointersMade",
    "Total Free Throws Made": "freeThrowsMade"
}

# === MVP Leaderboard ===
st.markdown("###  MVP Leaderboard (Fantasy Score)")

# --- Compute MVP score for each player ---
avg_players["mvp_score"] = (
    avg_players["points"]
    + avg_players["assists"] * 1.5
    + avg_players["reboundsTotal"] * 1.2
    + avg_players["steals"] * 3
    + avg_players["blocks"] * 3
    + avg_players["plusMinusPoints"] * 0.5
    - avg_players["turnovers"] * 1.5
)

# --- Filter: at least 15 minutes per game ---
filtered_mvp = avg_players[avg_players["numMinutes"] >= 15].copy()
top_mvp_players = filtered_mvp.sort_values("mvp_score", ascending=False).head(10)

# --- Add proper MVP rank ---
top_mvp_players = top_mvp_players.reset_index(drop=True)
top_mvp_players["Rank"] = top_mvp_players.index + 1

# --- Row color for top 3 ---
row_colors = {
    1: "#FFD700",  # Gold
    2: "#C0C0C0",  # Silver
    3: "#CD7F32",  # Bronze
}

# --- Build MVP leaderboard table ---
mvp_html = """
<style>
    table {
        width: 100%;
        border-collapse: collapse;
        font-family: Inter, sans-serif;
        margin-bottom: 40px;
    }
    th {
        background-color: #f0f2f6;
        padding: 10px;
        font-size: 16px;
        text-align: center;
    }
    td {
        padding: 12px;
        font-size: 15px;
        text-align: center;
    }
    td.name {
        font-weight: 600;
        text-align: left;
    }
    tr.gold { background-color: #FFF8DC; }   /* Subtle gold */
    tr.silver { background-color: #F8F8F8; } /* Light silver */
    tr.bronze { background-color: #FAF0E6; } /* Light bronze */
    img {
        width: 60px;
        border-radius: 8px;
        vertical-align: middle;
    }
</style>
<table>
    <thead>
        <tr>
            <th>Rank</th>
            <th></th>
            <th>Player</th>
            <th>PTS</th>
            <th>AST</th>
            <th>REB</th>
            <th>STL</th>
            <th>BLK</th>
            <th>TO</th>
            <th>+/-</th>
            <th>Score</th>
        </tr>
    </thead>
    <tbody>
"""

for _, row in top_mvp_players.iterrows():
    rank = row['Rank']
    row_class = ""
    if rank == 1:
        row_class = "gold"
    elif rank == 2:
        row_class = "silver"
    elif rank == 3:
        row_class = "bronze"

    mvp_html += f"""
        <tr class='{row_class}'>
            <td>{rank}</td>
            <td><img src="{row['playerImageURL']}"></td>
            <td class='name'>{row['firstName']} {row['lastName']}</td>
            <td>{round(row['points'], 1)}</td>
            <td>{round(row['assists'], 1)}</td>
            <td>{round(row['reboundsTotal'], 1)}</td>
            <td>{round(row['steals'], 1)}</td>
            <td>{round(row['blocks'], 1)}</td>
            <td>{round(row['turnovers'], 1)}</td>
            <td>{round(row['plusMinusPoints'], 1)}</td>
            <td><strong>{round(row['mvp_score'], 1)}</strong></td>
        </tr>
    """

mvp_html += "</tbody></table>"
st.components.v1.html(mvp_html, height=600, scrolling=True)


# === Display Leaderboard Heading ===
st.markdown("## Leaderboard")

# === Display Each Stat Category ===
for label, stat_col in top_stat_fields.items():
    st.markdown(f"#### {label}")

    # Filter: at least 15 minutes per game
    filtered_players = avg_players[avg_players["numMinutes"] >= 15]

    top_players = filtered_players.sort_values(stat_col, ascending=False).head(5)
    cols = st.columns(5)

    for idx, row in enumerate(top_players.itertuples(index=False)):
        stat_val = getattr(row, stat_col)
        stat_display = round(stat_val * 100, 2) if "Percentage" in stat_col else round(stat_val, 2)

        with cols[idx]:
            st.image(row.playerImageURL, width=100)
            st.markdown(f"**{row.firstName} {row.lastName}**")
            st.markdown(f"**{stat_display}{'%' if 'Percentage' in stat_col else ''}**")


# === Chat Section ===
st.markdown("---")
st.markdown("### 💬 Past Game Insights")
st.markdown("Ask about a game like `'Lakers vs Nuggets'`, `'Celtics vs Heat'`, etc.")

st.markdown("""
<div style='background-color:#f0f8ff; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
<h4 style='margin-bottom:10px;'>🔍 Type a matchup below:</h4>
<i>Use the format: <b>Team1 vs Team2</b></i>
</div>
""", unsafe_allow_html=True)

user_input = st.text_input(" ", "")  # Hidden label for cleaner UI

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

        import streamlit as st
        import pandas as pd
        import plotly.express as px
        
        # === TEAM LOGO MAP (must match filenames exactly) ===
        team_logo_map = {
            "Hawks": "atl.png", "Nets": "bkn.png", "Celtics": "bos.png", "Hornets": "cha.png", "Bulls": "chi.png",
            "Cavaliers": "cle.png", "Mavericks": "dal.png", "Nuggets": "den.png", "Pistons": "det.png", "Warriors": "gsw.png",
            "Rockets": "hou.png", "Pacers": "ind.png", "Clippers": "lac.png", "Lakers": "lal.png", "Grizzlies": "mem.png",
            "Heat": "mia.png", "Bucks": "mil.png", "Timberwolves": "min.png", "Pelicans": "nop.png", "Knicks": "nyk.png",
            "Thunder": "okc.png", "Magic": "orl.png", "76ers": "phl.png", "Suns": "phx.png", "Trail Blazers": "por.png",
            "Kings": "sac.png", "Spurs": "sas.png", "Raptors": "tor.png", "Jazz": "uth.png", "Wizards": "was.png"
        }
        
        # === Filter 2 teams for selected game ===
        team_stats = team_df[team_df["gameId"] == selected_gameId].copy()
        
        # === Compute Possessions & Ratings ===
        def estimate_possessions(row):
            return row["fieldGoalsAttempted"] + 0.44 * row["freeThrowsAttempted"] - row["reboundsOffensive"] + row["turnovers"]
        
        team_stats["possessions"] = team_stats.apply(estimate_possessions, axis=1)
        
        # Assign ratings
        team1, team2 = team_stats.iloc[0], team_stats.iloc[1]
        team_stats.loc[team_stats.index[0], "OffensiveRating"] = 100 * team1["teamScore"] / team1["possessions"]
        team_stats.loc[team_stats.index[0], "DefensiveRating"] = 100 * team2["teamScore"] / team1["possessions"]
        team_stats.loc[team_stats.index[1], "OffensiveRating"] = 100 * team2["teamScore"] / team2["possessions"]
        team_stats.loc[team_stats.index[1], "DefensiveRating"] = 100 * team1["teamScore"] / team2["possessions"]
        
        import streamlit as st

        # === Team Logo Map ===
        team_logo_map = {
            "Hawks": "atl.png",
            "Nets": "bkn.png",
            "Celtics": "bos.png",
            "Hornets": "cha.png",
            "Bulls": "chi.png",
            "Cavaliers": "cle.png",
            "Mavericks": "dal.png",
            "Nuggets": "den.png",
            "Pistons": "det.png",
            "Warriors": "gsw.png",
            "Rockets": "hou.png",
            "Pacers": "ind.png",
            "Clippers": "lac.png",
            "Lakers": "lal.png",
            "Grizzlies": "mem.png",
            "Heat": "mia.png",
            "Bucks": "mil.png",
            "Timberwolves": "min.png",
            "Pelicans": "nop.png",
            "Knicks": "nyk.png",
            "Thunder": "okc.png",
            "Magic": "orl.png",
            "76ers": "phl.png",
            "Suns": "phx.png",
            "Trail Blazers": "por.png",
            "Kings": "sac.png",
            "Spurs": "sas.png",
            "Raptors": "tor.png",
            "Jazz": "uth.png",
            "Wizards": "was.png"
        }
        
        # === Simulated team_stats dataframe (replace this with your real one) ===
        # team_stats = pd.DataFrame([...])
        
        # === Header ===
        st.markdown("###  Team Performance Cards")
        
        # === Fetch Team Info ===
        team1, team2 = team_stats.iloc[0], team_stats.iloc[1]
        t1_name, t2_name = team1["teamName"], team2["teamName"]
        t1_logo = team_logo_map.get(t1_name, "default.png")
        t2_logo = team_logo_map.get(t2_name, "default.png")
        
        # === Show Logos Side by Side (Left Aligned, Larger + Spaced) ===
        logo_col1, logo_col2 = st.columns([1, 1])
        
        with logo_col1:
            st.markdown(
                f"""
                <div style='text-align:left; padding-bottom:20px;'>
                    <img src='https://raw.githubusercontent.com/pcherian89/nba-game-analytics/main/{t1_logo}' 
                         style='height:180px; object-fit:contain; margin-bottom:20px;'/>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with logo_col2:
            st.markdown(
                f"""
                <div style='text-align:left; padding-bottom:20px;'>
                    <img src='https://raw.githubusercontent.com/pcherian89/nba-game-analytics/main/{t2_logo}' 
                         style='height:180px; object-fit:contain; margin-bottom:20px;'/>
                </div>
                """,
                unsafe_allow_html=True
            )



        
        # === Stat Fields to Compare ===
        team_compare_fields = {
            "Offensive Rating": "OffensiveRating",
            "Defensive Rating": "DefensiveRating",
            "Score": "teamScore",
            "Assists": "assists",
            "Rebounds Total": "reboundsTotal",
            "Steals": "steals",
            "Blocks": "blocks",
            "Fieldgoals %": "fieldGoalsPercentage",
            "Threepointers %": "threePointersPercentage",
            "Freethrows %": "freeThrowsPercentage",
            "Turnovers": "turnovers",
            "Plusminuspoints": "plusMinusPoints"
        }
        
        # === Stat Comparison Section ===
        
        for label, field in team_compare_fields.items():
            t1_val = team1.get(field, 0)
            t2_val = team2.get(field, 0)
            max_val = max(t1_val, t2_val, 1)
        
            st.markdown(f"**{label}**")
        
            bar_col1, bar_col2 = st.columns(2)
        
            with bar_col1:
                st.markdown(f"{t1_name}: {round(t1_val, 2)}")
                st.markdown(f"""
                    <div style='background-color:#eee; height:10px; border-radius:5px;'>
                        <div style='width:{(t1_val / max_val) * 100}%; background-color:green; height:10px; border-radius:5px;'></div>
                    </div>
                """, unsafe_allow_html=True)
        
            with bar_col2:
                st.markdown(f"{t2_name}: {round(t2_val, 2)}")
                st.markdown(f"""
                    <div style='background-color:#eee; height:10px; border-radius:5px;'>
                        <div style='width:{(t2_val / max_val) * 100}%; background-color:red; height:10px; border-radius:5px;'></div>
                    </div>
                """, unsafe_allow_html=True)



        
        def display_player_cards(players_df, team_label):
            st.subheader(f" {team_label} Player Cards")
        
            rows = [players_df.iloc[i:i+6] for i in range(0, len(players_df), 6)]
            for row_df in rows:
                cols = st.columns(len(row_df))
                for idx, player in enumerate(row_df.itertuples(index=False)):
                    full_name = f"{player.firstName} {player.lastName}"
                    player_id = int(player.personId)
                    image_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
        
                    with cols[idx]:
                        st.image(image_url, width=100, caption=full_name)
                        st.markdown(f"**Points:** {player.points}")
                        st.markdown(f"**Assists:** {player.assists}")
                        st.markdown(f"**Rebounds:** {player.reboundsTotal}")
                        st.markdown(f"**Turnovers:** {player.turnovers}")
                        st.markdown(f"**+/-:** {player.plusMinusPoints}")


        display_player_cards(home_players, home_team)
        display_player_cards(away_players, away_team)


        # === Combine Home & Away Players ===
        combined_players = pd.concat([home_players, away_players], ignore_index=True)
        
        # === Add Full Name Column ===
        combined_players["fullName"] = combined_players["firstName"] + " " + combined_players["lastName"]
        
        # === Add Rebound Total (if not already in the data) ===
        if "reboundsTotal" not in combined_players.columns:
            combined_players["reboundsTotal"] = (
                combined_players.get("reboundsOffensive", 0) + combined_players.get("reboundsDefensive", 0)
            )
        
        
        import streamlit as st
        import plotly.express as px
        
        # === User Selects Stat to View ===
        st.markdown("###  **View top players by:**")
        stat_option = st.selectbox("", ["points", "assists", "reboundsTotal", "turnovers", "plusMinusPoints"])
        
        # === Generate headshot URLs ===
        combined_players["playerImageURL"] = combined_players["personId"].apply(
            lambda pid: f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"
        )
        
        # === Filter Top 6 Players for the Selected Stat ===
        top6 = combined_players.sort_values(by=stat_option, ascending=False).head(6)
        
        # === Display Top 6 Headshots with Stat Value ===
        st.markdown("####  Top Player Cards")
        cols = st.columns(6)
        
        for i, row in enumerate(top6.itertuples()):
            with cols[i]:
                st.image(row.playerImageURL, width=90)
                st.markdown(f"**{row.fullName}**", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:16px;'>{stat_option.title()}: {getattr(row, stat_option)}</span>", unsafe_allow_html=True)
        
        # === Create Interactive Plotly Bar Chart ===
        fig = px.bar(
            top6,
            x=stat_option,
            y="fullName",
            color="playerteamName",
            orientation="h",
            title=f"Top 6 Players by {stat_option.replace('Points', ' Points').title()}",
            labels={stat_option: stat_option.title(), "fullName": "Player", "playerteamName": "Team"},
            color_discrete_sequence=["Green", "Red"]  # Customize color mapping
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
        
        # Filter: Played at least 15 minutes
        combined_players = combined_players[combined_players["numMinutes"].fillna(0) >= 15].dropna(
            subset=["OffensiveRating", "DefensiveRating"]
        )

        
        import streamlit as st
        import plotly.express as px
        
        # === Section Header ===
        st.subheader(" Player Impact Ratings ")
        
        # === Rating Selection ===
        rating_type = st.radio("Select rating type to display:", ["OffensiveRating", "DefensiveRating"])
        
        # === Generate Image URLs if missing ===
        if "playerImageURL" not in combined_players.columns:
            combined_players["playerImageURL"] = combined_players["personId"].apply(
                lambda pid: f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png"
            )
        
        # === Identify top 5 per team ===
        top_per_team = (
            combined_players.groupby("playerteamName", group_keys=False)
            .apply(lambda df: df.sort_values(by=rating_type, ascending=False).head(5))
        )
        
        # === Optional: Only keep 2 teams (if you expect just 2 teams in comparison) ===
        if len(top_per_team["playerteamName"].unique()) > 2:
            top2_teams = top_per_team["playerteamName"].value_counts().nlargest(2).index
            top_per_team = top_per_team[top_per_team["playerteamName"].isin(top2_teams)]
        
        # === Headshots for top 10 (5 per team) ===
        st.markdown("####  Top 5 Players per Team")
        cols = st.columns(10)
        for i, row in enumerate(top_per_team.itertuples()):
            with cols[i]:
                st.image(row.playerImageURL, width=80)
                st.markdown(f"**{row.fullName}**", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:15px;'>{rating_type}: {getattr(row, rating_type):.2f}</span>", unsafe_allow_html=True)
        
        # === Hover Settings ===
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
        
        # === Bar Chart for Top Players ===
        fig = px.bar(
            top_per_team,
            x="fullName",
            y=rating_type,
            color="playerteamName",
            title=f"Top 5 Players per Team by {rating_type} (Per Minute)",
            labels={"fullName": "Player", "playerteamName": "Team", rating_type: "Rating"},
            color_discrete_sequence=["Green", "Red"],
            hover_data=hover_cols
        )
        fig.update_layout(xaxis_tickangle=-45)
        
        st.plotly_chart(fig, use_container_width=True)

        # === MVP Comparison Cards with Dynamic Player Images ===
        st.subheader(" MVP Comparison – Player Cards")
        
        # Get top 3 players by Offensive Rating
        top3 = combined_players.sort_values(by="OffensiveRating", ascending=False).head(3).reset_index(drop=True)
        
        # Create 3 columns for layout
        cols = st.columns(3)
        
        for idx, row in top3.iterrows():
            with cols[idx % 3]:
                # Dynamically build image URL from personId
                image_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(row['personId'])}.png"
                st.image(image_url, width=120)
        
                # Display player name and stats
                st.markdown(f"**{row['fullName']}**")
                st.markdown(f"**Points:** {row['points']}")
                st.markdown(f"**Assists:** {row['assists']}")
                st.markdown(f"**Rebounds:** {row['reboundsTotal']}")
                st.markdown(f"**Turnovers:** {row['turnovers']}")
                st.markdown(f"**Offensive Rating:** {row['OffensiveRating']:.2f}")
                st.markdown(f"**Defensive Rating:** {row['DefensiveRating']:.2f}")
                st.markdown(f"**+/- Impact:** {row['plusMinusPoints']}")

        
        import io
        from fpdf import FPDF
        import plotly.graph_objects as go
        
        import streamlit as st
        import pandas as pd

        # === Custom CSS for spacing ===
        st.markdown("""
            <style>
            .element-container { padding-bottom: 0rem !important; }
            .stMarkdown p { margin-bottom: 0.3rem; }
            </style>
        """, unsafe_allow_html=True)
        # === Load player stats ===
        # Assuming combined_players DataFrame is already available
        
        # === Define Player Images CDN (use NBA CDN or local images ideally) ===
        def get_player_image_url(player_id):
            return f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"
        
        # === Dropdown to select player ===
        player_names = combined_players["fullName"].unique()
        st.markdown("### **Select a player to view scouting card:**")
        selected_player = st.selectbox("", player_names)

        
        # === Filter selected player's stats ===
        player_row = combined_players[combined_players["fullName"] == selected_player].iloc[0]
        
        # === Display headshot + name ===
        st.markdown(f"###  Scouting Card: {selected_player}")
        player_id = player_row["personId"]
        image_url = get_player_image_url(player_id)
        st.image(image_url, width=150)
        
        # === 3 Side-by-Side Columns ===
        col1, col2, col3 = st.columns(3)
        
        # === Column 1: Offensive Stats ===
        with col1:
            st.markdown("###  Offensive")
            st.markdown(f"**Points:** {player_row['points']}")
            st.markdown(f"**Assists:** {player_row['assists']}")
            st.markdown(f"**Turnovers:** {player_row['turnovers']}")
            st.markdown(f"**FG%:** {player_row['fieldGoalsPercentage']*100:.1f}%")
            st.markdown(f"**3P%:** {player_row['threePointersPercentage']*100:.1f}%")
            st.markdown(f"**FT%:** {player_row['freeThrowsPercentage']*100:.1f}%")
        
        # === Column 2: Defensive Stats ===
        with col2:
            st.markdown("###  Defensive")
            st.markdown(f"**Rebounds:** {player_row['reboundsTotal']}")
            st.markdown(f"**Steals:** {player_row['steals']}")
            st.markdown(f"**Blocks:** {player_row['blocks']}")
        
        # === Column 3: Summary Stats ===
        with col3:
            st.markdown("###  Summary")
            st.markdown(f"**Minutes Played:** {round(player_row['numMinutes'], 1)}")
            st.markdown(f"**Plus/Minus:** {player_row['plusMinusPoints']}")
            st.markdown(f"**Off Rating:** {round(player_row['OffensiveRating'], 2)}")
            st.markdown(f"**Def Rating:** {round(player_row['DefensiveRating'], 2)}")

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
        player_name = player_row["fullName"]
        minutes = player_row["numMinutes"]
        stats_text = f"""
        Points: {player_row['points']}
        Assists: {player_row['assists']}
        Turnovers: {player_row['turnovers']}
        FG%: {player_row['fieldGoalsPercentage']:.1%}
        3P%: {player_row['threePointersPercentage']:.1%}
        FT%: {player_row['freeThrowsPercentage']:.1%}
        Rebounds: {player_row['reboundsTotal']}
        Steals: {player_row['steals']}
        Blocks: {player_row['blocks']}
        Plus/Minus: {player_row['plusMinusPoints']}
        Offensive Rating: {player_row['OffensiveRating']:.2f}
        Defensive Rating: {player_row['DefensiveRating']:.2f}
        """

        
        # === Run the agent ===
        summary_output = summary_chain.run({
            "player_name": player_name,
            "minutes": minutes,
            "stats": stats_text
        })
        
        # === Display the scouting summary ===
        st.markdown("###  Scouting Summary Report")
        st.markdown(summary_output)

        import streamlit as st
        import pandas as pd
        
        # Ensure combined_players is already defined
        # e.g., combined_players = pd.read_csv("your_cleaned_player_data.csv")
        
        st.subheader(" Stats Comparison")
        
        # Only players who recorded minutes
        valid_players = combined_players[combined_players["numMinutes"].notna() & (combined_players["numMinutes"] > 0)]
        player_names = valid_players["fullName"].unique().tolist()
        
        # Select players
        col1, col2 = st.columns(2)
        with col1:
            player1 = st.selectbox("Select Player 1", player_names, key="p1")
        with col2:
            player2 = st.selectbox("Select Player 2", player_names, key="p2")
        
        # Get stats
        p1_stats = valid_players[valid_players["fullName"] == player1].iloc[0]
        p2_stats = valid_players[valid_players["fullName"] == player2].iloc[0]
        
        # Headshot function
        def get_headshot_url(personId):
            return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(personId)}.png"
        
        # Stats to compare
        compare_fields = {
            "Points": "points",
            "Assists": "assists",
            "Total Rebounds": "reboundsTotal",
            "Steals": "steals",
            "Blocks": "blocks",
            "Turnovers": "turnovers",
            "FG%": "fieldGoalsPercentage",
            "3P%": "threePointersPercentage",
            "FT%": "freeThrowsPercentage",
            "Offensive Rating": "OffensiveRating",
            "Defensive Rating": "DefensiveRating"
        }
        
        # Show profile and stat bars
        left, right = st.columns(2)
        
        with left:
            st.image(get_headshot_url(p1_stats["personId"]), width=180)
            st.markdown(f"### {player1}")
        
        with right:
            st.image(get_headshot_url(p2_stats["personId"]), width=180)
            st.markdown(f"### {player2}")
        
        for label, field in compare_fields.items():
            p1_val = p1_stats.get(field, 0)
            p2_val = p2_stats.get(field, 0)
            max_val = max(p1_val, p2_val, 1)
        
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(f"**{label}**")
        
            col_left, col_right = st.columns(2)
        
            with col_left:
                st.markdown(f"{player1}: {round(p1_val, 1)}")
                st.markdown(
                    f"""
                    <div style="background-color:#eee; height:10px; border-radius:5px;">
                        <div style="width:{(p1_val/max_val)*100}%; background-color:green; height:10px; border-radius:5px;"></div>
                    </div>
                    """, unsafe_allow_html=True)
        
            with col_right:
                st.markdown(f"{player2}: {round(p2_val, 1)}")
                st.markdown(
                    f"""
                    <div style="background-color:#eee; height:10px; border-radius:5px;">
                        <div style="width:{(p2_val/max_val)*100}%; background-color:red; height:10px; border-radius:5px;"></div>
                    </div>
                    """, unsafe_allow_html=True)



  
        # === AI-Generated Game Summary with Session Persistence ===
        st.subheader(" Game Summary")
        
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
            with st.spinner(" Generating game summary..."):
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
        # st.markdown("####  Game Summary")
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
            with st.spinner(" Analyzing game data..."):
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
        
elif user_input.strip() != "":
    st.info("⚠️ Please use the format: `Team1 vs Team2` (e.g., `Heat vs Knicks`)")
