import requests
import streamlit as st
import pandas as pd

st.title("🏀 NBA Game Viewer")
matchup = st.text_input("🤖 Which NBA matchup do you want to analyze today?", value="Knicks vs Pacers")
if st.button("Analyze Game"):

    with st.spinner("Fetching data..."):
        try:
            # 1️⃣ Trigger the n8n webhook and get full merged data
            url = "https://pcherian89.app.n8n.cloud/webhook-test/nba-chatbot-start"
            payload = {"matchup": matchup}
            res = requests.post(url, json=payload)
            data = res.json()

            # 2️⃣ Extract and split inside Python
            merged = data

            # Split team players and stats
            team1_players = pd.DataFrame(merged["team1Players"])
            team2_players = pd.DataFrame(merged["team2Players"])
            team_stats = pd.DataFrame(merged["teamStats"])

            # 3️⃣ Display
            st.subheader("👥 Team 1 Player Stats")
            st.dataframe(team1_players)

            st.subheader("👥 Team 2 Player Stats")
            st.dataframe(team2_players)

            st.subheader("📊 Team Stats")
            st.dataframe(team_stats)

        except Exception as e:
            st.error("Failed to fetch game data.")
            st.code(str(e))


