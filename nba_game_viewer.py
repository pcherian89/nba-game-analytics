import streamlit as st
import pandas as pd
import requests

st.title("🏀 NBA Game Viewer")

# Step 1: Input box for user to enter matchup
matchup = st.text_input("🤖 Which NBA matchup do you want to analyze today?", placeholder="e.g. Knicks vs Pacers")

if st.button("Analyze Game") and matchup:
    try:
        # Step 2: Send POST request with user input
        url = "https://pcherian89.app.n8n.cloud/webhook-test/nba-chatbot-start"
        response = requests.post(url, json={"text": matchup})
        data = response.json()

        # Step 3: Display tables
        team1_data = data["team1Players"]
        team2_data = data["team2Players"]
        team_stats = data["teamStats"]

        st.subheader("👥 Team 1 Player Stats")
        st.dataframe(pd.DataFrame(team1_data))

        st.subheader("👥 Team 2 Player Stats")
        st.dataframe(pd.DataFrame(team2_data))

        st.subheader("📊 Team Stats")
        st.dataframe(pd.DataFrame(team_stats))

    except Exception as e:
        st.error("Failed to fetch game data.")
        st.code(str(e))

