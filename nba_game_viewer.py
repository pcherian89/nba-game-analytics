import requests
import streamlit as st
import pandas as pd

st.title("🏀 NBA Game Viewer")

matchup = st.text_input("🤖 Which NBA matchup do you want to analyze today?", "Knicks vs Pacers")

if st.button("Analyze Game"):
    try:
        # Trigger n8n
        response = requests.post("https://pcherian89.app.n8n.cloud/webhook-test/nba-chatbot-start", json={"matchup": matchup})
        data = response.json()

        # 🧠 Separate items
        team1Players = []
        team2Players = []
        teamStats = []

        for item in data:
            if "personId" in item:
                if item["team"] in matchup:
                    if matchup.split(" vs ")[0] == item["team"]:
                        team1Players.append(item)
                    else:
                        team2Players.append(item)
            elif "teamName" in item:
                teamStats.append(item)

        st.subheader("👥 Team 1 Player Stats")
        st.dataframe(pd.DataFrame(team1Players))

        st.subheader("👥 Team 2 Player Stats")
        st.dataframe(pd.DataFrame(team2Players))

        st.subheader("📊 Team Stats")
        st.dataframe(pd.DataFrame(teamStats))

    except Exception as e:
        st.error("Failed to fetch game data.")
        st.code(str(e))

