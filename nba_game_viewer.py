import streamlit as st
import requests
import json

st.title("🏀 NBA Game Viewer")
st.markdown("🤖 **Which NBA matchup do you want to analyze today?**")

# Game input
matchup = st.text_input(" ", "Knicks vs Pacers")
team1, team2 = matchup.split(" vs ")

if st.button("Analyze Game"):
    try:
        # Send POST request to n8n Webhook
        response = requests.post(
            "https://pcherian89.app.n8n.cloud/webhook-test/nba-chatbot-start",
            json={"team1": team1.strip(), "team2": team2.strip()},
            timeout=30,
        )

        # Raise error if bad response
        response.raise_for_status()

        # Parse the response
        data = response.json()

        team1Players = data.get("team1Players", [])
        team2Players = data.get("team2Players", [])
        teamStats = data.get("teamStats", [])

        # Display Results
        st.subheader("👥 Team 1 Player Stats")
        if team1Players:
            st.dataframe(team1Players)
        else:
            st.warning("No data for Team 1")

        st.subheader("👥 Team 2 Player Stats")
        if team2Players:
            st.dataframe(team2Players)
        else:
            st.warning("No data for Team 2")

        st.subheader("📊 Team Stats")
        if teamStats:
            st.dataframe(teamStats)
        else:
            st.warning("No team stats available")

    except requests.exceptions.RequestException as e:
        st.error("Failed to fetch game data.")
        st.code(str(e))
    except json.JSONDecodeError as e:
        st.error("Received malformed JSON.")
        st.code(str(e))

