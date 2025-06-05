import streamlit as st
import pandas as pd
import requests

st.title("🏀 NBA Game Viewer")

# Send POST request to your webhook
url = "https://pcherian89.app.n8n.cloud/webhook-test/nba-chatbot-start"
response = requests.post(url)
data = response.json()

# Extract the sections
team1_data = data["team1Players"]
team2_data = data["team2Players"]
team_stats = data["teamStats"]

# Show output
st.subheader("👥 Team 1 Player Stats")
st.dataframe(pd.DataFrame(team1_data))

st.subheader("👥 Team 2 Player Stats")
st.dataframe(pd.DataFrame(team2_data))

st.subheader("📊 Team Stats")
st.dataframe(pd.DataFrame(team_stats))
