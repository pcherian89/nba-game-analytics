import streamlit as st
import pandas as pd

# Dummy data to simulate output from n8n
team1_data = [
    {"player": "Jalen Brunson", "points": 22, "rebounds": 6},
    {"player": "Josh Hart", "points": 12, "rebounds": 8}
]

team2_data = [
    {"player": "Tyrese Haliburton", "points": 25, "rebounds": 4},
    {"player": "Myles Turner", "points": 10, "rebounds": 9}
]

team_stats = [
    {"team": "Knicks", "eFG%": 54.2, "TO%": 11.0},
    {"team": "Pacers", "eFG%": 50.1, "TO%": 8.5}
]

# App UI
st.title("🏀 NBA Game Viewer")

st.subheader("👥 Team 1 Player Stats")
st.dataframe(pd.DataFrame(team1_data))

st.subheader("👥 Team 2 Player Stats")
st.dataframe(pd.DataFrame(team2_data))

st.subheader("📊 Team Stats")
st.dataframe(pd.DataFrame(team_stats))
