import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def load_data():
    data = pd.read_csv('NBA2024.csv')
    data.fillna(0, inplace=True)
    return data

def get_similarity_matrix(data, columns_for_similarity):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[columns_for_similarity])
    similarity_matrix = cosine_similarity(scaled_data)
    return pd.DataFrame(similarity_matrix, index=data['ID'], columns=data['ID'])

def find_similar_players(data, similarity_df, player_id, top_n=5):
    if player_id not in similarity_df.index:
        return f"Player ID {player_id} not found in the data."

    player_name = data.loc[data['ID'] == player_id, 'Player_Name'].values[0]
    similar_players = similarity_df[player_id].sort_values(ascending=False).iloc[1:top_n+1]

    similar_players_df = pd.DataFrame({
        'Player_Name': [data.loc[data['ID'] == p, 'Player_Name'].values[0] for p in similar_players.index],
        'Similarity_Score': similar_players.values
    })

    similar_players_df = pd.concat([
        pd.DataFrame({'Player_Name': [player_name], 'Similarity_Score': [1.0]}),
        similar_players_df
    ], ignore_index=True)

    return similar_players_df

def nba_player_similarity_app():
    st.title("NBA Player Similarity App")

    data = load_data()

    columns_for_similarity = st.multiselect(
        "Select columns for similarity comparison",
        data.columns,
        default=[
            'Average_Field_Goals_Made', 'Average_Field_Goals_Attempted',
            'Average_Field_Goal_Percentage', 'Average_Three_Pointers_Made',
            'Average_Three_Pointers_Attempted', 'Average_Three_Point_Percentage',
            'Average_Three_Percentage', 'Average_Two_Pointers_Made',
            'Average_Two_Pointers_Attempted', 'Average_Two_Point_Percentage',
            'Average_Free_Throws_Made', 'Average_Free_Throws_Attempted',
            'Average_Free_Throw_Percentage', 'Average_Points',
            'Average_Offensive_Rebounds', 'Average_Defensive_Rebounds',
            'Average_Total_Rebounds', 'Average_Assists', 'Average_Turnovers',
            'Average_Steals', 'Average_Personal_Fouls',
            'Total_Minutes_Played', 'Total_Field_Goals_Made',
            'Total_Field_Goals_Attempted', 'Total_Field_Goal_Percentage',
            'Total_Three_Pointers_Made', 'Total_Three_Pointers_Attempted',
            'Total_Three_Point_Percentage', 'Total_Three_Percentage',
            'Total_Two_Pointers_Made', 'Total_Two_Pointers_Attempted',
            'Total_Two_Point_Percentage', 'Total_Free_Throws_Made',
            'Total_Free_Throws_Attempted', 'Total_Free_Throw_Percentage',
            'Total_Points', 'Total_Offensive_Rebounds', 'Total_Defensive_Rebounds',
            'Total_Total_Rebounds', 'Total_Assists', 'Total_Turnovers',
            'Total_Steals', 'Total_Blocks', 'Total_Personal_Fouls'
        ]
    )

    player_id = st.text_input("Enter Player ID:", "")

    if player_id:
        similarity_df = get_similarity_matrix(data, columns_for_similarity)
        similar_players = find_similar_players(data, similarity_df, player_id)
        st.write(similar_players)

if __name__ == "__main__":
    nba_player_similarity_app()
