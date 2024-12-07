import streamlit as st
import pickle
import pandas as pd

teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies', 'Afghanistan',
         'Pakistan', 'Sri Lanka']

city = ['Victoria', 'Napier', 'Mount Maunganui', 'Auckland',
       'Southampton', 'Taunton', 'Cardiff', 'Chester-le-Street', 'Kanpur',
       'Nagpur', 'Bangalore', 'Lauderhill', 'Abu Dhabi', 'Hobart',
       'Wellington', 'Hamilton', 'Bloemfontein', 'Potchefstroom',
       'Barbados', 'Trinidad', 'Colombo', 'St Kitts', 'Jamaica', 'Nelson',
       'Ranchi', 'Birmingham', 'Manchester', 'Bristol', 'Delhi', 'Rajkot',
       'Thiruvananthapuram', 'Lahore', 'Johannesburg', 'Centurion',
       'Cape Town', 'Cuttack', 'Indore', 'Mumbai', 'Dhaka', 'Karachi',
       'Brisbane', 'Dehradun', 'Sylhet', 'Kolkata', 'Lucknow', 'Chennai',
       'Gros Islet', 'Basseterre', 'Visakhapatnam', 'Bengaluru',
       'Adelaide', 'Melbourne', 'Sydney', 'Canberra', 'Perth',
       'East London', 'Durban', 'Port Elizabeth', 'Chandigarh',
       'Hyderabad', 'Christchurch', 'Providence', 'Kandy', 'Chattogram',
       'Pune', 'Paarl', 'London', 'Nairobi', 'Nottingham', 'King City',
       'Guyana', 'St Lucia', 'Antigua', 'Mirpur', 'Hambantota',
       'Ahmedabad', 'St Vincent', 'Chittagong', 'Dominica', 'Dharmasala',
       'Dharamsala']

pipe = pickle.load(open('pipe.pkl', 'rb'))
st.title('T20 win prediction')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))

selected_city = st.selectbox('Select Host City', sorted(city))

target = st.number_input('Target')

col4, col5, col6 = st.columns(3)
with col4:
    score = st.number_input('Score')
with col5:
    overs = st.number_input('Overs Completed')
with col6:
    wickets = st.number_input('Wickets Out')

if st.button('Predict probability'):
    runs_left = target - score
    balls_left = 120-(overs*6)
    wickets = 10-wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets_left': [wickets],
                             'runs_y': [target], 'crr': [crr], 'rrr': [rrr]})

    st.table(input_df)
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")
