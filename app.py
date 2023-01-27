import streamlit as st
import pandas as pd
import numpy  as np
import joblib

# Setting up Sidebar
social_acc = ['Data Field Description', 'EDA', 'About App']
social_acc_nav = st.sidebar.radio('**INFORMATION SECTION**', social_acc)

if social_acc_nav == 'Data Field Description':
    st.sidebar.markdown("<h2 style='text-align: center;'> Data Field Description </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown("**Date:** The date you want to predict sales  for")
    st.sidebar.markdown("**Family:** identifies the type of product sold")
    st.sidebar.markdown("**Onpromotion:** gives the total number of items in a product family that are being promoted at a store at a given date")
    st.sidebar.markdown("**Store Number:** identifies the store at which the products are sold")
    st.sidebar.markdown("**Holiday Locale:** provide information about the locale where holiday is celebrated")

elif social_acc_nav == 'EDA':
    st.sidebar.markdown("<h2 style='text-align: center;'> Exploratory Data Analysis </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown('''The exploratory data analysis of this project can be find in a Jupyter notebook from the linl below''')
    st.sidebar.markdown("[Open Notebook](https://github.com/Kyei-frank/Regression-Project-Store-Sales--Time-Series-Forecasting/blob/main/project_workflow.ipynb)")

elif social_acc_nav == 'About App':
    st.sidebar.markdown("<h2 style='text-align: center;'> Sales Forecasting App </h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown("This App predicts the sales for product families sold at Favorita stores using regression model.")
    st.sidebar.markdown("")
    st.sidebar.markdown("[ Visit Github Repository for more information](https://github.com/Kyei-frank/Regression-Project-Store-Sales--Time-Series-Forecasting)")


# load best model
with open('model.pkl','rb') as file_1:
    model = joblib.load(file_1)

def greet(no_of_adults, no_of_children, type_of_meal_plan,
       required_car_parking_space, room_type_reserved, lead_time,
       arrival_month, arrival_date, market_segment_type,
       repeated_guest, no_of_previous_cancellations,
       no_of_previous_bookings_not_canceled, avg_price_per_room,
       no_of_special_requests):
  pred = model.predict([[no_of_adults, no_of_children, type_of_meal_plan,
       required_car_parking_space, room_type_reserved, lead_time,
       arrival_month, arrival_date, market_segment_type,
       repeated_guest, no_of_previous_cancellations,
       no_of_previous_bookings_not_canceled, avg_price_per_room,
       no_of_special_requests]])
  pred_proba = model.predict_proba([[no_of_adults, no_of_children, type_of_meal_plan,
       required_car_parking_space, room_type_reserved, lead_time,
       arrival_month, arrival_date, market_segment_type,
       repeated_guest, no_of_previous_cancellations,
       no_of_previous_bookings_not_canceled, avg_price_per_room,
       no_of_special_requests]])
  return list(pred)[0], float(list(pred_proba)[0][0]), "{:.2%}".format(float(list(pred_proba)[0][0]))
no_of_adults = st.slider('no_of_adults',0, 4, step=1)
no_of_children =  st.slider('no_of_children',0, 3, step=1)
type_of_meal_plan = st.selectbox('meal?',('breakfast', 'Not Selected', 'half board', 'full board'))
required_car_parking_space = st.checkbox('required_car_parking_space')
room_type_reserved = st.selectbox('room?',('standart', 'economy', 'deluxe'))
lead_time = st.slider('lead_time',0, 420, step=1)
arrival_month = st.slider('arrival_month',1, 12, step=1)
arrival_date = st.slider('arrival_date',1, 31, step=1)
market_segment_type = st.selectbox('market_segment_type?',('Online', 'Offline', 'Corporate', 'Aviation'))
repeated_guest = st.checkbox('repeated_guest')
no_of_previous_cancellations = st.selectbox('no_of_previous_cancellations?',(0, 1, 3, 'more than 5', 2))
no_of_previous_bookings_not_canceled = st.selectbox('no_of_previous_bookings_not_canceled?',(0, 1, 3, 'more than 5', 2))
avg_price_per_room = st.slider('avg_price_per_room',20, 300, step=1)
no_of_special_requests = st.selectbox('no_of_special_requests?',(0, 2, 3, 4, 5))

if st.button('Predict'):
    prediction = greet(no_of_adults, no_of_children, type_of_meal_plan,
       required_car_parking_space, room_type_reserved, lead_time,
       arrival_month, arrival_date, market_segment_type,
       repeated_guest, no_of_previous_cancellations,
       no_of_previous_bookings_not_canceled, avg_price_per_room,
       no_of_special_requests)
    st.write(f'You are most likely {prediction[0]} with your flight experience {prediction[1]}, {prediction[2]}')
