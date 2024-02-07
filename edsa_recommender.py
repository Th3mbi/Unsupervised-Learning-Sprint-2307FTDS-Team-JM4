"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from recommenders.Trail import content_generate_top_N_recommendations

# Data Loading
title_list = load_movie_titles('resources/data/usableData.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System", "Information", "About Us", "Solution Overview"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[1:2000])
        movie_2 = st.selectbox('Second Option',title_list[2001:4000])
        movie_3 = st.selectbox('Third Option',title_list[4001:8100])
        fav_movies = [movie_1,movie_2,movie_3]
        fav_movies = movie_1
        usableData = pd.read_csv("resources/data/usableData.csv")
        index = usableData[usableData['title'] == fav_movies].index[0]
        movie_Id = usableData.loc[index,'movieId']
        
        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_generate_top_N_recommendations(movie_Id=movie_Id, N=10)
                        
                    st.title("We think you'll like:")
                    top_recommendations
                    #for i,j in enumerate(top_recommendations):
                       # st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                movies = pd.read_csv("resources/data/movies.csv")
                try:
                    with st.spinner('Crunching the numbers...'):
                        predictor = joblib.load(open(os.path.join("resources/models/SVD.pkl"), "rb"))
                        usableData['predicted_ratingscore'] = usableData['movieId'].apply(lambda x: predictor.predict(146790, x).est)
                        usableData = usableData.sort_values(by=['predicted_ratingscore'], ascending=False)


                        top_recommendations = usableData.head(10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

            


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    # Building out the "Information" page
    if page_selection == "Information":
        st.markdown("<h1 style='color: black; font-weight: bold;'>Movie Recommender</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='color: black; font-weight: bold;'>Movie Recommendation Predict</h2>", unsafe_allow_html=True)
        st.info("Introduction")
        # You can read a markdown file from supporting resources folder
        # st.markdown("Some information here")
        st.markdown("<p style='color: black; font-size: 1.0em; font-weight: bold;'>" +"A movie recommender is a software designed to help users discover films aligning with their preferences. By employing various algorithms and data analysis, it analyzes user behavior and historical preferences to generate personalized suggestions, aiming to enhance the user experience. These systems, utilizing content-based filtering, collaborative filtering, or hybrid methods, provide accurate and relevant movie recommendations based on users' past interactions. The introduction to a recommendation algorithm underscores the challenge users face in navigating vast content. It emphasizes the need for personalized recommendations and outlines the dual strategies—content-based and collaborative filtering."+"</p>", unsafe_allow_html=True)

        st.image('resources/imgs/Information.png',use_column_width=True)
        st.markdown("<p style='color: black; font-size: 1.0em; font-weight: bold;'>" +" The algorithm's goal is to forecast a user's movie rating accurately, leveraging insights from historical preferences, paving the way for a recommendation system that seamlessly integrates both methods for precise and personalized predictions. "+"</p>", unsafe_allow_html=True)
        
    # Building About us page
    if page_selection == "About Us":
        st.markdown("<h1 style='color: black; font-weight: bold;'>Entertainment AI</h1>", unsafe_allow_html=True)
        #st.markdown("<h2 style='color: black; font-weight: bold;'>Movie Recommender Prediction</h2>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: black;'>Who we are</h1>", unsafe_allow_html=True)

        st.markdown("<p style='color: black; font-size: 1.2em; font-weight: bold;'>" +
            "We are Entertainment AI, a leading global technology company specializing in cloud infrastructure, software, and hardware solutions. We empower businesses to make smarter decisions and foster growth through insightful data analysis." +
            "</p>", unsafe_allow_html=True)
        col1, col2 = st.columns([1,1])
        col1.markdown("**<h3 style='color:black'>Vision</h3>**", unsafe_allow_html=True)
        col2.markdown("**<h3 style='color:black'>Mission</h3>**", unsafe_allow_html=True)
        col1.markdown("<p style='color: black; font-size: 1.0em; font-weight: bold;'>" +"Empowering personalized entertainment experiences, Entertainment AI envisions pioneering a recommendation algorithm that seamlessly combines content and collaborative filtering. Our goal is to accurately predict user movie ratings based on historical preferences, redefining the landscape of user-centric content discovery."+"</p>", unsafe_allow_html=True)
        col2.markdown("<p style='color: black; font-size: 1.0em; font-weight: bold;'>" +"Our mission at Entertainment AI is to engineer a cutting-edge recommendation algorithm, blending content and collaborative filtering to elevate user satisfaction by delivering precise and personalized movie recommendations. We aim to revolutionize the entertainment industry through innovative solutions that anticipate and exceed user expectations."+"</p>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: black;'>The team</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1.75, 1.5, 1.75])
        col1.markdown("**<h3 style='color:black'>Name</h3>**", unsafe_allow_html=True)
        col1.info("Sharonrose Khokhololo")
        col2.markdown("**<h3 style='color:black'>Role</h3>**", unsafe_allow_html=True)
        col2.info("Team Lead")
        col3.markdown("**<h3 style='color:black'>Email</h3>**", unsafe_allow_html=True)
        col3.info("morongwarose92@gmail.com")
        col1.info("Nino Palesa Tsolo")
        col2.info("Data Scientist")
        col3.info("tsolonino@gmail.com")
        col1.info("Daluxolo Hadebe")
        col2.info("Technical Lead")
        col3.info("sanelehadebe070@gmail.com")
        col1.info("Thembi Chauke")
        col2.info("Project Manager")
        col3.info("thembichauke@gmail.com")
        st.markdown("<h2 style='text-align: center; color: red;'>Clients</h1>", unsafe_allow_html=True)
        st.image('resources/imgs/clients.jpeg',use_column_width=True)

    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.markdown("<h1 style='text-align: center; color: black;'>Collaborative filtering</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: black; font-size: 1.2em; font-weight: bold;'>" +
                    "Collaborative filtering is a method used in recommendation systems to predict the preferences or interests of a user by collecting and analyzing information from many users. It relies on the assumption that users who have agreed in the past tend to agree again in the future." +
            "</p>", unsafe_allow_html=True)
 
    
       
 
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    
 

if __name__ == '__main__':
    main()
