import pandas as pd
import numpy as np
import joblib
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\GUVI - PROJECTS\project 5 _ NLP Sentiment Analysis\chatgpt_style_reviews_dataset.csv")
cleaned_data = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\GUVI - PROJECTS\project 5 _ NLP Sentiment Analysis\chatgpt_style_reviews_Cleaned_dataset.csv")
model = joblib.load(r"C:\Users\Dell\OneDrive\Desktop\GUVI - PROJECTS\project 5 _ NLP Sentiment Analysis\nb_sentiment_pipeline.pkl")

data["sentiment"] = model.predict(data["review"])
label_map = {0:"Negative",1:"Neutral",2:"Positive"}
data["sentiment"] = data["sentiment"].map(label_map)

n = st.sidebar.radio("🧠 NLP - Sentiment Analysis", (
        "🏡 Home",
        "📘 Introduction",
        "❓ Problem Statement",
        "📊 Exploratory Data Analysis",
        "😀😑😞 Live Sentiment Analyzer",
        "✅ Conclusion"))

if n == "🏡 Home":

    st.title("NATURAL LANGUAGE PROCESSING")
    
    st.markdown("---")

    st.image(r"C:\Users\Dell\OneDrive\Desktop\GUVI - PROJECTS\project 5 _ NLP Sentiment Analysis\NLP.jpg", use_container_width=True)

    st.markdown("---")

    
    st.markdown("By")
    st.write("**Girija Swaminathan**")
    
elif n == "📘 Introduction":
    
    st.title("📘 Introduction")

    st.header("🧠 What is Natural Language Processing (NLP)?")

    st.write(
        """
        Natural Language Processing (NLP) is a branch of Artificial Intelligence
        that enables computers to understand, interpret, and generate human language.
        
        It allows machines to read text, hear speech, analyze meaning,
        and respond like humans.
        """
    )

    st.info("In simple words: NLP helps computers understand human language.")

    st.markdown("---")

    st.header("💬 What is Sentiment Analysis?")

    st.write(
        """
        Sentiment Analysis is an NLP technique used to determine the emotional tone
        behind a piece of text.
        
        It identifies whether the sentiment is Positive, Negative, or Neutral.
        """
    )

    st.success("Example: 'I love this product' → Positive Sentiment")

    st.markdown("---")

    st.subheader("📊 Types of Sentiment")

    col1, col2, col3 = st.columns(3)

    col1.success("Positive 😀")
    col2.info("Neutral 😑")
    col3.error("Negative 😞")
    
elif n == "❓ Problem Statement":

    st.title("❓ Problem Statement")

    st.write(
        """
        In this project, we analyze user reviews of the ChatGPT application
        and classify them into Positive, Neutral, or Negative sentiments
        based on the opinion expressed.

        The objective is to understand customer satisfaction levels,
        identify common issues or concerns, and provide insights to
        improve the overall user experience of the application.
        """
    )

    st.markdown("---")

    st.subheader("🎯 Project Goals")

    st.write("""
    • Analyze user review text data  
    • Classify sentiment polarity  
    • Measure customer satisfaction  
    • Identify improvement areas  
    """)
elif n == "✅ Conclusion":

    st.title("✅ Conclusion")

    st.write(
        """
        In this project, we developed a Sentiment Analysis system
        using Natural Language Processing (NLP) techniques
        to analyze user reviews of the ChatGPT application.
        
        The system classifies reviews into Positive, Neutral,
        and Negative sentiments based on the opinion expressed.
        """
    )

    st.markdown("---")

    st.subheader("🔍 Key Achievements")

    st.write("""
    • Performed text preprocessing on raw review data  
    • Implemented Machine Learning and Deep Learning models  
    • Compared model performances  
    • Built an interactive Streamlit web application  
    """)

    st.markdown("---")

    st.subheader("🏆 Outcome")

    st.success(
        "The system successfully predicts user sentiment and helps "
        "understand customer satisfaction and feedback patterns."
    )

    st.markdown("---")    
    
elif n == "📊 Exploratory Data Analysis":

    st.title("📊 Exploratory Data Analysis")

    eda_q = st.sidebar.radio(
    "📊 EDA Questions",
    (
        "1️⃣ Overall Sentiment",
        "2️⃣ Sentiment by Rating",
        "3️⃣ Key Sentiment Phrases",
        "4️⃣ Sentiment Over Time",
        "5️⃣ Verified User Sentiment",
        "6️⃣ Review Length vs Sentiment",
        "7️⃣ Location-wise Sentiment",
        "8️⃣ Platform Sentiment",
        "9️⃣ Version-wise Sentiment",
        "🔟 Negative Feedback Themes"
    )
)
    if eda_q == "1️⃣ Overall Sentiment":
        st.subheader("📊 Overall Sentiment Distribution")

        sentiment_counts = cleaned_data["sentiment"].value_counts().reset_index()
        
        sentiment_counts.columns = ["Sentiment", "Count"]

        st.write("### Sentiment Count")
        st.dataframe(sentiment_counts)      
        
        
        st.write("### Sentiment Distribution — Pie Chart")

        fig, ax = plt.subplots()
        
        ax.pie(
        sentiment_counts["Count"],
        labels=sentiment_counts["Sentiment"],
        autopct="%1.1f%%",
        startangle=90
        )
        
        ax.axis("equal")
        
        st.pyplot(fig)      
          
        
    elif eda_q == "2️⃣ Sentiment by Rating":
        
        st.subheader("📊 Sentiment Variation by Rating")

        rating_sentiment = pd.crosstab(
            cleaned_data["rating"],
            cleaned_data["sentiment"]
        )

        st.write("### Rating vs Sentiment Table")
        st.dataframe(rating_sentiment)

        st.bar_chart(rating_sentiment)
        
        def detect_mismatch(row):
            rating = row["rating"]
            sentiment = row["sentiment"]

            if pd.isna(rating) or pd.isna(sentiment):
                return "Unknown"

            if rating >= 4 and sentiment == "Positive":
                return "Match"

            elif rating == 3 and sentiment == "Neutral":
                return "Match"
 
            elif rating <= 2 and sentiment == "Negative":
                return "Match"

            else:
                return "Mismatch"
            
            cleaned_data["Mismatch_Flag"] = (
    cleaned_data.apply(detect_mismatch, axis=1)
)  

            mismatch_counts = (
            cleaned_data["Mismatch_Flag"]
            .value_counts()
            )
            
            st.dataframe(mismatch_counts)
            st.bar_chart(mismatch_counts)       
        
    elif eda_q == "3️⃣ Key Sentiment Phrases":
        st.subheader("☁️ Word Clouds by Sentiment")

        sentiments = cleaned_data["sentiment"].unique()

        for s in sentiments:

            st.write(f"### {s} Reviews")

            text = " ".join(
                cleaned_data[cleaned_data["sentiment"] == s]["review"]
                .astype(str)
            )

            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate(text)

            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")

            st.pyplot(fig)
            
    elif eda_q == "4️⃣ Sentiment Over Time":

        # Clean date
        data["date"] = data["date"].replace("########", np.nan)

        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce",
            dayfirst=True
        )

        # Fill missing
        data = data.sort_values(["version", "date"])

        data["date_filled"] = (
            data.groupby("version")["date"]
            .transform(lambda x: x.ffill().bfill())
        )
        # Extract month
        data["month"] = (
            data["date_filled"]
            .dt.to_period("M")
            .astype(str)
        )       
        
        # Crosstab sentiment
        monthly_sentiment = pd.crosstab(
            data["month"],
            data["sentiment"]
        )
        # Extract Week
        data["week"] = (
            data["date_filled"]
            .dt.to_period("W")
            .astype(str)
        )
        
        weekly_sentiment = pd.crosstab(
                                    data["week"],
                                    data["sentiment"]
                                )
        st.subheader("📈 Monthly Sentiment Trend")

        st.line_chart(monthly_sentiment)

        st.subheader("📈 Weekly Sentiment Trend")

        st.line_chart(weekly_sentiment)
        
    elif eda_q == "5️⃣ Verified User Sentiment":
        
        st.subheader("📊 Verified Users Sentiment Analysis")
        
        verified_sentiment = pd.crosstab(
            data["verified_purchase"],
            data["sentiment"]
        )

        st.dataframe(verified_sentiment)

        fig, ax = plt.subplots(figsize=(8,5))

        verified_sentiment.plot(
            kind="bar",
            color=["red","yellow","green"],
            ax=ax
        )
        
        ax.set_title("Sentiment by Verified Purchase")
        ax.set_xlabel("Verified Purchase")
        ax.set_ylabel("Review Count")
        ax.legend(title="Sentiment", labels=["Negative","Neutral","Positive"])
               
        st.pyplot(fig)

    elif eda_q == "6️⃣ Review Length vs Sentiment":
        st.subheader("📊 Review Length vs Sentiment")        
        
        length_sentiment = data.groupby(
            "sentiment"
        )["review_length"].mean() 
        
        st.bar_chart(length_sentiment)
        st.markdown("___")
        st.dataframe(length_sentiment)
        
    elif eda_q == "7️⃣ Location-wise Sentiment": 
        
        st.subheader("📊 Location vs Sentiment")

        locations = data["location"].unique()

        filtered_data = data[
             data["location"].isin(locations)
         ]

        location_sentiment = pd.crosstab(
            filtered_data["location"],
            filtered_data["sentiment"]
        )
        
        st.bar_chart(location_sentiment)  
        st.markdown("___")
        st.dataframe(location_sentiment)
        
    elif eda_q == "8️⃣ Platform Sentiment":
        st.subheader("📊 Platform-wise Sentiment Analysis")

        platform_sentiment = pd.crosstab(
            data["platform"],
            data["sentiment"]
        )
        
        
        st.bar_chart(platform_sentiment)  
        st.markdown("___")
        st.dataframe(platform_sentiment)
        
    elif eda_q == "9️⃣ Version-wise Sentiment":
        st.subheader("📊 Sentiment Across ChatGPT Versions")

        version_sentiment = pd.crosstab(
            data["version"],
            data["sentiment"]
        )

        st.bar_chart(version_sentiment) 
        st.markdown("___")
        st.dataframe(version_sentiment)

    elif eda_q == "🔟 Negative Feedback Themes":
        st.subheader("☁️ Negative Review Word Cloud")

        negative_reviews = data[
            data["sentiment"] == "Negative"
        ]["review"]

        negative_text = " ".join(
            negative_reviews.astype(str)
        )

        wc_neg = WordCloud(
            width=1200,
            height=600,
            background_color="black",
            colormap="Reds"
        ).generate(negative_text)

        fig, ax = plt.subplots(figsize=(12,6))

        ax.imshow(wc_neg, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig)
        
elif n == "😀😑😞 Live Sentiment Analyzer" :
    st.title("NLP Sentiment Analyzer 😀😑😞 ")

      
    # User input
    review = st.text_area(
        "Enter your review:"
    ) 
    
    # Prediction
    if st.button("Analyze"):
        prediction = model.predict([review])[0]
        proba = model.predict_proba([review])[0]
        classes = model.classes_
    
        label_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }
    
        st.subheader("Prediction:")
    
        pred_label = label_map[prediction]
    
        if pred_label == "Positive":
            st.success(pred_label)
    
        elif pred_label == "Neutral":
            st.info(pred_label)
    
        else:
            st.error(pred_label)
    
        st.subheader("Confidence Distribution")
    
        for label, p in zip(classes, proba):
    
            sentiment = label_map[label]
            percent = f"{p*100:.2f}%"
    
            if sentiment == "Positive":
                st.success(f"{sentiment} → {percent}")
    
            elif sentiment == "Neutral":
                st.info(f"{sentiment} → {percent}")
    
            else:
                st.error(f"{sentiment} → {percent}")
            

    
   
    
    


    

    
    
          

         
            
    
        
        



            
            
   
        

