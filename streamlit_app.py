import requests
import sqlite3
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# NASA API Configuration
API_KEY = "uEl6Trd7OCAqNbzdBWFedJXA8ScCsDHFnlDzc2Mx"
NASA_API_URL = "https://api.nasa.gov/planetary/apod"
NASA_IMAGE_VIDEO_API_URL = "https://images-api.nasa.gov/search"
MARS_ROVER_PHOTOS_API_URL = "https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos"

# SQLite Database Manager
class DatabaseManager:
    def __init__(self, db_name="data.db"):
        self.connection = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        with self.connection:
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS apod (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    date TEXT UNIQUE,
                    explanation TEXT,
                    url TEXT
                )
            """)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS favorites (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    date TEXT UNIQUE,
                    explanation TEXT,
                    url TEXT
                )
            """)

    def insert_apod(self, apod_data):
        with self.connection:
            self.connection.execute("""
                INSERT OR IGNORE INTO apod (title, date, explanation, url)
                VALUES (?, ?, ?, ?)
            """, (apod_data["title"], apod_data["date"], apod_data["explanation"], apod_data["url"]))

    def insert_favorite(self, favorite_data):
        with self.connection:
            self.connection.execute("""
                INSERT OR IGNORE INTO favorites (title, date, explanation, url)
                VALUES (?, ?, ?, ?)
            """, (favorite_data["title"], favorite_data["date"], favorite_data["explanation"], favorite_data["url"]))

    def fetch_apod_by_date(self, date):
        with self.connection:
            cursor = self.connection.execute("SELECT * FROM apod WHERE date = ?", (date,))
            row = cursor.fetchone()
            if row:
                return {"id": row[0], "title": row[1], "date": row[2], "explanation": row[3], "url": row[4]}
        return None

    def fetch_all_apods(self):
        with self.connection:
            cursor = self.connection.execute("SELECT * FROM apod")
            return [
                {"id": row[0], "title": row[1], "date": row[2], "explanation": row[3], "url": row[4]}
                for row in cursor.fetchall()
            ]

    def fetch_favorites(self):
        with self.connection:
            cursor = self.connection.execute("SELECT * FROM favorites")
            return [
                {"id": row[0], "title": row[1], "date": row[2], "explanation": row[3], "url": row[4]}
                for row in cursor.fetchall()
            ]

class WordCloudGenerator:
    @staticmethod
    def generate_word_cloud(text):
        return WordCloud(width=800, height=400, background_color='white').generate(text)

    @staticmethod
    def display_word_cloud(wordcloud):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# NASA API Client
class NASAAPIClient:
    def fetch_apod(self, date=None):
        params = {"api_key": API_KEY}
        if date:
            params["date"] = date

        response = requests.get(NASA_API_URL, params=params)
        response.raise_for_status()
        return response.json()

    def search_images(self, query, media_type="image"):
        params = {
            "q": query,
            "media_type": media_type,  # Filter for images only
            "center": "JPL"  
        }
        response = requests.get(NASA_IMAGE_VIDEO_API_URL, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_mars_rover_photos(self, earth_date, camera="navcam"):
        params = {
            "api_key": API_KEY,
            "earth_date": earth_date,
            "camera": camera
        }
        response = requests.get(MARS_ROVER_PHOTOS_API_URL, params=params)
        response.raise_for_status()
        return response.json().get("photos", [])
    
def load_space_data():
    try:
        df = pd.read_csv('Space_Corrected.csv')
        df['Year'] = pd.to_datetime(df['Datum'], dayfirst=True, errors='coerce').dt.year
        df = df.dropna(subset=['Year'])  # Drop rows where the date couldn't be parsed
        df = df.groupby('Year').size().reset_index(name='MissionCount')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None



def load_star_visibility_data():
    try:
        df = pd.read_csv('Star_Visibility_Dataset.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def build_star_visibility_model(df):
    X = df[['Star_Magnitude', 'Viewing_Angle', 'Light_Pollution', 'Humidity']]
    y = df['Star_Visibility']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Calculate Adjusted R-squared
    r_squared = model.score(X_test, y_test)
    n = len(y_test)
    p = X_test.shape[1]
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    return model, adjusted_r_squared

def main():
    st.set_page_config(page_title="Cosmic Explorer")

    st.title("Cosmic Explorer")

    db_manager = DatabaseManager()
    nasa_client = NASAAPIClient()

    st.sidebar.header("Options")
    mode = st.sidebar.radio(
        "Select Mode",
        options=["Search NASA Images", "Search by Date", "Gallery View", "Favorite APODs", "Mars Rover Photos", "Word Cloud", "Predicted Star Visibility"]
    )

    if mode == "Search NASA Images":
        st.header("Search NASA Images")
        search_term = st.text_input("Enter a search term (e.g., Mars, Earth, Galaxy, etc.):")
        search = st.button("Search Images")

        if search:
            try:
                results = nasa_client.search_images(search_term)
                items = results.get("collection", {}).get("items", [])
                if not items:
                    st.warning(f"No results found for '{search_term}'.")
                else:
                    st.subheader(f"Results for '{search_term}':")
                    for item in items[:10]:  
                        data = item.get("data", [])[0]
                        links = item.get("links", [])
                        if links:
                            st.image(links[0]["href"], caption=data.get("title", "No Title"), use_container_width=True)
                            st.write(f"**Description:** {data.get('description', 'No description available.')}")
            except Exception as e:
                st.error(f"Error searching images: {e}")

    elif mode == "Search by Date":
        st.header("Search APOD by Date")
        date = st.date_input("Select a date", value=datetime.now().date())
        search = st.button("Fetch APOD")

        if search:
            try:
                # Check if APOD is already in the database
                stored_apod = db_manager.fetch_apod_by_date(date.strftime('%Y-%m-%d'))
                if stored_apod:
                    st.success("APOD already exists in the database.")
                    st.image(stored_apod["url"], caption=stored_apod["title"])
                    st.write(f"**Title:** {stored_apod['title']}")
                    st.write(f"**Date:** {stored_apod['date']}")
                    st.write(f"**Explanation:** {stored_apod['explanation']}")
                else:
                    # Fetch and display APOD
                    apod_data = nasa_client.fetch_apod(date=date.strftime('%Y-%m-%d'))
                    st.image(apod_data["url"], caption=apod_data["title"])
                    st.write(f"**Title:** {apod_data['title']}")
                    st.write(f"**Date:** {apod_data['date']}")
                    st.write(f"**Explanation:** {apod_data['explanation']}")

                    # Save to database
                    db_manager.insert_apod(apod_data)
                    st.info("APOD saved to the database.")
            except Exception as e:
                st.error(f"Error fetching APOD: {e}")

    elif mode == "Gallery View":
        st.header("APOD Gallery")
        apods = db_manager.fetch_all_apods()

        if not apods:
            st.warning("No APODs found in the database.")
        else:
            # Initialize session state for favorites
            if "favorites" not in st.session_state:
                st.session_state.favorites = {fav["date"] for fav in db_manager.fetch_favorites()}

            cols = st.columns(3)  
            for i, apod in enumerate(apods):
                with cols[i % 3]:
                    st.image(apod["url"], caption=apod["title"], use_container_width=True)

                    if apod["date"] in st.session_state.favorites:
                        # If the APOD is already in favorites, show "Remove from Favorites" button
                        if st.button(f"Remove {apod['title']} from Favorites", key=f"remove_{apod['id']}"):
                            # Remove from favorites in database and session state
                            with db_manager.connection:
                                db_manager.connection.execute("DELETE FROM favorites WHERE date = ?", (apod["date"],))
                            st.session_state.favorites.remove(apod["date"])
                            st.success(f"{apod['title']} removed from favorites.")
                            st.rerun()  
                    else:
                        # Otherwise, show "Add to Favorites" button
                        if st.button(f"Favorite {apod['title']}", key=f"favorite_{apod['id']}"):
                            # Add to favorites in database and session state
                            db_manager.insert_favorite(apod)
                            st.session_state.favorites.add(apod["date"])
                            st.success(f"{apod['title']} added to favorites.")
                            st.rerun()  

    elif mode == "Favorite APODs":
        st.header("My Favorite APODs")
        favorites = db_manager.fetch_favorites()

        if not favorites:
            st.warning("No favorite APODs found.")
        else:
            for favorite in favorites:
                st.image(favorite["url"], caption=favorite["title"])
                st.write(f"**Title:** {favorite['title']}")
                st.write(f"**Date:** {favorite['date']}")
                st.write(f"**Explanation:** {favorite['explanation']}")

    elif mode == "Mars Rover Photos":
        st.header("Mars Rover Photos Explorer")

        # Add calendar input for Earth date
        earth_date = st.date_input(
            "Select Earth Date",
            value=datetime(2022, 1, 1).date(),  
            min_value=datetime(2021, 2, 1).date(),  
            max_value=datetime.today().date()  
        )

        camera = st.selectbox("Camera", ["navcam", "fhaz", "rhaz", "mast", "chemcam"])

        if st.button("Load Mars Photos"):
            mars_photos = nasa_client.fetch_mars_rover_photos(earth_date.strftime("%Y-%m-%d"), camera)

            if mars_photos:
                st.subheader(f"Found {len(mars_photos)} photos")
                for photo in mars_photos[:5]:  
                    st.image(photo["img_src"], caption=f"{photo['camera']['full_name']} - {photo['earth_date']}")
            else:
                st.warning("No photos found for selected criteria")

    elif mode == "Word Cloud":
        st.header("Generate Word Clouds")
        cloud_tab = st.selectbox("Select Word Cloud Source", ["APOD Titles", "Favorite Titles"])

        if cloud_tab == "APOD Titles":
            apods = db_manager.fetch_all_apods()
            if apods:
                text = " ".join(apod["title"] for apod in apods)
                wordcloud = WordCloudGenerator.generate_word_cloud(text)
                st.subheader("Word Cloud from All APOD Titles")
                WordCloudGenerator.display_word_cloud(wordcloud)
            else:
                st.warning("No APODs available to generate a word cloud.")

        elif cloud_tab == "Favorite Titles":
            favorites = db_manager.fetch_favorites()
            if favorites:
                text = " ".join(favorite["title"] for favorite in favorites)
                wordcloud = WordCloudGenerator.generate_word_cloud(text)
                st.subheader("Word Cloud from Favorite Titles")
                WordCloudGenerator.display_word_cloud(wordcloud)
            else:
                st.warning("No favorite APODs available to generate a word cloud.")

    

    elif mode == "Predicted Star Visibility":
        st.title("Star Visibility Predictor")
        
        df = load_star_visibility_data()
        if df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(df.head())

            model, adjusted_r_squared = build_star_visibility_model(df)

            st.subheader("Model Performance")
            st.write(f"Adjusted R-Squared: {adjusted_r_squared:.4f}")

            st.subheader("Predict Star Visibility")
            star_magnitude = st.slider("Star Magnitude", min_value=float(df['Star_Magnitude'].min()), max_value=float(df['Star_Magnitude'].max()), value=float(df['Star_Magnitude'].mean()))
            viewing_angle = st.slider("Viewing Angle", min_value=float(df['Viewing_Angle'].min()), max_value=float(df['Viewing_Angle'].max()), value=float(df['Viewing_Angle'].mean()))
            light_pollution = st.slider("Light Pollution", min_value=float(df['Light_Pollution'].min()), max_value=float(df['Light_Pollution'].max()), value=float(df['Light_Pollution'].mean()))
            humidity = st.slider("Humidity", min_value=float(df['Humidity'].min()), max_value=float(df['Humidity'].max()), value=float(df['Humidity'].mean()))

            input_data = pd.DataFrame({
                "Star_Magnitude": [star_magnitude],
                "Viewing_Angle": [viewing_angle],
                "Light_Pollution": [light_pollution],
                "Humidity": [humidity]
            })

            prediction = model.predict(input_data)[0]

            # Ensure visibility is not negative
            if prediction < 0:
                prediction = 0

            st.write(f"Predicted Star Visibility: {prediction:.2f}")


if __name__ == "__main__":
    main()
