import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading dataset...")
df = pd.read_csv('tmdb_5000_credits.csv')

# Helper function to parse top 3 cast members
def extract_cast(cast_str):
    try:
        cast_list = []
        counter = 0
        for item in ast.literal_eval(cast_str):
            if counter < 3:
                # Remove spaces so "Sam Worthington" becomes "SamWorthington" (treated as single entity)
                cast_list.append(item['name'].replace(" ", ""))
                counter += 1
            else:
                break
        return " ".join(cast_list)
    except:
        return ""

# Helper function to parse Director from crew
def extract_director(crew_str):
    try:
        for item in ast.literal_eval(crew_str):
            if item.get('job') == 'Director':
                return item['name'].replace(" ", "")
        return ""
    except:
        return ""

print("Extracting features (Cast & Director)...")
df['cast_clean'] = df['cast'].apply(extract_cast)
df['director_clean'] = df['crew'].apply(extract_director)

# Combine movie title, main cast, and director into a unified feature tag
df['tags'] = df['title'] + " " + df['cast_clean'] + " " + df['director_clean']

# Vectorize tags using CountVectorizer
print("Vectorizing features...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

# Compute Cosine Similarity Matrix
print("Calculating Cosine Similarity matrix...")
similarity = cosine_similarity(vectors)

# Keep relevant columns for output dataframe
final_df = df[['movie_id', 'title']]

# Save trained artifacts
print("Saving model files...")
pickle.dump(final_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Model training completed successfully! Files saved: 'movies.pkl', 'similarity.pkl'")