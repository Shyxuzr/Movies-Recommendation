from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for your frontend

# Load trained model artifacts
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie_title, top_n=5):
    # Case-insensitive search for movie index
    matching_movies = movies[movies['title'].str.lower() == movie_title.lower()]
    
    if matching_movies.empty:
        return None

    movie_index = matching_movies.index[0]
    distances = similarity[movie_index]
    
    # Sort by similarity score in descending order (excluding the movie itself)
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n + 1]
    
    recommendations = []
    for i in movie_list:
        rec_movie = movies.iloc[i[0]]
        recommendations.append({
            "movie_id": int(rec_movie['movie_id']),
            "title": rec_movie['title'],
            "score": round(float(i[1]), 4)
        })
    return recommendations

@app.route('/api/movies', methods=['GET'])
def get_all_movies():
    """Returns list of all movie titles for dropdowns or search bars"""
    titles = movies['title'].tolist()
    return jsonify({"movies": titles})

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Returns top recommended movies based on user input"""
    data = request.get_json()
    
    if not data or 'movie' not in data:
        return jsonify({"error": "Please provide 'movie' title in request body"}), 400
    
    selected_movie = data['movie']
    results = recommend(selected_movie)
    
    if results is None:
        return jsonify({"error": f"Movie '{selected_movie}' not found in database."}), 404
    
    return jsonify({
        "selected_movie": selected_movie,
        "recommendations": results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)