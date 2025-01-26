from flask import Flask, request, render_template, jsonify
import openai
import requests
from newspaper import Article
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import random


app = Flask(__name__)

openai.api_key = "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"
bing_api_key = "b8f06ca3fe274be19276df0ce95b7ff1"


vectorizer = TfidfVectorizer()


DATABASE_FILE = "truthfulness_data.csv"


true_keywords = ["confirmed", "verified", "accurate", "true", "supported", "consistent", "legitimate"]
not_true_keywords = ["false", "unconfirmed", "misleading", "inaccurate", "debunked", "contradictory", "disproven"]
unknown_keywords = ["uncertain", "unknown", "speculative", "inconclusive", "ambiguous"]


def load_database():
    if os.path.exists(DATABASE_FILE):
        return pd.read_csv(DATABASE_FILE)
    else:
        return pd.DataFrame(columns=["query", "snippets", "openai_analysis", "truthfulness_score"])

def save_to_database(query, snippets, openai_analysis, truthfulness_score):
    db = load_database()
    new_entry = {
        "query": query,
        "snippets": " || ".join(snippets),
        "openai_analysis": openai_analysis,
        "truthfulness_score": truthfulness_score,
    }
    new_row = pd.DataFrame([new_entry])
    db = pd.concat([db, new_row], ignore_index=True)
    db.to_csv(DATABASE_FILE, index=False)

def parse_link(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception:
        return None

def search_bing(query, api_key):
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": f"Is it true that {query}", "count": 5, "textDecorations": True, "textFormat": "HTML"}
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        results = response.json()
        return results.get("webPages", {}).get("value", [])
    except requests.exceptions.RequestException:
        return []

def parse_link(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception:
        return None

def analyze_with_openai(content, input_query):
    """
    Use OpenAI for a two-pass analysis: 
    1. Analyze the content and generate an initial response.
    2. Evaluate the response for consistency and determine truthfulness.
    """
    try:
        initial_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a truth-verification assistant."},
                {
                    "role": "user",
                    "content": (
                        f"Analyze the following content to verify the truthfulness of the statement: '{input_query}'. "
                        f"Provide a detailed analysis of the key points supporting and contradicting the statement. "
                        f"Do not declare the statement as true or false yet: {content}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=350,
        )
        initial_analysis = initial_response["choices"][0]["message"]["content"]

        
        final_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a truth-verification assistant."},
                {
                    "role": "user",
                    "content": (
                        f"Based on the following analysis, determine if the statement is likely true, false, or uncertain. "
                        f"The analysis is: {initial_analysis} "
                        f"Your response must begin with one of these phrases: "
                        f"'This statement is likely true,' 'This statement is false,' or 'This statement is uncertain.' "
                        f"Follow this opening statement with a clear explanation supporting your conclusion."
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=350,
        )
        return final_response["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"Error during OpenAI analysis: {e}")
        return None


def analyze_openai_response(response):
    """
    Analyze OpenAI's response and assign a score based on the overall content.
    Prioritize consistency between the opening statement and the main analysis.
    """
    response = response.lower()

    
    opening_statement = response.split('.')[0]

    
    if "likely true" in opening_statement:
        score = 0.9
    elif "false" in opening_statement or "not true" in opening_statement:
        score = 0.1
    elif "uncertain" in opening_statement:
        score = 0.5
    else:
        score = 0.5  

    
    if "misleading" in response or "not true" in response or "false" in response:
        score = min(score, 0.1)  
    elif "accurate" in response or "clearly true" in response:
        score = max(score, 0.9)  

    return score

def analyze_excerpts_with_keywords(snippets):
    """
    Analyze snippets for keywords and assign a score.
    """
    true_count = sum(any(word in snippet.lower() for word in true_keywords) for snippet in snippets)
    not_true_count = sum(any(word in snippet.lower() for word in not_true_keywords) for snippet in snippets)
    unknown_count = sum(any(word in snippet.lower() for word in unknown_keywords) for snippet in snippets)

    total_count = true_count + not_true_count + unknown_count
    if total_count == 0:
        return 0.5  # Neutral score if no keywords are found

    # Weighted average of the keyword analysis
    return (true_count * 0.9 + unknown_count * 0.5 + not_true_count * 0.1) / total_count

def compute_similarity(input_text, snippets):
    texts = [input_text] + snippets
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return np.mean(similarities)

def predict_truthfulness(openai_analysis, search_results, input_text):
    snippets = [result["snippet"] for result in search_results if "snippet" in result]
    if not snippets:
        return 0.0

    cosine_sim = compute_similarity(input_text, snippets)
    openai_score = analyze_openai_response(openai_analysis)
    keyword_score = analyze_excerpts_with_keywords(snippets)

    # Weights: OpenAI (60%), Cosine Similarity (20%), Keywords (20%)
    combined_score = (openai_score * 0.6) + (cosine_sim * 0.2) + (keyword_score * 0.2)
    return combined_score * 100

@app.route("/facts", methods=["GET"])
def get_fact():
    facts = [
        "A fake news website can spread fake news faster than real news.",
        "Social media proficiency does not correlate with digital literacy.",
        "Schools are incorporating digital literacy into curricula.",
        "Businesses are developing countermeasures against misinformation.",
        "Fake news examples are not new.",
        "66% of U.S. consumers believe that 76% or more of the news on social media is biased.",
        "60% globally say news organizations regularly report false stories.",
        "82% of Argentinians reported seeing deliberately false stories often, compared to lower percentages in Germany, Japan, and South Korea.",
        "60% of U.S. journalists express high concern about possible press freedom limitations.",
        "94% of journalists see made-up news as a significant problem.",
        "38.2% of U.S. news consumers unknowingly shared fake news on social media.",
        "16% of overall respondents found news content on Twitter accurate, with stark contrasts based on political alignment.",
        "66% of bots discussing COVID-19 were spreading misinformation.",
        "47% of U.S. adults encountered a significant amount of made-up news about COVID-19.",
        "3x increase in video deep fakes and 8x increase in voice deep fakes from 2022 to 2023.",
        "An estimated 500,000 deepfakes were shared on social media in 2023."
    ]
    return jsonify({"fact": random.choice(facts)})

@app.route("/")
def home():
    return render_template("landing.html")

@app.route('/about')
def about():
    return render_template('about.html') 

@app.route("/query", methods=["GET", "POST"])
def query_page():
    if request.method == "GET":
        return render_template("query.html")
    if request.method == "POST":
        query = request.form.get("query")
        if not query:
            return render_template("results.html", error="No query provided.")

        search_results = search_bing(query, bing_api_key)
        snippets = [result["snippet"] for result in search_results]
        analysis = analyze_with_openai(" ".join(snippets), query)
        truthfulness_score = round(predict_truthfulness(analysis, search_results, query), 2)

        save_to_database(query, snippets, analysis, truthfulness_score)

        return render_template(
            "results.html",
            query=query,
            truthfulness_score=truthfulness_score,
            openai_analysis=analysis,
            similar_articles=[
                {"name": result["name"], "url": result["url"]} for result in search_results
            ],
        )
    
@app.route("/links", methods=["GET", "POST"])
def links_page():
    if request.method == "GET":
        return render_template("links.html")

    if request.method == "POST":
        url = request.form.get("url")
        if not url:
            return render_template("results.html", error="No URL provided.")

        # Parse the content from the link
        parsed_content = parse_link(url)
        if not parsed_content:
            return render_template("results.html", error="Could not parse the link.")

        # Fetch related articles from Bing
        search_results = search_bing(parsed_content[:100], bing_api_key)
        snippets = [result["snippet"] for result in search_results]

        # Analyze with OpenAI
        openai_analysis = analyze_with_openai(" ".join(snippets), parsed_content)

        # Predict truthfulness using updated logic
        truthfulness_score = round(predict_truthfulness(openai_analysis, search_results, parsed_content), 2)

        # Save the data to the database
        save_to_database(url, snippets, openai_analysis, truthfulness_score)

        # Render results
        return render_template(
            "results.html",
            query=url,
            truthfulness_score=truthfulness_score,
            openai_analysis=openai_analysis,
            similar_articles=[
                {"name": result["name"], "url": result["url"]} for result in search_results
            ],
        )
    
@app.route("/link", methods=["POST"])
def handle_link():
    url = request.form.get("url")
    if not url:
        return render_template("results.html", error="No URL provided.")

    parsed_content = parse_link(url)
    if not parsed_content:
        return render_template("results.html", error="Could not parse the link.")

    search_results = search_bing(parsed_content[:100], bing_api_key)
    snippets = [result["snippet"] for result in search_results]
    analysis = analyze_with_openai(" ".join(snippets), parsed_content)
    truthfulness_score = round(predict_truthfulness(analysis, search_results, parsed_content), 2)

    save_to_database(url, snippets, analysis, truthfulness_score)

    return render_template(
        "results.html",
        query=url,
        truthfulness_score=truthfulness_score,
        openai_analysis=analysis,
        similar_articles=[
            {"name": result["name"], "url": result["url"]} for result in search_results
        ],
    )

if __name__ == "__main__":
    app.run(debug=True)
