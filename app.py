"""
Enhanced Book Recommendation Agent - HuggingFace Deployment
Built with semantic search, sentiment analysis, and ML enrichments
"""

import gradio as gr
import pandas as pd
import numpy as np
import requests
import json
from huggingface_hub import InferenceClient
import os
from sklearn.metrics.pairwise import cosine_similarity

print("="*70)
print("🚀 Loading Book Recommendation System...")
print("="*70)

# ==================== Load All Data ====================

print("\n[1/3] Loading enriched books...")
try:
    enriched_books_df = pd.read_csv('fully_enriched_books.csv')
    print(f"   ✅ Loaded {len(enriched_books_df)} books")
except Exception as e:
    print(f"   ❌ Error loading books: {e}")
    raise

print("\n[2/3] Loading embeddings...")
try:
    embeddings = np.load('book_embeddings.npy')
    print(f"   ✅ Loaded embeddings: {embeddings.shape}")
except Exception as e:
    print(f"   ❌ Error loading embeddings: {e}")
    raise

print("\n[3/3] Initializing LLM...")
HF_TOKEN = os.environ.get('HF_TOKEN')
if HF_TOKEN:
    try:
        hf_client = InferenceClient(token=HF_TOKEN)
        print("   ✅ HuggingFace client initialized")
    except Exception as e:
        print(f"   ⚠️ Warning: Could not initialize HF client: {e}")
        hf_client = None
else:
    print("   ⚠️ HF_TOKEN not found - running without LLM")
    hf_client = None

print("\n" + "="*70)
print("✅ ALL COMPONENTS LOADED SUCCESSFULLY!")
print("="*70)

# ==================== Open Library API ====================

class OpenLibraryAPI:
    """Wrapper for Open Library API"""
    BASE_URL = "https://openlibrary.org"
    
    @staticmethod
    def search_books(query, limit=10):
        """Search for books by title, author, or keyword"""
        url = f"{OpenLibraryAPI.BASE_URL}/search.json"
        params = {'q': query, 'limit': limit}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error searching books: {e}")
            return None
    
    @staticmethod
    def search_by_genre(genre, limit=20):
        """Search books by genre/subject"""
        url = f"{OpenLibraryAPI.BASE_URL}/search.json"
        params = {'subject': genre, 'limit': limit}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error searching by genre: {e}")
            return None

# ==================== Recommendation Engine ====================

class BookRecommendationEngine:
    """Recommendation engine with semantic search and enriched data"""
    
    def __init__(self, enriched_df, embeddings):
        self.enriched_books = enriched_df
        self.embeddings = embeddings
        self.ol_api = OpenLibraryAPI()
        self.isbn_list = enriched_df['ISBN'].tolist()
    
    def semantic_search(self, query_text, n=10):
        """Search books using semantic similarity"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Load model for query encoding
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            model.eval()
            
            # Encode query
            inputs = tokenizer(query_text, return_tensors='pt', truncation=True, max_length=128)
            with torch.no_grad():
                query_embedding = model(**inputs).last_hidden_state.mean(dim=1).numpy()
            
            # Find similar books
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            top_indices = similarities.argsort()[-n:][::-1]
            
            recommendations = []
            for idx in top_indices:
                book = self.enriched_books.iloc[idx]
                recommendations.append({
                    'title': book['Book-Title'],
                    'author': book['Book-Author'],
                    'similarity': round(float(similarities[idx]), 3),
                    'genre': book.get('detected_genre', 'N/A'),
                    'sentiment': book.get('sentiment_score', 0),
                    'rating': book.get('avg_rating', 0),
                    'confidence': book.get('recommendation_confidence', 0)
                })
            
            return recommendations
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def recommend_by_sentiment(self, sentiment='Positive', n=10):
        """Recommend books based on sentiment category"""
        try:
            sentiment_books = self.enriched_books[
                self.enriched_books['sentiment_category'] == sentiment
            ].sort_values('recommendation_confidence', ascending=False)
            
            recommendations = []
            for idx, book in sentiment_books.head(n).iterrows():
                recommendations.append({
                    'title': book['Book-Title'],
                    'author': book['Book-Author'],
                    'sentiment': book['sentiment_score'],
                    'sentiment_category': book['sentiment_category'],
                    'rating': book['avg_rating'],
                    'confidence': book.get('recommendation_confidence', 0)
                })
            return recommendations
        except Exception as e:
            print(f"Error in sentiment recommendations: {e}")
            return []
    
    def recommend_by_genre(self, genre, n=10):
        """Recommend by genre using Open Library API"""
        try:
            result = self.ol_api.search_by_genre(genre, limit=n*2)
            if not result or 'docs' not in result:
                return []
            
            recommendations = []
            for doc in result['docs'][:n]:
                recommendations.append({
                    'title': doc.get('title', 'N/A'),
                    'author': doc.get('author_name', ['N/A'])[0] if doc.get('author_name') else 'N/A',
                    'year': doc.get('first_publish_year', 'N/A'),
                    'genre': genre
                })
            return recommendations
        except Exception as e:
            print(f"Error in genre recommendations: {e}")
            return []
    
    def recommend_by_multiple_filters(self, genre=None, sentiment_min=0.5, 
                                     reading_level=None, min_confidence=0.6, n=10):
        """Advanced filtering with multiple criteria"""
        try:
            filtered = self.enriched_books.copy()
            
            if genre:
                filtered = filtered[filtered['detected_genre'] == genre]
            
            if sentiment_min:
                filtered = filtered[filtered['sentiment_score'] >= sentiment_min]
            
            if reading_level:
                filtered = filtered[filtered['reading_level'] == reading_level]
            
            if min_confidence:
                filtered = filtered[filtered['recommendation_confidence'] >= min_confidence]
            
            filtered = filtered.sort_values('recommendation_confidence', ascending=False)
            
            recommendations = []
            for idx, book in filtered.head(n).iterrows():
                recommendations.append({
                    'title': book['Book-Title'],
                    'author': book['Book-Author'],
                    'genre': book['detected_genre'],
                    'reading_level': book.get('reading_level', 'N/A'),
                    'sentiment': book['sentiment_score'],
                    'confidence': book['recommendation_confidence'],
                    'rating': book.get('avg_rating', 0)
                })
            
            return recommendations
        except Exception as e:
            print(f"Error in multi-filter recommendations: {e}")
            return []
    
    def recommend_popular(self, n=10, min_sentiment=0.6):
        """Popular books with sentiment filter"""
        try:
            qualified = self.enriched_books[
                self.enriched_books['sentiment_score'] >= min_sentiment
            ].sort_values('recommendation_confidence', ascending=False)
            
            recommendations = []
            for idx, book in qualified.head(n).iterrows():
                recommendations.append({
                    'title': book['Book-Title'],
                    'author': book['Book-Author'],
                    'rating': book['avg_rating'],
                    'sentiment': book['sentiment_score'],
                    'sentiment_category': book['sentiment_category'],
                    'confidence': book['recommendation_confidence']
                })
            return recommendations
        except Exception as e:
            print(f"Error in popular recommendations: {e}")
            return []
    
    def search_books(self, query, n=10):
        """Search for specific books"""
        try:
            result = self.ol_api.search_books(query, limit=n)
            
            if not result or 'docs' not in result:
                return []
            
            recommendations = []
            for doc in result['docs'][:n]:
                recommendations.append({
                    'title': doc.get('title', 'N/A'),
                    'author': doc.get('author_name', ['N/A'])[0] if doc.get('author_name') else 'N/A',
                    'year': doc.get('first_publish_year', 'N/A')
                })
            return recommendations
        except Exception as e:
            print(f"Error in search: {e}")
            return []

# Initialize engine
print("\n🔧 Initializing recommendation engine...")
rec_engine = BookRecommendationEngine(
    enriched_df=enriched_books_df,
    embeddings=embeddings
)
print("✅ Recommendation engine ready!")

# ==================== Helper Functions ====================

def format_recommendations(recommendations, limit=5):
    """Format recommendations for display"""
    if not recommendations:
        return "No recommendations found."
    
    formatted = ""
    for i, book in enumerate(recommendations[:limit], 1):
        title = book.get('title', 'N/A')
        author = book.get('author', 'N/A')
        
        formatted += f"**{i}. {title}**\n"
        formatted += f"   *by {author}*\n"
        
        if book.get('rating') and book['rating'] > 0:
            formatted += f"   📊 Rating: {book['rating']:.1f}/10\n"
        if book.get('sentiment') and book['sentiment'] > 0:
            formatted += f"   😊 Sentiment: {book['sentiment']:.2f}\n"
        if book.get('confidence') and book['confidence'] > 0:
            formatted += f"   ✨ Confidence: {book['confidence']:.2f}\n"
        if book.get('genre') and book['genre'] != 'N/A':
            formatted += f"   📚 Genre: {book['genre']}\n"
        if book.get('year') and book['year'] != 'N/A':
            formatted += f"   📅 Year: {book['year']}\n"
        if book.get('similarity'):
            formatted += f"   🔍 Similarity: {book['similarity']:.2f}\n"
        
        formatted += "\n"
    
    return formatted

def get_bot_response(user_message):
    """Generate bot response based on user message"""
    if not user_message or not user_message.strip():
        return "Please ask me something! For example: 'recommend fantasy books' or 'show me uplifting stories'"
    
    try:
        message_lower = user_message.lower()
        recommendations = []
        context = ""
        
        if any(word in message_lower for word in ['about', 'theme', 'deal with', 'explore', 'stories about']):
            recommendations = rec_engine.semantic_search(user_message, n=5)
            if recommendations:
                context = "📚 **Books matching your theme:**\n\n" + format_recommendations(recommendations)
            else:
                context = "I couldn't find books matching that theme. Try rephrasing or ask for a specific genre!"
        
        elif any(word in message_lower for word in ['positive', 'uplifting', 'happy', 'feel-good', 'inspiring']):
            recommendations = rec_engine.recommend_by_sentiment('Very Positive', n=5)
            if recommendations:
                context = "😊 **Uplifting books with great reviews:**\n\n" + format_recommendations(recommendations)
        
        elif any(word in message_lower for word in ['easy', 'simple', 'beginner']):
            recommendations = rec_engine.recommend_by_multiple_filters(
                reading_level='Easy', 
                min_confidence=0.5, 
                n=5
            )
            if recommendations:
                context = "📖 **Easy-to-read books:**\n\n" + format_recommendations(recommendations)
        
        elif any(word in message_lower for word in ['popular', 'best', 'top', 'trending', 'highly rated']):
            recommendations = rec_engine.recommend_popular(n=5, min_sentiment=0.6)
            if recommendations:
                context = "⭐ **Most popular books with great reviews:**\n\n" + format_recommendations(recommendations)
        
        elif any(genre in message_lower for genre in ['fantasy', 'sci-fi', 'science fiction', 'mystery', 'romance', 'thriller', 'horror', 'biography', 'history']):
            for genre in ['fantasy', 'science fiction', 'sci-fi', 'mystery', 'romance', 'thriller', 'horror', 'biography', 'history']:
                if genre in message_lower:
                    search_genre = 'science fiction' if genre == 'sci-fi' else genre
                    recommendations = rec_engine.recommend_by_genre(search_genre, n=5)
                    if recommendations:
                        context = f"📚 **{search_genre.title()} books:**\n\n" + format_recommendations(recommendations)
                    break
        
        else:
            recommendations = rec_engine.search_books(user_message, n=5)
            if recommendations:
                context = "🔍 **Search results:**\n\n" + format_recommendations(recommendations)
            else:
                recommendations = rec_engine.recommend_popular(n=5)
                context = "⭐ **Here are some popular books you might enjoy:**\n\n" + format_recommendations(recommendations)
        
        bot_response = context
        
        if HF_TOKEN and hf_client and context:
            try:
                system_prompt = """You are a friendly and enthusiastic book recommendation assistant. 
When given book recommendations, present them naturally and add brief, engaging commentary about why someone might enjoy them. 
Keep responses concise but warm. Don't just repeat the list - add personality and insights."""
                
                user_prompt = f"Based on these recommendations, respond to the user in a friendly way:\n\n{context}\n\nUser's request: {user_message}"
                
                response = hf_client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="mistralai/Mistral-7B-Instruct-v0.2",
                    max_tokens=500,
                    temperature=0.7
                )
                
                llm_response = response.choices[0].message.content
                bot_response = f"{llm_response}\n\n---\n\n{context}"
                
            except Exception as e:
                print(f"LLM error: {e}")
                bot_response = context
        
        return bot_response
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}\n\nTry asking: 'recommend fantasy books' or 'show me popular books'"
        print(f"Error in get_bot_response: {e}")
        return error_msg

# ==================== Gradio Interface ====================

custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
"""

with gr.Blocks(title="Book Recommendation Agent") as demo:
    gr.Markdown("""
    # 📚 Enhanced Book Recommendation Agent
    
    <div style='text-align: center; padding: 10px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 20px;'>
        <h3>AI-Powered Book Discovery System</h3>
        <p>Powered by ML, Sentiment Analysis & Semantic Search • 5,640+ Books • 90K+ Ratings</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                avatar_images=(
                    "https://api.dicebear.com/7.x/thumbs/svg?seed=user", 
                    "https://api.dicebear.com/7.x/bottts/svg?seed=bot"
                ),
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me for book recommendations... (e.g., 'uplifting fantasy books')",
                    show_label=False,
                    scale=9
                )
                submit = gr.Button("📤 Send", scale=1, variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Try These:")
            gr.Examples(
                examples=[
                    "Show me the best books",
                    "Popular science fiction",
                    "Easy mystery novels",
                ],
                inputs=msg,
            )
            
            gr.Markdown("""
            ### ✨ Features
            - 🎯 **Semantic Search** - Find books by theme
            - 😊 **Sentiment Analysis** - Filter by mood
            - 📊 **5,640 Enriched Books** - Full metadata
            - 🤖 **AI-Powered** - Natural conversations
            - 📚 **Multiple Genres** - 10+ categories
            
            ### 🎓 How It Works
            1. **Type your request** naturally
            2. **AI understands** your preferences  
            3. **ML finds** perfect matches
            4. **Get recommendations** with ratings
            """)
    
    gr.Markdown("""
    ---
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <strong>Tech Stack:</strong> Mistral-7B • Sentence Transformers • scikit-learn • Gradio<br>
        <strong>Data:</strong> Book-Crossing Dataset • Open Library API<br>
        Built with ❤️ for book lovers everywhere 📖
    </div>
    """)
    
    def respond(message, history):
        """Handle user message and generate response"""
        if history is None:
            history = []
        
        bot_message = get_bot_response(message)
        
        # use messages-style history even if older Gradio still accepts it
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_message})
        
        return "", history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=custom_css)
