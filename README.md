---
title: Book Recommendation Agent
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: apache-2.0
---

# 📚 Enhanced Book Recommendation Agent

An AI-powered book recommendation system that combines machine learning, sentiment analysis, and semantic search to help you discover your next great read.

## 🌟 Features

- **Semantic Search**: Find books by themes and concepts
- **Sentiment Analysis**: 5,640 books analyzed for sentiment
- **Genre Classification**: 10+ genres automatically detected
- **ML-Powered**: Trained on 90,000+ ratings
- **Conversational AI**: Natural language interface powered by Mistral-7B

## 🚀 How to Use

Simply chat with the agent! Try asking:
- "Show me uplifting science fiction books"
- "Books about overcoming adversity"
- "What are the best fantasy books?"
- "Positive mystery novels"

## 🛠️ Technical Stack

- **ML Models**: SVD, KNN (Collaborative Filtering)
- **Embeddings**: Sentence Transformers (384D)
- **LLM**: Mistral-7B-Instruct (HuggingFace Inference)
- **Data**: Book-Crossing Dataset + Open Library API
- **Framework**: Gradio

## 📊 Dataset

- 5,640 books with full enrichment
- 90,556 user ratings
- Sentiment scores for all books
- 20 topic clusters
- Reading level classifications

## 🔧 Configuration

This Space requires a `HF_TOKEN` secret for full LLM functionality.

## 📝 License

Apache 2.0