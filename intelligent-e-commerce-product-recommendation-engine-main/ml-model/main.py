#!/usr/bin/env python3
"""
Intelligent E-commerce Product Recommendation Engine - Main Entry Point
Author: Jessie Borras
Website: jessiedev.xyz
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.recommendation_engine import RecommendationEngine
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize components
data_processor = DataProcessor()
model_trainer = ModelTrainer()
recommendation_engine = RecommendationEngine()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "ML Recommendation Engine"})

@app.route('/recommendations/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    """
    Get personalized recommendations for a user
    """
    try:
        num_recommendations = int(request.args.get('count', 10))
        
        recommendations = recommendation_engine.get_recommendations(
            user_id=user_id,
            num_recommendations=num_recommendations
        )
        
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        })
    
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
        return jsonify({"error": "Failed to get recommendations"}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """
    Train or retrain the recommendation model
    """
    try:
        # Process training data
        processed_data = data_processor.process_training_data()
        
        # Train model
        model_trainer.train(processed_data)
        
        # Update recommendation engine with new model
        recommendation_engine.load_model()
        
        return jsonify({"message": "Model training completed successfully"})
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({"error": "Failed to train model"}), 500

@app.route('/similar-products/<product_id>', methods=['GET'])
def get_similar_products(product_id):
    """
    Get similar products based on content-based filtering
    """
    try:
        num_similar = int(request.args.get('count', 5))
        
        similar_products = recommendation_engine.get_similar_products(
            product_id=product_id,
            num_similar=num_similar
        )
        
        return jsonify({
            "product_id": product_id,
            "similar_products": similar_products,
            "count": len(similar_products)
        })
    
    except Exception as e:
        logger.error(f"Error getting similar products for {product_id}: {str(e)}")
        return jsonify({"error": "Failed to get similar products"}), 500

@app.route('/user-interaction', methods=['POST'])
def record_interaction():
    """
    Record user interaction for future model improvements
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['user_id', 'product_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Store interaction data
        data_processor.store_interaction(data)
        
        return jsonify({"message": "Interaction recorded successfully"})
    
    except Exception as e:
        logger.error(f"Error recording interaction: {str(e)}")
        return jsonify({"error": "Failed to record interaction"}), 500

if __name__ == '__main__':
    logger.info("Starting ML Recommendation Engine...")
    
    # Load existing model if available
    try:
        recommendation_engine.load_model()
        logger.info("Loaded existing model")
    except Exception as e:
        logger.warning(f"Could not load existing model: {str(e)}")
        logger.info("Will train new model when needed")
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)