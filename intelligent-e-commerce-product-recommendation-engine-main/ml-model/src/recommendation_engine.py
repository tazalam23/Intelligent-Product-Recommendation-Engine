"""
Recommendation Engine - Core ML Logic
Author: Jessie Borras
Website: jessiedev.xyz
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import joblib
import os
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Hybrid recommendation engine using both collaborative and content-based filtering
    """
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.collaborative_model = None
        self.content_model = None
        self.tfidf_vectorizer = None
        self.product_features = None
        self.user_item_matrix = None
        self.product_metadata = None
        self.user_profiles = None
        
        # Model parameters
        self.n_components = 50  # For SVD dimensionality reduction
        self.n_neighbors = 20   # For k-NN collaborative filtering
        
    def load_model(self):
        """Load trained models from disk"""
        try:
            # Load collaborative filtering model
            collaborative_path = os.path.join(self.model_dir, 'collaborative_model.pkl')
            if os.path.exists(collaborative_path):
                self.collaborative_model = joblib.load(collaborative_path)
                logger.info("Loaded collaborative filtering model")
            
            # Load content-based model
            content_path = os.path.join(self.model_dir, 'content_model.pkl')
            if os.path.exists(content_path):
                self.content_model = joblib.load(content_path)
                logger.info("Loaded content-based model")
            
            # Load TF-IDF vectorizer
            tfidf_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
            if os.path.exists(tfidf_path):
                self.tfidf_vectorizer = joblib.load(tfidf_path)
                logger.info("Loaded TF-IDF vectorizer")
            
            # Load product features
            features_path = os.path.join(self.model_dir, 'product_features.pkl')
            if os.path.exists(features_path):
                self.product_features = joblib.load(features_path)
                logger.info("Loaded product features")
            
            # Load user-item matrix
            matrix_path = os.path.join(self.model_dir, 'user_item_matrix.pkl')
            if os.path.exists(matrix_path):
                self.user_item_matrix = joblib.load(matrix_path)
                logger.info("Loaded user-item matrix")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def save_model(self):
        """Save trained models to disk"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            
            if self.collaborative_model is not None:
                joblib.dump(self.collaborative_model, 
                           os.path.join(self.model_dir, 'collaborative_model.pkl'))
            
            if self.content_model is not None:
                joblib.dump(self.content_model, 
                           os.path.join(self.model_dir, 'content_model.pkl'))
            
            if self.tfidf_vectorizer is not None:
                joblib.dump(self.tfidf_vectorizer, 
                           os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'))
            
            if self.product_features is not None:
                joblib.dump(self.product_features, 
                           os.path.join(self.model_dir, 'product_features.pkl'))
            
            if self.user_item_matrix is not None:
                joblib.dump(self.user_item_matrix, 
                           os.path.join(self.model_dir, 'user_item_matrix.pkl'))
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def train_collaborative_filtering(self, user_item_matrix: pd.DataFrame):
        """Train collaborative filtering model using SVD and k-NN"""
        try:
            # Store user-item matrix
            self.user_item_matrix = user_item_matrix
            
            # Apply SVD for dimensionality reduction
            svd = TruncatedSVD(n_components=self.n_components, random_state=42)
            user_features = svd.fit_transform(user_item_matrix)
            
            # Train k-NN model for finding similar users
            knn_model = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                metric='cosine',
                algorithm='brute'
            )
            knn_model.fit(user_features)
            
            self.collaborative_model = {
                'svd': svd,
                'knn': knn_model,
                'user_features': user_features
            }
            
            logger.info("Collaborative filtering model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering: {str(e)}")
            raise
    
    def train_content_based_filtering(self, product_data: pd.DataFrame):
        """Train content-based filtering using TF-IDF and cosine similarity"""
        try:
            # Create product feature text by combining relevant columns
            product_data['combined_features'] = (
                product_data['name'].fillna('') + ' ' +
                product_data['category'].fillna('') + ' ' +
                product_data['description'].fillna('') + ' ' +
                product_data['brand'].fillna('')
            )
            
            # Create TF-IDF vectors
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                product_data['combined_features']
            )
            
            # Compute cosine similarity matrix
            content_similarity = cosine_similarity(tfidf_matrix)
            
            self.content_model = content_similarity
            self.product_features = {
                'tfidf_matrix': tfidf_matrix,
                'product_ids': product_data['product_id'].tolist(),
                'product_data': product_data
            }
            
            logger.info("Content-based filtering model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training content-based filtering: {str(e)}")
            raise
    
    def get_collaborative_recommendations(self, user_id: str, num_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using collaborative filtering"""
        try:
            if self.collaborative_model is None or self.user_item_matrix is None:
                logger.warning("Collaborative model not available")
                return []
            
            # Check if user exists in the matrix
            if user_id not in self.user_item_matrix.index:
                logger.warning(f"User {user_id} not found in training data")
                return []
            
            # Get user index
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            user_vector = self.collaborative_model['user_features'][user_idx].reshape(1, -1)
            
            # Find similar users
            distances, indices = self.collaborative_model['knn'].kneighbors(user_vector)
            similar_users = indices[0][1:]  # Exclude the user themselves
            
            # Get recommendations based on similar users' preferences
            recommendations = {}
            user_rated_items = set(self.user_item_matrix.loc[user_id][
                self.user_item_matrix.loc[user_id] > 0
            ].index)
            
            for similar_user_idx in similar_users:
                similar_user_id = self.user_item_matrix.index[similar_user_idx]
                similar_user_items = self.user_item_matrix.loc[similar_user_id][
                    self.user_item_matrix.loc[similar_user_id] > 0
                ]
                
                for item, rating in similar_user_items.items():
                    if item not in user_rated_items:
                        if item not in recommendations:
                            recommendations[item] = []
                        recommendations[item].append(rating)
            
            # Calculate average scores and sort
            final_recommendations = []
            for item, ratings in recommendations.items():
                avg_score = np.mean(ratings)
                final_recommendations.append({
                    'product_id': item,
                    'score': avg_score,
                    'method': 'collaborative'
                })
            
            # Sort by score and return top recommendations
            final_recommendations.sort(key=lambda x: x['score'], reverse=True)
            return final_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {str(e)}")
            return []
    
    def get_content_recommendations(self, user_id: str, num_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using content-based filtering"""
        try:
            if self.content_model is None or self.product_features is None:
                logger.warning("Content-based model not available")
                return []
            
            # Get user's purchase history (this would come from your data)
            # For now, we'll use a placeholder
            user_liked_products = self._get_user_liked_products(user_id)
            
            if not user_liked_products:
                return []
            
            # Calculate content-based recommendations
            product_ids = self.product_features['product_ids']
            recommendations = {}
            
            for liked_product in user_liked_products:
                if liked_product in product_ids:
                    product_idx = product_ids.index(liked_product)
                    similarities = self.content_model[product_idx]
                    
                    for idx, similarity in enumerate(similarities):
                        product_id = product_ids[idx]
                        if product_id != liked_product and product_id not in user_liked_products:
                            if product_id not in recommendations:
                                recommendations[product_id] = []
                            recommendations[product_id].append(similarity)
            
            # Calculate average similarity scores
            final_recommendations = []
            for product_id, similarities in recommendations.items():
                avg_score = np.mean(similarities)
                final_recommendations.append({
                    'product_id': product_id,
                    'score': avg_score,
                    'method': 'content'
                })
            
            # Sort by score and return top recommendations
            final_recommendations.sort(key=lambda x: x['score'], reverse=True)
            return final_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting content recommendations: {str(e)}")
            return []
    
    def get_recommendations(self, user_id: str, num_recommendations: int = 10) -> List[Dict]:
        """
        Get hybrid recommendations combining collaborative and content-based filtering
        """
        try:
            # Get recommendations from both methods
            collaborative_recs = self.get_collaborative_recommendations(
                user_id, num_recommendations * 2
            )
            content_recs = self.get_content_recommendations(
                user_id, num_recommendations * 2
            )
            
            # Combine and weight the recommendations
            combined_recommendations = {}
            
            # Weight collaborative filtering recommendations
            for rec in collaborative_recs:
                product_id = rec['product_id']
                if product_id not in combined_recommendations:
                    combined_recommendations[product_id] = {'scores': [], 'methods': []}
                combined_recommendations[product_id]['scores'].append(rec['score'] * 0.6)
                combined_recommendations[product_id]['methods'].append('collaborative')
            
            # Weight content-based recommendations
            for rec in content_recs:
                product_id = rec['product_id']
                if product_id not in combined_recommendations:
                    combined_recommendations[product_id] = {'scores': [], 'methods': []}
                combined_recommendations[product_id]['scores'].append(rec['score'] * 0.4)
                combined_recommendations[product_id]['methods'].append('content')
            
            # Calculate final scores
            final_recommendations = []
            for product_id, data in combined_recommendations.items():
                final_score = sum(data['scores'])
                methods_used = list(set(data['methods']))
                
                final_recommendations.append({
                    'product_id': product_id,
                    'score': final_score,
                    'methods': methods_used
                })
            
            # Sort by final score and return top recommendations
            final_recommendations.sort(key=lambda x: x['score'], reverse=True)
            return final_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting hybrid recommendations: {str(e)}")
            return []
    
    def get_similar_products(self, product_id: str, num_similar: int = 5) -> List[Dict]:
        """Get similar products using content-based filtering"""
        try:
            if self.content_model is None or self.product_features is None:
                return []
            
            product_ids = self.product_features['product_ids']
            
            if product_id not in product_ids:
                logger.warning(f"Product {product_id} not found")
                return []
            
            product_idx = product_ids.index(product_id)
            similarities = self.content_model[product_idx]
            
            # Get similar products
            similar_indices = np.argsort(similarities)[::-1][1:num_similar+1]  # Exclude self
            
            similar_products = []
            for idx in similar_indices:
                similar_product_id = product_ids[idx]
                similarity_score = similarities[idx]
                
                similar_products.append({
                    'product_id': similar_product_id,
                    'similarity_score': float(similarity_score)
                })
            
            return similar_products
            
        except Exception as e:
            logger.error(f"Error getting similar products: {str(e)}")
            return []
    
    def _get_user_liked_products(self, user_id: str) -> List[str]:
        """
        Get products that the user has liked/purchased
        This is a placeholder - in a real implementation, this would query your database
        """
        # This would be replaced with actual database query
        # For now, return empty list
        return []