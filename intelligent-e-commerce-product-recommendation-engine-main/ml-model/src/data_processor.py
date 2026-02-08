"""
Data Processor - Handle data loading, processing and storage
Author: Jessie Borras
Website: jessiedev.xyz
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, Tuple
import json
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handle data processing for the recommendation engine
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_user_interactions(self, file_path: str = None) -> pd.DataFrame:
        """Load user interaction data from file or database"""
        try:
            if file_path is None:
                file_path = os.path.join(self.data_dir, 'user_interactions.csv')
            
            if os.path.exists(file_path):
                interactions_df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(interactions_df)} user interactions")
                return interactions_df
            else:
                logger.warning(f"Interactions file {file_path} not found, creating empty dataset")
                return pd.DataFrame(columns=['user_id', 'product_id', 'rating', 'timestamp'])
                
        except Exception as e:
            logger.error(f"Error loading user interactions: {str(e)}")
            return pd.DataFrame(columns=['user_id', 'product_id', 'rating', 'timestamp'])
    
    def load_product_data(self, file_path: str = None) -> pd.DataFrame:
        """Load product metadata"""
        try:
            if file_path is None:
                file_path = os.path.join(self.data_dir, 'products.csv')
            
            if os.path.exists(file_path):
                products_df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(products_df)} products")
                return products_df
            else:
                logger.warning(f"Products file {file_path} not found, creating sample data")
                return self._create_sample_product_data()
                
        except Exception as e:
            logger.error(f"Error loading product data: {str(e)}")
            return self._create_sample_product_data()
    
    def _create_sample_product_data(self) -> pd.DataFrame:
        """Create sample product data for testing"""
        sample_products = [
            {
                'product_id': 'prod_001',
                'name': 'Wireless Bluetooth Headphones',
                'category': 'Electronics',
                'brand': 'TechBrand',
                'description': 'High-quality wireless headphones with noise cancellation',
                'price': 149.99
            },
            {
                'product_id': 'prod_002',
                'name': 'Running Shoes',
                'category': 'Sports',
                'brand': 'SportsBrand',
                'description': 'Comfortable running shoes for daily exercise',
                'price': 89.99
            },
            {
                'product_id': 'prod_003',
                'name': 'Coffee Maker',
                'category': 'Kitchen',
                'brand': 'KitchenBrand',
                'description': 'Automatic coffee maker with programmable features',
                'price': 79.99
            },
            {
                'product_id': 'prod_004',
                'name': 'Smartphone',
                'category': 'Electronics',
                'brand': 'TechBrand',
                'description': 'Latest smartphone with advanced camera and features',
                'price': 699.99
            },
            {
                'product_id': 'prod_005',
                'name': 'Yoga Mat',
                'category': 'Sports',
                'brand': 'FitnessBrand',
                'description': 'Non-slip yoga mat for workout and meditation',
                'price': 29.99
            }
        ]
        
        return pd.DataFrame(sample_products)
    
    def create_user_item_matrix(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction matrix from interactions data"""
        try:
            # Handle different interaction types (view, purchase, rating)
            if 'rating' in interactions_df.columns:
                # Use explicit ratings
                matrix_data = interactions_df.groupby(['user_id', 'product_id'])['rating'].mean().unstack(fill_value=0)
            else:
                # Create implicit ratings based on interaction frequency
                interaction_counts = interactions_df.groupby(['user_id', 'product_id']).size().reset_index(name='count')
                # Convert counts to ratings (log scale for better distribution)
                interaction_counts['implicit_rating'] = np.log1p(interaction_counts['count'])
                matrix_data = interaction_counts.pivot(
                    index='user_id', 
                    columns='product_id', 
                    values='implicit_rating'
                ).fillna(0)
            
            logger.info(f"Created user-item matrix with {matrix_data.shape[0]} users and {matrix_data.shape[1]} items")
            return matrix_data
            
        except Exception as e:
            logger.error(f"Error creating user-item matrix: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_product_features(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess product features for content-based filtering"""
        try:
            # Clean and standardize text features
            text_columns = ['name', 'description', 'category', 'brand']
            for col in text_columns:
                if col in products_df.columns:
                    products_df[col] = products_df[col].astype(str).str.lower().str.strip()
            
            # Handle missing values
            products_df = products_df.fillna({
                'name': 'unknown',
                'description': 'no description',
                'category': 'other',
                'brand': 'unknown',
                'price': products_df['price'].median() if 'price' in products_df.columns else 0
            })
            
            logger.info(f"Preprocessed {len(products_df)} product features")
            return products_df
            
        except Exception as e:
            logger.error(f"Error preprocessing product features: {str(e)}")
            return products_df
    
    def process_training_data(self) -> Dict[str, Any]:
        """Process all data needed for model training"""
        try:
            # Load raw data
            interactions_df = self.load_user_interactions()
            products_df = self.load_product_data()
            
            # Preprocess data
            products_df = self.preprocess_product_features(products_df)
            user_item_matrix = self.create_user_item_matrix(interactions_df)
            
            # Additional processing for cold start problem
            popular_products = self._get_popular_products(interactions_df)
            
            processed_data = {
                'user_item_matrix': user_item_matrix,
                'products_df': products_df,
                'interactions_df': interactions_df,
                'popular_products': popular_products
            }
            
            logger.info("Successfully processed training data")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing training data: {str(e)}")
            return {}
    
    def _get_popular_products(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Get popular products for handling cold start problem"""
        try:
            if interactions_df.empty:
                return pd.DataFrame(columns=['product_id', 'popularity_score'])
            
            # Calculate popularity based on interaction frequency
            popularity = interactions_df.groupby('product_id').size().reset_index(name='interaction_count')
            
            # Add rating-based popularity if ratings are available
            if 'rating' in interactions_df.columns:
                avg_ratings = interactions_df.groupby('product_id')['rating'].mean().reset_index(name='avg_rating')
                popularity = popularity.merge(avg_ratings, on='product_id', how='left')
                # Combined popularity score
                popularity['popularity_score'] = (
                    popularity['interaction_count'] * 0.7 + 
                    popularity['avg_rating'].fillna(3.0) * 0.3
                )
            else:
                popularity['popularity_score'] = popularity['interaction_count']
            
            # Normalize popularity scores
            max_score = popularity['popularity_score'].max()
            if max_score > 0:
                popularity['popularity_score'] = popularity['popularity_score'] / max_score
            
            return popularity.sort_values('popularity_score', ascending=False)
            
        except Exception as e:
            logger.error(f"Error calculating popular products: {str(e)}")
            return pd.DataFrame(columns=['product_id', 'popularity_score'])
    
    def store_interaction(self, interaction_data: Dict[str, Any]):
        """Store a single user interaction"""
        try:
            # Add timestamp if not provided
            if 'timestamp' not in interaction_data:
                interaction_data['timestamp'] = datetime.now().isoformat()
            
            # Convert interaction type to rating if needed
            if 'rating' not in interaction_data and 'interaction_type' in interaction_data:
                interaction_type_mapping = {
                    'view': 1,
                    'add_to_cart': 2,
                    'purchase': 5,
                    'like': 4,
                    'share': 3
                }
                interaction_data['rating'] = interaction_type_mapping.get(
                    interaction_data['interaction_type'], 1
                )
            
            # Load existing interactions
            interactions_file = os.path.join(self.data_dir, 'user_interactions.csv')
            
            if os.path.exists(interactions_file):
                existing_interactions = pd.read_csv(interactions_file)
            else:
                existing_interactions = pd.DataFrame(columns=[
                    'user_id', 'product_id', 'rating', 'timestamp', 'interaction_type'
                ])
            
            # Add new interaction
            new_interaction = pd.DataFrame([interaction_data])
            updated_interactions = pd.concat([existing_interactions, new_interaction], ignore_index=True)
            
            # Save updated interactions
            updated_interactions.to_csv(interactions_file, index=False)
            
            logger.info(f"Stored interaction for user {interaction_data['user_id']} and product {interaction_data['product_id']}")
            
        except Exception as e:
            logger.error(f"Error storing interaction: {str(e)}")
    
    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """Get interaction history for a specific user"""
        try:
            interactions_df = self.load_user_interactions()
            user_history = interactions_df[interactions_df['user_id'] == user_id]
            
            logger.info(f"Retrieved {len(user_history)} interactions for user {user_id}")
            return user_history
            
        except Exception as e:
            logger.error(f"Error getting user history: {str(e)}")
            return pd.DataFrame()
    
    def get_product_stats(self, product_id: str) -> Dict[str, Any]:
        """Get statistics for a specific product"""
        try:
            interactions_df = self.load_user_interactions()
            product_interactions = interactions_df[interactions_df['product_id'] == product_id]
            
            if product_interactions.empty:
                return {'interaction_count': 0, 'average_rating': 0}
            
            stats = {
                'interaction_count': len(product_interactions),
                'unique_users': product_interactions['user_id'].nunique(),
                'average_rating': product_interactions['rating'].mean() if 'rating' in product_interactions.columns else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting product stats: {str(e)}")
            return {'interaction_count': 0, 'average_rating': 0}
    
    def clean_old_interactions(self, days_to_keep: int = 365):
        """Clean old interaction data to manage storage"""
        try:
            interactions_df = self.load_user_interactions()
            
            if 'timestamp' in interactions_df.columns:
                # Convert timestamp to datetime
                interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
                
                # Keep only recent interactions
                cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
                recent_interactions = interactions_df[interactions_df['timestamp'] >= cutoff_date]
                
                # Save cleaned data
                interactions_file = os.path.join(self.data_dir, 'user_interactions.csv')
                recent_interactions.to_csv(interactions_file, index=False)
                
                removed_count = len(interactions_df) - len(recent_interactions)
                logger.info(f"Cleaned {removed_count} old interactions, kept {len(recent_interactions)} recent ones")
            
        except Exception as e:
            logger.error(f"Error cleaning old interactions: {str(e)}")