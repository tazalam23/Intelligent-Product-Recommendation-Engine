"""
Model Trainer - Train and evaluate recommendation models
Author: Jessie Borras
Website: jessiedev.xyz
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Dict, Any, Tuple
from .recommendation_engine import RecommendationEngine
import joblib
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Train and evaluate recommendation models
    """
    
    def __init__(self):
        self.recommendation_engine = RecommendationEngine()
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train recommendation models on processed data
        """
        try:
            logger.info("Starting model training...")
            
            user_item_matrix = processed_data.get('user_item_matrix')
            products_df = processed_data.get('products_df')
            
            if user_item_matrix is None or user_item_matrix.empty:
                logger.warning("No user-item matrix available, skipping collaborative filtering training")
                collaborative_metrics = {}
            else:
                # Train collaborative filtering model
                logger.info("Training collaborative filtering model...")
                collaborative_metrics = self._train_collaborative_filtering(user_item_matrix)
            
            if products_df is None or products_df.empty:
                logger.warning("No product data available, skipping content-based filtering training")
                content_metrics = {}
            else:
                # Train content-based filtering model
                logger.info("Training content-based filtering model...")
                content_metrics = self._train_content_based_filtering(products_df)
            
            # Save trained models
            self.recommendation_engine.save_model()
            
            # Combine metrics
            training_metrics = {
                'collaborative_filtering': collaborative_metrics,
                'content_based_filtering': content_metrics,
                'training_status': 'completed'
            }
            
            logger.info("Model training completed successfully")
            return training_metrics
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return {'training_status': 'failed', 'error': str(e)}
    
    def _train_collaborative_filtering(self, user_item_matrix: pd.DataFrame) -> Dict[str, float]:
        """Train collaborative filtering model and return metrics"""
        try:
            if user_item_matrix.empty or user_item_matrix.shape[0] < 2 or user_item_matrix.shape[1] < 2:
                logger.warning("Insufficient data for collaborative filtering")
                return {}
            
            # Split data for evaluation
            train_matrix, test_matrix = self._split_user_item_matrix(user_item_matrix)
            
            # Train the collaborative filtering model
            self.recommendation_engine.train_collaborative_filtering(train_matrix)
            
            # Evaluate model performance
            metrics = self._evaluate_collaborative_model(train_matrix, test_matrix)
            
            logger.info(f"Collaborative filtering training completed. RMSE: {metrics.get('rmse', 'N/A')}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering: {str(e)}")
            return {'error': str(e)}
    
    def _train_content_based_filtering(self, products_df: pd.DataFrame) -> Dict[str, float]:
        """Train content-based filtering model and return metrics"""
        try:
            if products_df.empty:
                logger.warning("No product data available for content-based filtering")
                return {}
            
            # Train the content-based filtering model
            self.recommendation_engine.train_content_based_filtering(products_df)
            
            # Evaluate content-based model (basic metrics)
            metrics = self._evaluate_content_model(products_df)
            
            logger.info(f"Content-based filtering training completed. Products processed: {len(products_df)}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training content-based filtering: {str(e)}")
            return {'error': str(e)}
    
    def _split_user_item_matrix(self, user_item_matrix: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split user-item matrix for training and testing"""
        try:
            # Create masks for training and testing
            train_matrix = user_item_matrix.copy()
            test_matrix = user_item_matrix.copy()
            
            # For each user, randomly select some ratings for testing
            np.random.seed(42)
            
            for user_id in user_item_matrix.index:
                user_ratings = user_item_matrix.loc[user_id]
                non_zero_indices = user_ratings[user_ratings > 0].index.tolist()
                
                if len(non_zero_indices) > 1:
                    # Randomly select test indices
                    n_test = max(1, int(len(non_zero_indices) * test_size))
                    test_indices = np.random.choice(non_zero_indices, size=n_test, replace=False)
                    
                    # Set test ratings to 0 in training matrix
                    train_matrix.loc[user_id, test_indices] = 0
                    
                    # Keep only test ratings in test matrix, set others to 0
                    test_mask = test_matrix.columns.isin(test_indices)
                    test_matrix.loc[user_id, ~test_mask] = 0
                else:
                    # If user has only one rating, keep it in training
                    test_matrix.loc[user_id, :] = 0
            
            return train_matrix, test_matrix
            
        except Exception as e:
            logger.error(f"Error splitting user-item matrix: {str(e)}")
            return user_item_matrix, pd.DataFrame()
    
    def _evaluate_collaborative_model(self, train_matrix: pd.DataFrame, test_matrix: pd.DataFrame) -> Dict[str, float]:
        """Evaluate collaborative filtering model performance"""
        try:
            if test_matrix.empty:
                return {}
            
            predictions = []
            actuals = []
            
            # Get predictions for test users
            for user_id in test_matrix.index:
                test_ratings = test_matrix.loc[user_id]
                test_items = test_ratings[test_ratings > 0].index.tolist()
                
                if not test_items:
                    continue
                
                # Get recommendations for this user
                user_recommendations = self.recommendation_engine.get_collaborative_recommendations(
                    user_id, num_recommendations=len(test_items) * 2
                )
                
                # Match predictions with actual test ratings
                for rec in user_recommendations:
                    product_id = rec['product_id']
                    if product_id in test_items:
                        predictions.append(rec['score'])
                        actuals.append(test_ratings[product_id])
            
            if len(predictions) == 0:
                logger.warning("No predictions available for evaluation")
                return {}
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Normalize predictions to rating scale (assuming 1-5)
            if len(predictions) > 0:
                pred_min, pred_max = predictions.min(), predictions.max()
                if pred_max > pred_min:
                    predictions = 1 + 4 * (predictions - pred_min) / (pred_max - pred_min)
            
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            # Coverage (percentage of items that can be recommended)
            total_items = len(train_matrix.columns)
            recommended_items = len(set(rec['product_id'] for rec in user_recommendations))
            coverage = recommended_items / total_items if total_items > 0 else 0
            
            metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'coverage': float(coverage),
                'n_predictions': len(predictions)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating collaborative model: {str(e)}")
            return {'error': str(e)}
    
    def _evaluate_content_model(self, products_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate content-based filtering model"""
        try:
            # Basic metrics for content-based model
            metrics = {
                'products_processed': len(products_df),
                'features_created': 1 if self.recommendation_engine.content_model is not None else 0
            }
            
            # Test similarity calculation for a sample product
            if len(products_df) > 1:
                sample_product_id = products_df['product_id'].iloc[0]
                similar_products = self.recommendation_engine.get_similar_products(
                    sample_product_id, num_similar=min(5, len(products_df) - 1)
                )
                
                metrics['similarity_test_passed'] = len(similar_products) > 0
                metrics['avg_similarity_score'] = np.mean([
                    p['similarity_score'] for p in similar_products
                ]) if similar_products else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating content model: {str(e)}")
            return {'error': str(e)}
    
    def cross_validate(self, processed_data: Dict[str, Any], k_folds: int = 5) -> Dict[str, Any]:
        """Perform k-fold cross-validation"""
        try:
            user_item_matrix = processed_data.get('user_item_matrix')
            
            if user_item_matrix is None or user_item_matrix.empty:
                logger.warning("No data available for cross-validation")
                return {}
            
            logger.info(f"Starting {k_folds}-fold cross-validation...")
            
            fold_metrics = []
            users = list(user_item_matrix.index)
            np.random.shuffle(users)
            fold_size = len(users) // k_folds
            
            for fold in range(k_folds):
                logger.info(f"Processing fold {fold + 1}/{k_folds}")
                
                # Split users into train and validation sets
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < k_folds - 1 else len(users)
                val_users = users[val_start:val_end]
                train_users = [u for u in users if u not in val_users]
                
                # Create train and validation matrices
                train_fold = user_item_matrix.loc[train_users]
                val_fold = user_item_matrix.loc[val_users]
                
                # Train model on fold
                temp_engine = RecommendationEngine()
                temp_engine.train_collaborative_filtering(train_fold)
                
                # Evaluate on validation set
                fold_rmse = self._evaluate_fold(temp_engine, train_fold, val_fold)
                fold_metrics.append(fold_rmse)
            
            # Calculate average metrics
            avg_rmse = np.mean(fold_metrics)
            std_rmse = np.std(fold_metrics)
            
            cv_results = {
                'cv_rmse_mean': float(avg_rmse),
                'cv_rmse_std': float(std_rmse),
                'cv_folds': k_folds,
                'fold_scores': [float(score) for score in fold_metrics]
            }
            
            logger.info(f"Cross-validation completed. Average RMSE: {avg_rmse:.4f} ± {std_rmse:.4f}")
            return cv_results
            
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}")
            return {'error': str(e)}
    
    def _evaluate_fold(self, model_engine: RecommendationEngine, train_data: pd.DataFrame, val_data: pd.DataFrame) -> float:
        """Evaluate a single fold"""
        try:
            predictions = []
            actuals = []
            
            for user_id in val_data.index:
                if user_id in train_data.index:
                    # Get user's actual ratings in validation set
                    actual_ratings = val_data.loc[user_id]
                    actual_items = actual_ratings[actual_ratings > 0].index.tolist()
                    
                    if not actual_items:
                        continue
                    
                    # Get recommendations
                    recommendations = model_engine.get_collaborative_recommendations(
                        user_id, num_recommendations=len(actual_items) * 2
                    )
                    
                    # Match predictions with actuals
                    for rec in recommendations:
                        if rec['product_id'] in actual_items:
                            predictions.append(rec['score'])
                            actuals.append(actual_ratings[rec['product_id']])
            
            if len(predictions) > 0:
                # Normalize predictions
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                pred_min, pred_max = predictions.min(), predictions.max()
                if pred_max > pred_min:
                    predictions = 1 + 4 * (predictions - pred_min) / (pred_max - pred_min)
                
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                return rmse
            
            return float('inf')  # Return high error if no predictions
            
        except Exception as e:
            logger.error(f"Error evaluating fold: {str(e)}")
            return float('inf')
    
    def hyperparameter_tuning(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        try:
            user_item_matrix = processed_data.get('user_item_matrix')
            
            if user_item_matrix is None or user_item_matrix.empty:
                logger.warning("No data available for hyperparameter tuning")
                return {}
            
            logger.info("Starting hyperparameter tuning...")
            
            # Parameters to tune
            n_components_options = [20, 50, 100]
            n_neighbors_options = [10, 20, 50]
            
            best_params = {}
            best_score = float('inf')
            results = []
            
            for n_components in n_components_options:
                for n_neighbors in n_neighbors_options:
                    logger.info(f"Testing n_components={n_components}, n_neighbors={n_neighbors}")
                    
                    # Create temporary engine with these parameters
                    temp_engine = RecommendationEngine()
                    temp_engine.n_components = n_components
                    temp_engine.n_neighbors = n_neighbors
                    
                    # Perform cross-validation
                    cv_results = self._quick_cv(temp_engine, user_item_matrix)
                    avg_score = cv_results.get('avg_rmse', float('inf'))
                    
                    results.append({
                        'n_components': n_components,
                        'n_neighbors': n_neighbors,
                        'avg_rmse': avg_score
                    })
                    
                    if avg_score < best_score:
                        best_score = avg_score
                        best_params = {
                            'n_components': n_components,
                            'n_neighbors': n_neighbors
                        }
            
            # Update recommendation engine with best parameters
            self.recommendation_engine.n_components = best_params.get('n_components', 50)
            self.recommendation_engine.n_neighbors = best_params.get('n_neighbors', 20)
            
            tuning_results = {
                'best_params': best_params,
                'best_score': float(best_score),
                'all_results': results
            }
            
            logger.info(f"Hyperparameter tuning completed. Best params: {best_params}, Best RMSE: {best_score:.4f}")
            return tuning_results
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            return {'error': str(e)}
    
    def _quick_cv(self, model_engine: RecommendationEngine, user_item_matrix: pd.DataFrame, k_folds: int = 3) -> Dict[str, float]:
        """Quick cross-validation for hyperparameter tuning"""
        try:
            users = list(user_item_matrix.index)
            np.random.shuffle(users)
            fold_size = len(users) // k_folds
            
            fold_scores = []
            
            for fold in range(k_folds):
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < k_folds - 1 else len(users)
                val_users = users[val_start:val_end]
                train_users = [u for u in users if u not in val_users]
                
                train_fold = user_item_matrix.loc[train_users]
                val_fold = user_item_matrix.loc[val_users]
                
                model_engine.train_collaborative_filtering(train_fold)
                fold_score = self._evaluate_fold(model_engine, train_fold, val_fold)
                fold_scores.append(fold_score)
            
            avg_rmse = np.mean([score for score in fold_scores if score != float('inf')])
            
            return {'avg_rmse': float(avg_rmse)}
            
        except Exception as e:
            logger.error(f"Error in quick cross-validation: {str(e)}")
            return {'avg_rmse': float('inf')}