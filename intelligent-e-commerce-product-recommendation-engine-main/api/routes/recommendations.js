/**
 * Recommendations Routes
 * Author: Jessie Borras
 * Website: jessiedev.xyz
 */

const express = require('express');
const axios = require('axios');
const { body, param, query, validationResult } = require('express-validator');
const auth = require('../middleware/auth');
const User = require('../models/User');
const Product = require('../models/Product');
const Interaction = require('../models/Interaction');

const router = express.Router();

// ML service configuration
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5000';

/**
 * Get personalized recommendations for a user
 * GET /api/recommendations/:userId
 */
router.get('/:userId', [
  param('userId').isMongoId().withMessage('Invalid user ID'),
  query('count').optional().isInt({ min: 1, max: 50 }).withMessage('Count must be between 1 and 50'),
  query('type').optional().isIn(['collaborative', 'content', 'hybrid']).withMessage('Invalid recommendation type')
], async (req, res) => {
  try {
    // Check validation errors
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        error: 'Validation failed',
        details: errors.array()
      });
    }

    const { userId } = req.params;
    const { count = 10, type = 'hybrid' } = req.query;

    // Verify user exists
    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    // Get recommendations from ML service
    const mlResponse = await axios.get(`${ML_SERVICE_URL}/recommendations/${userId}`, {
      params: { count, type },
      timeout: 10000
    });

    const recommendations = mlResponse.data.recommendations || [];

    // Enrich recommendations with product details
    const enrichedRecommendations = await Promise.all(
      recommendations.map(async (rec) => {
        try {
          const product = await Product.findOne({ productId: rec.product_id });
          return {
            ...rec,
            product: product ? {
              id: product._id,
              name: product.name,
              category: product.category,
              price: product.price,
              image: product.image,
              rating: product.averageRating,
              inStock: product.stock > 0
            } : null
          };
        } catch (error) {
          console.error(`Error enriching recommendation for product ${rec.product_id}:`, error);
          return rec;
        }
      })
    );

    // Filter out recommendations without product data
    const validRecommendations = enrichedRecommendations.filter(rec => rec.product !== null);

    // Update user's last recommendation timestamp
    await User.findByIdAndUpdate(userId, {
      lastRecommendationAt: new Date()
    });

    res.json({
      userId,
      recommendations: validRecommendations,
      count: validRecommendations.length,
      type,
      generatedAt: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error getting recommendations:', error);
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: 'Recommendation service unavailable',
        message: 'The ML service is currently unavailable. Please try again later.'
      });
    }

    res.status(500).json({
      error: 'Failed to get recommendations',
      message: 'An unexpected error occurred while fetching recommendations'
    });
  }
});

/**
 * Get similar products based on a product
 * GET /api/recommendations/similar/:productId
 */
router.get('/similar/:productId', [
  param('productId').notEmpty().withMessage('Product ID is required'),
  query('count').optional().isInt({ min: 1, max: 20 }).withMessage('Count must be between 1 and 20')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        error: 'Validation failed',
        details: errors.array()
      });
    }

    const { productId } = req.params;
    const { count = 5 } = req.query;

    // Verify product exists
    const product = await Product.findOne({ productId });
    if (!product) {
      return res.status(404).json({ error: 'Product not found' });
    }

    // Get similar products from ML service
    const mlResponse = await axios.get(`${ML_SERVICE_URL}/similar-products/${productId}`, {
      params: { count },
      timeout: 10000
    });

    const similarProducts = mlResponse.data.similar_products || [];

    // Enrich with product details
    const enrichedSimilarProducts = await Promise.all(
      similarProducts.map(async (similar) => {
        try {
          const similarProduct = await Product.findOne({ productId: similar.product_id });
          return {
            ...similar,
            product: similarProduct ? {
              id: similarProduct._id,
              productId: similarProduct.productId,
              name: similarProduct.name,
              category: similarProduct.category,
              price: similarProduct.price,
              image: similarProduct.image,
              rating: similarProduct.averageRating,
              inStock: similarProduct.stock > 0
            } : null
          };
        } catch (error) {
          console.error(`Error enriching similar product ${similar.product_id}:`, error);
          return similar;
        }
      })
    );

    const validSimilarProducts = enrichedSimilarProducts.filter(item => item.product !== null);

    res.json({
      productId,
      baseProduct: {
        id: product._id,
        productId: product.productId,
        name: product.name,
        category: product.category
      },
      similarProducts: validSimilarProducts,
      count: validSimilarProducts.length
    });

  } catch (error) {
    console.error('Error getting similar products:', error);
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: 'Recommendation service unavailable',
        message: 'The ML service is currently unavailable. Please try again later.'
      });
    }

    res.status(500).json({
      error: 'Failed to get similar products',
      message: 'An unexpected error occurred while fetching similar products'
    });
  }
});

/**
 * Get trending/popular products
 * GET /api/recommendations/trending
 */
router.get('/trending', [
  query('count').optional().isInt({ min: 1, max: 50 }).withMessage('Count must be between 1 and 50'),
  query('category').optional().isString().withMessage('Category must be a string'),
  query('timeframe').optional().isIn(['day', 'week', 'month']).withMessage('Invalid timeframe')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        error: 'Validation failed',
        details: errors.array()
      });
    }

    const { count = 20, category, timeframe = 'week' } = req.query;

    // Calculate date range based on timeframe
    const now = new Date();
    const timeframeHours = {
      day: 24,
      week: 168, // 7 days
      month: 720 // 30 days
    };

    const startDate = new Date(now.getTime() - (timeframeHours[timeframe] * 60 * 60 * 1000));

    // Build aggregation pipeline
    const pipeline = [
      {
        $match: {
          createdAt: { $gte: startDate },
          ...(category && { category })
        }
      },
      {
        $group: {
          _id: '$productId',
          interactionCount: { $sum: 1 },
          uniqueUsers: { $addToSet: '$userId' },
          avgRating: { $avg: '$rating' },
          lastInteraction: { $max: '$createdAt' }
        }
      },
      {
        $addFields: {
          uniqueUserCount: { $size: '$uniqueUsers' },
          trendingScore: {
            $add: [
              { $multiply: ['$interactionCount', 0.4] },
              { $multiply: ['$uniqueUserCount', 0.4] },
              { $multiply: [{ $ifNull: ['$avgRating', 3] }, 0.2] }
            ]
          }
        }
      },
      {
        $sort: { trendingScore: -1 }
      },
      {
        $limit: parseInt(count)
      }
    ];

    const trendingData = await Interaction.aggregate(pipeline);

    // Enrich with product details
    const trendingProducts = await Promise.all(
      trendingData.map(async (item) => {
        try {
          const product = await Product.findOne({ productId: item._id });
          return {
            product: product ? {
              id: product._id,
              productId: product.productId,
              name: product.name,
              category: product.category,
              price: product.price,
              image: product.image,
              rating: product.averageRating,
              inStock: product.stock > 0
            } : null,
            trendingScore: item.trendingScore,
            interactionCount: item.interactionCount,
            uniqueUsers: item.uniqueUserCount,
            lastInteraction: item.lastInteraction
          };
        } catch (error) {
          console.error(`Error enriching trending product ${item._id}:`, error);
          return null;
        }
      })
    );

    const validTrendingProducts = trendingProducts.filter(item => item.product !== null);

    res.json({
      trendingProducts: validTrendingProducts,
      count: validTrendingProducts.length,
      timeframe,
      category: category || 'all',
      generatedAt: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error getting trending products:', error);
    res.status(500).json({
      error: 'Failed to get trending products',
      message: 'An unexpected error occurred while fetching trending products'
    });
  }
});

/**
 * Retrain the ML model
 * POST /api/recommendations/retrain
 */
router.post('/retrain', auth, async (req, res) => {
  try {
    // Only allow admin users to retrain
    if (req.user.role !== 'admin') {
      return res.status(403).json({
        error: 'Forbidden',
        message: 'Only admin users can trigger model retraining'
      });
    }

    // Trigger ML model retraining
    const mlResponse = await axios.post(`${ML_SERVICE_URL}/train`, {}, {
      timeout: 300000 // 5 minutes timeout for training
    });

    res.json({
      message: 'Model retraining initiated successfully',
      trainingId: mlResponse.data.trainingId || 'unknown',
      estimatedTime: '5-10 minutes',
      startedAt: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error initiating model retraining:', error);
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: 'ML service unavailable',
        message: 'The ML service is currently unavailable. Please try again later.'
      });
    }

    res.status(500).json({
      error: 'Failed to initiate model retraining',
      message: 'An unexpected error occurred while starting model retraining'
    });
  }
});

/**
 * Get recommendation quality metrics
 * GET /api/recommendations/metrics
 */
router.get('/metrics', auth, async (req, res) => {
  try {
    // Only allow admin users to view metrics
    if (req.user.role !== 'admin') {
      return res.status(403).json({
        error: 'Forbidden',
        message: 'Only admin users can view recommendation metrics'
      });
    }

    const { timeframe = 'week' } = req.query;

    // Calculate basic metrics from interactions
    const now = new Date();
    const timeframeHours = {
      day: 24,
      week: 168,
      month: 720
    };

    const startDate = new Date(now.getTime() - (timeframeHours[timeframe] * 60 * 60 * 1000));

    const metrics = await Interaction.aggregate([
      {
        $match: {
          createdAt: { $gte: startDate }
        }
      },
      {
        $group: {
          _id: null,
          totalInteractions: { $sum: 1 },
          uniqueUsers: { $addToSet: '$userId' },
          uniqueProducts: { $addToSet: '$productId' },
          avgRating: { $avg: '$rating' },
          clickThroughRate: {
            $avg: {
              $cond: [{ $eq: ['$type', 'click'] }, 1, 0]
            }
          },
          conversionRate: {
            $avg: {
              $cond: [{ $eq: ['$type', 'purchase'] }, 1, 0]
            }
          }
        }
      },
      {
        $project: {
          _id: 0,
          totalInteractions: 1,
          uniqueUsers: { $size: '$uniqueUsers' },
          uniqueProducts: { $size: '$uniqueProducts' },
          avgRating: { $round: ['$avgRating', 2] },
          clickThroughRate: { $round: ['$clickThroughRate', 3] },
          conversionRate: { $round: ['$conversionRate', 3] }
        }
      }
    ]);

    const metricsData = metrics[0] || {
      totalInteractions: 0,
      uniqueUsers: 0,
      uniqueProducts: 0,
      avgRating: 0,
      clickThroughRate: 0,
      conversionRate: 0
    };

    res.json({
      metrics: metricsData,
      timeframe,
      generatedAt: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error getting recommendation metrics:', error);
    res.status(500).json({
      error: 'Failed to get metrics',
      message: 'An unexpected error occurred while fetching recommendation metrics'
    });
  }
});

module.exports = router;