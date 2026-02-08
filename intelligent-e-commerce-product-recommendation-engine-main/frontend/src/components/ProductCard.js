/**
 * Product Card Component - Display product information in card format
 * Author: Jessie Borras
 * Website: jessiedev.xyz
 */

import React from 'react';
import {
  Card,
  CardContent,
  CardMedia,
  Typography,
  Button,
  Box,
  Rating,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  ShoppingCart as CartIcon,
  Favorite as FavoriteIcon,
  FavoriteBorder as FavoriteBorderIcon,
  Share as ShareIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';

// Hooks
import { useCart } from '../hooks/useCart';
import { useWishlist } from '../hooks/useWishlist';
import { useInteractions } from '../hooks/useInteractions';

const ProductCard = ({ 
  product, 
  recommendation, 
  showRecommendationInfo = false,
  className = '',
  ...props 
}) => {
  const navigate = useNavigate();
  const { addToCart, isLoading: cartLoading } = useCart();
  const { addToWishlist, removeFromWishlist, isInWishlist } = useWishlist();
  const { trackInteraction } = useInteractions();
  
  const isWishlisted = isInWishlist(product.id);

  const handleCardClick = (e) => {
    // Prevent navigation if clicking on buttons
    if (e.target.closest('button')) {
      return;
    }
    
    // Track view interaction
    trackInteraction({
      productId: product.productId || product.id,
      type: 'view',
      source: showRecommendationInfo ? 'recommendation' : 'browse',
      ...(recommendation && {
        recommendationScore: recommendation.score,
        recommendationMethods: recommendation.methods,
      }),
    });
    
    navigate(`/products/${product.productId || product.id}`);
  };

  const handleAddToCart = async (e) => {
    e.stopPropagation();
    
    try {
      await addToCart({
        productId: product.productId || product.id,
        quantity: 1,
      });
      
      // Track add to cart interaction
      trackInteraction({
        productId: product.productId || product.id,
        type: 'add_to_cart',
        source: showRecommendationInfo ? 'recommendation' : 'browse',
      });
      
      toast.success(`${product.name} added to cart!`);
    } catch (error) {
      toast.error('Failed to add product to cart');
    }
  };

  const handleWishlistToggle = async (e) => {
    e.stopPropagation();
    
    try {
      if (isWishlisted) {
        await removeFromWishlist(product.id);
        toast.success('Removed from wishlist');
      } else {
        await addToWishlist(product.id);
        toast.success('Added to wishlist');
      }
    } catch (error) {
      toast.error('Failed to update wishlist');
    }
  };

  const handleShare = async (e) => {
    e.stopPropagation();
    
    const shareData = {
      title: product.name,
      text: `Check out this product: ${product.name}`,
      url: `${window.location.origin}/products/${product.productId || product.id}`,
    };

    if (navigator.share) {
      try {
        await navigator.share(shareData);
      } catch (error) {
        // User cancelled share
      }
    } else {
      // Fallback: copy to clipboard
      try {
        await navigator.clipboard.writeText(shareData.url);
        toast.success('Product link copied to clipboard!');
      } catch (error) {
        toast.error('Failed to copy link');
      }
    }
    
    // Track share interaction
    trackInteraction({
      productId: product.productId || product.id,
      type: 'share',
      source: showRecommendationInfo ? 'recommendation' : 'browse',
    });
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(price);
  };

  return (
    <Card
      className={`product-card ${className}`}
      onClick={handleCardClick}
      sx={{
        cursor: 'pointer',
        transition: 'all 0.3s ease-in-out',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: '0 8px 25px rgba(0,0,0,0.2)',
        },
        position: 'relative',
        overflow: 'visible',
      }}
      {...props}
    >
      {/* Recommendation badge */}
      {showRecommendationInfo && recommendation && (
        <Box
          sx={{
            position: 'absolute',
            top: 8,
            right: 8,
            zIndex: 1,
          }}
        >
          <Tooltip title={`Recommendation Score: ${(recommendation.score * 100).toFixed(0)}%`}>
            <Chip
              label={`${(recommendation.score * 100).toFixed(0)}% match`}
              size="small"
              color="primary"
              sx={{
                fontWeight: 'bold',
                boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
              }}
            />
          </Tooltip>
        </Box>
      )}

      {/* Stock status */}
      {!product.inStock && (
        <Box
          sx={{
            position: 'absolute',
            top: 8,
            left: 8,
            zIndex: 1,
          }}
        >
          <Chip
            label="Out of Stock"
            size="small"
            color="error"
            sx={{
              fontWeight: 'bold',
              boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
            }}
          />
        </Box>
      )}

      {/* Product Image */}
      <CardMedia
        component="img"
        height="200"
        image={product.image || '/placeholder-product.jpg'}
        alt={product.name}
        sx={{
          objectFit: 'cover',
          transition: 'transform 0.3s ease-in-out',
          '&:hover': {
            transform: 'scale(1.05)',
          },
        }}
      />

      <CardContent sx={{ pb: 1 }}>
        {/* Product Category */}
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ textTransform: 'uppercase', letterSpacing: 1 }}
        >
          {product.category}
        </Typography>

        {/* Product Name */}
        <Typography
          variant="h6"
          component="div"
          sx={{
            fontWeight: 600,
            mb: 1,
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
            lineHeight: 1.3,
          }}
        >
          {product.name}
        </Typography>

        {/* Rating */}
        {product.rating && (
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Rating value={product.rating} readOnly size="small" precision={0.1} />
            <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
              ({product.rating})
            </Typography>
          </Box>
        )}

        {/* Price */}
        <Typography
          variant="h5"
          color="primary"
          sx={{ fontWeight: 700, mb: 2 }}
        >
          {formatPrice(product.price)}
        </Typography>

        {/* Recommendation Methods */}
        {showRecommendationInfo && recommendation?.methods && (
          <Box sx={{ mb: 2 }}>
            {recommendation.methods.map((method) => (
              <Chip
                key={method}
                label={method}
                size="small"
                variant="outlined"
                sx={{ mr: 0.5, mb: 0.5, fontSize: '0.75rem' }}
              />
            ))}
          </Box>
        )}

        {/* Action Buttons */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Button
            variant="contained"
            startIcon={<CartIcon />}
            onClick={handleAddToCart}
            disabled={!product.inStock || cartLoading}
            fullWidth
            sx={{ flex: 1 }}
          >
            {product.inStock ? 'Add to Cart' : 'Out of Stock'}
          </Button>

          <IconButton
            onClick={handleWishlistToggle}
            color={isWishlisted ? 'error' : 'default'}
            aria-label={isWishlisted ? 'Remove from wishlist' : 'Add to wishlist'}
          >
            {isWishlisted ? <FavoriteIcon /> : <FavoriteBorderIcon />}
          </IconButton>

          <IconButton
            onClick={handleShare}
            aria-label="Share product"
          >
            <ShareIcon />
          </IconButton>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ProductCard;