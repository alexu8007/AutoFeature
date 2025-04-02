"""
Text-Based Transformer

This module provides transformers for extracting features from text columns.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Union
from collections import Counter
from autofeature.feature_generation.base import BaseTransformer


class TextBasedTransformer(BaseTransformer):
    """Transformer for extracting features from text columns.
    
    This transformer creates new features by extracting statistics and patterns
    from text data, such as length, word count, special character counts, etc.
    """
    
    TEXT_FEATURES = {
        'char_count': lambda x: len(str(x)),
        'word_count': lambda x: len(str(x).split()),
        'unique_word_count': lambda x: len(set(str(x).lower().split())),
        'stop_word_count': lambda x, stop_words: sum(1 for word in str(x).lower().split() 
                                                   if word in stop_words),
        'special_char_count': lambda x: len([c for c in str(x) if not c.isalnum() and not c.isspace()]),
        'numeric_char_count': lambda x: len([c for c in str(x) if c.isdigit()]),
        'uppercase_char_count': lambda x: len([c for c in str(x) if c.isupper()]),
        'lowercase_char_count': lambda x: len([c for c in str(x) if c.islower()]),
        'space_count': lambda x: str(x).count(' '),
        'mean_word_length': lambda x: np.mean([len(word) for word in str(x).split()]) 
                                      if len(str(x).split()) > 0 else 0,
        'punctuation_count': lambda x: len([c for c in str(x) if c in '.,;:!?-()[]{}\'\"']),
        'sentence_count': lambda x: len([s for s in re.split(r'[.!?]+', str(x)) if len(s.strip()) > 0]),
    }
    
    # English stop words (a small subset, can be expanded)
    DEFAULT_STOP_WORDS = set([
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
        'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
        'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'i', 'me', 'my', 'myself',
        'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
        'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'having', 'do', 'does', 'did', 'doing', 'would', 'should', 'could',
        'ought', 'now'
    ])
    
    def __init__(self, features: List[str] = None, text_columns: List[str] = None,
                 stop_words: List[str] = None, custom_features: Dict[str, Any] = None,
                 include_tfidf: bool = False, max_tfidf_features: int = 100,
                 ngram_range: tuple = (1, 1)):
        """Initialize the text-based transformer.
        
        Args:
            features: Text features to extract
            text_columns: Text columns to transform (if None, auto-detect)
            stop_words: List of stop words (if None, use default)
            custom_features: Custom text features to extract
            include_tfidf: Whether to include TF-IDF features
            max_tfidf_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
        """
        super().__init__()
        self.features = features or ['char_count', 'word_count', 'special_char_count', 
                                     'numeric_char_count', 'uppercase_char_count']
        self.text_columns = text_columns
        self.stop_words = set(stop_words) if stop_words is not None else self.DEFAULT_STOP_WORDS
        self.custom_features = custom_features or {}
        self.include_tfidf = include_tfidf
        self.max_tfidf_features = max_tfidf_features
        self.ngram_range = ngram_range
        self.detected_text_columns = []
        self.tfidf_vectorizer = None
        
        # Validate features
        for feature in self.features:
            if feature not in self.TEXT_FEATURES and feature not in self.custom_features:
                raise ValueError(f"Feature '{feature}' is not supported. "
                                f"Supported features are: {list(self.TEXT_FEATURES.keys())}. "
                                f"Alternatively, provide a custom feature.")
                
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TextBasedTransformer':
        """Fit the transformer to the data by identifying text columns.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            self: The fitted transformer
        """
        self._validate_input(X)
        
        # If text columns are provided, use those
        if self.text_columns:
            self.detected_text_columns = [col for col in self.text_columns if col in X.columns]
        else:
            # Otherwise, detect text columns
            self.detected_text_columns = []
            for col in X.columns:
                # Check if column is string type
                if X[col].dtype == 'object' or X[col].dtype.name == 'string':
                    # Check if column contains mostly text (heuristic)
                    sample = X[col].dropna().sample(min(100, len(X[col]))).astype(str)
                    avg_length = sample.str.len().mean()
                    avg_words = sample.str.split().str.len().mean()
                    
                    # Simple heuristic: longer texts with multiple words are likely text columns
                    if avg_length > 5 and avg_words > 1.5:
                        self.detected_text_columns.append(col)
        
        # Fit TF-IDF vectorizer if needed
        if self.include_tfidf and self.detected_text_columns:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                # Combine all text columns for vectorization
                corpus = []
                for _, row in X.iterrows():
                    text = ' '.join([str(row[col]) for col in self.detected_text_columns 
                                    if col in row and pd.notna(row[col])])
                    corpus.append(text)
                
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.max_tfidf_features,
                    stop_words=list(self.stop_words) if self.stop_words else None,
                    ngram_range=self.ngram_range
                )
                self.tfidf_vectorizer.fit(corpus)
            except ImportError:
                # Disable TF-IDF if scikit-learn is not available
                self.include_tfidf = False
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data by extracting text-based features.
        
        Args:
            X: Input features
            
        Returns:
            pd.DataFrame: Transformed data with new features
        """
        if not self.is_fitted:
            raise ValueError("Transformer is not fitted yet. Call fit or fit_transform first.")
        
        self._validate_input(X)
        result = X.copy()
        
        # Extract basic text features
        for col in self.detected_text_columns:
            if col not in X.columns:
                continue
                
            for feature_name in self.features:
                try:
                    # Generate the new feature
                    new_col_name = f"{col}_{feature_name}"
                    
                    # Apply the feature extraction function
                    if feature_name in self.TEXT_FEATURES:
                        if feature_name == 'stop_word_count':
                            result[new_col_name] = X[col].apply(
                                lambda x: self.TEXT_FEATURES[feature_name](x, self.stop_words)
                            )
                        else:
                            result[new_col_name] = X[col].apply(self.TEXT_FEATURES[feature_name])
                    else:
                        result[new_col_name] = X[col].apply(self.custom_features[feature_name])
                    
                    # Register the feature
                    self._register_feature(new_col_name, {
                        'source_column': col,
                        'feature': feature_name,
                        'type': 'text_feature'
                    })
                except Exception:
                    continue
            
            # Add lexical diversity (unique words / total words)
            try:
                diversity_col = f"{col}_lexical_diversity"
                word_count = X[col].apply(lambda x: len(str(x).split()))
                unique_word_count = X[col].apply(lambda x: len(set(str(x).lower().split())))
                
                # Avoid division by zero
                result[diversity_col] = np.where(
                    word_count > 0,
                    unique_word_count / word_count,
                    0
                )
                
                self._register_feature(diversity_col, {
                    'source_column': col,
                    'feature': 'lexical_diversity',
                    'type': 'text_feature'
                })
            except Exception:
                pass
        
        # Add TF-IDF features if enabled
        if self.include_tfidf and self.tfidf_vectorizer and self.detected_text_columns:
            try:
                # Combine all text columns for vectorization
                corpus = []
                for _, row in X.iterrows():
                    text = ' '.join([str(row[col]) for col in self.detected_text_columns 
                                    if col in row and pd.notna(row[col])])
                    corpus.append(text)
                
                # Transform the corpus
                tfidf_matrix = self.tfidf_vectorizer.transform(corpus)
                
                # Convert to DataFrame
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                for i, feature_name in enumerate(feature_names):
                    # Create a valid column name
                    valid_feature_name = re.sub(r'\W+', '_', feature_name)
                    new_col_name = f"tfidf_{valid_feature_name}"
                    
                    # Add the TF-IDF feature
                    result[new_col_name] = tfidf_matrix[:, i].toarray().flatten()
                    
                    # Register the feature
                    self._register_feature(new_col_name, {
                        'source_columns': self.detected_text_columns,
                        'feature': feature_name,
                        'type': 'tfidf'
                    })
            except Exception:
                pass
        
        return result 