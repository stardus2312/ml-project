"""
Script to create sample ML models for demonstration purposes
Run this script to generate sample .pkl files if you don't have the actual models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

def create_sample_data():
    """Create sample drug quality dataset"""
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    
    data = {
        'molecular_weight': np.random.normal(200, 100, n_samples),
        'logp': np.random.normal(2, 2, n_samples),
        'hbd': np.random.randint(0, 6, n_samples),
        'hba': np.random.randint(0, 10, n_samples),
        'rotb': np.random.randint(0, 10, n_samples),
        'tpsa': np.random.normal(80, 40, n_samples),
        'aromatic_rings': np.random.randint(0, 4, n_samples),
        'heavy_atoms': np.random.randint(10, 50, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on some rules
    # High quality: low molecular weight, good logp, reasonable properties
    quality_score = (
        (df['molecular_weight'] < 300).astype(int) * 2 +
        (df['logp'] > 0).astype(int) * 2 +
        (df['logp'] < 5).astype(int) * 2 +
        (df['hbd'] <= 5).astype(int) +
        (df['hba'] <= 10).astype(int) +
        (df['rotb'] <= 7).astype(int) +
        (df['tpsa'] < 140).astype(int) +
        (df['aromatic_rings'] <= 3).astype(int)
    )
    
    # Convert to quality categories
    df['quality'] = pd.cut(quality_score, bins=[0, 5, 10, 15], labels=['Low', 'Medium', 'High'])
    
    return df

def train_models():
    """Train sample models"""
    print("Creating sample dataset...")
    df = create_sample_data()
    
    # Prepare features and target
    feature_columns = [
        'molecular_weight', 'logp', 'hbd', 'hba', 'rotb', 
        'tpsa', 'aromatic_rings', 'heavy_atoms'
    ]
    
    X = df[feature_columns]
    y = df['quality']
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    
    print("Training Decision Tree model...")
    # Decision Tree (as substitute for Decision_bayes_model)
    dt_model = RandomForestClassifier(n_estimators=100, random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    models['Decision_bayes_model.pkl'] = dt_model
    
    print("Training Logistic Regression model...")
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    models['logistic_regression_model.pkl'] = lr_model
    
    print("Training Naive Bayes model...")
    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    models['naive_bayes_model.pkl'] = nb_model
    
    # Save models
    for filename, model in models.items():
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Saved {filename}")
    
    # Save scaler and label encoder
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print("✅ All models saved successfully!")
    
    # Print model performance
    print("\n📊 Model Performance:")
    for name, model in models.items():
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        print(f"{name}: Train={train_score:.3f}, Test={test_score:.3f}")
    
    return models, scaler, le

if __name__ == "__main__":
    train_models()


