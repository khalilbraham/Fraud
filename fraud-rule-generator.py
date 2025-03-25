import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import re

class FraudRuleGenerator:
    def __init__(self, min_score=50, max_score=70, target_precision=0.3):
        """
        Initialize the fraud rule generator.
        
        Parameters:
        - min_score: Lower bound of the fraud score range to analyze (default: 50)
        - max_score: Upper bound of the fraud score range to analyze (default: 70)
        - target_precision: Target precision for the generated rules (default: 0.3)
        """
        self.min_score = min_score
        self.max_score = max_score
        self.target_precision = target_precision
        self.categorical_features = []
        self.id_features = []
        self.encoder = None
        self.dt_model = None
        self.amount_thresholds = [3000, 10000, 25000, 50000, 100000]  # Based on the example rules
        
    def preprocess_data(self, df, categorical_features=None, id_features=None):
        """
        Preprocess the data for decision tree training.
        
        Parameters:
        - df: DataFrame containing the fraud data
        - categorical_features: List of categorical feature names to one-hot encode
        - id_features: List of ID-like features to handle specially
        
        Returns:
        - Preprocessed DataFrame ready for model training
        """
        # Make a copy to avoid modifying the original dataframe
        processed_df = df.copy()
        
        # Store feature types
        self.categorical_features = categorical_features or []
        self.id_features = id_features or []
        
        # Handle ID features by creating aggregated features
        # We'll do this BEFORE filtering to ensure we have enough data for reliable statistics
        for id_feature in self.id_features:
            print(f"Creating features for {id_feature}...")
            
            # --- MERCHANT-SPECIFIC FEATURES ---
            if id_feature == 'mch_id_trimmed':
                # 1. Calculate merchant-specific fraud rates
                mch_fraud_rate = df.groupby(id_feature)['is_fraud'].mean().reset_index()
                mch_fraud_rate.columns = [id_feature, f'merchant_fraud_rate']
                
                # 2. Calculate merchant transaction volume
                mch_count = df.groupby(id_feature).size().reset_index()
                mch_count.columns = [id_feature, f'merchant_tx_count']
                
                # 3. Calculate merchant average transaction amount
                mch_avg_amount = df.groupby(id_feature)['amount'].mean().reset_index()
                mch_avg_amount.columns = [id_feature, f'merchant_avg_amount']
                
                # 4. Calculate variability in merchant transaction amounts
                mch_std_amount = df.groupby(id_feature)['amount'].std().reset_index()
                mch_std_amount.columns = [id_feature, f'merchant_amount_std']
                
                # 5. Calculate merchant fraud amount rate
                def fraud_amount_rate(group):
                    total_amount = group['amount'].sum()
                    fraud_amount = group.loc[group['is_fraud'] == 1, 'amount'].sum()
                    return fraud_amount / total_amount if total_amount > 0 else 0
                    
                mch_fraud_amount_rate = df.groupby(id_feature).apply(fraud_amount_rate).reset_index()
                mch_fraud_amount_rate.columns = [id_feature, f'merchant_fraud_amount_rate']
                
                # 6. Calculate recent merchant fraud rate (last 30 days)
                if 'timestamp' in df.columns:
                    recent_df = df[df['timestamp'] >= df['timestamp'].max() - pd.Timedelta(days=30)]
                    if not recent_df.empty:
                        recent_fraud_rate = recent_df.groupby(id_feature)['is_fraud'].mean().reset_index()
                        recent_fraud_rate.columns = [id_feature, f'merchant_recent_fraud_rate']
                        processed_df = processed_df.merge(recent_fraud_rate, on=id_feature, how='left')
                
                # 7. Calculate merchant fraud rate by amount tier
                for i, threshold in enumerate(self.amount_thresholds):
                    if i == 0:
                        # Small amounts
                        small_amounts = df[df['amount'] <= threshold]
                        if not small_amounts.empty:
                            small_fraud_rate = small_amounts.groupby(id_feature)['is_fraud'].mean().reset_index()
                            small_fraud_rate.columns = [id_feature, f'merchant_small_amount_fraud_rate']
                            processed_df = processed_df.merge(small_fraud_rate, on=id_feature, how='left')
                    elif i == len(self.amount_thresholds) - 1:
                        # Large amounts
                        large_amounts = df[df['amount'] > threshold]
                        if not large_amounts.empty:
                            large_fraud_rate = large_amounts.groupby(id_feature)['is_fraud'].mean().reset_index()
                            large_fraud_rate.columns = [id_feature, f'merchant_large_amount_fraud_rate']
                            processed_df = processed_df.merge(large_fraud_rate, on=id_feature, how='left')
                
                # 8. Calculate velocity metrics if timestamp is available
                if 'timestamp' in df.columns:
                    # Number of transactions per day
                    df['date'] = pd.to_datetime(df['timestamp']).dt.date
                    tx_per_day = df.groupby([id_feature, 'date']).size().reset_index()
                    tx_per_day.columns = [id_feature, 'date', 'daily_count']
                    
                    # Calculate statistics on daily counts
                    daily_stats = tx_per_day.groupby(id_feature)['daily_count'].agg(['mean', 'std', 'max']).reset_index()
                    daily_stats.columns = [id_feature, 'merchant_daily_tx_mean', 'merchant_daily_tx_std', 'merchant_daily_tx_max']
                    processed_df = processed_df.merge(daily_stats, on=id_feature, how='left')
                
                # 9. Calculate cross-device metrics if available
                if 'device' in df.columns:
                    device_counts = df.groupby(id_feature)['device'].nunique().reset_index()
                    device_counts.columns = [id_feature, 'merchant_device_count']
                    processed_df = processed_df.merge(device_counts, on=id_feature, how='left')
                
                # 10. Calculate merchant-country risk
                if 'merchant_country_code' in df.columns:
                    # Get country fraud rates
                    country_fraud_rates = df.groupby('merchant_country_code')['is_fraud'].mean().reset_index()
                    country_fraud_rates.columns = ['merchant_country_code', 'country_fraud_rate']
                    
                    # Get merchant countries
                    merchant_countries = df[[id_feature, 'merchant_country_code']].drop_duplicates()
                    
                    # Merge to get country risk for each merchant
                    merchant_country_risk = merchant_countries.merge(country_fraud_rates, on='merchant_country_code', how='left')
                    merchant_country_risk = merchant_country_risk[[id_feature, 'country_fraud_rate']]
                    merchant_country_risk.columns = [id_feature, 'merchant_country_risk']
                    
                    processed_df = processed_df.merge(merchant_country_risk, on=id_feature, how='left')
                
                # Merge all merchant features back to the processed dataframe
                processed_df = processed_df.merge(mch_fraud_rate, on=id_feature, how='left')
                processed_df = processed_df.merge(mch_count, on=id_feature, how='left')
                processed_df = processed_df.merge(mch_avg_amount, on=id_feature, how='left')
                processed_df = processed_df.merge(mch_std_amount, on=id_feature, how='left')
                processed_df = processed_df.merge(mch_fraud_amount_rate, on=id_feature, how='left')
                
                # Create merchant risk tiers based on fraud rate percentiles
                fraud_rate_col = 'merchant_fraud_rate'
                # Fill NaN with 0 for the bucketing
                processed_df[fraud_rate_col] = processed_df[fraud_rate_col].fillna(0)
                
                # Create risk tiers (low, medium, high, very high)
                percentiles = [0, 50, 90, 95, 100]
                labels = ['low', 'medium', 'high', 'very_high']
                bins = np.percentile(processed_df[fraud_rate_col].unique(), percentiles)
                # Ensure bins are unique
                bins = np.unique(bins)
                if len(bins) >= 2:  # Need at least 2 bins to categorize
                    processed_df['merchant_risk_tier'] = pd.cut(
                        processed_df[fraud_rate_col], 
                        bins=bins, 
                        labels=labels[:len(bins)-1],
                        include_lowest=True
                    )
                    # One-hot encode the risk tier
                    risk_dummies = pd.get_dummies(
                        processed_df['merchant_risk_tier'], 
                        prefix='merchant_risk'
                    )
                    processed_df = pd.concat([processed_df, risk_dummies], axis=1)
                    processed_df.drop('merchant_risk_tier', axis=1, inplace=True)
            
            # --- ISSUER-SPECIFIC FEATURES ---
            elif id_feature == 'issuer_member':
                # 1. Calculate issuer-specific fraud rates
                issuer_fraud_rate = df.groupby(id_feature)['is_fraud'].mean().reset_index()
                issuer_fraud_rate.columns = [id_feature, f'issuer_fraud_rate']
                
                # 2. Calculate issuer transaction volume
                issuer_count = df.groupby(id_feature).size().reset_index()
                issuer_count.columns = [id_feature, f'issuer_tx_count']
                
                # 3. Calculate issuer average transaction amount
                issuer_avg_amount = df.groupby(id_feature)['amount'].mean().reset_index()
                issuer_avg_amount.columns = [id_feature, f'issuer_avg_amount']
                
                # 4. Calculate variability in issuer transaction amounts
                issuer_std_amount = df.groupby(id_feature)['amount'].std().reset_index()
                issuer_std_amount.columns = [id_feature, f'issuer_amount_std']
                
                # 5. Calculate issuer fraud amount rate
                def fraud_amount_rate(group):
                    total_amount = group['amount'].sum()
                    fraud_amount = group.loc[group['is_fraud'] == 1, 'amount'].sum()
                    return fraud_amount / total_amount if total_amount > 0 else 0
                    
                issuer_fraud_amount_rate = df.groupby(id_feature).apply(fraud_amount_rate).reset_index()
                issuer_fraud_amount_rate.columns = [id_feature, f'issuer_fraud_amount_rate']
                
                # 6. Calculate issuer authorization rate (if available)
                if 'trans_status' in df.columns:
                    auth_rate = df.groupby(id_feature).apply(
                        lambda x: (x['trans_status'] == 'Y').mean()
                    ).reset_index()
                    auth_rate.columns = [id_feature, 'issuer_auth_rate']
                    processed_df = processed_df.merge(auth_rate, on=id_feature, how='left')
                
                # 7. Calculate issuer-country metrics (if available)
                if 'issuer_bin_profile_issuing_cou' in df.columns:
                    # Issuer country fraud rates
                    country_fraud_rates = df.groupby('issuer_bin_profile_issuing_cou')['is_fraud'].mean().reset_index()
                    country_fraud_rates.columns = ['issuer_bin_profile_issuing_cou', 'issuer_country_fraud_rate']
                    
                    # Get issuer countries 
                    issuer_countries = df[[id_feature, 'issuer_bin_profile_issuing_cou']].drop_duplicates()
                    
                    # Merge to get country risk for each issuer
                    issuer_country_risk = issuer_countries.merge(
                        country_fraud_rates, 
                        on='issuer_bin_profile_issuing_cou', 
                        how='left'
                    )
                    issuer_country_risk = issuer_country_risk[[id_feature, 'issuer_country_fraud_rate']]
                    
                    processed_df = processed_df.merge(issuer_country_risk, on=id_feature, how='left')
                
                # 8. Calculate 3DS usage rate by issuer (if available)
                if 'three_ds_mode' in df.columns:
                    threeds_rate = df.groupby(id_feature).apply(
                        lambda x: (x['three_ds_mode'] != 'N').mean()
                    ).reset_index()
                    threeds_rate.columns = [id_feature, 'issuer_3ds_usage_rate']
                    processed_df = processed_df.merge(threeds_rate, on=id_feature, how='left')
                
                # 9. Calculate cross-border transaction metrics
                if 'issuer_bin_profile_issuing_cou' in df.columns and 'merchant_country_code' in df.columns:
                    df['is_cross_border'] = (df['issuer_bin_profile_issuing_cou'] != df['merchant_country_code']).astype(int)
                    cross_border_rate = df.groupby(id_feature)['is_cross_border'].mean().reset_index()
                    cross_border_rate.columns = [id_feature, 'issuer_cross_border_rate'] 
                    
                    # Cross-border fraud rate
                    cross_border_fraud = df[df['is_cross_border'] == 1].groupby(id_feature)['is_fraud'].mean().reset_index()
                    cross_border_fraud.columns = [id_feature, 'issuer_cross_border_fraud_rate']
                    
                    processed_df = processed_df.merge(cross_border_rate, on=id_feature, how='left')
                    processed_df = processed_df.merge(cross_border_fraud, on=id_feature, how='left')
                
                # Merge all issuer features back to the processed dataframe
                processed_df = processed_df.merge(issuer_fraud_rate, on=id_feature, how='left')
                processed_df = processed_df.merge(issuer_count, on=id_feature, how='left')
                processed_df = processed_df.merge(issuer_avg_amount, on=id_feature, how='left')
                processed_df = processed_df.merge(issuer_std_amount, on=id_feature, how='left')
                processed_df = processed_df.merge(issuer_fraud_amount_rate, on=id_feature, how='left')
                
                # Create issuer risk tiers based on fraud rate percentiles
                fraud_rate_col = 'issuer_fraud_rate'
                # Fill NaN with 0 for the bucketing
                processed_df[fraud_rate_col] = processed_df[fraud_rate_col].fillna(0)
                
                # Create risk tiers (low, medium, high, very high)
                percentiles = [0, 50, 90, 95, 100]
                labels = ['low', 'medium', 'high', 'very_high']
                bins = np.percentile(processed_df[fraud_rate_col].unique(), percentiles)
                # Ensure bins are unique
                bins = np.unique(bins)
                if len(bins) >= 2:  # Need at least 2 bins to categorize
                    processed_df['issuer_risk_tier'] = pd.cut(
                        processed_df[fraud_rate_col], 
                        bins=bins, 
                        labels=labels[:len(bins)-1],
                        include_lowest=True
                    )
                    # One-hot encode the risk tier
                    risk_dummies = pd.get_dummies(
                        processed_df['issuer_risk_tier'], 
                        prefix='issuer_risk'
                    )
                    processed_df = pd.concat([processed_df, risk_dummies], axis=1)
                    processed_df.drop('issuer_risk_tier', axis=1, inplace=True)
            
            # --- GENERAL ID FEATURES (for any other ID features) ---
            else:
                # 1. Calculate overall fraud rate per ID
                id_fraud_rate = df.groupby(id_feature)['is_fraud'].mean().reset_index()
                id_fraud_rate.columns = [id_feature, f'{id_feature}_fraud_rate']
                
                # 2. Calculate transaction count per ID (volume indicator)
                id_count = df.groupby(id_feature).size().reset_index()
                id_count.columns = [id_feature, f'{id_feature}_count']
                
                # 3. Calculate average transaction amount per ID
                id_avg_amount = df.groupby(id_feature)['amount'].mean().reset_index()
                id_avg_amount.columns = [id_feature, f'{id_feature}_avg_amount']
                
                # 4. Calculate the standard deviation of amounts per ID (variability indicator)
                id_std_amount = df.groupby(id_feature)['amount'].std().reset_index()
                id_std_amount.columns = [id_feature, f'{id_feature}_std_amount']
                
                # 5. Calculate fraud amount rate
                def fraud_amount_rate(group):
                    total_amount = group['amount'].sum()
                    fraud_amount = group.loc[group['is_fraud'] == 1, 'amount'].sum()
                    return fraud_amount / total_amount if total_amount > 0 else 0
                    
                id_fraud_amount_rate = df.groupby(id_feature).apply(fraud_amount_rate).reset_index()
                id_fraud_amount_rate.columns = [id_feature, f'{id_feature}_fraud_amount_rate']
                
                # Merge all features back to the processed dataframe
                processed_df = processed_df.merge(id_fraud_rate, on=id_feature, how='left')
                processed_df = processed_df.merge(id_count, on=id_feature, how='left')
                processed_df = processed_df.merge(id_avg_amount, on=id_feature, how='left')
                processed_df = processed_df.merge(id_std_amount, on=id_feature, how='left')
                processed_df = processed_df.merge(id_fraud_amount_rate, on=id_feature, how='left')
                
                # Create risk tiers based on fraud rate percentiles
                fraud_rate_col = f'{id_feature}_fraud_rate'
                # Fill NaN with 0 for the bucketing
                processed_df[fraud_rate_col] = processed_df[fraud_rate_col].fillna(0)
                
                # Create risk tiers (low, medium, high, very high)
                percentiles = [0, 50, 90, 95, 100]
                labels = ['low', 'medium', 'high', 'very_high']
                bins = np.percentile(processed_df[fraud_rate_col].unique(), percentiles)
                # Ensure bins are unique
                bins = np.unique(bins)
                if len(bins) >= 2:  # Need at least 2 bins to categorize
                    processed_df[f'{id_feature}_risk_tier'] = pd.cut(
                        processed_df[fraud_rate_col], 
                        bins=bins, 
                        labels=labels[:len(bins)-1],
                        include_lowest=True
                    )
                    # One-hot encode the risk tier
                    risk_dummies = pd.get_dummies(
                        processed_df[f'{id_feature}_risk_tier'], 
                        prefix=f'{id_feature}_risk'
                    )
                    processed_df = pd.concat([processed_df, risk_dummies], axis=1)
                    processed_df.drop(f'{id_feature}_risk_tier', axis=1, inplace=True)
            
            # Drop the original ID feature as we now have more meaningful features
            processed_df.drop(id_feature, axis=1, inplace=True)
        
        # Now filter transactions with scores in the target range
        mask = (processed_df['fraud_score'] >= self.min_score) & (processed_df['fraud_score'] < self.max_score)
        processed_df = processed_df[mask].copy()
        
        # One-hot encode categorical features (but with a limited cardinality check)
        if self.categorical_features:
            safe_categorical_features = []
            for feature in self.categorical_features:
                # Only one-hot encode if cardinality is reasonably low
                if processed_df[feature].nunique() <= 50:  # Arbitrary threshold, adjust as needed
                    safe_categorical_features.append(feature)
                else:
                    print(f"Warning: Feature {feature} has high cardinality ({processed_df[feature].nunique()} values). "
                          f"Converting to frequency encoding instead of one-hot.")
                    # Use frequency encoding for high cardinality categoricals
                    value_counts = processed_df[feature].value_counts(normalize=True)
                    processed_df[f'{feature}_freq'] = processed_df[feature].map(value_counts)
                    processed_df.drop(feature, axis=1, inplace=True)
            
            if safe_categorical_features:
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_features = self.encoder.fit_transform(processed_df[safe_categorical_features])
                feature_names = self.encoder.get_feature_names_out(safe_categorical_features)
                
                # Create a DataFrame with the encoded features
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=processed_df.index)
                
                # Drop the original categorical features and join the encoded ones
                processed_df = processed_df.drop(safe_categorical_features, axis=1)
                processed_df = pd.concat([processed_df, encoded_df], axis=1)
        
        # Fill NaN values
        processed_df = processed_df.fillna(-1)
        
        return processed_df
    
    def train_model(self, X, y, amount_col='amount', max_depth=5):
        """
        Train a decision tree model to identify fraudulent transactions.
        
        Parameters:
        - X: Feature DataFrame
        - y: Target Series (is_fraud)
        - amount_col: Name of the column containing transaction amounts
        - max_depth: Maximum depth of the decision tree
        
        Returns:
        - Trained decision tree model
        """
        # Add amount range features based on thresholds
        X_with_ranges = X.copy()
        for i in range(len(self.amount_thresholds) + 1):
            if i == 0:
                X_with_ranges[f'amount_range_{i}'] = (X[amount_col] <= self.amount_thresholds[i])
            elif i == len(self.amount_thresholds):
                X_with_ranges[f'amount_range_{i}'] = (X[amount_col] > self.amount_thresholds[i-1])
            else:
                X_with_ranges[f'amount_range_{i}'] = ((X[amount_col] > self.amount_thresholds[i-1]) & 
                                                     (X[amount_col] <= self.amount_thresholds[i]))
        
        # Remove the amount column to avoid redundancy
        X_model = X_with_ranges.drop(amount_col, axis=1)
        
        # Train the decision tree
        self.dt_model = DecisionTreeClassifier(max_depth=max_depth, 
                                             min_samples_leaf=100,  # Prevent overfitting with minimum samples per leaf
                                             class_weight='balanced')  # Balance class weights
        self.dt_model.fit(X_model, y)
        
        # Store feature names for rule generation
        self.feature_names = X_model.columns.tolist()
        
        return self.dt_model
    
    def extract_rules(self, X, y, amount_col='amount', score_col='community_fraud_score_amount_d'):
        """
        Extract rules from the trained decision tree model.
        
        Parameters:
        - X: Feature DataFrame used for training
        - y: Target Series (is_fraud)
        - amount_col: Name of the column containing transaction amounts
        - score_col: Name of the column containing community fraud scores
        
        Returns:
        - List of rule dictionaries with conditions and performance metrics
        """
        if self.dt_model is None:
            raise ValueError("Model has not been trained yet. Call train_model first.")
        
        # Get the tree structure in text format
        tree_rules = export_text(self.dt_model, feature_names=self.feature_names)
        
        # Parse the tree text to extract decision paths
        paths = self._parse_tree_text(tree_rules)
        
        # Generate rules from the paths
        rules = []
        for path in paths:
            # Only keep paths that lead to fraud prediction (class 1)
            if path['class'] == 1:
                # Convert the path conditions to a rule
                rule = self._path_to_rule(path, X, y, amount_col, score_col)
                if rule:
                    rules.append(rule)
        
        # Sort rules by amount_recall (highest first)
        rules.sort(key=lambda x: x['amount_recall'], reverse=True)
        
        return rules
    
    def _parse_tree_text(self, tree_text):
        """
        Parse the text representation of the decision tree to extract paths.
        
        Parameters:
        - tree_text: Text representation of the decision tree
        
        Returns:
        - List of path dictionaries
        """
        lines = tree_text.split('\n')
        paths = []
        current_path = []
        
        for line in lines:
            if not line.strip():
                continue
                
            # Calculate the depth by counting leading spaces (divided by 4)
            depth = (len(line) - len(line.lstrip())) // 4
            
            # If we're going back up the tree, remove deeper nodes from current path
            if depth < len(current_path):
                current_path = current_path[:depth]
            
            # Parse the node content
            content = line.strip()
            
            if 'class:' in content:
                # Leaf node with class prediction
                class_val = int(content.split('class:')[1].strip())
                
                # Create a copy of the current path and add the class
                path_copy = current_path.copy()
                paths.append({
                    'conditions': path_copy,
                    'class': class_val
                })
            else:
                # Decision node with feature test
                if '<=' in content:
                    feature, value = content.split('<=')
                    current_path.append((feature.strip(), '<=', float(value.strip())))
                elif '>' in content:
                    feature, value = content.split('>')
                    current_path.append((feature.strip(), '>', float(value.strip())))
        
        return paths
    
    def _path_to_rule(self, path, X, y, amount_col, score_col):
        """
        Convert a decision path to a rule and evaluate its performance.
        
        Parameters:
        - path: Dictionary containing the path conditions and predicted class
        - X: Feature DataFrame
        - y: Target Series (is_fraud)
        - amount_col: Name of the column containing transaction amounts
        - score_col: Name of the column containing community fraud scores
        
        Returns:
        - Rule dictionary with conditions and performance metrics
        """
        # Create a mask to identify transactions matching all conditions
        mask = pd.Series(True, index=X.index)
        amount_conditions = []
        score_conditions = []
        other_conditions = []
        
        for feature, op, value in path['conditions']:
            # Check if this is an amount range feature
            if feature.startswith('amount_range_'):
                # Find the corresponding amount thresholds
                range_idx = int(feature.split('_')[-1])
                
                if range_idx == 0:
                    # amount <= first threshold
                    amount_conditions.append(f"(event.{amount_col} <= {self.amount_thresholds[0]})")
                    if op == '<=':
                        if value < 0.5:  # False
                            mask &= ~(X[amount_col] <= self.amount_thresholds[0])
                        else:  # True
                            mask &= (X[amount_col] <= self.amount_thresholds[0])
                    else:  # '>'
                        if value > 0.5:  # True
                            mask &= (X[amount_col] <= self.amount_thresholds[0])
                        else:  # False
                            mask &= ~(X[amount_col] <= self.amount_thresholds[0])
                            
                elif range_idx == len(self.amount_thresholds):
                    # amount > last threshold
                    amount_conditions.append(f"(event.{amount_col} > {self.amount_thresholds[-1]})")
                    if op == '<=':
                        if value < 0.5:  # False
                            mask &= ~(X[amount_col] > self.amount_thresholds[-1])
                        else:  # True
                            mask &= (X[amount_col] > self.amount_thresholds[-1])
                    else:  # '>'
                        if value > 0.5:  # True
                            mask &= (X[amount_col] > self.amount_thresholds[-1])
                        else:  # False
                            mask &= ~(X[amount_col] > self.amount_thresholds[-1])
                else:
                    # amount > previous threshold and <= current threshold
                    lower = self.amount_thresholds[range_idx-1]
                    upper = self.amount_thresholds[range_idx]
                    amount_conditions.append(f"(event.{amount_col} > {lower} and event.{amount_col} <= {upper})")
                    
                    if op == '<=':
                        if value < 0.5:  # False
                            mask &= ~((X[amount_col] > lower) & (X[amount_col] <= upper))
                        else:  # True
                            mask &= ((X[amount_col] > lower) & (X[amount_col] <= upper))
                    else:  # '>'
                        if value > 0.5:  # True
                            mask &= ((X[amount_col] > lower) & (X[amount_col] <= upper))
                        else:  # False
                            mask &= ~((X[amount_col] > lower) & (X[amount_col] <= upper))
            
            # Handle original features
            elif feature in X.columns:
                if feature == score_col:
                    # For score features, we create a separate condition
                    if op == '<=':
                        score_conditions.append(f"event.{feature} < {value:.3f}")
                        mask &= (X[feature] <= value)
                    else:  # '>'
                        score_conditions.append(f"event.{feature} >= {value:.3f}")
                        mask &= (X[feature] > value)
                else:
                    # For other features
                    if op == '<=':
                        other_conditions.append(f"event.{feature} <= {value}")
                        mask &= (X[feature] <= value)
                    else:  # '>'
                        other_conditions.append(f"event.{feature} > {value}")
                        mask &= (X[feature] > value)
        
        # If no transactions match the conditions, return None
        if not mask.any():
            return None
        
        # Evaluate rule performance
        matches = X[mask]
        true_positives = y[mask].sum()
        precision = true_positives / len(matches) if len(matches) > 0 else 0
        
        # Calculate amount recall (sum of fraud amounts captured / total fraud amounts)
        total_fraud_amount = (X[y == 1][amount_col]).sum()
        captured_fraud_amount = (matches[y[mask] == 1][amount_col]).sum()
        amount_recall = captured_fraud_amount / total_fraud_amount if total_fraud_amount > 0 else 0
        
        # Skip rules with precision below target
        if precision < self.target_precision:
            return None
        
        # Format the rule as a string similar to the example
        rule_components = []
        
        # Add amount and score conditions together
        if amount_conditions and score_conditions:
            score_parts = []
            for amt_cond in amount_conditions:
                for score_cond in score_conditions:
                    score_parts.append(f"{amt_cond} and {score_cond}")
            
            if score_parts:
                rule_components.append("(" + " or ".join(score_parts) + ")")
        
        # Add other conditions
        rule_components.extend(other_conditions)
        
        # Combine all parts with 'and'
        rule_str = " and ".join(rule_components)
        
        return {
            'rule_string': rule_str,
            'precision': precision,
            'amount_recall': amount_recall,
            'num_transactions': len(matches),
            'num_frauds': true_positives,
            'captured_fraud_amount': captured_fraud_amount
        }
    
    def generate_optimized_ruleset(self, rules, max_rules=5):
        """
        Generate an optimized set of rules to maximize amount recall while maintaining target precision.
        
        Parameters:
        - rules: List of rules generated from extract_rules
        - max_rules: Maximum number of rules to include in the final ruleset
        
        Returns:
        - Optimized ruleset as a string
        """
        # Sort rules by amount recall (highest first)
        sorted_rules = sorted(rules, key=lambda x: x['amount_recall'], reverse=True)
        
        # Select top rules up to max_rules
        selected_rules = sorted_rules[:max_rules]
        
        # Combine rules into a single ruleset with 'or' operator
        combined_rule = " or ".join([f"({rule['rule_string']})" for rule in selected_rules])
        
        # Calculate the performance metrics for the combined ruleset
        total_captured_amount = sum(rule['captured_fraud_amount'] for rule in selected_rules)
        total_transactions = sum(rule['num_transactions'] for rule in selected_rules)
        total_frauds = sum(rule['num_frauds'] for rule in selected_rules)
        
        # Create the final output with metrics
        ruleset = {
            'combined_rule': combined_rule,
            'rules': selected_rules,
            'metrics': {
                'total_rules': len(selected_rules),
                'average_precision': sum(rule['precision'] for rule in selected_rules) / len(selected_rules),
                'total_captured_fraud_amount': total_captured_amount,
                'total_transactions': total_transactions,
                'total_frauds': total_frauds
            }
        }
        
        return ruleset

# Example usage
def example_workflow():
    # Load sample data (replace with your actual data loading)
    # df = pd.read_csv('fraud_transactions.csv')
    
    # For demonstration, creating a synthetic dataset
    np.random.seed(42)
    n_samples = 10000
    
    df = pd.DataFrame({
        'fraud_score': np.random.uniform(0, 100, n_samples),
        'is_fraud': np.random.binomial(1, 0.1, n_samples),  # 10% fraud rate
        'amount': np.random.exponential(5000, n_samples),  # Transaction amounts
        'community_fraud_score_amount_d': np.random.uniform(0, 1, n_samples),  # Fraud score
        'mch_id_trimmed': np.random.choice(['0160859', '0245789', '0389012'], n_samples),
        'issuer_member': np.random.choice(['20041', '30567', '45678'], n_samples),
        'panalias_hab_cp_bill_count_45d': np.random.randint(0, 5, n_samples),
        'panalias_hab_ipaddress_count_4': np.random.randint(0, 10, n_samples),
        'ship_address_usage_ind': np.random.choice(['01', '02', '03', '04', '05'], n_samples),
        'isres_dpt_estimated_fact': np.random.randint(0, 2, n_samples),
        'is_correct_bill_dept': np.random.choice([True, False], n_samples),
        'card_profile_dep': np.random.choice(['0', '1', '2', '3', '4', 'nan', ''], n_samples)
    })
    
    # Scale the fraud likelihood based on the fraud_score
    df['is_fraud'] = np.where(
        (df['fraud_score'] >= 70), 
        np.random.binomial(1, 0.8, n_samples),  # 80% chance of fraud if score >= 70
        np.where(
            (df['fraud_score'] >= 50) & (df['fraud_score'] < 70),
            np.random.binomial(1, 0.3, n_samples),  # 30% chance of fraud if score in [50,70)
            np.random.binomial(1, 0.05, n_samples)  # 5% chance of fraud otherwise
        )
    )
    
    # Initialize the rule generator
    generator = FraudRuleGenerator(min_score=50, max_score=70, target_precision=0.3)
    
    # Identify categorical and ID features
    categorical_features = ['ship_address_usage_ind']
    id_features = ['mch_id_trimmed', 'issuer_member']
    
    # Preprocess the data
    processed_df = generator.preprocess_data(df, 
                                            categorical_features=categorical_features, 
                                            id_features=id_features)
    
    # Split into features and target
    X = processed_df.drop('is_fraud', axis=1)
    y = processed_df['is_fraud']
    
    # Train the model
    generator.train_model(X, y, amount_col='amount', max_depth=4)
    
    # Extract rules
    rules = generator.extract_rules(X, y, amount_col='amount', score_col='community_fraud_score_amount_d')
    
    # Generate optimized ruleset
    ruleset = generator.generate_optimized_ruleset(rules, max_rules=3)
    
    print("Generated Rule:")
    print(ruleset['combined_rule'])
    print("\nRule Metrics:")
    for key, value in ruleset['metrics'].items():
        print(f"{key}: {value}")
    
    return ruleset

if __name__ == "__main__":
    example_workflow()
