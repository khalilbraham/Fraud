import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

class SparkFraudRuleGenerator:
    def __init__(self, min_score=50, max_score=70, target_precision=0.3):
        """
        Initialize the Spark-based fraud rule generator.
        
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
        self.dt_model = None
        self.feature_names = []
        self.feature_importances = None
        self.amount_thresholds = [3000, 10000, 25000, 50000, 100000]  # Based on the example rules
        
        # Create or get Spark session
        self.spark = SparkSession.builder \
            .appName("FraudRuleGenerator") \
            .config("spark.sql.session.timeZone", "UTC") \
            .getOrCreate()
        
    def preprocess_data(self, sdf, categorical_features=None, id_features=None):
        """
        Preprocess the data for decision tree training using Spark.
        
        Parameters:
        - sdf: Spark DataFrame containing the fraud data
        - categorical_features: List of categorical feature names to encode
        - id_features: List of ID-like features to handle specially
        
        Returns:
        - Processed Spark DataFrame ready for model training
        """
        # Store feature types
        self.categorical_features = categorical_features or []
        self.id_features = id_features or []
        
        # Register the DataFrame as a temp view for SQL queries
        sdf.createOrReplaceTempView("transactions")
        
        # Create a base processed DataFrame for feature engineering
        processed_sdf = sdf
        
        # Handle merchant ID features
        if 'mch_id_trimmed' in self.id_features:
            # Calculate merchant fraud rate
            merchant_fraud_rate = self.spark.sql("""
                SELECT mch_id_trimmed, 
                       AVG(CAST(is_fraud AS DOUBLE)) AS merchant_fraud_rate,
                       COUNT(*) AS merchant_tx_count,
                       AVG(amount) AS merchant_avg_amount,
                       STDDEV(amount) AS merchant_amount_std
                FROM transactions
                GROUP BY mch_id_trimmed
            """)
            
            # Calculate merchant fraud amount rate
            merchant_fraud_amount = self.spark.sql("""
                SELECT mch_id_trimmed,
                       SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) / SUM(amount) AS merchant_fraud_amount_rate
                FROM transactions
                GROUP BY mch_id_trimmed
            """)
            
            # Calculate small amount fraud rate
            merchant_small_fraud = self.spark.sql(f"""
                SELECT mch_id_trimmed,
                       AVG(CAST(is_fraud AS DOUBLE)) AS merchant_small_amount_fraud_rate
                FROM transactions
                WHERE amount <= {self.amount_thresholds[0]}
                GROUP BY mch_id_trimmed
            """)
            
            # Calculate large amount fraud rate
            merchant_large_fraud = self.spark.sql(f"""
                SELECT mch_id_trimmed,
                       AVG(CAST(is_fraud AS DOUBLE)) AS merchant_large_amount_fraud_rate
                FROM transactions
                WHERE amount > {self.amount_thresholds[-1]}
                GROUP BY mch_id_trimmed
            """)
            
            # Join merchant metrics to the processed DataFrame
            processed_sdf = processed_sdf.join(merchant_fraud_rate, on="mch_id_trimmed", how="left")
            processed_sdf = processed_sdf.join(merchant_fraud_amount, on="mch_id_trimmed", how="left")
            processed_sdf = processed_sdf.join(merchant_small_fraud, on="mch_id_trimmed", how="left")
            processed_sdf = processed_sdf.join(merchant_large_fraud, on="mch_id_trimmed", how="left")
            
            # Create merchant risk tiers - We'll use Spark's UDF for this
            # First, collect distinct fraud rates to calculate percentiles (small operation)
            merchant_rates = merchant_fraud_rate.select("merchant_fraud_rate").distinct().collect()
            merchant_rates = [row["merchant_fraud_rate"] for row in merchant_rates if row["merchant_fraud_rate"] is not None]
            
            if merchant_rates:
                percentiles = [0, 50, 90, 95, 100]
                bins = np.percentile(merchant_rates, percentiles)
                bins = np.unique(bins)
                
                # Define a UDF to assign risk tiers
                def assign_merchant_risk_tier(fraud_rate):
                    if fraud_rate is None:
                        return "unknown"
                    for i in range(1, len(bins)):
                        if fraud_rate <= bins[i]:
                            if i == 1:
                                return "low"
                            elif i == 2:
                                return "medium"
                            elif i == 3:
                                return "high"
                            else:
                                return "very_high"
                    return "unknown"
                
                # Register the UDF
                assign_merchant_risk_tier_udf = F.udf(assign_merchant_risk_tier)
                
                # Apply the UDF to create risk tiers
                processed_sdf = processed_sdf.withColumn(
                    "merchant_risk_tier", 
                    assign_merchant_risk_tier_udf(F.col("merchant_fraud_rate"))
                )
                
                # One-hot encode the risk tier
                indexer = StringIndexer(
                    inputCol="merchant_risk_tier", 
                    outputCol="merchant_risk_tier_idx",
                    handleInvalid="keep"
                )
                encoder = OneHotEncoder(
                    inputCols=["merchant_risk_tier_idx"],
                    outputCols=["merchant_risk_tier_enc"]
                )
                
                pipeline = Pipeline(stages=[indexer, encoder])
                processed_sdf = pipeline.fit(processed_sdf).transform(processed_sdf)
                
                # Drop the intermediate columns
                processed_sdf = processed_sdf.drop("merchant_risk_tier", "merchant_risk_tier_idx")
            
            # Drop the original merchant ID
            processed_sdf = processed_sdf.drop("mch_id_trimmed")
        
        # Handle issuer ID features
        if 'issuer_member' in self.id_features:
            # Calculate issuer fraud rate
            issuer_fraud_rate = self.spark.sql("""
                SELECT issuer_member, 
                       AVG(CAST(is_fraud AS DOUBLE)) AS issuer_fraud_rate,
                       COUNT(*) AS issuer_tx_count,
                       AVG(amount) AS issuer_avg_amount,
                       STDDEV(amount) AS issuer_amount_std
                FROM transactions
                GROUP BY issuer_member
            """)
            
            # Calculate issuer fraud amount rate
            issuer_fraud_amount = self.spark.sql("""
                SELECT issuer_member,
                       SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) / SUM(amount) AS issuer_fraud_amount_rate
                FROM transactions
                GROUP BY issuer_member
            """)
            
            # Calculate issuer auth rate if available
            if "trans_status" in sdf.columns:
                issuer_auth_rate = self.spark.sql("""
                    SELECT issuer_member,
                           AVG(CASE WHEN trans_status = 'Y' THEN 1.0 ELSE 0.0 END) AS issuer_auth_rate
                    FROM transactions
                    GROUP BY issuer_member
                """)
                processed_sdf = processed_sdf.join(issuer_auth_rate, on="issuer_member", how="left")
            
            # Calculate issuer 3DS usage rate if available
            if "three_ds_mode" in sdf.columns:
                issuer_3ds_rate = self.spark.sql("""
                    SELECT issuer_member,
                           AVG(CASE WHEN three_ds_mode != 'N' THEN 1.0 ELSE 0.0 END) AS issuer_3ds_usage_rate
                    FROM transactions
                    GROUP BY issuer_member
                """)
                processed_sdf = processed_sdf.join(issuer_3ds_rate, on="issuer_member", how="left")
            
            # Calculate cross-border metrics if available
            if "issuer_bin_profile_issuing_cou" in sdf.columns and "merchant_country_code" in sdf.columns:
                # First add is_cross_border column
                processed_sdf = processed_sdf.withColumn(
                    "is_cross_border",
                    F.when(F.col("issuer_bin_profile_issuing_cou") != F.col("merchant_country_code"), 1).otherwise(0)
                )
                
                # Calculate cross-border rates
                issuer_cross_border = self.spark.sql("""
                    SELECT issuer_member,
                           AVG(CAST(is_cross_border AS DOUBLE)) AS issuer_cross_border_rate,
                           AVG(CASE WHEN is_cross_border = 1 THEN CAST(is_fraud AS DOUBLE) ELSE NULL END) 
                               AS issuer_cross_border_fraud_rate
                    FROM transactions
                    GROUP BY issuer_member
                """)
                processed_sdf = processed_sdf.join(issuer_cross_border, on="issuer_member", how="left")
            
            # Join issuer metrics to the processed DataFrame
            processed_sdf = processed_sdf.join(issuer_fraud_rate, on="issuer_member", how="left")
            processed_sdf = processed_sdf.join(issuer_fraud_amount, on="issuer_member", how="left")
            
            # Create issuer risk tiers
            issuer_rates = issuer_fraud_rate.select("issuer_fraud_rate").distinct().collect()
            issuer_rates = [row["issuer_fraud_rate"] for row in issuer_rates if row["issuer_fraud_rate"] is not None]
            
            if issuer_rates:
                percentiles = [0, 50, 90, 95, 100]
                bins = np.percentile(issuer_rates, percentiles)
                bins = np.unique(bins)
                
                # Define a UDF to assign risk tiers
                def assign_issuer_risk_tier(fraud_rate):
                    if fraud_rate is None:
                        return "unknown"
                    for i in range(1, len(bins)):
                        if fraud_rate <= bins[i]:
                            if i == 1:
                                return "low"
                            elif i == 2:
                                return "medium"
                            elif i == 3:
                                return "high"
                            else:
                                return "very_high"
                    return "unknown"
                
                # Register the UDF
                assign_issuer_risk_tier_udf = F.udf(assign_issuer_risk_tier)
                
                # Apply the UDF to create risk tiers
                processed_sdf = processed_sdf.withColumn(
                    "issuer_risk_tier", 
                    assign_issuer_risk_tier_udf(F.col("issuer_fraud_rate"))
                )
                
                # One-hot encode the risk tier
                indexer = StringIndexer(
                    inputCol="issuer_risk_tier", 
                    outputCol="issuer_risk_tier_idx",
                    handleInvalid="keep"
                )
                encoder = OneHotEncoder(
                    inputCols=["issuer_risk_tier_idx"],
                    outputCols=["issuer_risk_tier_enc"]
                )
                
                pipeline = Pipeline(stages=[indexer, encoder])
                processed_sdf = pipeline.fit(processed_sdf).transform(processed_sdf)
                
                # Drop the intermediate columns
                processed_sdf = processed_sdf.drop("issuer_risk_tier", "issuer_risk_tier_idx")
            
            # Drop the original issuer ID
            processed_sdf = processed_sdf.drop("issuer_member")
        
        # Handle other ID features generically
        for id_feature in [f for f in self.id_features if f not in ["mch_id_trimmed", "issuer_member"]]:
            # Calculate basic metrics
            id_metrics = self.spark.sql(f"""
                SELECT {id_feature}, 
                       AVG(CAST(is_fraud AS DOUBLE)) AS {id_feature}_fraud_rate,
                       COUNT(*) AS {id_feature}_tx_count,
                       AVG(amount) AS {id_feature}_avg_amount,
                       STDDEV(amount) AS {id_feature}_std_amount,
                       SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) / SUM(amount) AS {id_feature}_fraud_amount_rate
                FROM transactions
                GROUP BY {id_feature}
            """)
            
            # Join to processed DataFrame
            processed_sdf = processed_sdf.join(id_metrics, on=id_feature, how="left")
            
            # Create risk tiers
            id_rates = id_metrics.select(f"{id_feature}_fraud_rate").distinct().collect()
            id_rates = [row[f"{id_feature}_fraud_rate"] for row in id_rates if row[f"{id_feature}_fraud_rate"] is not None]
            
            if id_rates:
                percentiles = [0, 50, 90, 95, 100]
                bins = np.percentile(id_rates, percentiles)
                bins = np.unique(bins)
                
                # Define a UDF to assign risk tiers
                def assign_risk_tier(fraud_rate):
                    if fraud_rate is None:
                        return "unknown"
                    for i in range(1, len(bins)):
                        if fraud_rate <= bins[i]:
                            if i == 1:
                                return "low"
                            elif i == 2:
                                return "medium"
                            elif i == 3:
                                return "high"
                            else:
                                return "very_high"
                    return "unknown"
                
                # Register the UDF
                assign_risk_tier_udf = F.udf(assign_risk_tier)
                
                # Apply the UDF to create risk tiers
                processed_sdf = processed_sdf.withColumn(
                    f"{id_feature}_risk_tier", 
                    assign_risk_tier_udf(F.col(f"{id_feature}_fraud_rate"))
                )
                
                # One-hot encode the risk tier
                indexer = StringIndexer(
                    inputCol=f"{id_feature}_risk_tier", 
                    outputCol=f"{id_feature}_risk_tier_idx",
                    handleInvalid="keep"
                )
                encoder = OneHotEncoder(
                    inputCols=[f"{id_feature}_risk_tier_idx"],
                    outputCols=[f"{id_feature}_risk_tier_enc"]
                )
                
                pipeline = Pipeline(stages=[indexer, encoder])
                processed_sdf = pipeline.fit(processed_sdf).transform(processed_sdf)
                
                # Drop the intermediate columns
                processed_sdf = processed_sdf.drop(f"{id_feature}_risk_tier", f"{id_feature}_risk_tier_idx")
            
            # Drop the original ID feature
            processed_sdf = processed_sdf.drop(id_feature)
        
        # Handle categorical features
        string_indexers = []
        one_hot_encoders = []
        output_cols = []
        
        for feature in self.categorical_features:
            # Check cardinality first
            distinct_count = sdf.select(feature).distinct().count()
            
            if distinct_count <= 50:  # Reasonable threshold for one-hot encoding
                indexer_output = f"{feature}_index"
                encoder_output = f"{feature}_vec"
                
                string_indexers.append(
                    StringIndexer(
                        inputCol=feature,
                        outputCol=indexer_output,
                        handleInvalid="keep"
                    )
                )
                
                one_hot_encoders.append(
                    OneHotEncoder(
                        inputCols=[indexer_output],
                        outputCols=[encoder_output]
                    )
                )
                
                output_cols.append(encoder_output)
            else:
                # For high cardinality features, use frequency encoding
                feature_freq = self.spark.sql(f"""
                    SELECT {feature}, COUNT(*) / (SELECT COUNT(*) FROM transactions) AS {feature}_freq
                    FROM transactions
                    GROUP BY {feature}
                """)
                
                # Join to processed DataFrame
                processed_sdf = processed_sdf.join(feature_freq, on=feature, how="left")
                
                # Drop the original categorical feature
                processed_sdf = processed_sdf.drop(feature)
        
        # Apply the categorical encoding pipeline if needed
        if string_indexers and one_hot_encoders:
            categorical_pipeline = Pipeline(stages=string_indexers + one_hot_encoders)
            processed_sdf = categorical_pipeline.fit(processed_sdf).transform(processed_sdf)
            
            # Drop the original categorical features and intermediate indexed columns
            for feature in self.categorical_features:
                if feature in processed_sdf.columns:
                    processed_sdf = processed_sdf.drop(feature)
                
                indexer_output = f"{feature}_index"
                if indexer_output in processed_sdf.columns:
                    processed_sdf = processed_sdf.drop(indexer_output)
        
        # Add amount range features for decision tree
        for i, threshold in enumerate(self.amount_thresholds):
            if i == 0:
                processed_sdf = processed_sdf.withColumn(
                    f"amount_range_{i}",
                    F.when(F.col("amount") <= threshold, 1.0).otherwise(0.0)
                )
            elif i == len(self.amount_thresholds):
                processed_sdf = processed_sdf.withColumn(
                    f"amount_range_{i}",
                    F.when(F.col("amount") > self.amount_thresholds[i-1], 1.0).otherwise(0.0)
                )
            else:
                processed_sdf = processed_sdf.withColumn(
                    f"amount_range_{i}",
                    F.when(
                        (F.col("amount") > self.amount_thresholds[i-1]) & 
                        (F.col("amount") <= threshold),
                        1.0
                    ).otherwise(0.0)
                )
        
        # Filter transactions with scores in the target range
        processed_sdf = processed_sdf.filter(
            (F.col("fraud_score") >= self.min_score) & 
            (F.col("fraud_score") < self.max_score)
        )
        
        # Fill NaN values
        for column in processed_sdf.columns:
            processed_sdf = processed_sdf.withColumn(
                column,
                F.when(F.col(column).isNull(), -1).otherwise(F.col(column))
            )
        
        return processed_sdf
    
    def train_model(self, sdf, label_col="is_fraud", amount_col="amount", max_depth=5):
        """
        Train a decision tree model using Spark ML.
        
        Parameters:
        - sdf: Processed Spark DataFrame
        - label_col: Name of the target column
        - amount_col: Name of the column containing transaction amounts
        - max_depth: Maximum depth of the decision tree
        
        Returns:
        - Trained decision tree model
        """
        # Create a list of features to use (excluding the label and amount)
        feature_cols = [col for col in sdf.columns if col != label_col and col != amount_col]
        self.feature_names = feature_cols
        
        # Create a vector assembler for the features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="keep"
        )
        
        # Create the decision tree classifier
        dt = DecisionTreeClassifier(
            labelCol=label_col,
            featuresCol="features",
            maxDepth=max_depth,
            minInstancesPerNode=100,  # Prevent overfitting
            seed=42
        )
        
        # Create the pipeline
        pipeline = Pipeline(stages=[assembler, dt])
        
        # Train the model
        self.dt_model = pipeline.fit(sdf)
        
        # Extract feature importances
        tree_model = self.dt_model.stages[-1]
        self.feature_importances = list(zip(feature_cols, tree_model.featureImportances.toArray()))
        self.feature_importances.sort(key=lambda x: x[1], reverse=True)
        
        return self.dt_model
    
    def extract_rules(self, sdf, label_col="is_fraud", amount_col="amount", score_col="community_fraud_score_amount_d"):
        """
        Extract rules from the trained decision tree model.
        
        Parameters:
        - sdf: Processed Spark DataFrame
        - label_col: Name of the target column
        - amount_col: Name of the column containing transaction amounts
        - score_col: Name of the column containing community fraud scores
        
        Returns:
        - List of rule dictionaries with conditions and performance metrics
        """
        if self.dt_model is None:
            raise ValueError("Model has not been trained yet. Call train_model first.")
        
        # Get the decision tree model from the pipeline
        tree_model = self.dt_model.stages[-1]
        
        # Extract decision paths (this requires converting to PMML and parsing, or collecting rules manually)
        # As Spark doesn't provide direct access to decision paths, we'll use an alternative approach
        # We'll convert the model to a string representation and extract rules
        
        # Since we can't easily extract decision paths from Spark's decision tree model directly,
        # we'll use model predictions and analyze patterns in the data to construct rules
        
        # Add predictions to the DataFrame
        predictions = self.dt_model.transform(sdf)
        
        # Register the DataFrame as a temp view for SQL queries
        predictions.createOrReplaceTempView("predictions")
        
        # Get the important features based on feature importance
        top_features = [f[0] for f in self.feature_importances[:10]]  # Take top 10 features
        
        # Analyze patterns in the data to construct rules
        rules = []
        
        # For each amount range, generate rules based on important features
        for i in range(len(self.amount_thresholds) + 1):
            amount_condition = ""
            if i == 0:
                amount_condition = f"amount <= {self.amount_thresholds[0]}"
            elif i == len(self.amount_thresholds):
                amount_condition = f"amount > {self.amount_thresholds[-1]}"
            else:
                amount_condition = f"amount > {self.amount_thresholds[i-1]} AND amount <= {self.amount_thresholds[i]}"
            
            # Find patterns where the model predicts fraud with high confidence
            patterns_query = f"""
                SELECT 
                    {amount_condition} as amount_condition,
                    {', '.join(top_features)} 
                FROM predictions
                WHERE prediction = 1.0 AND probability[1] >= 0.7
                GROUP BY {amount_condition}, {', '.join(top_features)}
                HAVING COUNT(*) >= 50
            """
            
            try:
                patterns = self.spark.sql(patterns_query)
                
                if patterns.count() > 0:
                    # For each pattern, construct a rule
                    for pattern in patterns.collect():
                        rule_conditions = [f"({amount_condition})"]
                        
                        # Add conditions for important features
                        for feature in top_features:
                            value = pattern[feature]
                            if value is not None and value != -1:  # Skip null or default values
                                # Handle numeric and categorical features differently
                                if isinstance(value, (int, float)):
                                    # For score features, use narrower ranges
                                    if feature == score_col:
                                        lower_bound = max(0, value - 0.05)
                                        upper_bound = min(1, value + 0.05)
                                        rule_conditions.append(f"event.{feature} >= {lower_bound:.3f} AND event.{feature} < {upper_bound:.3f}")
                                    else:
                                        rule_conditions.append(f"event.{feature} = {value}")
                                else:
                                    rule_conditions.append(f"event.{feature} = '{value}'")
                        
                        # Construct the rule string
                        rule_string = " AND ".join(rule_conditions)
                        
                        # Evaluate the rule
                        rule_query = f"""
                            SELECT 
                                COUNT(*) as num_transactions,
                                SUM(CAST({label_col} AS INT)) as num_frauds,
                                SUM(CASE WHEN {label_col} = 1 THEN {amount_col} ELSE 0 END) as captured_fraud_amount
                            FROM predictions
                            WHERE {rule_string.replace('event.', '')}
                        """
                        
                        rule_stats = self.spark.sql(rule_query).collect()[0]
                        
                        num_transactions = rule_stats["num_transactions"]
                        num_frauds = rule_stats["num_frauds"]
                        captured_fraud_amount = rule_stats["captured_fraud_amount"]
                        
                        # Calculate precision
                        precision = num_frauds / num_transactions if num_transactions > 0 else 0
                        
                        # Calculate amount recall
                        total_fraud_amount_query = f"""
                            SELECT SUM(CASE WHEN {label_col} = 1 THEN {amount_col} ELSE 0 END) as total_fraud_amount
                            FROM predictions
                        """
                        
                        total_fraud_amount = self.spark.sql(total_fraud_amount_query).collect()[0]["total_fraud_amount"]
                        amount_recall = captured_fraud_amount / total_fraud_amount if total_fraud_amount > 0 else 0
                        
                        # Filter rules with precision below target
                        if precision >= self.target_precision:
                            rules.append({
                                'rule_string': rule_string,
                                'precision': precision,
                                'amount_recall': amount_recall,
                                'num_transactions': num_transactions,
                                'num_frauds': num_frauds,
                                'captured_fraud_amount': captured_fraud_amount
                            })
            except Exception as e:
                print(f"Error generating rules for amount range {i}: {str(e)}")
        
        # Sort rules by amount recall (highest first)
        rules.sort(key=lambda x: x['amount_recall'], reverse=True)
        
        return rules
    
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
        
        # Format the score ranges in the amount conditions
        formatted_rules = []
        for rule in selected_rules:
            # Parse the rule string
            rule_parts = rule['rule_string'].split(' AND ')
            formatted_parts = []
            
            for part in rule_parts:
                # Handle community_fraud_score ranges
                if "community_fraud_score_amount_d" in part and ">=" in part and "AND" in part and "<" in part:
                    # Extract the bounds
                    lower_bound = float(part.split(">=")[1].split("AND")[0].strip())
                    upper_bound = float(part.split("<")[1].strip())
                    formatted_parts.append(f"event.community_fraud_score_amount_d >= {lower_bound:.3f} AND event.community_fraud_score_amount_d < {upper_bound:.3f}")
                # Handle amount ranges
                elif "amount" in part and ">" in part and "<=" in part:
                    # Extract the bounds
                    lower_bound = float(part.split(">")[1].split("AND")[0].strip())
                    upper_bound = float(part.split("<=")[1].strip())
                    formatted_parts.append(f"event.amount > {lower_bound} AND event.amount <= {upper_bound}")
                elif "amount" in part and "<=" in part:
                    # Extract the bound
                    upper_bound = float(part.split("<=")[1].strip())
                    formatted_parts.append(f"event.amount <= {upper_bound}")
                elif "amount" in part and ">" in part:
                    # Extract the bound
                    lower_bound = float(part.split(">")[1].strip())
                    formatted_parts.append(f"event.amount > {lower_bound}")
                else:
                    # Keep other conditions as is, but replace any "event." prefixes
                    formatted_parts.append(part if part.startswith("event.") else f"event.{part}")
            
            # Join the formatted parts
            formatted_rule = " AND ".join(formatted_parts)
            formatted_rules.append(formatted_rule)
        
        # Combine rules into a single ruleset with 'or' operator
        combined_rule = " OR ".join([f"({rule})" for rule in formatted_rules])
        
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

# Example usage in a Spark environment
def example_spark_workflow():
    """
    Example workflow for using the SparkFraudRuleGenerator with sample data.
    """
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("FraudRuleGeneratorExample") \
        .config("spark.sql.session.timeZone", "UTC") \
        .getOrCreate()
    
    # For demonstration, creating a synthetic dataset
    # In a real environment, you would load your data from storage
    # For example: sdf = spark.read.parquet("hdfs:///path/to/fraud/data")
    
    # Create a sample dataframe with random data
    import random
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType
    
    # Define schema
    schema = StructType([
        StructField("fraud_score", DoubleType(), True),
        StructField("is_fraud", IntegerType(), True),
        StructField("amount", DoubleType(), True),
        StructField("community_fraud_score_amount_d", DoubleType(), True),
        StructField("mch_id_trimmed", StringType(), True),
        StructField("issuer_member", StringType(), True),
        StructField("panalias_hab_cp_bill_count_45d", IntegerType(), True),
        StructField("panalias_hab_ipaddress_count_4", IntegerType(), True),
        StructField("ship_address_usage_ind", StringType(), True),
        StructField("isres_dpt_estimated_fact", IntegerType(), True),
        StructField("is_correct_bill_dept", BooleanType(), True),
        StructField("card_profile_dep", StringType(), True),
        StructField("merchant_country_code", StringType(), True),
        StructField("trans_status", StringType(), True),
        StructField("three_ds_mode", StringType(), True)
    ])
    
    # Generate sample data
    n_samples = 100000  # Use a larger number in production
    
    # Create merchant and issuer IDs
    merchant_ids = [f"MERCH{i:06d}" for i in range(1000)]
    issuer_ids = [f"ISS{i:05d}" for i in range(200)]
    countries = ["FR", "US", "UK", "DE", "ES", "IT", "JP", "CN", "AU", "BR"]
    ship_inds = ["01", "02", "03", "04", "05"]
    
    # Generate random data
    data = []
    for _ in range(n_samples):
        fraud_score = random.uniform(0, 100)
        
        # Set fraud label based on score (higher score = higher fraud probability)
        if fraud_score >= 70:
            is_fraud = 1 if random.random() < 0.8 else 0  # 80% chance of fraud if score >= 70
        elif 50 <= fraud_score < 70:
            is_fraud = 1 if random.random() < 0.3 else 0  # 30% chance of fraud if score in [50,70)
        else:
            is_fraud = 1 if random.random() < 0.05 else 0  # 5% chance of fraud otherwise
        
        # Generate amount with exponential distribution (higher amounts are less common)
        amount = random.expovariate(1/5000)  # Mean of 5000
        
        # Community fraud score correlated with is_fraud
        base_score = random.uniform(0.5, 0.9) if is_fraud else random.uniform(0.1, 0.7)
        community_score = min(max(base_score + random.uniform(-0.1, 0.1), 0), 1)
        
        # Random categorical values
        mch_id = random.choice(merchant_ids)
        issuer_member = random.choice(issuer_ids)
        ship_address_usage_ind = random.choice(ship_inds)
        merchant_country = random.choice(countries)
        issuer_country = random.choice(countries)
        
        # Transaction status - higher rejection rate for fraud
        trans_status = "N" if (is_fraud and random.random() < 0.4) else "Y"
        
        # 3DS mode - more likely for higher risk transactions
        three_ds_mode = random.choice(["N", "C", "F"]) if fraud_score < 50 else random.choice(["C", "F", "F", "F"])
        
        # Other features
        bill_count = random.randint(0, 5)
        ip_count = random.randint(0, 10)
        is_dpt_estimated = random.randint(0, 1)
        is_correct_bill = random.choice([True, False])
        card_profile_dep = random.choice(["0", "1", "2", "3", "4", "nan", ""])
        
        # Add the row
        data.append((
            fraud_score, is_fraud, amount, community_score, mch_id, issuer_member,
            bill_count, ip_count, ship_address_usage_ind, is_dpt_estimated,
            is_correct_bill, card_profile_dep, merchant_country, trans_status, three_ds_mode
        ))
    
    # Create Spark DataFrame
    sdf = spark.createDataFrame(data, schema)
    
    # Initialize the rule generator
    generator = SparkFraudRuleGenerator(min_score=50, max_score=70, target_precision=0.3)
    
    # Identify categorical and ID features
    categorical_features = ['ship_address_usage_ind', 'card_profile_dep']
    id_features = ['mch_id_trimmed', 'issuer_member']
    
    # Preprocess the data
    processed_sdf = generator.preprocess_data(
        sdf,
        categorical_features=categorical_features,
        id_features=id_features
    )
    
    # Train the model
    generator.train_model(processed_sdf, label_col="is_fraud", amount_col="amount", max_depth=4)
    
    # Extract rules
    rules = generator.extract_rules(
        processed_sdf,
        label_col="is_fraud",
        amount_col="amount",
        score_col="community_fraud_score_amount_d"
    )
    
    # Generate optimized ruleset
    ruleset = generator.generate_optimized_ruleset(rules, max_rules=3)
    
    print("Generated Rule:")
    print(ruleset['combined_rule'])
    print("\nRule Metrics:")
    for key, value in ruleset['metrics'].items():
        print(f"{key}: {value}")
    
    return ruleset

if __name__ == "__main__":
    example_spark_workflow()
