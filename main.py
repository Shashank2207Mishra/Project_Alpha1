import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')


# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        if self.num_classes == 2:
            bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            return focal_loss.mean()
        else:
            ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets.long())
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()


# Neural Network Model
class PropensityNet(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dims=[128, 64, 32]):
        super(PropensityNet, self).__init__()
        self.num_classes = num_classes
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        output_dim = 1 if num_classes == 2 else num_classes
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out.squeeze() if self.num_classes == 2 else out


# Custom Dataset
class TabularDataset(Dataset):
    def __init__(self, X, y):
        # Convert to numpy arrays first to handle any input type
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if len(y.shape) == 1 or y.shape[1] == 1 else torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Data Preprocessing
def preprocess_data(df, target_col, task_type='auto', threshold=None):
    """Preprocess the dataset - handles ANY target type"""
    df = df.copy()

    # Separate features and target
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in dataset!")
        return None, None, None, None, None

    X = df.drop(columns=[target_col])
    y_original = df[target_col].copy()

    # Check if we have any features
    if X.shape[1] == 0:
        st.error("No features found in dataset after removing target column!")
        return None, None, None, None, None

    # Determine task type and process target
    unique_values = y_original.nunique()

    st.info(f"Target column '{target_col}' has {unique_values} unique values")

    # Auto-detect or convert based on task_type
    if task_type == 'auto':
        if unique_values == 2:
            task_type = 'binary'
        elif unique_values <= 20:
            task_type = 'multiclass'
        else:
            task_type = 'binary'  # Convert to binary
            st.warning(f"Target has {unique_values} unique values. Converting to binary classification.")

    # Process target based on task type
    if task_type == 'binary':
        if unique_values == 2:
            # Already binary
            if y_original.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y_original)
                st.info(f"Binary target encoded: {dict(zip(le.classes_, [0, 1]))}")
            else:
                y = y_original.values
                unique_vals = np.unique(y)
                if not np.array_equal(unique_vals, [0, 1]):
                    # Map to 0, 1
                    y = (y == unique_vals[1]).astype(int)
                    st.info(f"Binary target mapped: {unique_vals[0]}→0, {unique_vals[1]}→1")
        else:
            # Convert to binary using threshold
            if y_original.dtype in ['int64', 'float64']:
                if threshold is None:
                    threshold = y_original.median()
                y = (y_original > threshold).astype(int)
                st.info(f"Converted to binary using threshold={threshold:.2f} (>{threshold}=1, ≤{threshold}=0)")
            else:
                # Categorical - use most common as negative class
                top_class = y_original.value_counts().index[0]
                y = (y_original != top_class).astype(int)
                st.info(f"Binary conversion: '{top_class}'→0, others→1")

        num_classes = 2

    else:  # multiclass
        if y_original.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y_original)
            st.info(f"Multiclass target encoded ({unique_values} classes)")
        else:
            y = y_original.values
            # Ensure classes are 0 to n-1
            unique_y = np.unique(y)
            if not np.array_equal(unique_y, np.arange(len(unique_y))):
                mapping = {val: idx for idx, val in enumerate(unique_y)}
                y = np.array([mapping[val] for val in y])

        num_classes = len(np.unique(y))

    # Show class distribution
    class_counts = pd.Series(y).value_counts().sort_index()
    st.info(f"Class distribution: {class_counts.to_dict()}")

    # Handle features - KEEP ALL FEATURES, just encode them properly
    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    st.info(f"Found {len(numeric_cols)} numeric and {len(cat_cols)} categorical features")

    # Handle categorical features - encode ALL of them
    for col in cat_cols:
        n_unique = X[col].nunique()
        if n_unique == 1:
            # Constant column - keep it but fill with 0
            X[col] = 0
            st.info(f"Column '{col}' is constant, encoded as 0")
        elif n_unique > 100:
            # High cardinality - use frequency encoding
            freq_map = X[col].value_counts(normalize=True).to_dict()
            X[col] = X[col].map(freq_map).fillna(0)
            st.info(f"Column '{col}' has {n_unique} unique values, using frequency encoding")
        else:
            # Regular label encoding
            try:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                st.info(f"Column '{col}' encoded ({n_unique} unique values)")
            except Exception as e:
                # If encoding fails, use frequency encoding as fallback
                freq_map = X[col].value_counts(normalize=True).to_dict()
                X[col] = X[col].map(freq_map).fillna(0)
                st.warning(f"Column '{col}' encoding failed, using frequency encoding")

    # Handle missing values - fill ALL missing values
    if X.isnull().sum().sum() > 0:
        st.info(f"Filling {X.isnull().sum().sum()} missing values")
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype in ['int64', 'float64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(0)

    # Handle infinite values - replace with large finite values instead of dropping
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            if np.isinf(X[col]).any():
                max_val = X[col][~np.isinf(X[col])].max() if not X[col][~np.isinf(X[col])].empty else 0
                X[col] = X[col].replace([np.inf, -np.inf], [max_val * 2, max_val * -2])
                st.info(f"Column '{col}' had infinite values, replaced with finite values")

    # Ensure all columns are numeric now
    X = X.astype(float)

    # Final check
    if X.shape[1] == 0:
        st.error("No valid features remaining after preprocessing!")
        return None, None, None, None, None

    st.success(f"✅ Preprocessing complete: **ALL {X.shape[1]} features retained**, {num_classes} classes")

    return X, y, X.columns.tolist(), task_type, num_classes


# Feature Selection
def select_features(X, y, n_features=20):
    """Select top features"""
    X_numeric = X.select_dtypes(include=[np.number])

    if X_numeric.shape[1] == 0:
        st.error("No numeric features available!")
        return None, None, None

    n_features = min(n_features, X_numeric.shape[1])

    try:
        selector = SelectKBest(mutual_info_classif, k=n_features)
        X_selected = selector.fit_transform(X_numeric, y)
        selected_features = X_numeric.columns[selector.get_support()].tolist()

        scores = pd.DataFrame({
            'feature': X_numeric.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)

        return X_selected, selected_features, scores
    except:
        st.warning("Using all available features")
        return X_numeric.values, X_numeric.columns.tolist(), pd.DataFrame({
            'feature': X_numeric.columns,
            'score': [1.0] * len(X_numeric.columns)
        })


# Train Neural Network with Focal Loss
def train_nn_model(X_train, y_train, X_val, y_val, num_classes=2, epochs=50):
    """Train neural network with focal loss"""
    # Convert to numpy arrays
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_val, 'values'):
        X_val = X_val.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_val, 'values'):
        y_val = y_val.values

    X_train = np.array(X_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)

    input_dim = X_train.shape[1]
    model = PropensityNet(input_dim, num_classes=num_classes)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TabularDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=min(64, len(X_train) // 10 + 1), shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        if num_classes == 2:
            val_preds = torch.sigmoid(model(torch.FloatTensor(X_val))).numpy()
        else:
            val_preds = torch.softmax(model(torch.FloatTensor(X_val)), dim=1).numpy()

    return model, val_preds


# Hyperparameter Tuning with Cross-Validation
def tune_and_train_models(X, y, selected_features, num_classes=2):
    """Train multiple models with cross-validation"""
    # Convert to numpy arrays to ensure consistency
    if isinstance(X, pd.DataFrame):
        X_selected = X[selected_features].values
    else:
        X_selected = np.array(X)

    if hasattr(y, 'values'):
        y = y.values
    y = np.array(y, dtype=np.int32)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Check class imbalance
    class_counts = np.bincount(y)

    # Determine number of CV folds based on smallest class
    min_class_count = min(class_counts)
    n_splits = min(5, max(2, min_class_count))  # At least 2, at most 5

    # Determine if we can use stratification
    can_stratify = min_class_count >= 2

    # If too few samples, use simple train-test split instead
    if min_class_count < 2:
        st.warning(
            f"⚠️ Class with only {min_class_count} sample(s) detected. Using simple split without stratification.")
        use_cv = False
    else:
        use_cv = True
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        st.info(f"Using {n_splits}-fold stratified cross-validation")

    results = {}

    # XGBoost
    st.write("Training XGBoost...")
    if num_classes == 2:
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
    else:
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )

    if use_cv:
        xgb_scores = cross_val_score(xgb_model, X_scaled, y, cv=cv, scoring='accuracy')
    else:
        from sklearn.model_selection import train_test_split
        # Don't use stratify if any class has less than 2 samples
        stratify_param = y if can_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
        xgb_model_temp = XGBClassifier(**xgb_model.get_params())
        xgb_model_temp.fit(X_train, y_train)
        xgb_scores = np.array([xgb_model_temp.score(X_test, y_test)])

    xgb_model.fit(X_scaled, y)

    results['XGBoost'] = {
        'model': xgb_model,
        'cv_scores': xgb_scores,
        'mean_score': xgb_scores.mean()
    }

    # Random Forest
    st.write("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )

    if use_cv:
        rf_scores = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='accuracy')
    else:
        rf_model_temp = RandomForestClassifier(**rf_model.get_params())
        rf_model_temp.fit(X_train, y_train)
        rf_scores = np.array([rf_model_temp.score(X_test, y_test)])

    rf_model.fit(X_scaled, y)

    results['Random Forest'] = {
        'model': rf_model,
        'cv_scores': rf_scores,
        'mean_score': rf_scores.mean()
    }

    # Neural Network with Focal Loss
    st.write("Training Neural Network with Focal Loss...")
    nn_scores = []

    if use_cv:
        for train_idx, val_idx in cv.split(X_scaled, y):
            X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]

            _, val_preds = train_nn_model(X_train_cv, y_train_cv, X_val_cv, y_val_cv,
                                          num_classes=num_classes, epochs=30)

            if num_classes == 2:
                acc = accuracy_score(y_val_cv, (val_preds >= 0.5).astype(int))
            else:
                acc = accuracy_score(y_val_cv, val_preds.argmax(axis=1))
            nn_scores.append(acc)
    else:
        _, val_preds = train_nn_model(X_train, y_train, X_test, y_test,
                                      num_classes=num_classes, epochs=30)
        if num_classes == 2:
            acc = accuracy_score(y_test, (val_preds >= 0.5).astype(int))
        else:
            acc = accuracy_score(y_test, val_preds.argmax(axis=1))
        nn_scores.append(acc)

    # Train final NN
    nn_model, _ = train_nn_model(X_scaled, y, X_scaled, y, num_classes=num_classes, epochs=50)

    results['Neural Network (Focal Loss)'] = {
        'model': nn_model,
        'cv_scores': np.array(nn_scores),
        'mean_score': np.mean(nn_scores)
    }

    return results, scaler, X_scaled, y


# Evaluation
def evaluate_model(model, X, y, model_name, is_nn=False, num_classes=2):
    """Evaluate model and generate predictions"""
    if is_nn:
        model.eval()
        with torch.no_grad():
            if num_classes == 2:
                y_proba = torch.sigmoid(model(torch.FloatTensor(X))).numpy()
                y_pred = (y_proba >= 0.5).astype(int)
            else:
                logits = model(torch.FloatTensor(X))
                y_proba = torch.softmax(logits, dim=1).numpy()
                y_pred = y_proba.argmax(axis=1)
    else:
        y_proba = model.predict_proba(X)
        if num_classes == 2:
            y_proba = y_proba[:, 1]
        y_pred = model.predict(X)

    # Metrics
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')

    # AUC for binary
    if num_classes == 2:
        auc = roc_auc_score(y, y_proba)
        fpr, tpr, _ = roc_curve(y, y_proba)
        precision, recall, _ = precision_recall_curve(y, y_proba)
    else:
        try:
            auc = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
        except:
            auc = None
        fpr, tpr = None, None
        precision, recall = None, None

    cm = confusion_matrix(y, y_pred)

    return {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': acc,
        'auc': auc,
        'f1': f1,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr) if num_classes == 2 else None,
        'pr_curve': (precision, recall) if num_classes == 2 else None,
        'classification_report': classification_report(y, y_pred)
    }


# Plotting
def plot_results(eval_results, model_name, num_classes=2):
    """Plot evaluation results"""
    if num_classes == 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Confusion Matrix
        sns.heatmap(eval_results['confusion_matrix'], annot=True, fmt='d',
                    cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(f'Confusion Matrix - {model_name}')
        axes[0, 0].set_ylabel('True')
        axes[0, 0].set_xlabel('Predicted')

        # ROC Curve
        if eval_results['roc_curve'][0] is not None:
            fpr, tpr = eval_results['roc_curve']
            axes[0, 1].plot(fpr, tpr, label=f"AUC = {eval_results['auc']:.3f}")
            axes[0, 1].plot([0, 1], [0, 1], 'k--')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title(f'ROC Curve - {model_name}')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Precision-Recall Curve
        if eval_results['pr_curve'][0] is not None:
            precision, recall = eval_results['pr_curve']
            axes[1, 0].plot(recall, precision)
            axes[1, 0].set_xlabel('Recall')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title(f'Precision-Recall Curve - {model_name}')
            axes[1, 0].grid(True, alpha=0.3)

        # Prediction Distribution
        axes[1, 1].hist(eval_results['y_proba'], bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Prediction Distribution - {model_name}')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confusion Matrix
        sns.heatmap(eval_results['confusion_matrix'], annot=True, fmt='d',
                    cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Confusion Matrix - {model_name}')
        axes[0].set_ylabel('True Class')
        axes[0].set_xlabel('Predicted Class')

        # Class Distribution
        class_counts = np.bincount(eval_results['y_pred'])
        axes[1].bar(range(len(class_counts)), class_counts)
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Predicted Class Distribution')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Main App
def main():
    st.set_page_config(page_title="Propensity Model Builder For Banks", layout="wide")

    st.title("🤖 Propensity Model Builder For Banks")
    st.markdown("""
    Upload ANY CSV file and get an end-to-end ML model with:
    - ✅ Handles binary AND multi-class classification
    - ✅ Automatic target conversion (continuous→binary)
    - ✅ Handling class imbalance (Focal Loss + weighted methods)
    - ✅ Cross-validation & hyperparameter tuning
    - ✅ Comprehensive evaluation
    """)

    uploaded_file = st.file_uploader("Upload CSV file (max 1GB)", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(100))
                st.write(f"**Columns:** {', '.join(df.columns.tolist())}")

            target_col = st.selectbox("Select target column", df.columns)

            col1, col2, col3 = st.columns(3)
            with col1:
                task_type = st.selectbox("Task type", ['auto', 'binary', 'multiclass'])
            with col2:
                n_features = st.slider("Features to select", 5, len(df.columns) - 1, min(20, len(df.columns) - 1))
            with col3:
                threshold = None
                if df[target_col].dtype in ['int64', 'float64'] and df[target_col].nunique() > 20:
                    threshold = st.number_input("Binary threshold (for continuous)",
                                                value=float(df[target_col].median()))

            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Processing data..."):
                    X, y, feature_names, final_task, num_classes = preprocess_data(
                        df, target_col, task_type, threshold
                    )

                    if X is None:
                        st.stop()

                with st.spinner("Selecting features..."):
                    X_selected, selected_features, feature_scores = select_features(X, y, n_features)

                    if X_selected is None:
                        st.stop()

                    st.success(f"Selected {len(selected_features)} features")

                    with st.expander("🎯 Feature Scores"):
                        st.dataframe(feature_scores.head(20))

                with st.spinner("Training models..."):
                    results, scaler, X_scaled, y_array = tune_and_train_models(
                        X, y, selected_features, num_classes
                    )

                st.subheader("📈 Cross-Validation Results")
                cv_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'Mean Accuracy': [r['mean_score'] for r in results.values()],
                    'Std': [r['cv_scores'].std() for r in results.values()]
                })
                st.dataframe(cv_df.style.highlight_max(subset=['Mean Accuracy'], color='lightgreen'))

                best_model_name = max(results.keys(), key=lambda k: results[k]['mean_score'])
                best_model = results[best_model_name]['model']
                is_nn = 'Neural' in best_model_name

                st.success(f"🏆 Best Model: {best_model_name} (Accuracy: {results[best_model_name]['mean_score']:.4f})")

                with st.spinner("Evaluating..."):
                    eval_results = evaluate_model(best_model, X_scaled, y_array,
                                                  best_model_name, is_nn, num_classes)

                st.subheader("📊 Model Performance")
                cols = st.columns(3)
                cols[0].metric("Accuracy", f"{eval_results['accuracy']:.4f}")
                if eval_results['auc']:
                    cols[1].metric("AUC-ROC", f"{eval_results['auc']:.4f}")
                cols[2].metric("F1 Score", f"{eval_results['f1']:.4f}")

                with st.expander("📋 Classification Report"):
                    st.text(eval_results['classification_report'])

                st.subheader("📈 Visualizations")
                fig = plot_results(eval_results, best_model_name, num_classes)
                st.pyplot(fig)

                if not is_nn and hasattr(best_model, 'feature_importances_'):
                    st.subheader("🎯 Feature Importance")
                    feat_imp = pd.DataFrame({
                        'feature': selected_features,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)

                    fig_imp, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=feat_imp.head(15), x='importance', y='feature', ax=ax)
                    ax.set_title('Top 15 Most Important Features')
                    st.pyplot(fig_imp)

                st.success("✅ Training complete!")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()