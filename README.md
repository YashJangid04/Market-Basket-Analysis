# Market Basket Analysis Report

## Introduction
In the highly competitive landscape of retail, understanding consumer purchasing patterns is paramount. Market basket analysis, which investigates associations between items purchased together, provides valuable insights into product affinities and customer behavior at the point of sale. By leveraging both supervised and unsupervised machine learning techniques, retailers can not only predict outcomes based on historical data but also uncover latent groupings within product assortments, leading to more informed decisions in inventory planning, layout optimization, and targeted promotions.

This report presents a two‑pronged analytical approach:
1. **Classification Evaluation**: When ground‑truth labels are available, we evaluate classification performance using accuracy, precision, recall, and visualize results via a confusion matrix heatmap.  
2. **Unsupervised Segmentation & Clustering**: In the absence of labels, we segment aisle descriptions using TF‑IDF vectorization, cosine similarity, KMeans clustering, and PCA for visualization.

---

## Methodology

1. **Data Loading & Exploration**  
   - Load the provided `Market Basket Analysis.csv` containing `aisle_id` and `aisle` fields.  
   - Inspect data types, check for missing values, and review basic aisle name distributions.

2. **Classification Workflow (Conditional)**  
   - Verify existence of `true_label` and `predicted_label` columns.  
   - If present, compute a confusion matrix and display as a heatmap.  
   - Calculate accuracy, precision (weighted), and recall (weighted).

3. **Unsupervised Segmentation & Clustering**  
   - **TF‑IDF Vectorization**: Convert aisle names into numerical feature vectors capturing term importance.  
   - **Cosine Similarity**: Measure pairwise textual similarity between aisle names and visualize with a heatmap.  
   - **KMeans Clustering**: Group aisles into _k_ clusters (default k=5).  
   - **PCA Visualization**: Reduce TF‑IDF dimensions to 2D for scatter‑plot visualization of cluster structure.

4. **Visualization & Output**  
   - Render all heatmaps and scatterplots with clear titles and axis labels.  
   - Print cluster assignments and classification metrics to the console or notebook.

---

## Code

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("10. Market Basket Analysis.csv")

# Check for classification columns
has_classification = {'true_label', 'predicted_label'}.issubset(df.columns)

if has_classification:
    # --- Classification Block ---
    y_true = df['true_label']
    y_pred = df['predicted_label']
    labels = sorted(set(y_true) | set(y_pred))
    
    # Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Metrics
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision (weighted):", precision_score(y_true, y_pred, average='weighted'))
    print("Recall (weighted):", recall_score(y_true, y_pred, average='weighted'))

else:
    # --- Clustering Block ---
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['aisle'])
    
    # Cosine similarity heatmap
    sim = cosine_similarity(X)
    sns.heatmap(sim, cmap="YlGnBu",
                xticklabels=df['aisle'], yticklabels=df['aisle'])
    plt.title('Aisle Similarity Heatmap')
    plt.show()
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['cluster'] = clusters
    
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X.toarray())
    plt.figure(figsize=(8,6))
    plt.scatter(coords[:,0], coords[:,1], c=clusters, cmap='Set2')
    for i, name in enumerate(df['aisle']):
        plt.annotate(name, (coords[i,0], coords[i,1]), fontsize=8, alpha=0.7)
    plt.title('Aisle Clusters (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
    
    # Display cluster assignments
    print(df[['aisle','cluster']])
