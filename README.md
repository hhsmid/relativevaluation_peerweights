# relativevaluation_peerweights
Python code for the paper "Machine Learning-Based Relative Valuation: A Unified Framework with Peer Weights from Clustering and Tree-Based Models"

The "data" folder includes the script for data collection and preparation. The "models" folder includes scripts for model execution and peer weight computations. The "results" folder includes scripts for reproducing evaluation metrics, test procedures, and regressions.


Abstract:

This thesis proposes a data-driven and interpretable framework for relative valuation, in which machine learning models predict firm value using peer-weighted averages of comparable firms. Peer weights are derived from the structural properties of clustering and tree-based models under a unified formal definition, and applied to standardized valuation metrics known as valuation multiples. The framework is evaluated empirically on U.S. firm-level data using K-Means, Hierarchical Agglomerative Clustering, Gaussian Mixture Models, Fuzzy C-Means, Gradient Boosting Machines, and Random Forests. Tree-based models consistently achieve higher valuation accuracy than clustering methods, albeit at higher computational cost. Gradient Boosting Machines and Random Forests achieve comparable valuation accuracy but generate markedly different peer-weight structures. To assess whether valuation errors capture mispricing, I construct portfolios sorted by model-implied valuation errors. Portfolios based on tree-based models yield statistically significant abnormal returns up to 15\% annually, providing evidence that these models effectively proxy for unobserved fundamental value.


Keywords: Relative Valuation, Peer Weights, Clustering, Random Forest, Gradient Boosting
