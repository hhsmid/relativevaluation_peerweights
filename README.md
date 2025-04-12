# relativevaluation_peerweights
Python code for the paper "Machine Learning-Based Relative Valuation: A Unified Framework with Peer Weights from Clustering and Tree-Based Models"

This repository includes all the code needed to reproduce the analysis from the paper.

The "data" folder includes the script for data collection and preparation. The "models" folder includes scripts for model execution and peer weight computations. The "results" folder includes scripts for reproducing evaluation metrics, test procedures, and regressions.


Abstract:

This thesis develops a unified machine learning framework for relative valuation, in which clustering and tree-based models are used to predict a firm’s valuation multiple---standardized metric of firm value---as a weighted average of its peers. Peer weights are derived from the models' underlying structures, enabling data-driven, but interpretable, peer selection and weighting across model classes. Gradient Boosting Machines and Random Forests deliver similarly high valuation accuracy---outperforming clustering methods---despite producing distinct peer weight patterns, highlighting the effectiveness of diverse peer weighting schemes across model classes. Clustering-based methods underperform in valuation accuracy compared to tree-based models, but offer computational advantages in constrained settings. To assess whether the valuation prediction errors reflect deviations from firms' fundamental value, I employ portfolio sorting and evaluate risk-adjusted returns in an asset pricing framework. I find that valuation errors from tree-based models capture deviations from fundamental value, generating statistically and economically significant returns unexplained by standard risk factors. The proposed framework advances explainable machine learning in finance, offering researchers a rigorous model comparison tool and practitioners a scalable, accurate alternative to traditional valuation methods.


Keywords: Relative Valuation, Peer Weights, Clustering, Random Forest, Gradient Boosting
