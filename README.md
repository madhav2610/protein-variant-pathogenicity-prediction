🧬 Protein Variant Pathogenicity Prediction using Transformer Representations

📌 Overview
This project investigates how different representations of protein sequences influence the prediction of mutation pathogenicity. The goal is to evaluate whether transformer-based embeddings can improve prediction performance over traditional mutation-based features.
Instead of treating this as a pure modeling task, the project focuses on representation learning — specifically, how different ways of encoding mutations affect downstream classification.

🎯 Problem Statement
Given:

A protein sequence
A mutation (position + amino acid change)
Predict:

Whether the mutation is pathogenic (1) or benign (0)
Formally:

(sequence, mutation_position, wildtype_aa, mutated_aa) → label
This is a binary classification problem with biological relevance in disease understanding and genomic analysis.

⚙️ Methodology
The project is structured around comparing three levels of information:

Mutation-only features (baseline)
Local sequence context (position embedding)
Global mutation effect (embedding difference)

🧱 Dataset
Source: ClinVar Missense Mutations Dataset (April 2023)
Sequence Mapping: Protein sequences mapped via UniProt accessions to ensure accurate wild-type and mutated alignments.
Total samples: ~75,000 (55,599 train / 19,642 test)
Labels: Binary (0 = benign, 1 = pathogenic)
Features:
Protein sequence (amino acid string)
Mutation position
Wild-type amino acid (WT)
Mutated amino acid (MT)
Preprocessing:
Train/test split at protein level to prevent data leakage
Sequences longer than 1,024 tokens skipped (~41% of training data) due to GPU memory constraints — noted as a key limitation
Class imbalance handled via class_weight='balanced' in all classifiers
Zero-vector rows removed before training (rows corresponding to skipped sequences), reducing effective test evaluation to ~10,720 samples

🧪 Models & Experiments
🔹 Baseline Model — Mutation Features Only
Model: Random Forest
Input Features:
Mutation position
Wild-type amino acid
Mutated amino acid
Goal:
Establish a lower bound performance using only mutation-level information.

🔹 Strategy 2 — Position Embedding (Local Context)
Model: Logistic Regression
Embedding Source: Transformer model (protein language model)
Pipeline:

Sequence → Transformer → Extract embedding at mutation position → Classifier
Key Idea:
The mutation's local context within the protein sequence is critical for determining its effect.

🔹 Strategy 3 — Difference Embedding (Global Effect)
Model: Logistic Regression
Pipeline:

Original sequence → embedding  
Mutated sequence → embedding  
Difference = mutated − original → classifier
Key Idea:
The mutation effect can be modeled as a change in representation space.

📊 Results
Model                                F1 Score
Baseline (Random Forest)             0.537
Strategy 3 (Difference Embedding)    0.604
Strategy 2 (Position Embedding)      0.724

📈 Detailed Performance (Both Embedding Models)

Strategy 2 — Position Embedding
Accuracy: 0.76
F1 Score: 0.724

Class        Precision    Recall    F1
0 (Benign)   0.78         0.80      0.79
1 (Pathogenic) 0.73       0.72      0.72

Strategy 3 — Difference Embedding
Accuracy: 0.69
F1 Score: 0.604

Class        Precision    Recall    F1
0 (Benign)   0.69         0.79      0.74
1 (Pathogenic) 0.67       0.55      0.60

🧠 Key Insights
1. Mutation-only features are insufficient
The baseline model achieved an F1 score of ~0.53, indicating that mutation identity and position alone do not capture enough biological context.

2. Transformer embeddings significantly improve performance
Both embedding-based approaches outperform the baseline, confirming that pretrained protein models encode meaningful structural and functional information.

3. Local context > global difference representation
The most important finding:

Embedding at the mutation position (local context) outperforms global embedding difference.
This suggests:

The effect of a mutation is primarily determined by its immediate structural environment
Global sequence-level differences introduce noise or dilute signal

4. Difference embeddings underperform due to information loss
Subtracting embeddings:

Removes absolute context
Introduces noise
Fails to capture nonlinear biological effects

⚠️ Limitations
Sequences longer than 1,024 amino acids were skipped entirely (~41% of training data), which may introduce selection bias toward shorter proteins
Difference embedding is a simplistic approximation of mutation effect
No structural (3D) information included
Dataset quality depends on ClinVar annotation completeness

🚀 Future Improvements
Use concatenation instead of subtraction for mutation representation
Fine-tune transformer model instead of using frozen embeddings
Incorporate structural data (e.g., AlphaFold features)
Explore attention-based interpretability
Improve handling of long protein sequences

🧠 Conclusion
This project demonstrates that:

Transformer-based protein representations significantly improve mutation effect prediction
Local context around mutation sites is more informative than global embedding differences
Careful representation design matters more than model complexity
The results highlight the importance of biologically meaningful feature extraction when applying deep learning to sequence-based problems.

🛠️ Tech Stack
Python
PyTorch
Transformers (Hugging Face)
Scikit-learn
NumPy, Pandas

📌 Key Takeaway
This is not just a modeling task — it is a study of how different representations of biological sequences impact predictive performance.

🔗 Author
Madhav Takkar
Bioinformatics & AI/ML
