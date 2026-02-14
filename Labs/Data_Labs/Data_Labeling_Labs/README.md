# Data Labeling with Snorkel – MLOps Lab

## The Significance of Data Labeling and the Role of Snorkel

In machine learning (ML) and artificial intelligence (AI), data forms the foundation upon which models are built and refined. However, raw data alone is insufficient for supervised learning — it must be labeled with meaningful annotations. Data labeling provides the structured signals that allow models to learn relationships between inputs and outputs.

High-quality labeled data:

- Enables supervised learning models to generalize effectively  
- Serves as ground truth for evaluation  
- Supports domain-specific insight generation  
- Improves reliability and robustness of ML systems  

However, manual labeling at scale is expensive, slow, and prone to inconsistencies and bias.

---

## Why Snorkel?

Snorkel is a weak supervision framework that allows practitioners to programmatically generate labels using **labeling functions (LFs)** instead of manually labeling large datasets.

Snorkel:
- Combines noisy labeling rules using a Label Model
- Learns LF accuracies and correlations
- Produces probabilistic training labels
- Enables scalable data labeling workflows

This lab walks through the core components of Snorkel using spam classification as a running example.

---

# Spam Tutorials

This lab contains three tutorials:

### `01_spam_tutorial`
Demonstrates:
- Writing heuristic labeling functions (LFs)
- Combining LFs with the Label Model
- Training a classifier using probabilistic labels

### `02_spam_data_augmentation_tutorial`
Demonstrates:
- Writing transformation functions (TFs)
- Applying augmentation policies
- Expanding training data via programmatic augmentation

### `03_spam_data_slicing_tutorial`
Demonstrates:
- Writing slicing functions (SFs)
- Monitoring model performance on specific data subsets
- Evaluating slice-aware performance

---

# Custom Modifications (Not in Original Tutorial)

The following enhancements were added beyond the original Snorkel examples:

## 🔹 Notebook 01 – Additional Heuristic Labeling Functions

Added custom spam-detection LFs:
- URL detection (`has_url`)
- Money symbol detection (`has_money_symbols`)
- Phone number detection (`has_phone_number`)
- Excess punctuation detection (`excessive_punct`)
- High uppercase ratio detection (`all_caps_ratio`)

These heuristics expand spam coverage using structural and formatting patterns commonly found in spam comments.

---

## Notebook 02 – Additional Transformation Function

Added custom transformation function:

- `normalize_excess_punct`

This function reduces repeated punctuation (e.g., "!!!", "???") to a single character, improving augmentation diversity while maintaining semantic meaning.

---

## Notebook 03 – Additional Slicing Function

Added custom slice:

- `has_url_slice`

This slice monitors performance specifically on comments containing URLs, allowing more granular evaluation of spam-related patterns.

---

# Summary

This lab demonstrates how weak supervision via Snorkel can:

- Replace manual labeling with programmatic heuristics  
- Combine noisy signals using a principled Label Model  
- Improve dataset coverage through augmentation  
- Monitor performance on targeted data slices  

The custom additions extend the original tutorials to better capture structural spam patterns and enhance augmentation and monitoring capabilities.
