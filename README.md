# Categorizing Risk: A Data Driven Approach to Online Casino Games
**INST414 - Module 4 Assignment**
**Author:** Oscar Fowler

## Project Overview
This repository contains the Python code and analysis used to mathematically categorize the modern online casino market into distinct risk versus reward profiles. Using a synthetic dataset of 1.2 million casino games, this project applies K-Means clustering to identify actionable groupings for Responsible Gambling Regulators.

## Methodology
The analysis was conducted using Python and the `scikit-learn` library. 
* **Features:** RTP (Return to Player), Minimum Bet, and Maximum Win.
* **Preprocessing:** `StandardScaler` was used to normalize the data, ensuring the massive payout ceilings did not mathematically dominate the percentage based RTP values.
* **Clustering:** K-Means clustering was applied using Euclidean distance. The optimal number of clusters (k=3) was determined mathematically using the Elbow Method.
* **Sampling:** Due to memory and platform constraints, a reproducible random sample of 20,000 rows (`random_state=42`) was used for the final analysis.

## Files in this Repository
* `casino_clustering.py`: The core Python script containing the data cleaning pipeline, scaling, and K-Means logic.
* `elbow_plot.png`: The generated visualization used to justify the selection of k=3.

## Note on Dataset Size
**The dataset used for this project (`online_casino_games_dataset_v2.csv`) is approximately 250 MB and exceeds GitHub's strict 100 MB file size limit.** It has been intentionally excluded from this repository via `.gitignore` to allow the code to push successfully. 

To run this code locally, you must download the "Online Casino Games Dataset — 1.2M Records" from Kaggle and place the CSV in the same root directory as the Python script.
