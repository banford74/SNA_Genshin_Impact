# SNA_Genshin_Impact
A integrated package used for sentiment analysis and emotion word analysis. This package is first applied in one project related to Japanese discussion texts of Genshin Impact. Based on BERT and Japanese pre-trained model "cl-tohoku/bert-base-japanese", this model was then fine-tuned by texts related to ACG culture, especially those from "5ch".

## Packages Required
### 1.Data Mining: 
Scrapy
### 2.Model Training and Fine-tuning: 
transformers
### 3.Sentiment Analysis: 
pandas, torch
### 4.Emotion Word Analysis: 
os, numpy, pandas, logging, argparse, tqdm

## Description for each directory
### 1."Data Mining-Scrapy": 
Used for mining data from open online forum
### 2."Data Mined": 
Mined data for analysis in this project (from 5ch)
### 3."Emothion Word Analysis-Lexion": 
Lexion-based analysis for different emotion(positive: anticipation, joy, surprise, trust; negative: anger, sadness, disgust, fear) and related intensity (valence-degree of pleasantness, arousal-degree of intensity, dominace-degree of control).
### 4."Sentiment Analysis-BERT": 
BERT-based maching learning model for sentiment analysis, especially for ACG-related texts.

## Code of Cite
### 1.Pre-trained Model: 
cl-tohoku/bert-base-japanese
### 2.NRC-Lexion: 
-Mohammad, S. M., & Turney, P. D. Crowdsourcing a Word-Emotion Association Lexicon[R/OL]. arXiv,2013:1308.6297.
### 3.Data-Mined, fine-tuned model, integrated package: 
-Chu, Z.Y., & Huang, M.S. Affective Interaction in Hybrid ACG Culture: Computaional Grounded Study on Japanese Players of Genshin Impact [Under Review]. 