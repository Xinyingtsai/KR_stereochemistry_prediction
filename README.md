# KR Stereochemistry Prediction

This repository contains code and data for predicting ketoreductase (KR) stereochemistry in type I polyketide synthases (PKSs) using two complementary machine-learning approaches:

**A site-specific (residue-level) classification model**

**A protein language model (PLM)–based classification model**

The goal of this project is to improve stereochemical prediction of KR domains beyond traditional motif-based rules by leveraging sequence-level and representation-learning approaches.

## Project Overview

Ketoreductase (KR) domains control the stereochemistry of β-hydroxy groups in polyketide products. 
In this project, we explore two strategies:

**1. Site-specific classification based on aligned KR sequences and residue-level features**

**2. PLM-based classification using embeddings from protein language models (ESM)**


## Repository Structure

(1) **TypeA_aligned.fasta**: Multiple sequence alignment of A-type KR domains

(2) **TypeB_aligned.fasta**: Multiple sequence alignment of B-type KR domains

(3) **KRc.xlsx**: Raw data and labels for KR sequences

(4) **KR_Extraction.ipynb**: KR Domain Extraction (MIBiG 4.0)

(5) **Site_specific.ipynb**: Site-specific Classification Model

(6) **ESM.ipynb**: PLM-based Classification Model

## How to Run

Prepare KR sequences and data

Run KR_Extraction.ipynb to extract and preprocess KR domains

Run Site_specific.ipynb for residue-level classification

Run ESM.ipynb for PLM-based classification

## Applications

Predicting KR stereochemistry in uncharacterized PKS biosynthetic gene clusters

Assisting rational PKS engineering and module design

Bridging sequence diversity gaps not covered by rule-based methods
