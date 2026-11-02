# arXiv Scientific Dataset Quality Report

## 1. Overview & Schema Integrity
- **Total Records:** 287,421
- **Schema Valid:** True
- **Missing Columns:** None (all 10 canonical columns present)

## 2. Null & Missing Values
- **Null Fields across all columns:** 0 (0.00% missing values)
- **Empty / Malformed Summaries:** 0

## 3. Identifiers & Multi-Version Papers
- **Unique Full IDs:** 287,421 (100% unique primary identifiers)
- **Duplicate Full IDs:** 0
- **Unique Base IDs (papers regardless of version):** 287,395
- **Multi-Version Paper Entries:** 26 papers have multiple version revisions (e.g. v1, v2, v3) in the dataset.
- **Version Breakdown:**
  - Version 1: 183,130 papers
  - Version 10: 33 papers
  - Version 11: 20 papers
  - Version 12: 5 papers
  - Version 13: 7 papers
  - Version 14: 6 papers
  - Version 15: 5 papers
  - Version 16: 2 papers
  - Version 17: 3 papers
  - Version 2: 68,549 papers
  - Version 20: 1 papers
  - Version 21: 2 papers
  - Version 3: 23,669 papers
  - Version 4: 7,806 papers
  - Version 5: 2,703 papers
  - Version 6: 894 papers
  - Version 7: 367 papers
  - Version 8: 139 papers
  - Version 9: 80 papers

## 4. Title & Content Statistics
- **Unique Normalized Titles:** 287,299
- **Duplicate Normalized Titles:** 122 (mostly identical papers cross-listed across subdisciplines or versions)
- **Abstract Word Count (Summary):**
  - Min: 1 words
  - 25th Percentile: 140.0 words
  - Median: 170.0 words
  - Mean: 170.96 words
  - 75th Percentile: 202.0 words
  - Max: 552 words
  - Abstracts under 10 words: 19

## 5. Temporal Range & Date Parsing
- **Publication Date Parse Failures:** 0 (0.00%)
- **Earliest Publication Date:** 1993-08-01 00:00:00
- **Latest Publication Date:** 2026-06-18 00:00:00

## 6. Categories & Disciplines
- **Unique Category Names:** 146
- **Unique Category Taxonomy Codes:** 147
- **Top 10 Categories by Paper Count:**
  - **Machine Learning:** 80,452 papers (27.99%)
  - **Computer Vision and Pattern Recognition:** 71,106 papers (24.74%)
  - **Computation and Language (Natural Language Processing):** 37,251 papers (12.96%)
  - **Artificial Intelligence:** 28,726 papers (9.99%)
  - **Machine Learning (Statistics):** 13,892 papers (4.83%)
  - **Neural and Evolutionary Computing:** 6,715 papers (2.34%)
  - **Robotics:** 5,254 papers (1.83%)
  - **Cryptography and Security:** 4,376 papers (1.52%)
  - **Image and Video Processing:** 2,771 papers (0.96%)
  - **Information Retrieval:** 2,712 papers (0.94%)
