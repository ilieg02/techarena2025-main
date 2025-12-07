# VIGO Team – Huawei TechArena 2025 Submission

This repository contains the code developed by the VIGO team for the Huawei TechArena 2025 competition (#top7).

## Overview
All code in this repository is written in Python and showcases our solutions for the competition’s challenges.

## The Challenge
Huawei gave us an interesting challenge which theme was spend less, do more.
Using this theme, the challenge was to upload a LLM inference pipeline which is the most acurate and precise with minimal latency. 


For this purpose, we used Qwen3 1.7B as our main large language model. We employed below strategies to achieve the results. 
## Contents
  * Fine-tuning Program
  * Rag trainer and extractor
  * Difficulty Classifier
  * Dynamic Prompts for varied subject queries
  * A calculator tool as an assitive tool for mathemtical problems.
  * Final pipeline (load.py) which combines above mentioned methods

## Results

  * Overall Accuray : 68.77%
  * Latency : 98.35%
  * Algebra : 51.15%
  * Geography : 71.92%
  * History : 83.43%
  * Chinese : 43.61%

(The results were evaluated by LLM deployed by organizers)
