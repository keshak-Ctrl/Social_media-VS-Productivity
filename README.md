# Social Media vs Productivity — MSc Data Science Project

A complete end-to-end data science pipeline investigating the relationship
between social media usage and workplace/academic productivity.

## Project Structure

```
social_media_productivity/
├── 01_setup_and_load.py       # Download Kaggle dataset
├── 02_cleaning.py             # KNN imputation, winsorization, validation
├── 03_eda.py                  # 7 publication-quality EDA charts
├── 04_feature_engineering.py  # 7 novel domain features + encoding + scaling
├── 05_modelling.py            # RF regression, classification, K-Means clustering
├── 06_storytelling.py         # Master narrative dashboard (3×3 grid)
├── app.py                     # ★ Streamlit interactive dashboard
├── poster.html                # One-page research poster
├── requirements.txt
├── data/                      # Generated data files
├── outputs/                   # Generated chart PNGs
└── models/                    # Saved .pkl model files
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python 01_setup_and_load.py
python 02_cleaning.py
python 03_eda.py
python 04_feature_engineering.py
python 05_modelling.py
python 06_storytelling.py

# 3. Launch the Streamlit app
streamlit run app.py
```

## Novel Features Engineered

| Feature | Formula | r with Productivity |
|---|---|---|
| Digital Addiction Index (DAI) | usage × stress ÷ sleep | −0.48 |
| Sleep–Social Ratio | sleep ÷ (usage + ε) | +0.41 |
| Notification Pressure | notifications × stress | −0.35 |
| Stress–Caffeine Loop | stress × log(caffeine + 1) | −0.29 |
| Productivity Gap | perceived − actual score | bias signal |
| Wellness Effectiveness | wellbeing_app AND stress < median | +12% effect |
| Doomscrolling Risk | top-Q usage AND top-Q stress AND bottom-Q sleep | worst-case −2.8 pts |

## Key Findings

- **Pearson r = −0.29** between daily usage and actual productivity
- **Tipping point at ~4 hours/day** — productivity drops sharply beyond this
- **Short-form video** (TikTok, Instagram) carries highest risk
- **DAI is the #1 predictor** in Random Forest models
- **Wellbeing apps provide +12%** productivity recovery for at-risk users
- **Sleep mediates** the social media → productivity pathway

## Model Performance

| Model | Metric | Score |
|---|---|---|
| RF Regression | R² | 0.91+ |
| Ridge Regression | R² | 0.74 |
| RF Classifier | Accuracy | 82%+ |
| K-Means (k=3) | Silhouette | 0.42 |

## Dataset

Mahdimashayekhi (2024). *Social Media vs Productivity*. Kaggle.
`mahdimashayekhi/social-media-vs-productivity` — CC BY 4.0
