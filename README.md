# **Flight Delay Prediction**

Final project developed for the Statistics for Data Science Specialization at Pontifícia Universidade Católica de Minas Gerais.

## **Project Summary**

For years, commercial flights have been frequently used by many people for various purposes. Still, problems related to this industry are widely recognized by passengers, with flight delays among the main ones.

Several machine learning models can be used in an attempt to track these patterns, learning from the past and predicting future flights.

A diverse feature set can help a lot when it comes to notifying people earlier about the delay, better allocate workforce, set prices to avoid losses or to prevent the buyer from arriving too early at the airport or reserving time-sensitive accommodations based on past similar (or even identical) flights.

Flight delays are more than just a small problem. In spite of the passengers' problems, mainly the missed flight connections or lost reservations, they cost a lot to airlines and threaten [their historically low gross margin](https://www.iata.org/en/iata-repository/publications/economic-reports/airline-profits-hit-record-high-but-margins-stay-thin/). 

## **Dataset**

The USA government, through the Bureau of Transportation Statistics (BTS) of the Department of Transportation (DoT), provides a report called [On-Time : Reporting Carrier On-Time Performance (1987-present)](https://transtats.bts.gov/TableInfo.asp?QO_fu146_anzr=b0-gvzr&V0s1_b0yB=D&gnoyr_VQ=FGJ). From it, several sampling versions were created and published online. The specific dataset used at this study has sampled flights that occurred between January 2019 and August 2023.

The dataset used is not included in this repository due to its size (around 600 MB and 3 million records). It was downloaded on 13 June 2026 from [Kaggle](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/data). A representative random sample (`data/sample_flights.csv`) generated from the original dataset is included for demonstration purposes and it can be regenerated using `scripts/create_sample.py`.

## **Project Pipeline**

```mermaid
flowchart LR
    A([Dataset]) --> B([Data Processing])
    B --> C([Feature Engineering])
    C --> H([Feature Enhancement])
    H --> D([Model Training])
    D --> E([Model Evaluation])
    E --> F([SHAP Analysis])
    F --> G([REST API])
```

## **Highlights**

- End-to-end ML pipeline;
- Important variables created during feature engineering;
- SHAP-based model interpretation;
- REST API for future flight predictions.

## **Main Results**

Accuracy, f1 score, precision and recall metrics were used to evaluate all trained models. Since it is a classification task, f1 score and recall were slightly preferred over the others.

Since metrics were close among all models, fully featured XGBoost was chosen the best one. This model scores were:
| Metric | Score |
  |---------|----------|
  | Accuracy | 0.6680 |
  | Precision | 0.3009 |
  | Recall | 0.6135 |
  | F1 | 0.4037 |

Setting all the features allows more insights at the built API and that added to this decision.

SHAP TreeExplainer was used to create table and charts on features importance, helping select the best for new training iterations.

An API was built and it allows users to input future flights information to guide buying and pricing decisions, providing historical concerns surrounding the route, but no final decision is taken, because people should take it.

## **Technologies**

The project was developed in Python 3.12 using the following libraries and tools:

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- FastAPI
- MLflow
- DagsHub
- Matplotlib

The complete list of dependencies and their versions is available in `requirements.txt`.

## **Repository Structure**

```
flight-delay-prediction/
│
├── .gitignore
│
├── README.md
|
├── requirements.txt
|
├── data/                                  #sample dataset for tests replication
│   └── sample_flights.csv
|
├── eda/                                   #original dataset exploration and analysis
│   └── data_exploration.py
│
├── images/                                #images directory
│   ├── eda/                               #charts and tables from eda analysis
│   |   ├── airline_codes.png
│   |   ├── airline_frequency_delay.png
│   |   ├── airport_codes_and_cities.png
│   |   ├── correlation_heatmap.png
│   |   ├── day_delay_rates.png
│   |   ├── delay_by_states.png
│   |   ├── delay_distributions.png
│   |   ├── geographic_coverage.png
│   |   ├── missing_values.png
│   |   ├── month_delay_rates.png
│   |   └── volume_over_time.png
│   |
│   └── models/                             #charts and tables from SHAP feature analysis
│       ├── feature_imnportance.png
│       ├── feature_imnportance_table.png
│       ├── summary_plot.png
│       └── summary_plot_bar.png
|
└── scripts/                                #complete pipeline codes
    ├── config.py
    ├── create_sample.py
    ├── data_processing.py
    ├── extra_features.py
    ├── feature_engineering.py
    ├── modeling.py
    |
    ├── api/                                #REST API codes
    |   ├── api.py
    |   ├── insights.py
    |   ├── predictor.py
    |   ├── schemas.py
    |   └── train_api.py
    |
    ├── evaluation and analysis/            #model metrics and SHAP analysis
    |   ├── model_evaluation.py
    |   └── shap_analysis.ipynb
    |
    └── models/                             #each model train algorithm
        ├── lightgbm.py
        ├── logistic_regression.py
        ├── random_forest.py
        └── xgboost.py
```

## **Other References**

[USA Regions and Divisions](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf) | 
[Flight Distance Classification](https://www.iata.org/en/iata-repository/publications/economic-reports/regional-air-connectivity-in-the-united-states/)
