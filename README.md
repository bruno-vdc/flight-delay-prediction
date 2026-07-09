# Work in Progress - Flight Delay Prediction

Final project developed for the Statistics for Data Science Specialization at Pontifícia Universidade Católica de Minas Gerais.

## Business Problem

Despite being an old industry, the US commercial flight industry underwent major changes after the Airline Deregulation Act of 1978, which increased competition and contributed to making air travel more accessible and more used. However, it is still operating at tight margins.
Flight delays are relatively common and cause several losses to the companies, due to mismanagement of the workforce at the airports and extra fuel usage caused by higher cruising speeds, holding patterns, etc. According to International Air Transport Association (IATA), labor and fuel were expected to be the two largest costs to airlines for 2026. Besides the problems companies have in such situations, flight delays are also inconvenient for customers, who lose hours of work or even reservations because of the longer-than-expected time at the airport or flight.
This study proposes the use of machine learning to predict which flights are likely to be delayed, helping airlines to plan their operations, improve resource allocation and notify customers in advance and helping passengers make more informed travel decisions and book hotel accommodations or similar services.

## Dataset

The USA government, throught the Bureau of Transportation Statistics (BTS) of the Department of Transportation (DoT), provides a reporting called [On-Time : Reporting Carrier On-Time Performance (1987-present)](https://transtats.bts.gov/TableInfo.asp?QO_fu146_anzr=b0-gvzr&V0s1_b0yB=D&gnoyr_VQ=FGJ), which has registers of over 230 million domestic flights ocurred during the period present at its title. From it, several sampling versions were created and published online. The specific piece of data used at this study has sampling flights happened between January 2019 and August 2023. Not all of the columns from the original BTS report are included, but the main ones are, such as flight date, airline, origin and destination airports and cities, scheduled and actual departure, arrival and elapsed times, distance and delays.

The dataset used is not included in this repository due to its size (around 600 MB and 3 million records). It was downloaded on 13 June 2026 from [Kaggle](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/data). A representative random sample (`data/sample_flights.csv`) generated from the original dataset is included for demonstration purposes and for running the project's automated tests. The sample dataset can be regenerated using `scripts/create_sample.py`.
