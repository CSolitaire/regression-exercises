# Regression Project: Estimating Home Value

### Author: Corey Solitaire

## Description: 
Regression project estimating home vale with zillow database

## Project Organization

Generated with [ryans_codeup_data_science_mvp](https://github.com/RyanMcCall/ryans_codeup_data_science_mvp)

Modified from [datasciencemvp](https://github.com/cliffclive/datasciencemvp/)

```
├── README.md               <- The top-level README for developers using this project.
│
├── data                    <- All of the data for the project
│   ├── modeling            <- The prepared, processed and split datasets for modeling.
│   ├── prepared            <- The prepared datasets for exploration
│   └── raw                 <- The original, immutable data
│
├── main.py                 <- The main python script that calls all src scripts
│
├── mvp.ipynb               <- The main notebook for the project
│
├── src                     <- The source code for use in this project
│   ├── __init__.py         <- Makes src a Python module
│   ├── acquire.py          <- The script to download or generate data and store it in
│   │                          data/raw/
│   ├── explore.py          <- The script for creating any visuals that need to be stored
│   │                          in visuals/generated_graphics/
│   ├── model.py            <- The script for preprocessing, modeling, and interpreting
│   └── prepare.py          <- The script for preparing the raw data and storing it in
│                              data/prepared/
│
└── visuals                 <- All project visuals
    ├── external_visuals    <- Visuals brought from outside the project
    ├── generated_graphics  <- Visuals generated from the project
    └── presentation        <- A copy of your presentation
```

## Data Dictionary

| Tax Distribution | Definition |
| Single Unit Property | The term housing unit refers to a single unit within a larger structure that can be used by an individual or household to eat, sleep, and live. The unit can be in any type of residence such as a house, apartment, mobile home, or may also be a single unit in a group of rooms. Essentially, a housing unit is deemed to be a separate living quarter where the occupants live and eat separately from other residents of the structure or building. They also have direct access from the building's exterior or through a common hallway.
| --- | --- |
| Feature 1 | Definition 1 |
| Feature 2 | Definition 2 |

| Target | Definition |
| --- | --- |
| Target 1 | Definition 1 |