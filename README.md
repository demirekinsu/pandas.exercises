# Pandas Data Analysis Exercises

A collection of 23 practical exercises using the Titanic and Tips datasets to practice data exploration, filtering, cleaning, aggregation, and feature creation with pandas.

## Datasets

Both datasets are loaded using Seaborn:

- **Titanic:** Passenger demographics, ticket information, and survival outcomes.
- **Tips:** Restaurant bills, tips, party sizes, and dining details.

## Topics Covered

### Data Exploration
- Inspecting DataFrame structure and data types
- Counting values and unique observations
- Identifying missing values
- Converting columns to categorical types

### Filtering and Selection
- Selecting rows based on single and multiple conditions
- Combining conditions with AND and OR operators
- Selecting specific columns with `loc`

### Data Cleaning
- Dropping columns
- Filling missing values using the mode and median

### Grouping and Aggregation
- Summarizing survival outcomes by passenger class and sex
- Analyzing restaurant bills by day and meal time
- Calculating sums, counts, minimums, maximums, and means

### Feature Creation and Sorting
- Creating an age indicator using a function, `apply`, and `lambda`
- Combining bill and tip amounts into a new column
- Selecting the top 30 records by combined bill and tip amount

## Requirements

- Python 3
- pandas
- seaborn
- numpy

Install the dependencies:

```bash
pip install pandas seaborn numpy
```

## Usage

Run the script from its containing directory:

```bash
python pandas_exercises.py
```

An internet connection may be required when loading the datasets for the first time.

Most exercises use standalone expressions. To view their results, run them interactively in an IDE or notebook, or wrap them in `print()`.

## File

- `pandas_exercises.py`: Exercise descriptions and solutions, with comments in Turkish.

## Learning Objective

Build familiarity with common pandas operations and strengthen the data preparation and exploration skills used in data analysis.
