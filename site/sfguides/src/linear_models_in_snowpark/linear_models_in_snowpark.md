author: Luke Merrick
id: linear_models_in_snowpark
summary: Quickly apply basic machine learning to your data in Snowflake with linear models
categories: data-science-&-ml
environments: web
status: Published
feedback link: https://github.com/Snowflake-Labs/sfguides/issues
tags: Snowpark Python, Machine Learning, Data Science

# Fit Simple Linear Models in Snowflake

<!-- ------------------------ -->

## Overview

Duration: 2

### What are Linear Models?

Linear models constitute a family machine learning algorithms which make predictions of a target variable by taking a linear combination (weighted average) of input variables.

Linear models are very fast, both in fitting and prediction, require minimal tuning, and are often robust against overfitting. Their simple linear form also enables transparency and interpretability. Though other techniques often deliver accuracy improvements, ML practitioners often begin projects by use linear models both as predictive baselines and for non-predictive analyses into the way that predictive features relate to the target variable.

In the statistical literature, linear models are often handled under the theretical umbrella of Generalized Linear Models (GLMs), and statisticians have built up a great deal of theory that lets linear models not just produce predictions, but also enable sophisticated explanations and analyses of data. Additionally, statistical theory also intimately [links linear modeling to Analysis of Variance (ANOVA)](https://en.wikipedia.org/wiki/Analysis_of_variance#Textbook_analysis_using_a_normal_distribution).

#### References

- [Wikipedia: Linear model](https://en.wikipedia.org/wiki/Linear_model)
- [Wikipedia: Generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model)
- [Wikipedia: Regression analysis](https://en.wikipedia.org/wiki/Regression_analysis)
- [Wikipedia: Analysis of variance](https://en.wikipedia.org/wiki/Analysis_of_variance)
- [Scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)

### What You’ll Learn

This Quickstart will cover

- How Snowpark Python Stored Procedures let you to extend Snowflake with additional statistical and ML functionality.
- How to perform a basic regression analysis in Snowflake.
- How Snowflake Stages let you save and reuse models natively in Snowflake.

### Prerequisites

- A Snowflake account with [Anaconda Packages enabled by ORGADMIN](https://docs.snowflake.com/en/developer-guide/udf/python/udf-python-packages.html#using-third-party-packages-from-anaconda). If you do not have a Snowflake account, you can register for a [free trial account](https://signup.snowflake.com/).
- A Snowflake account login with ACCOUNTADMIN role. If you have this role in your environment, you may choose to use it. If not, you will need to 1) Register for a free trial, 2) Use a different role that has the ability to create database, schema, tables, stages, tasks, user-defined functions, and stored procedures OR 3) Use an existing database and schema in which you are able to create the mentioned objects.

<!-- ------------------------ -->

## Setup 1 of 2: Loading Demo Data

Duration: 3

> aside negative
> **Got your own data? You can skip this part!**
>
> If you'd like to run linear models only on your own data and feel comfortable adapting some simple SQL queries to use the table(s) of your own choosing, it is safe to skip this data loading step.

### [Optional] Make a new dedicated warehouse, database, and schema for this quickstart

If you would like to create a [warehouse](https://docs.snowflake.com/en/sql-reference/sql/create-warehouse.html), [database](https://docs.snowflake.com/en/sql-reference/sql/create-database.html) and [schema](https://docs.snowflake.com/en/sql-reference/sql/create-schema.html) specifically for this demo data, feel free to do so with the snippet below.

```sql
USE ROLE ACCOUNTADMIN;

CREATE OR REPLACE WAREHOUSE TEMPORARY_WAREHOUSE;
CREATE OR REPLACE DATABASE TEMPORARY_DB;
CREATE OR REPLACE SCHEMA TEMPORARY_SCHEMA;

USE WAREHOUSE TEMPORARY_WAREHOUSE;
USE TEMPORARY_DB.TEMPORARY_SCHEMA;
```

### Load the demo data into Snowflake

We're going to borrow the demo data from the [Getting Started with Data Engineering and ML using Snowpark for Python](https://quickstarts.snowflake.com/guide/getting_started_with_dataengineering_ml_using_snowpark_python/index.html?index=..%2F..index#1) quickstart.

```sql
-- Create tables for our demo data
CREATE or REPLACE TABLE CAMPAIGN_SPEND (
  CAMPAIGN VARCHAR(60),
  CHANNEL VARCHAR(60),
  DATE DATE,
  TOTAL_CLICKS NUMBER(38,0),
  TOTAL_COST NUMBER(38,0),
  ADS_SERVED NUMBER(38,0)
);
CREATE or REPLACE TABLE MONTHLY_REVENUE (
  YEAR NUMBER(38,0),
  MONTH NUMBER(38,0),
  REVENUE FLOAT
);

-- Load demo data from CSV
COPY into CAMPAIGN_SPEND
  FROM 's3://sfquickstarts/ad-spend-roi-snowpark-python-scikit-learn-streamlit/campaign_spend/'
  FILE_FORMAT = (type = 'CSV', skip_header = 1);
COPY into MONTHLY_REVENUE
  FROM 's3://sfquickstarts/ad-spend-roi-snowpark-python-scikit-learn-streamlit/monthly_revenue/'
  FILE_FORMAT = (type = 'CSV', skip_header = 1);
```

## Setup 2 of 2: Install Linear Modeling Stored Procedures

Duration: 3

[Snowpark Python](https://docs.snowflake.com/en/developer-guide/snowpark/python/index) makes it possible to add powerful ML functionality to Snowflake. Other tutorials (**TODO**: link examples) cover the step-by-step details of how to do this, and it is a great idea to check those out if you're curious. All you need to do for this Quickstart is to run the below SQL script by pasting it into a new Snowsight worksheet and running the whole thing with `ctrl` + `shift` + `enter`.

**TODO**: Add script to this github.

[SQL Script](https://snowflake.com)

**TODO**: Add screenshot gif.

**TODO**: Remove this large text box for copy-paste

```sql
-- Our main implementation is a Python procedure that stores its output to a temporary table provided by the caller
create or replace procedure _linear_regression(output_table_name varchar, input_table_name varchar, target_column varchar)
  returns varchar
  language python
  runtime_version = '3.8'
  packages = ('snowflake-snowpark-python', 'numpy','pandas')
  handler = 'linreg_sproc'
as $$
import numpy as np
import pandas as pd


def contains_nans(a):
    # Fast and memory efficient, see: https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
    return np.isnan(np.dot(a.ravel(), a.ravel()))


def expand_to_include_constant_columns(nonconstant_mask, vector):
    output = np.zeros(nonconstant_mask.shape[0], dtype=vector.dtype)
    output[nonconstant_mask] = vector
    return output


def linear_regression(df: pd.DataFrame, target: str, dtype=np.float64):
    GLOBAL_NAME = "__global__"
    INTERCEPT_NAME = "__intercept__"
    assert INTERCEPT_NAME not in df.columns

    # Extract feature matrix and target vector.
    x = df.drop(columns=target).to_numpy(dtype=dtype)
    y = df.loc[:, target].to_numpy(dtype=dtype)
    assert not contains_nans(x)
    assert not contains_nans(y)
    assert x.ndim == 2
    assert y.ndim == 1

    # Drop constant columns.
    #   See: https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/preprocessing/_data.py#L74
    x_mean, x_var = np.mean(x, axis=0), np.var(x, axis=0, ddof=0)
    nonconstant_mask = x_var > (
        x.shape[0] * x_var * np.finfo(dtype).eps + (x.shape[0] * x_mean * np.finfo(dtype).eps) ** 2
    )
    x = x[:, nonconstant_mask]
    x_mean = x_mean[nonconstant_mask]
    x_var = x_var[nonconstant_mask]

    # Z-scale features and target to zero mean and unit variance.
    x_offset, x_scale = x_mean, np.sqrt(x_var)
    y_offset, y_scale = np.mean(y), np.std(y, ddof=0)
    x_scaled = (x - x_offset) / x_scale
    y_scaled = (y - y_offset) / y_scale

    # Run least squares.
    coefficients, residual, rank, singluar_values = np.linalg.lstsq(x_scaled, y_scaled, rcond=None)

    # Undo scaling on intercept and coefficients.
    coefficients_unscaled = y_scale * (coefficients / x_scale)
    intercept_unscaled = y_offset - np.dot(coefficients_unscaled, x_offset)

    # Compute R^2 score.
    residuals = y_scaled - x_scaled @ coefficients
    residual_mse = np.mean(np.power(residuals, 2))
    total_mse = np.mean(np.power(y_scaled, 2))
    mse_reduction = total_mse - residual_mse
    r2_score = mse_reduction / total_mse

    # Reconstitute zero coefficients for constant columns.
    coefficients = expand_to_include_constant_columns(nonconstant_mask, coefficients)

    # Return output.
    return pd.DataFrame({
        "variable": [GLOBAL_NAME, INTERCEPT_NAME] + [c for c in df.columns if c != target],
        "r2_score": np.concatenate([np.asarray([r2_score]), np.full(coefficients.shape[0] + 1, np.nan)]),
        "coefficient": np.concatenate([np.asarray([np.nan, intercept_unscaled]), coefficients_unscaled]),
        "coefficient_for_scaled_data": np.concatenate([np.asarray([np.nan, 0]), coefficients]),
    })

def linreg_sproc(session, output_table_name: str, input_table_name: str, target_column: str) -> str:
    df_input = session.table(input_table_name).select("*").to_pandas()
    df_output = linear_regression(df=df_input, target=target_column.upper())
    session.write_pandas(
        df_output,
        table_name=output_table_name,
        quote_identifiers=False,
        auto_create_table=False,
        overwrite=True,
    )
    return "SUCCESS"
$$;

-- In order to let the user easily run a regression and get a table of results, we wrap our Python procedure
-- in this SQL procedure which creates, reads from, and tears down a temporary table for our Python procedure.
create or replace procedure linear_regression(table_name varchar, target_column varchar)
    returns table(
        variable varchar,
        r2_score float,
        coefficient float,
        coefficient_for_scaled_data float
    )
    language sql
as
declare
    res resultset;
begin
    create or replace temporary table _temporary_linear_regression_result_table (
        variable varchar,
        r2_score float,
        coefficient float,
        coefficient_for_scaled_data float
    );
	call _linear_regression('_temporary_linear_regression_result_table', :table_name, :target_column);
	res := (select * from _temporary_linear_regression_result_table);
    drop table _temporary_linear_regression_result_table;
	return table(res);
end;
```

> aside positive
> **Recommnded reading**
>
> If you are comfortable with Python, SQL, and Snowpark concepts, you may find it interesting to read the script, as it is pretty well commented and serves as a relatively minimal example of extending Snowflake with Python stored procedures. This is not required to follow along in this tutorial, though.

> aside positive
> **The code is open-source!**
>
> [All Snowflake Quickstarts are licensed under the permissive Apache License 2.0](https://github.com/Snowflake-Labs/sfquickstarts/blob/master/LICENSE), and since the above script is bundled with this Quickstart, it is covered by the same license. This means you are free to alter and use it as you please, though please be aware that it is provided for demonstration, and _it is **not** an official Snowflake product_.

### Troubleshooting

**TODO**: Offer advice on why the script might have failed (improper Anaconda setup, improper permissions, etc.)

<!-- ------------------------ -->

## Your First Regression Analysis in Snowflake

Duration: 5

### Teeing up the regression problem

Let's use regression analysis to investigate our marketing spend. First let's construct a query that places our total monthly spend across marketing channels side-by-side with our total monthly revenue. To make it easy to pass it into a Stored Procedure for running our regression, we'll store the query results into a temporary table called `input_data`.

```sql
create or replace temporary table input_data as
with
    -- Aggregate spend to monthly granularity per channel.
    monthly_per_channel_spend as (
        select
            year(date) as year,
            month(date) as month,
            channel,
            sum(total_cost) as total_cost
        from campaign_spend
        group by year, month, channel
    ),
    -- Pivot channel to become a set of columns.
    pivot_per_channel_spend as (
        select
            year,
            month,
            "'search_engine'" as search_engine_spend,
            "'social_media'" as social_media_spend,
            "'video'" as video_spend,
            "'email'" as email_spend
        from monthly_per_channel_spend
        pivot (sum(total_cost) for channel in ('search_engine', 'social_media', 'video', 'email'))
    )
-- Join revenue and spend.
select
    spend.search_engine_spend,
    spend.social_media_spend,
    spend.video_spend,
    spend.email_spend,
    revenue.revenue
from
    pivot_per_channel_spend as spend
join monthly_revenue as revenue
on spend.year=revenue.year and spend.month=revenue.month
order by spend.year, spend.month;
```

We can now take a look at our data:

```sql
select * from input_data limit 10;
```

**TODO**: Add Snowsight screenshot of output.

### Running the regression analysis

Thanks to our setup step, there is a `linear_regression` procedure loaded into our current database and schema. All we have to do is call it, passing the name of the table to use as input data and the name of the column to use as the target of the regression.

```sql
call linear_regression('input_data', 'revenue');
```

**TODO**: Add Snowsight screenshot.

<!-- ------------------------ -->

## Making Predictions

Duration: 5

<!-- ------------------------ -->

## Saving Models for Reuse

Duration: 5

<!-- ------------------------ -->

## [Optional] Cleaning Up

Duration: 1

If you created a temporary warehouse, db, and schema during the data setup step, you can clean them up now as well. This will remove the data we loaded, the procedures we installed, and the models and predictions that we saved.

```sql
USE ROLE ACCOUNTADMIN;

DROP WAREHOUSE TEMPORARY_WAREHOUSE;
DROP DATABASE TEMPORARY_DATABASE;
DROP SCHEMA TEMPORARY_SCHEMA;
```
