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

**TODO:** Provide a CSV and instructions for loading it into Snowflake. Probably best to copy from another tutorial.

## Setup 2 of 2: Install Linear Modeling Stored Procedures

Duration: 3

[Snowpark Python](https://docs.snowflake.com/en/developer-guide/snowpark/python/index) makes it possible to add powerful ML functionality to Snowflake. Other tutorials (**TODO**: link examples) cover the step-by-step details of how to do this, and it is a great idea to check those out if you're curious. All you need to do for this Quickstart is to run the below SQL script by pasting it into a new Snowsight worksheet and running the whole thing with `ctrl` + `shift` + `enter`.

**TODO**: Add script to this github.

[SQL Script](https://snowflake.com)

**TODO**: Add screenshot gif.

> aside positive
> **Recommnded reading**
>
> If you are comfortable with Python, SQL, and Snowpark concepts, you may find it interesting to read the script, as it is pretty well commented and serves as a relatively minimal example of extending Snowflake with Python stored procedures. This is not required to follow along in this tutorial, though.

> aside positive
> **The code is open-source!**
>
> [All Snowflake Quickstarts are licensed under the permissive Apache License 2.0](https://github.com/Snowflake-Labs/sfquickstarts/blob/master/LICENSE), and since the above script is bundled with this Quickstart, it is covered by the same license. This means you are free to alter and use it as you please, though please be aware that it is provided for demonstration, and *it is **not** an official Snowflake product*.


### Troubleshooting

**TODO**: Offer advice on why the script might have failed (improper Anaconda setup, improper permissions, etc.)

<!-- ------------------------ -->
## Your First Regression Analysis
Duration: 5

<!-- ------------------------ -->
## Saving Models for Reuse
Duration: 5


<!-- ------------------------ -->
## Cleaning Up
Duration: 2

### Dropping the Demo Data

If you loaded the demo data into Snowflake and want to remove it now that you have completed the tutorial, feel free!

**TODO**: Correct the below snippet.

```sql
DROP STAGE @linear-model-demo-data-loading;
DROP SCHEMA linear-model-demo;
```

### Dropping Saved Models

If you want to wipe out the models we saved in this tutorial, you can do so by deleting the Stage.

**TODO**: Correct the below snippet.

```sql
DROP STAGE @linear-model-demo-saved-models;
```
### Removing the Linear Modeling Stored Procedures

If you want to uninstall the linear modeling stored procedures we installed in the beginning, this snippet should have you covered.

**TODO**: Correct the below snippet.

```sql
DROP PROCEDURE XYZ;
DROP PROCEDURE XYZ2;
...
```
