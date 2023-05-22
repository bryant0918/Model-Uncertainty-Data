import pyspark
from pyspark.sql import SparkSession
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as MCE

import os


# --------------------- Resilient Distributed Datasets --------------------- #

### Problem 1
def word_count(filename='huck_finn.txt'):
    """
    A function that counts the number of occurrences unique occurrences of each
    word. Sorts the words by count in descending order.
    Parameters:
        filename (str): filename or path to a text file
    Returns:
        word_counts (list): list of (word, count) pairs for the 20 most used words
    """
    # Start my session and open the file
    spark = SparkSession.builder.appName("app_name").getOrCreate()
    words = spark.sparkContext.textFile(filename)

    # Flatten the words and make a dictionary
    words = words.flatMap(lambda row: row.split())
    diction = words.map(lambda word: (word, 1))

    # Count words and sort them
    counts = diction.reduceByKey(lambda x, y: x + y)
    sorted = counts.sortBy(lambda row: row[1], ascending=False).take(20)
    spark.stop()

    return sorted


# print(word_count())


### Problem 2
def monte_carlo(n=10 ** 5, parts=6):
    """
    Runs a Monte Carlo simulation to estimate the value of pi.
    Parameters:
        n (int): number of sample points per partition
        parts (int): number of partitions
    Returns:
        pi_est (float): estimated value of pi
    """
    # Start spark session and paralellize
    spark = SparkSession.builder.appName("app_name").getOrCreate()
    x = spark.sparkContext.parallelize(np.random.uniform(-1, 1, (n * parts, 2)), parts)

    # Filter and count
    number = x.filter(lambda row: la.norm(row, axis=0) <= 1).count()

    spark.stop()

    # Return approximation
    return 4 * number / (n * parts)


# print(monte_carlo())


# ------------------------------- DataFrames ------------------------------- #

### Problem 3
def titanic_df(filename='titanic.csv'):
    """
    Calculates some statistics from the titanic data.

    Returns: the number of women on-board, the number of men on-board,
             the survival rate of women,
             and the survival rate of men in that order.
    """
    # Start spark session
    spark = SparkSession.builder.appName("app_name").getOrCreate()

    # Define schema and read in file
    schema = ('survived INT, pclass INT, name STRING, sex STRING, age FLOAT, sibsp INT, parch INT, fare FLOAT')
    titanic = spark.read.csv(filename, schema=schema)

    # Filter by sex
    females = titanic.filter(titanic.sex == "female")
    males = titanic.filter(titanic.sex == "male")

    # Filter by survived
    survived_males = males.filter(males.survived == 1)
    survived_females = females.filter(females.survived == 1)

    spark.stop()

    return females.count(), males.count(), survived_females.count() / females.count(), survived_males.count() / males.count()


# print(titanic_df())


### Problem 4
def crime_and_income(crimefile='london_crime_by_lsoa.csv',
                     incomefile='london_income_by_borough.csv', major_cat='Robbery'):
    """
    Explores crime by borough and income for the specified major_cat
    Parameters:
        crimefile (str): path to csv file containing crime dataset
        incomefile (str): path to csv file containing income dataset
        major_cat (str): major or general crime category to analyze
    returns:
        numpy array: borough names sorted by percent months with crime, descending
    """
    # Start Session and load in files
    spark = SparkSession.builder.appName("app_name").getOrCreate()
    crime = spark.read.csv(crimefile, header=True, inferSchema=True)
    income = spark.read.csv(incomefile, header=True, inferSchema=True)

    # Get crimes by borough
    crimes_by_borough = crime.filter(crime.major_category == major_cat).groupBy("borough").sum('value')

    # Join the tables and select columns
    joined = crimes_by_borough.join(income, on="borough").sort('sum(value)', ascending=False)
    df = joined.select(['borough', 'sum(value)', 'median-08-16']).collect()

    # Cast as array to plot
    df = np.array(df)
    plt.scatter(df[:, 1].astype(int), df[:, 2].astype(float))
    plt.xlabel("Crimes")
    plt.xticks(rotation=90)
    plt.ylabel("Income")
    plt.title("Crime by Income")
    plt.show()

    spark.stop()
    return df


# print(crime_and_income())

### Problem 5
def titanic_classifier(filename='titanic.csv'):
    """
    Implements a classifier model to predict who survived the Titanic.
    Parameters:
        filename (str): path to the dataset
    Returns:
        metrics (tuple): a tuple of metrics gauging the performance of the model
            ('accuracy', 'weightedRecall', 'weightedPrecision')
    """
    # Start Spark Session
    spark = SparkSession.builder.appName("app_name").getOrCreate()

    # Define schema and read in file
    schema = ('survived INT, pclass INT, name STRING, sex STRING, age FLOAT, sibsp INT, parch INT, fare FLOAT')
    titanic = spark.read.csv(filename, schema=schema)

    # Code in the lab file
    sex_binary = StringIndexer(inputCol='sex', outputCol='sex_binary')
    onehot = OneHotEncoder(inputCols=['pclass'], outputCols=['pclass_onehot'])

    features = ['sex_binary', 'pclass_onehot', 'age', 'sibsp', 'parch', 'fare']
    features_col = VectorAssembler(inputCols=features, outputCol='features')

    pipeline = Pipeline(stages=[sex_binary, onehot, features_col])
    titanic = pipeline.fit(titanic).transform(titanic)
    titanic = titanic.drop('pclass', 'name', 'sex')
    train, test = titanic.randomSplit([0.75, 0.25], seed=11)

    # Use random forest instead of LogReg
    rf = RandomForestClassifier(labelCol='survived', featuresCol='features')

    # change my estimator paramMaps
    tvs = TrainValidationSplit(estimator=rf,
                               estimatorParamMaps=[{rf.maxDepth: 3}, {rf.minInstancesPerNode: 3}],
                               evaluator=MCE(labelCol='survived'),
                               trainRatio=0.75, seed=11)

    # Train model
    clf = tvs.fit(train)

    # Get results
    results = clf.bestModel.evaluate(test)
    answer = (results.accuracy, results.weightedRecall, results.weightedPrecision)
    spark.stop()

    return answer


# titanic_classifier()

