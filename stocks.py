from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col, regexp_replace, year, month, dayofmonth
import pandas as pd

# create a SparkSession
spark = SparkSession.builder.appName('KMeansClustering').getOrCreate()

# read in the csv file using spark
data = spark.read.csv('/input/dow_jones_index.csv', header=True, inferSchema=True)

# remove nulls
data = data.na.drop()

# remove the dollar sign from the High, Low, and Close columns
data = data.withColumn("high", regexp_replace(col("high"), "\$", ""))
data = data.withColumn("low", regexp_replace(col("low"), "\$", ""))
data = data.withColumn("close", regexp_replace(col("close"), "\$", ""))
data = data.withColumn("next_weeks_open", regexp_replace(col("next_weeks_open"), "\$", ""))
data = data.withColumn("next_weeks_close", regexp_replace(col("next_weeks_close"), "\$", ""))

# split the date column into separate year, month, and day columns
data = data.withColumn('year', year('date')) \
           .withColumn('month', month('date')) \
           .withColumn('day', dayofmonth('date'))

# combine the relevant columns into a single feature vector
assembler = VectorAssembler(inputCols=['open', 'high', 'low', 'close', 'volume', 'percent_change_price',\
                                       'percent_change_volume_over_last_wk', 'previous_weeks_volume',\
                                       "next_weeks_open","next_weeks_close",'year', 'month', 'day',\
                                        "percent_change_next_weeks_price", "days_to_next_dividend",\
                                        "percent_return_next_dividend"], outputCol='features')
data = assembler.transform(data)

# create a KMeans model with k=3 clusters
kmeans = KMeans(featuresCol='features', k=2, seed=1)

# fit the KMeans model to the data
model = kmeans.fit(data)

# make predictions on the data
predictions = model.transform(data)

# show the cluster assignments for each data point
predictions.select('quarter', 'stock', 'date', 'open', 'high', 'low', 'close', 'volume',\
                   'percent_change_price', 'percent_change_volume_over_last_wk', 'previous_weeks_volume',\
                   "next_weeks_open","next_weeks_close","percent_change_next_weeks_price", "days_to_next_dividend",\
                    "percent_return_next_dividend",'features','prediction').show()