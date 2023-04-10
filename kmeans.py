from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# create a SparkSession
spark = SparkSession.builder.appName('KMeansClustering').getOrCreate()

# create a dataframe from the dataset
data = spark.read.csv('/input/data.csv', header=True, inferSchema=True)

# combine all columns into a single feature vector
assembler = VectorAssembler(inputCols=data.columns, outputCol='features')
data = assembler.transform(data)

# create a KMeans model with k=3 clusters
kmeans = KMeans(featuresCol='features', k=2, seed=1)

# fit the KMeans model to the data
model = kmeans.fit(data)

# make predictions on the data
predictions = model.transform(data)

# show the cluster assignments for each data point
predictions.select('Exp', 'isEmployed', 'PreviousJob', 'EducationLevel', 'TopTierSchool', 'Internship', 'Hired', 'prediction').show()

