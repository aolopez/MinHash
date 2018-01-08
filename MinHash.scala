import scala.math.pow
import scala.collection.mutable.WrappedArray
import org.apache.spark.sql.functions.{udf, explode}
import org.apache.spark.ml.feature.{NGram, HashingTF, MinHashLSH}

//Preprocess data to join all strings into a large one.
val base_path = "<address>"
val rawText = sqlContext.read.format("com.databricks.spark.avro").load(base_path)
val preprocessing  = udf { s: WrappedArray[String] => s.mkString(" ").replaceAll("\\s", " ").split("") }
val stringToArray = udf {s: String => s.split("")}
val arrayToString = udf {s: Array[String] => s.mkString("")}
val arrayToStringWSpace = udf {s: WrappedArray[String] => s.mkString(" ")}

val rawTextTrim = rawText.
                    .distinct()
                    .withColumn("text",preprocessing(col("text")))

// Create 6-gram
val ngram = new NGram()
              .setN(6)
              .setInputCol("text")
              .setOutputCol("ngrams")

val ngramDF = ngram
                .transform(rawTextTrim)
                .drop("text")

// Create a one-hot encoding for the ngrams
val hashingTF = new HashingTF()
                    .setInputCol("ngrams")
                    .setOutputCol("rawVectors")
                    .setNumFeatures(pow(2,30).toInt)
                    .setBinary(true)

val featurizedData = hashingTF
                      .transform(ngramDF)
                      .drop("ngrams")
                      .distinct()

// Compute MinHash
val mh = new MinHashLSH()
              .setNumHashTables(400)
              .setInputCol("rawVectors")
              .setOutputCol("mh_values")

val model = mh.fit(featurizedData)

// Compute the similarity matrix between documents
model
.approxSimilarityJoin(featurizedData,featurizedData,1.0)
