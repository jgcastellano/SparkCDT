/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package example

import java.io.File

import org.apache.spark.ml.classification.CredalDecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, monotonically_increasing_id}

object npiAnpiComparisonExample {
  def main(args: Array[String]): Unit = {
    val trainPath = "data/susy-10k-tra.data"
    val testPath = "data/susy-10k-tst.data"

    Seq(trainPath, testPath).filterNot(path => new File(path).exists()).foreach { path =>
      throw new IllegalArgumentException(
        s"Dataset not found: $path. Run this example from the project root.")
    }

    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("NPI and ANPI decision tree comparison")
      .getOrCreate()

    try {
      spark.sparkContext.setLogLevel("ERROR")
      val sourceLabelColumn = "_c18"

      def loadDataset(path: String): DataFrame = {
        spark.read
          .format("csv")
          .option("inferSchema", value = true)
          .option("header", value = false)
          .load(path)
          .withColumn("label", col(sourceLabelColumn).cast("double"))
          .drop(sourceLabelColumn)
      }

      val trainingData = loadDataset(trainPath)
      val rawTestData = loadDataset(testPath)
      val featureColumns = trainingData.columns.filterNot(_ == "label")
      val assembler = new VectorAssembler()
        .setInputCols(featureColumns)
        .setOutputCol("features")

      val assembledTrainingData = assembler.transform(trainingData)
      val testData = assembler.transform(rawTestData)
        .withColumn("comparison_row_id", monotonically_increasing_id())
        .cache()
      val testRowCount = testData.count()

      def predict(impurity: String): DataFrame = {
        val classifier = new CredalDecisionTreeClassifier()
          .setImpurity(impurity)
          .setMaxDepth(10)
        classifier.fit(assembledTrainingData).transform(testData)
      }

      val npiPredictions = predict("npi")
      val anpiPredictions = predict("anpi")
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")

      val npiAccuracy = evaluator.evaluate(npiPredictions)
      val anpiAccuracy = evaluator.evaluate(anpiPredictions)

      val pairedPredictions = npiPredictions
        .select(col("comparison_row_id"), col("prediction").as("npi_prediction"))
        .join(
          anpiPredictions.select(
            col("comparison_row_id"), col("prediction").as("anpi_prediction")),
          Seq("comparison_row_id"))
      val matchingPredictions = pairedPredictions
        .filter(col("npi_prediction") === col("anpi_prediction"))
        .count()
      val agreement = matchingPredictions.toDouble / testRowCount

      println("------------------------------------------------------------")
      println("NPI-M vs A-NPI-M decision tree comparison (SUSY)")
      println(s"Test rows:                 $testRowCount")
      println(f"NPI-M accuracy:            $npiAccuracy%.8f")
      println(f"A-NPI-M accuracy:          $anpiAccuracy%.8f")
      println(f"Accuracy difference:       ${anpiAccuracy - npiAccuracy}%+.8f (A-NPI-M - NPI-M)")
      println(f"Prediction agreement:      ${agreement * 100}%.2f%%")
      println("------------------------------------------------------------")
      println("These are empirical results for this dataset and tree configuration.")

      testData.unpersist()
    } finally {
      spark.stop()
    }
  }
}