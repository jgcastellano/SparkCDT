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
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.CredalDecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object cdtExample {
  def main(arg: Array[String]): Unit = {

    // 1. Check data files (Best practices for GitHub repositories)
    val ftrain = "data/susy-10k-tra.data" // Relative path to project root
    val ftest  = "data/susy-10k-tst.data"

    if (!new File(ftrain).exists() || !new File(ftest).exists()) {
      System.err.println(
        s"""
           |ERROR: The example datasets cannot be found.
           |Please ensure that a 'data/' folder exists in the root of the project
           |and that it contains the files '$ftrain' and '$ftest'.
           |""".stripMargin)
      System.exit(1)
    }

    // Create a Spark session
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("Example of Credal Decision Trees for Apache Spark")
      .getOrCreate()
    
    // Suppress Spark informational messages to only see our output
    spark.sparkContext.setLogLevel("ERROR") 

    println("----------------------------------------------------------------------------")
    println("- CREDAL DECISION TREE CLASSIFICATION IN SPARK")
    println("----------------------------------------------------------------------------")

    // Read the training dataset
    var dfTra = spark.read
      .format("csv")
      .option("inferSchema", value = true)
      .option("header", value = false)
      .load(ftrain)

    // Read the test dataset
    var dfTst = spark.read
      .format("csv")
      .option("inferSchema", value = true)
      .option("header", value = false)
      .load(ftest)

    //Identify the class column and cast it to an integer value
    dfTra=dfTra.withColumn("label",dfTra("_c18").cast("Integer"))
    dfTst=dfTst.withColumn("label",dfTst("_c18").cast("Integer"))

    //Buil a transform to pack the features attributes into a single vector
    val assembler = new VectorAssembler()
      .setInputCols(dfTra.columns.slice(0,dfTra.columns.length-2) )
      .setOutputCol("features")

    // Prepare the evaluator to measure Accuracy
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")          
      .setPredictionCol("prediction") 
      .setMetricName("accuracy")

    // ---------------------------------------------------------
    // 1. CLASSIC DECISION TREE: GINI
    // ---------------------------------------------------------
    val dtGini = new CredalDecisionTreeClassifier()
      .setImpurity("gini")
      .setMaxDepth(10)

    val pipelineGini = new Pipeline().setStages(Array(assembler, dtGini))
    val modelGini = pipelineGini.fit(dfTra)
    val predGini = modelGini.transform(dfTst)

    println(s"DT (Gini) Accuracy:            ${evaluator.evaluate(predGini)}")

    // ---------------------------------------------------------
    // 2. CLASSIC DECISION TREE: ENTROPY
    // ---------------------------------------------------------
    val dtEntropy = new CredalDecisionTreeClassifier()
      .setImpurity("entropy")
      .setMaxDepth(10)

    val pipelineEntropy = new Pipeline().setStages(Array(assembler, dtEntropy))
    val modelEntropy = pipelineEntropy.fit(dfTra)
    val predEntropy = modelEntropy.transform(dfTst)

    println(s"DT (Entropy) Accuracy:         ${evaluator.evaluate(predEntropy)}")

    // ---------------------------------------------------------
    // 3. CREDAL DECISION TREE: NPI
    // ---------------------------------------------------------
    val dtNPI = new CredalDecisionTreeClassifier()
      .setImpurity("npi")
      .setMaxDepth(10)

    val pipelineNPI = new Pipeline().setStages(Array(assembler, dtNPI))
    val modelNPI = pipelineNPI.fit(dfTra)
    val predNPI = modelNPI.transform(dfTst)

    println(s"DT (NPI) Accuracy:             ${evaluator.evaluate(predNPI)}")

    // ---------------------------------------------------------
    // 4. CREDAL DECISION TREE: IDM
    // ---------------------------------------------------------
    val dtIDM = new CredalDecisionTreeClassifier()
      .setImpurity("idm")
      .setS(1.0)
      .setMaxDepth(10)

    val pipelineIDM = new Pipeline().setStages(Array(assembler, dtIDM))
    val modelIDM = pipelineIDM.fit(dfTra)
    val predIDM = modelIDM.transform(dfTst)

    println(s"DT (IDM s=1.0) Accuracy:          ${evaluator.evaluate(predIDM)}")
    
    println("----------------------------------------------------------------------------")
    
    // Close the session
    spark.stop()
  }
}
