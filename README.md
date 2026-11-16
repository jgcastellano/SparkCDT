# Credal Decision Trees for Apache Spark

An extension for Apache Spark MLlib that implements **Credal Decision Trees (CDT)** for Big Data classification problems. 

Standard decision trees in Spark use classical impurity measures (like Gini or Entropy). This project extends Spark's internal tree architecture to support imprecise probabilities, offering greater robustness when dealing with noise, missing data, or high uncertainty. By using **Package Membership**, this implementation integrates directly into the `org.apache.spark` namespace, allowing seamless access to Spark's private APIs and optimized communication between the Driver and Executors without the need for complex ClassLoader configurations.

This repository implements three credal impurity criteria:
1. **IDM** (Imprecise Dirichlet Model)
2. **NPI** (exact NPI-M for multinomial data)
3. **ANPI** (approximate NPI-M for multinomial data)

## Citation

This implementation was developed for the following research paper. If you use this code in your research, please cite:

> **Using imprecise probabilities in decision trees for Big Data problems** (2026) 
> *Javier G. Castellano, María Isabel A. Benítez, Serafín Moral-García, Carlos J Mantas, Joaquín Abellán* 
> (To be Published)

## Requirements

* **Apache Spark:** 4.x (Tested on Spark 4.1.1)
* **Scala:** 2.13 
* **SBT:** Standard Scala build tool

## Compilation

To build the project and generate the JAR file, clone this repository and run SBT from the root directory:

```bash
sbt clean package

```

This will generate a JAR file in the `./target/scala-2.13/` folder, for example: `sparkcdt_2.13-1.4.jar`.

## Usage 1: Running the provided examples (Spark-Submit / SBT)

We provide an example class (`example.cdtExample`) that loads a toy dataset and evaluates the different credal impurity measures.

Run the example using the following command from the root of the project:

```bash
/path/to/spark/bin/spark-submit \
  --class example.cdtExample \
  ./target/scala-2.13/sparkcdt_2.13-1.4.jar

```

To compare the NPI-M and A-NPI-M decision trees on the same SUSY training and test data, run:

```bash
sbt "runMain example.npiAnpiComparisonExample"
```



## Usage 2: Integrating CDTs into your own Scala code

If you want to use the `CredalDecisionTreeClassifier` in your own Spark ML pipelines, it acts as a drop-in replacement for Spark's standard `DecisionTreeClassifier`.

### 1. Import the classes

```scala
import org.apache.spark.ml.classification.CredalDecisionTreeClassifier

```

### 2. Instantiate and train the model

Hyperparameters like **s** (IDM) are managed directly through the Estimator's API, following the standard Spark Param pattern.
```scala
// Create the CDT Classifier
val cdt = new CredalDecisionTreeClassifier()
  .setImpurity("idm")      // Options: "idm", "npi", "anpi", "gini", "entropy"
  .setS(1.0)               // Specific for IDM (default is 1.0)
  .setMaxDepth(10)
  .setLabelCol("label")
  .setFeaturesCol("features")

// Train the model
val model = cdt.fit(trainingData)

// Make predictions
val predictions = model.transform(testData)

```
### 3. Running with `spark-submit`

Once you have compiled both your application and the SparkCDT library, use `spark-submit` to execute the job. 

**Important Note:** Since this project uses the **Package Membership** architecture, you must ensure the library JAR is available in the classpath of all nodes. The easiest way to achieve this is by using the `--jars` flag.

#### Running your own application:
```bash
$SPARK_HOME/bin/spark-submit \
  --class com.yourproject.Main \
  --master local[*] \
  --jars ./target/scala-2.13/sparkcdt_2.13-1.4.jar \
  /path/to/your/main_project.jar
  ```


## License

This project is licensed under the **Apache License 2.0**, the same license used by the Apache Spark project. See the header of the source files for more details.


