# Credal Decision Trees for Apache Spark

An extension for Apache Spark MLlib that implements **Credal Decision Trees (CDT)** for Big Data classification problems. 

Standard decision trees in Spark use classical impurity measures (like Gini or Entropy). This project extends Spark's internal tree architecture to support imprecise probabilities, offering greater robustness when dealing with noise, missing data, or high uncertainty.

This repository implements three credal impurity criteria:
1. **IDM** (Imprecise Dirichlet Model)
2. **NPI** (Nonparametric Predictive Inference)
3. **epsilon-Contamination** ($\epsilon$-Contamination)

## Citation

This implementation was developed for the following research paper. If you use this code in your research, please cite:

> **Using imprecise probabilities in decision trees for Big Data problems**(2026) 
> *Javier G. Castellano, Maria Isabel A. Benítez, Serafín Moral García, Carlos J Mantas, Joaquín Abellán Mulero* 
> (To be Published)

## ⚙️ Requirements

* **Apache Spark:** 4.x (Tested on Spark 4.1.1)
* **Scala:** 2.13 
* **SBT:** Standard Scala build tool

## 🔨 Compilation

To build the project and generate the JAR file, clone this repository and run SBT from the root directory:

```bash
sbt clean package

```

This will generate a JAR file in the `./target/scala-2.13/` folder, for example: `sparkcdt_2.13-1.4.jar`.

## Usage 1: Running the provided example (Spark-Submit)

We provide an example class (`example.cdtExample`) that loads a toy dataset and evaluates the different credal impurity measures.

**⚠️ Important Note on Spark Architecture (The ClassLoader issue):**
Because this library deeply integrates with Spark's internal MLlib architecture (modifying private package classes via *Package Shadowing*), you cannot simply submit the job normally. If you do, Spark's Default ClassLoader will prioritize its internal unmodified classes, leading to `IllegalArgumentException` or `ClassCastException`.

To solve this, we must force both the Spark Driver and the Spark Executors to load our custom JAR **before** the native Spark libraries. You can do this by using `--driver-class-path` and `spark.executor.extraClassPath`.

Run the example using the following command from the root of the project:

```bash
/path/to/spark/bin/spark-submit \
  --class example.cdtExample \
  --driver-class-path ./target/scala-2.13/sparkcdt_2.13-1.4.jar \
  --conf  "spark.executor.extraClassPath=./target/scala-2.13/sparkcdt_2.13-1.4.jar" \
  ./target/scala-2.13/sparkcdt_2.13-1.4.jar

```

## Usage 2: Integrating CDTs into your own Scala code

If you want to use the `CredalDecisionTreeClassifier` in your own Spark ML pipelines, it acts as a drop-in replacement for Spark's standard `DecisionTreeClassifier`.

### 1. Import the classes

```scala
import org.apache.spark.ml.classification.CredalDecisionTreeClassifier
import org.apache.spark.mllib.tree.impurity.{IDMParameter, EContaminationParameter}

```

### 2. Configure hyperparameters (if needed)

If you are using IDM or E-Contamination, configure their specific hyperparameters globally before training the tree:

```scala
// For IDM (Hyperparameter 's')
IDMParameter.setS(1.0)

// For epsilon-Contamination (Hyperparameter 'epsilon')
EContaminationParameter.setE(0.15)

```

### 3. Instantiate and train the model

Just like any standard Spark ML Estimator, set the `credalImpurity` parameter and fit the model to your DataFrame:

```scala
// Create the CDT Classifier
val cdt = new CredalDecisionTreeClassifier()
  .setCredalImpurity("idm") // Options: "gini", "entropy", "npi", "idm", "econtamination"
  .setMaxDepth(10)
  .setLabelCol("label")
  .setFeaturesCol("features")

// Train the model
val model = cdt.fit(trainingData)

// Make predictions
val predictions = model.transform(testData)

```

## License

This project is licensed under the **Apache License 2.0**, the same license used by the Apache Spark project. See the header of the source files for more details.


