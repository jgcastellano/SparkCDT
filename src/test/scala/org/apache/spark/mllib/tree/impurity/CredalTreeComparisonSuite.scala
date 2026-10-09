package org.apache.spark.mllib.tree.impurity

import org.apache.spark.ml.classification.CredalDecisionTreeClassifier
import org.apache.spark.ml.linalg.{SQLDataTypes, Vectors}
import org.apache.spark.ml.tree.{CategoricalSplit, ContinuousSplit, InternalNode, Node, Split}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

class CredalTreeComparisonSuite extends AnyFunSuite with BeforeAndAfterAll {
  private var spark: SparkSession = _

  private val schema = StructType(Seq(
    StructField("id", IntegerType, nullable = false),
    StructField("label", DoubleType, nullable = false),
    StructField("features", SQLDataTypes.VectorType, nullable = false)
  ))

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("CredalTreeComparisonSuite")
      .config("spark.ui.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
  }

  override protected def afterAll(): Unit = {
    try {
      if (spark != null) spark.stop()
    } finally {
      super.afterAll()
    }
  }

  private case class TreeCase(name: String, samples: Seq[(Double, Double)])
  private case class TreeResult(
      split: String,
      predictions: Seq[Double],
      structure: String,
      rootImpurity: Double,
      rootGain: Double)

  private val cases = Seq(
    TreeCase("binary-clear-split", Seq(
      0.0 -> 0.0, 0.0 -> 0.0, 1.0 -> 0.0, 1.0 -> 0.0,
      2.0 -> 1.0, 2.0 -> 1.0, 3.0 -> 1.0, 3.0 -> 1.0)),
    TreeCase("binary-tied-positive-candidates", Seq(
      0.0 -> 0.0, 1.0 -> 0.0, 2.0 -> 1.0, 3.0 -> 1.0,
      4.0 -> 1.0, 5.0 -> 1.0, 6.0 -> 0.0, 7.0 -> 0.0)),
    TreeCase("multiclass-empty-branch-classes", Seq(
      0.0 -> 0.0, 0.0 -> 0.0, 1.0 -> 0.0,
      2.0 -> 1.0, 2.0 -> 1.0, 3.0 -> 2.0, 3.0 -> 2.0,
      4.0 -> 2.0, 4.0 -> 2.0))
  )

  private def dataFrame(samples: Seq[(Double, Double)]): DataFrame = {
    val rows = samples.zipWithIndex.map { case ((feature, label), id) =>
      Row(id, label, Vectors.dense(feature))
    }
    spark.createDataFrame(spark.sparkContext.parallelize(rows, 2), schema)
  }

  private def splitLabel(split: Split): String = split match {
    case continuous: ContinuousSplit =>
      s"continuous(feature=${continuous.featureIndex},threshold=${continuous.threshold})"
    case categorical: CategoricalSplit =>
      s"categorical(feature=${categorical.featureIndex},left=" +
        s"${categorical.leftCategories.toSeq.sorted.mkString(",")})"
  }

  private def describeSplit(node: Node): String = node match {
    case internal: InternalNode =>
      s"${splitLabel(internal.split)}(${describeSplit(internal.leftChild)}," +
        s"${describeSplit(internal.rightChild)})"
    case _ => "leaf"
  }

  private def formatNumber(value: Double): String =
    java.lang.String.format(java.util.Locale.ROOT, "%.6f", Double.box(value))

  private def fit(samples: Seq[(Double, Double)], impurity: String): TreeResult = {
    val input = dataFrame(samples)
    val model = new CredalDecisionTreeClassifier()
      .setImpurity(impurity)
      .setMaxDepth(2)
      .setMaxBins(16)
      .setMinInstancesPerNode(1)
      .setSeed(8675309L)
      .fit(input)

    val root = model.rootNode
    val predictions = model.transform(input)
      .select("id", "prediction")
      .orderBy("id")
      .collect()
      .map(_.getAs[Double]("prediction")).toSeq

    TreeResult(
      root match {
        case internal: InternalNode => splitLabel(internal.split)
        case _ => "leaf"
      },
      predictions,
      describeSplit(root),
      root.impurity,
      root match {
        case internal: InternalNode => internal.gain
        case _ => 0.0
      }
    )
  }

  test("small deterministic datasets compare criterion, selected split, tree and predictions") {
    val reportRows = cases.map { treeCase =>
      val npi = fit(treeCase.samples, "npi")
      val anpi = fit(treeCase.samples, "anpi")
      val labelCounts = treeCase.samples.map(_._2).groupBy(identity).toSeq.sortBy(_._1)
        .map { case (_, rows) => rows.size.toDouble }.toArray
      val total = labelCounts.sum
      val (npiLower, npiUpper) = NPI.singletonBounds(labelCounts, total)
      val (anpiLower, anpiUpper) = ANPI.singletonBounds(labelCounts, total)
      val maxBoundsDifference = (npiLower.zip(anpiLower) ++ npiUpper.zip(anpiUpper))
        .map { case (a, b) => math.abs(a - b) }.max
      val predictionsAgree = npi.predictions == anpi.predictions
      val structuresAgree = npi.structure == anpi.structure
      val npiBounds = npiLower.zip(npiUpper).map { case (lo, hi) =>
        s"[${formatNumber(lo)},${formatNumber(hi)}]"
      }
      val anpiBounds = anpiLower.zip(anpiUpper).map { case (lo, hi) =>
        s"[${formatNumber(lo)},${formatNumber(hi)}]"
      }
      val row = Seq(
        treeCase.name,
        labelCounts.toSeq.mkString("[", ",", "]"),
        npiBounds.mkString("[", ",", "]"),
        anpiBounds.mkString("[", ",", "]"),
        formatNumber(maxBoundsDifference),
        s"${formatNumber(npi.rootImpurity)} / ${formatNumber(anpi.rootImpurity)}",
        s"${formatNumber(npi.rootGain)} / ${formatNumber(anpi.rootGain)}",
        s"${npi.split} / ${anpi.split}",
        structuresAgree.toString,
        s"${npi.predictions.mkString(",")} / ${anpi.predictions.mkString(",")}",
        predictionsAgree.toString
      )

      assert(npi.predictions.length == treeCase.samples.length)
      assert(anpi.predictions.length == treeCase.samples.length)
      assert(npi.predictions.forall(_.isFinite) && anpi.predictions.forall(_.isFinite))
      assert(maxBoundsDifference <= 1e-12,
        s"Singleton bounds differ for ${treeCase.name}: $maxBoundsDifference")
      if (treeCase.name == "binary-clear-split") {
        assert(npi.split != "leaf", "NPI-M should find the known positive-gain binary split")
        assert(anpi.split != "leaf", "A-NPI-M should find the known positive-gain binary split")
      }
      if (treeCase.name == "binary-tied-positive-candidates") {
        assert(npi.rootGain > 1e-12 && anpi.rootGain > 1e-12,
          s"The tied-candidate fixture must have materially positive best-split gain; " +
            s"NPI-M=${npi.rootGain}, A-NPI-M=${anpi.rootGain}")
        assert(npi.split != "leaf" && anpi.split != "leaf",
          "Both criteria should select one of the positive tied candidates")
      }
      row
    }

    println("NPI/A-NPI tree comparison (gain and split are empirical, not assumed equivalent):")
    println("case | label-counts | NPI singleton bounds | A-NPI singleton bounds | max-bound-diff | root-criterion NPI/A-NPI | root-gain NPI/A-NPI | split NPI/A-NPI | structures-agree | predictions NPI/A-NPI | predictions-agree")
    reportRows.foreach(row => println(row.mkString(" | ")))
  }

  test("fixed seed gives repeatable selected split and predictions") {
    val samples = cases.head.samples
    val npiFirst = fit(samples, "npi")
    val npiSecond = fit(samples, "npi")
    val anpiFirst = fit(samples, "anpi")
    val anpiSecond = fit(samples, "anpi")

    assert(npiFirst.split == npiSecond.split)
    assert(npiFirst.predictions == npiSecond.predictions)
    assert(anpiFirst.split == anpiSecond.split)
    assert(anpiFirst.predictions == anpiSecond.predictions)
  }
}
