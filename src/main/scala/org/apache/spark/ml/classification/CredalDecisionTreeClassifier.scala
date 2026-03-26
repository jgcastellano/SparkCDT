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

package org.apache.spark.ml.classification

import org.apache.hadoop.fs.Path
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._

import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
// BEGIN CREDAL MODIFICATION: Add DoubleParam and Param for hyperparameters
import org.apache.spark.ml.param.{ParamMap, DoubleParam, Param}
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.{DecisionTreeModel, Node, TreeClassifierParams}
import org.apache.spark.ml.tree.DecisionTreeModelReadWrite._
// BEGIN CREDAL MODIFICATION: Replace RandomForest with CredalRandomForest
import org.apache.spark.ml.tree.impl.CredalRandomForest
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.DatasetUtils._
import org.apache.spark.ml.util.Instrumentation.instrumented
// BEGIN CREDAL MODIFICATION: Import our impurity classes
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo, Strategy => OldStrategy, CredalStrategy}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel => OldDecisionTreeModel}
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType

/**
 * Decision tree learning algorithm (http://en.wikipedia.org/wiki/Decision_tree_learning)
 * for classification.
 * It supports both binary and multiclass labels, as well as both continuous and categorical
 * features.
 */
@Since("1.4.0")
class CredalDecisionTreeClassifier @Since("1.4.0") (
    @Since("1.4.0") override val uid: String)
  extends ProbabilisticClassifier[Vector, CredalDecisionTreeClassifier, DecisionTreeClassificationModel]
  with DecisionTreeClassifierParams with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("credal-dtc"))

  // --- BEGIN CREDAL MODIFICATION: Specific parameters ---
  val s = new DoubleParam(this, "s", "Hyperparameter for IDM", (x: Double) => x >= 0)
  def getS: Double = $(s)
  def setS(value: Double): this.type = set(s, value)
  setDefault(s -> 1.0)

  val epsilon = new DoubleParam(this, "epsilon", "Contamination parameter", (x: Double) => x >= 0 && x <= 1)
  def getEpsilon: Double = $(epsilon)
  def setEpsilon(value: Double): this.type = set(epsilon, value)
  setDefault(epsilon -> 0.2)

  // Create a custom parameter to avoid conflicting with Spark's strict "impurity" validator
  val credalImpurityStr = new Param[String](this, "credalImpurityStr", "Credal Impurity (idm, npi, econt)")
  def getCredalImpurityStr: String = $(credalImpurityStr)
  setDefault(credalImpurityStr -> "idm")
  // --- END CREDAL MODIFICATION ---

  // Override parameter setters from parent trait for Java API compatibility.

  /** @group setParam */
  @Since("1.4.0")
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group setParam */
  @Since("3.0.0")
  def setMinWeightFractionPerNode(value: Double): this.type = set(minWeightFractionPerNode, value)

  @Since("1.4.0")
  def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

  /**
   * Specifies how often to checkpoint the cached node IDs.
   * E.g. 10 means that the cache will get checkpointed every 10 iterations.
   * This is only used if cacheNodeIds is true and if the checkpoint directory is set in
   * [[org.apache.spark.SparkContext]].
   * Must be at least 1.
   * (default = 10)
   * @group setParam
   */
  @Since("1.4.0")
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** @group setParam */
  @Since("1.4.0")
  def setImpurity(value: String): this.type = {
    // --- BEGIN CREDAL MODIFICATION: Bypass the validator ---
    val supported = Set("gini", "entropy", "idm", "npi", "econt")
    require(supported.contains(value.toLowerCase), s"Impurity $value not supported.")
    set(credalImpurityStr, value.toLowerCase)
    // If it is a classical impurity, set the native parameter
    if (value.toLowerCase == "gini" || value.toLowerCase == "entropy") {
      set(impurity, value.toLowerCase)
    }
    this
    // --- END CREDAL MODIFICATION ---
  }

  /** @group setParam */
  @Since("1.6.0")
  def setSeed(value: Long): this.type = set(seed, value)

  /**
   * Sets the value of param [[weightCol]].
   * If this is not set or empty, we treat all instance weights as 1.0.
   * Default is not set, so all instances have weight one.
   *
   * @group setParam
   */
  @Since("3.0.0")
  def setWeightCol(value: String): this.type = set(weightCol, value)

  override protected def train(
      dataset: Dataset[_]): DecisionTreeClassificationModel = instrumented { instr =>
    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    val categoricalFeatures = MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    val instances = dataset.select(
      checkClassificationLabels($(labelCol), Some(numClasses)),
      checkNonNegativeWeights(get(weightCol)),
      checkNonNanVectors($(featuresCol))
    ).rdd.map { case Row(l: Double, w: Double, v: Vector) => Instance(l, w, v)
    }.setName("training instances")

    // --- BEGIN CREDAL MODIFICATION: Prepare strategy and call engine ---
    val impurityObj: Impurity = $(credalImpurityStr) match {
      case "idm"   => new IDM($(s))
      case "npi"   => NPI.asInstanceOf[Impurity]
      case "econt" => new EContamination($(epsilon))
      case "gini"  => Gini
      case "entropy" => Entropy
      case _       => Gini
    }

    val strategy = CredalStrategy.createClassificationStrategy(
      impurity = impurityObj,
      maxDepth = $(maxDepth),
      numClasses = numClasses,
      maxBins = $(maxBins),
      categoricalFeaturesInfo = categoricalFeatures
    )
    strategy.minInstancesPerNode = $(minInstancesPerNode)
    strategy.minInfoGain = $(minInfoGain)
    strategy.maxMemoryInMB = $(maxMemoryInMB)
    // REMOVED: strategy.forceIndexBins = $(forceIndexBins)
    // We don't need this parameter in the CDT: we always use index-based bins
    // --- END CREDAL MODIFICATION ---

    require(!strategy.bootstrap, "DecisionTreeClassifier does not need bootstrap sampling")
    instr.logNumClasses(numClasses)
    // Quitamos temporalmente 'impurity' original de este log para evitar fallos del framework de logs si no es Gini/Entropy
    instr.logParams(this, labelCol, featuresCol, predictionCol, rawPredictionCol,
      probabilityCol, leafCol, maxDepth, maxBins, minInstancesPerNode, minInfoGain,
      maxMemoryInMB, cacheNodeIds, checkpointInterval, seed, thresholds)

    // Explicit call to CredalRandomForest
    val trees = CredalRandomForest.run(instances, strategy, numTrees = 1, featureSubsetStrategy = "all",
      seed = $(seed), instr = Some(instr), parentUID = Some(uid))

    trees.head.asInstanceOf[DecisionTreeClassificationModel]
  }

  /** (private[ml]) Create a Strategy instance to use with the old API. */
  private[ml] def getOldStrategy(
      categoricalFeatures: Map[Int, Int],
      numClasses: Int): OldStrategy = {
    super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity,
      subsamplingRate = 1.0)
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): CredalDecisionTreeClassifier = defaultCopy(extra)
}

@Since("1.4.0")
object CredalDecisionTreeClassifier extends DefaultParamsReadable[CredalDecisionTreeClassifier] {
  /** Accessor for supported impurities: entropy, gini */
  @Since("1.4.0")
  final val supportedImpurities: Array[String] = Array("gini", "entropy", "idm", "npi", "econt")

  @Since("2.0.0")
  override def load(path: String): CredalDecisionTreeClassifier = super.load(path)
}
