package org.apache.spark.ml.classification

import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo, Strategy => OldStrategy}

class CredalDecisionTreeClassifier(override val uid: String)
  extends DecisionTreeClassifier(uid) {

  def this() = this(Identifiable.randomUID("credal_dtc"))

  // 1. Creamos un nuevo parámetro para evitar chocar con la validación restrictiva de Spark
  final val credalImpurity: Param[String] = new Param[String](this, "credalImpurity",
    s"Criterion used for information gain calculation. Supported options: " +
      s"${CredalDecisionTreeClassifier.supportedImpurities.mkString(", ")}",
    ParamValidators.inArray(CredalDecisionTreeClassifier.supportedImpurities))

  setDefault(credalImpurity -> "gini")

  // Setter para nuestro nuevo parámetro
  def setCredalImpurity(value: String): this.type = set(credalImpurity, value)

  // ¡MAGIA!: Al heredar de DecisionTreeClassifier ya tenemos setMaxDepth(), 
  // setSeed(), train(), etc. No necesitamos reescribirlos.

  // 2. Interceptamos la estrategia interna para inyectar nuestra Impurity credal
  private[ml] override def getOldStrategy(
      categoricalFeatures: Map[Int, Int],
      numClasses: Int): OldStrategy = {

    val oldImpurity = $(credalImpurity).toLowerCase(java.util.Locale.ROOT) match {
      case "gini" => org.apache.spark.mllib.tree.impurity.Gini
      case "entropy" => org.apache.spark.mllib.tree.impurity.Entropy
      case "npi" => org.apache.spark.mllib.tree.impurity.NPI
      case "idm" => org.apache.spark.mllib.tree.impurity.IDM
      case "econtamination" => org.apache.spark.mllib.tree.impurity.EContamination
      
      case _ => throw new IllegalArgumentException(s"Impurity ${$(credalImpurity)} no soportado.")
    }

    // Llamamos al método protegido de la clase abuelo en Spark
    super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, oldImpurity, subsamplingRate = 1.0)
  }

  override def copy(extra: ParamMap): CredalDecisionTreeClassifier = defaultCopy(extra)
}

object CredalDecisionTreeClassifier {
  final val supportedImpurities: Array[String] = Array("gini", "entropy", "npi", "idm", "econtamination")
}
