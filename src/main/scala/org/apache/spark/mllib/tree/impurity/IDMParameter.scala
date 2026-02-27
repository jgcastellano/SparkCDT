package org.apache.spark.mllib.tree.impurity

object IDMParameter {
  private var s: Double = 1.0 // s hyperparameter of the Imprecise Dirichlet Method

  // Getter for the s property 
  def getS: Double = s

  // Setter for the s property
  def setS(newValue: Double): Unit = {
    s = newValue
  }
}
