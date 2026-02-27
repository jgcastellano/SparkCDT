package org.apache.spark.mllib.tree.impurity

object EContaminationParameter {
  private var e: Double = 0.2 // epsilon hyperparameter of the e-contamination transformation

  // Getter for the e property 
  def getE: Double = e

  // Setter for the e property
  def setE(newValue: Double): Unit = {
    e = newValue
  }
}
