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
package org.apache.spark.mllib.tree.impurity

import org.apache.spark.annotation.DeveloperApi

/**
 * Base trait for impurity measures based on credal sets.
 */
@DeveloperApi
trait CredalImpurity extends Impurity {
  def name: String
}

object CredalImpurity {
  def apply(name: String, s: Double = 1.0, epsilon: Double = 0.2): CredalImpurity = {
    name.toLowerCase match {
      case "idm"   => new IDM(s)
      case "npi"   => NPI.asInstanceOf[CredalImpurity]
      case "econt" | "e-cont" => new EContamination(epsilon)
      case _ => throw new IllegalArgumentException(s"Unknown credal impurity: $name")
    }
  }
}

/**
 * Aggregator used by the Distributed Decision Tree engine.
 */
private[spark] class CredalImpurityAggregator(val impurity: CredalImpurity, numClasses: Int)
  extends ImpurityAggregator(numClasses + 1) with Serializable {

  def update(
      allStats: Array[Double],
      offset: Int,
      label: Double,
      numSamples: Int,
      sampleWeight: Double): Unit = {
    if (label < 0 || label >= numClasses) {
      throw new IllegalArgumentException(s"Label $label out of range.")
    }
    allStats(offset + label.toInt) += numSamples * sampleWeight
    allStats(offset + statsSize - 1) += numSamples
  }

  def getCalculator(allStats: Array[Double], offset: Int): CredalImpurityCalculator = {
    new CredalImpurityCalculator(
      impurity,
      allStats.view.slice(offset, offset + statsSize - 1).toArray,
      allStats(offset + statsSize - 1).toLong
    )
  }
}

/**
 * THIS IS THE ONLY CALCULATOR CLASS.
 * Responsible for calling the specific impurity's 'calculate' method.
 */
private[spark] class CredalImpurityCalculator(
    val impurity: CredalImpurity,
    stats: Array[Double],
    var rawCount: Long) extends ImpurityCalculator(stats) {

  override def copy: CredalImpurityCalculator = 
    new CredalImpurityCalculator(impurity, stats.clone(), rawCount)

  // Here we delegate the mathematical calculation to the serialized instance (IDM, NPI, ECont)
  override def calculate(): Double = {
    if (count == 0) return 0.0
    impurity.calculate(stats, stats.sum)
  }

  override def count: Double = stats.sum

  override def predict: Double = if (count == 0) 0 else indexOfLargestArrayElement(stats)

  override def prob(label: Double): Double = {
    val lbl = label.toInt
    if (count == 0 || lbl < 0 || lbl >= stats.length) 0 else stats(lbl) / count
  }
}
