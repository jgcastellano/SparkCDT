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

import org.apache.spark.annotation.Since
import scala.collection.mutable.ArrayBuffer

/**
 * Object to compute imprecise entropy using the approximate NPI-M implementation
 * (A-NPI-M) for multinomial data.
 * Based on: Abellán, J., Baker, R. M., & Coolen, F. P. A. (2011).
 * "Maximising entropy on the nonparametric predictive inference model for
 * multinomial data". European Journal of Operational Research, 212(1), 112–122.
 * https://doi.org/10.1016/j.ejor.2011.01.020
 */
@Since("1.0.0")
object ANPI extends CredalImpurity {

  override def name: String = "anpi"

  private val MINDIFF = 0.00001

  /**
   * Entry point for impurity calculation in Spark.
   * @param counts Array with the counts for each label.
   * @param totalCount Total sum of the counts.
   */
  override def calculate(counts: Array[Double], totalCount: Double): Double = {
    if (totalCount == 0) {
      return 0.0
    }

    // A-NPI-M transformation to obtain the approximate maximum-entropy distribution
    val pmax = maximumEntropyDistribution(counts, totalCount)

    // Shannon entropy calculation over the pmax distribution
    var impurity = 0.0
    var j = 0
    while (j < pmax.length) {
      if (pmax(j) > 0) {
        impurity -= pmax(j) * scala.math.log(pmax(j))
      }
      j += 1
    }
    impurity
  }

  /**
   * Computes the NPI-M probability intervals [pl, pu] and applies the A-NPI-M
   * procedure to find an approximate maximum-entropy distribution.
   */
  private[impurity] def maximumEntropyDistribution(
      array: Array[Double], sum: Double): Array[Double] = {
    val (pl, pu) = singletonBounds(array, sum)
    val indexes = ArrayBuffer.range(0, array.length)
    maxEntropy(pl, pu, indexes)
  }

  /** Singleton lower and upper probabilities represented by A-NPI-M. */
  private[impurity] def singletonBounds(
      counts: Array[Double], sum: Double): (Array[Double], Array[Double]) = {
    val n = counts.length
    val pl = new Array[Double](n)
    val pu = new Array[Double](n)

    var i = 0
    while (i < n) {
      pl(i) = math.max(0.0, (counts(i) - 1.0) / sum)
      pu(i) = math.min(1.0, (counts(i) + 1.0) / sum)
      i += 1
    }
    (pl, pu)
  }

  /**
  * Recursive A-NPI-M procedure that approximately maximizes entropy in the
  * credal set defined by NPI-M probability intervals, as described by
  * Abellán et al. (2011).
   */
  private def maxEntropy(pl: Array[Double], pu: Array[Double],
                         indexes: ArrayBuffer[Int]): Array[Double] = {

    val sumPl = pl.sum

    // Base case: the sum of probabilities is (almost) 1
    if (math.abs(1.0 - sumPl) < 0.001) {
      return pl.clone()
    }

    // Remove indices where there is no margin for increase (pl == pu)
    var i = 0
    while (i < indexes.length) {
      if (math.abs(pl(indexes(i)) - pu(indexes(i))) < MINDIFF) {
        indexes.remove(i)
      } else {
        i += 1
      }
    }

    if (indexes.isEmpty) {
      return pl.clone()
    }

    // Logic to search for minimal indices and distribute remaining mass
    val men = minIndex(pl, indexes)
    val sig = sigIndex(pl, men, indexes)
    val nme = nMin(pl, men, indexes)
    val minVal = pl(men)

    i = 0
    while (i < pl.length) {
      if (math.abs(pl(i) - minVal) < MINDIFF) {
        if (sig != men) {
          pl(i) = pl(i) + math.min(
            math.min(pu(i) - pl(i), pl(sig) - minVal),
            (1.0 - sumPl) / nme
          )
        } else {
          pl(i) = pl(i) + math.min(
            math.min(pu(i) - pl(i), (1.0 - sumPl) / nme),
            1.0
          )
        }
      }
      i += 1
    }

    maxEntropy(pl, pu, indexes)
  }

  // Technical helpers for index search
  private def minIndex(p: Array[Double], indexes: ArrayBuffer[Int]): Int = {
    var min = Double.MaxValue
    var index = -1
    for (i <- indexes) {
      if (p(i) < min) {
        min = p(i)
        index = i
      }
    }
    index
  }

  private def sigIndex(p: Array[Double], indexlowest: Int,
                       indexes: ArrayBuffer[Int]): Int = {
    var minDiffValue = Double.MaxValue
    var index = indexlowest
    var found = false

    for (idx <- indexes) {
      if (math.abs(p(idx) - p(indexlowest)) > MINDIFF && p(idx) < minDiffValue) {
        minDiffValue = p(idx)
        index = idx
        found = true
      }
    }
    if (found) index else indexlowest
  }

  private def nMin(p: Array[Double], indexlowest: Int,
                   indexes: ArrayBuffer[Int]): Int = {
    var count = 0
    for (i <- indexes) {
      if (math.abs(p(i) - p(indexlowest)) < MINDIFF) count += 1
    }
    count
  }

  override def calculate(count: Double, sum: Double, sumSquares: Double): Double =
    throw new UnsupportedOperationException("ANPI does not support regression.")

  def instance: this.type = this
}

/**
 * Specific calculator for ANPI.
 */
private[spark] class ANPICalculator(stats: Array[Double], var rawCount: Long)
  extends ImpurityCalculator(stats) {

  override def copy: ANPICalculator = new ANPICalculator(stats.clone(), rawCount)
  override def calculate(): Double = ANPI.calculate(stats, stats.sum)
  override def count: Double = stats.sum
  override def predict: Double = if (count == 0) 0 else indexOfLargestArrayElement(stats)

  override def prob(label: Double): Double = {
    val lbl = label.toInt
    if (count == 0) 0 else stats(lbl) / count
  }
}
