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

/**
 * Object to compute imprecise entropy using the exact NPI-M algorithm for
 * multinomial data.
 * Based on: Abellán, J., Baker, R. M., & Coolen, F. P. A. (2011).
 * "Maximising entropy on the nonparametric predictive inference model for
 * multinomial data". European Journal of Operational Research, 212(1), 112–122.
 * https://doi.org/10.1016/j.ejor.2011.01.020
 */
@Since("1.0.0")
object NPI extends CredalImpurity {

  override def name: String = "npi"

  /**
   * Calculates Shannon entropy of the maximum-entropy NPI-M distribution.
   * NPI-M is defined for multinomial frequency counts, so counts must be
   * non-negative integers.
   * @param counts Array with the counts for each label.
   * @param totalCount Total sum of the counts.
   */
  override def calculate(counts: Array[Double], totalCount: Double): Double = {
    if (totalCount == 0.0) {
      return 0.0
    }

    val pmax = maximumEntropyDistribution(counts, totalCount)
    entropy(pmax)
  }

  /**
   * Computes the exact NPI-M maximum-entropy distribution used by `calculate`.
   * Package visibility is provided so tests can validate the distribution itself.
   */
  private[impurity] def maximumEntropyDistribution(
      counts: Array[Double], totalCount: Double): Array[Double] = {
    require(counts.nonEmpty, "NPI requires at least one class when totalCount is positive.")
    require(counts.forall(c => c >= 0.0 && c.isFinite),
      "NPI requires finite, non-negative integer class counts.")
    require(counts.forall(c => c == math.rint(c)),
      "NPI-M is defined for integer multinomial counts; fractional counts are not supported.")
    require(totalCount.isFinite && totalCount > 0.0,
      "NPI requires a finite, positive totalCount.")
    require(math.abs(counts.sum - totalCount) <= 1e-8 * math.max(1.0, totalCount),
      "NPI requires totalCount to equal the sum of class counts.")
    require(counts.forall(_ <= Int.MaxValue.toDouble),
      "NPI class counts exceed the supported integer range.")

    // Compute variables used by the exact Max NPI-M algorithm.
    val k = counts.length
    val n = totalCount
    val integerCounts = counts.map(_.toInt)
    val maxCount = integerCounts.max

    // K(i) is the number of classes with frequency i. Keep one trailing
    // zero entry so the K(i + 1) access at the maximum frequency is safe.
    val K = new Array[Int](maxCount + 2)
    integerCounts.foreach { count => K(count) += 1 }
    val Kprime = integerCounts.count(_ > 1)
    val pmax = new Array[Double](k)

    // CASE K' == 0: all observed frequencies are 0 or 1.
    if (Kprime == 0) {
      java.util.Arrays.fill(pmax, 1.0 / k)
    } else if (K(0) > Kprime) {
      val numerator = K(0) + K(1)
      val denominator = Kprime + K(1)
      val beta = numerator / denominator
      val h = numerator % denominator
      val indexes = integerCounts.zipWithIndex.sortBy(_._1).map(_._2)

      if (h < K(1) + 1) {
        var j = 0
        val firstGroupEnd = beta * (K(1) + 1) + h
        while (j < firstGroupEnd) {
          pmax(indexes(j)) = (K(1) + 1).toDouble / (n * firstGroupEnd)
          j += 1
        }
        while (j < numerator) {
          pmax(indexes(j)) = 1.0 / (n * beta)
          j += 1
        }
        while (j < k) {
          pmax(indexes(j)) = (integerCounts(indexes(j)) - 1.0) / n
          j += 1
        }
      } else {
        var j = 0
        val firstGroupEnd = h * (beta + 1)
        while (j < firstGroupEnd) {
          pmax(indexes(j)) = 1.0 / (n * (beta + 1))
          j += 1
        }
        while (j < numerator) {
          pmax(indexes(j)) = 1.0 / (n * beta)
          j += 1
        }
        while (j < k) {
          pmax(indexes(j)) = (integerCounts(indexes(j)) - 1.0) / n
          j += 1
        }
      }
    } else {
      // CASE K(0) <= K': distribute the remaining mass over adjacent
      // frequency groups, as specified by the exact NPI-M algorithm.
      var mass = Kprime - K(0)
      var j = 0
      while (j < k) {
        if (integerCounts(j) <= 1) pmax(j) = 1.0 / n
        else pmax(j) = (integerCounts(j) - 1.0) / n
        j += 1
      }

      var i = 1
      while (mass > 0) {
        val groupSize = K(i) + K(i + 1)
        if (groupSize < mass) {
          j = 0
          while (j < k) {
            if (integerCounts(j) == i || integerCounts(j) == i + 1) {
              pmax(j) += 1.0 / n
              mass -= 1
            }
            j += 1
          }
        } else {
          val groupCount = math.min(mass + 1 + K(i), groupSize)
          var remaining = groupCount
          val indexes = integerCounts.zipWithIndex.sortBy(_._1).map(_._2)
          j = 0
          while (j < k && remaining > 0) {
            val count = integerCounts(indexes(j))
            if (count == i || count == i + 1) {
              pmax(indexes(j)) += mass.toDouble / (n * groupCount)
              remaining -= 1
            }
            j += 1
          }
          mass = 0
        }
        i += 1
      }
    }

    pmax
  }

  /** Singleton lower and upper probabilities under the NPI-M model. */
  private[impurity] def singletonBounds(
      counts: Array[Double], totalCount: Double): (Array[Double], Array[Double]) = {
    require(counts.nonEmpty, "NPI requires at least one class.")
    require(totalCount.isFinite && totalCount > 0.0,
      "NPI singleton bounds require a finite, positive totalCount.")
    require(counts.forall(c => c >= 0.0 && c.isFinite && c == math.rint(c)),
      "NPI singleton bounds require finite, non-negative integer class counts.")
    require(math.abs(counts.sum - totalCount) <= 1e-8 * math.max(1.0, totalCount),
      "NPI requires totalCount to equal the sum of class counts.")

    val lower = counts.map(count => math.max(0.0, (count - 1.0) / totalCount))
    val upper = counts.map(count => math.min(1.0, (count + 1.0) / totalCount))
    (lower, upper)
  }

  private def entropy(probabilities: Array[Double]): Double = {
    var impurity = 0.0
    var j = 0
    while (j < probabilities.length) {
      if (probabilities(j) > 0.0) {
        impurity -= probabilities(j) * scala.math.log(probabilities(j))
      }
      j += 1
    }
    impurity
  }

  override def calculate(count: Double, sum: Double, sumSquares: Double): Double =
    throw new UnsupportedOperationException("NPI does not support regression.")

  def instance: this.type = this
}
