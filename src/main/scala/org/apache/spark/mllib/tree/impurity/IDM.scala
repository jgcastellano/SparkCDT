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
 * Class to compute imprecise entropy using the Imprecise Dirichlet Model (IDM).
 * As a class (not an object), the parameter `s` is serialized and sent to workers.
 * Reference: J. Abellan, S. Moral. "Upper entropy of credal sets..."
 * Int. J. Approx. Reason., 39 (2005).
 *
 * @param s Hyperparameter for the IDM method.
 */
@Since("1.0.0")
class IDM(val s: Double) extends CredalImpurity {

  override def name: String = "idm"

  /**
   * Impurity calculation for multiclass classification.
   * Based on the optimized implementations `calculateS1` and `calculateCountsS`.
   */
  override def calculate(counts: Array[Double], totalCount: Double): Double = {
    if (totalCount == 0) {
      return 0.0
    }

    if (s == 1.0) {
      calculateS1(counts, totalCount)
    } else {
      val newCounts = calculateCountsS(counts, s)
      val total = totalCount + s
      var impreciseInfoGain = 0.0
      var i = 0
      while (i < newCounts.length) {
        val freq = newCounts(i) / total
        if (freq > 0) {
          impreciseInfoGain -= freq * scala.math.log(freq)
        }
        i += 1
      }
      impreciseInfoGain
    }
  }

  /**
   * Specific case for s=1.0 (mass optimization).
   */
  private def calculateS1(counts: Array[Double], totalCount: Double): Double = {
    val (minIndexes, _) = getMinIndexes(counts)
    val total = totalCount + 1.0
    val mass = 1.0 / minIndexes.length
    var impreciseInfoGain = 0.0
    var classIndex = 0

    while (classIndex < counts.length) {
      val classCount = counts(classIndex)
      val freq = if (minIndexes.contains(classIndex)) {
        (classCount + mass) / total
      } else {
        classCount / total
      }
      if (freq > 0) {
        impreciseInfoGain -= freq * scala.math.log(freq)
      }
      classIndex += 1
    }
    impreciseInfoGain
  }

  /**
   * General case for any value of s (recursive distribution of mass).
   */
  private def calculateCountsS(counts: Array[Double], currentS: Double): Array[Double] = {
    val (minIndexes, _) = getMinIndexes(counts)

    var massToDistribute = currentS
    if (currentS > minIndexes.length) {
      massToDistribute = minIndexes.length.toDouble
    }

    val newCounts = Array.fill(counts.length)(0.0)
    var classIndex = 0
    while (classIndex < counts.length) {
      val classCount = counts(classIndex)
      if (minIndexes.contains(classIndex)) {
        newCounts(classIndex) = classCount + ( massToDistribute / minIndexes.length )
      } else {
        newCounts(classIndex) = classCount
      }
      classIndex += 1
    }

    val remainingMass = currentS - massToDistribute
    if (remainingMass > 0.001) {
      calculateCountsS(newCounts, remainingMass)
    } else {
      newCounts
    }
  }

  /**
   * Helper to find indices with the minimum frequency.
   */
  private def getMinIndexes(counts: Array[Double]): (List[Int], Double) = {
    counts.zipWithIndex.foldLeft((List.empty[Int], Double.MaxValue)) {
      case ((indexes, currentMin), (value, index)) =>
        if (value < currentMin) (List(index), value)
        else if (value == currentMin) (index :: indexes, currentMin)
        else (indexes, currentMin)
    }
  }

  override def calculate(count: Double, sum: Double, sumSquares: Double): Double =
    throw new UnsupportedOperationException("IDM does not support regression.")

  def instance: this.type = this
}

/**
 * Factory to maintain compatibility with Spark style.
 */
object IDM {
  def apply(s: Double): IDM = new IDM(s)
}
