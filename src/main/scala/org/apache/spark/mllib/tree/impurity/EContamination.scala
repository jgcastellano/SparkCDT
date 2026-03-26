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
 * Class to compute imprecise entropy using the epsilon-contamination model.
 * The interval for each category is defined as [(1-eps)*pi, (1-eps)*pi + eps].
 * @param epsilon Contamination parameter (hyperparameter).
 */
@Since("1.0.0")
class EContamination(val epsilon: Double) extends CredalImpurity {

  override def name: String = "econt"
  
  private val MINDIFF = 0.00001

  /**
   * Impurity calculation for multiclass classification.
   * @param counts Array with the counts for each label.
   * @param totalCount Total sum of the counts.
   */
  override def calculate(counts: Array[Double], totalCount: Double): Double = {
    if (totalCount == 0) {
      return 0.0
    }

    // Transformation using the instance epsilon (serialized)
    val pmax = eContaminationTransformation(counts, totalCount)

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
   * Transforms counts into probability intervals and searches for the
   * maximum-entropy distribution.
   */
  private def eContaminationTransformation(array: Array[Double], sum: Double): Array[Double] = {
    val n = array.length
    val cfactor = (1.0 - epsilon) / sum
    
    val pl = new Array[Double](n)
    val pu = new Array[Double](n)
    val indexes = ArrayBuffer.empty[Int]
    
    var i = 0
    while (i < n) {
      //val p_i = array(i) / sum
      //val lower = math.max(0.0, (1.0 - epsilon) * p_i)
      //val upper = math.min(1.0, (1.0 - epsilon) * p_i + epsilon)
      val lower = math.max(0.0, cfactor * array(i))
      val upper = math.min(1.0, lower + epsilon)
      pl(i) = lower
      pu(i) = upper
      
      if (upper - lower >= MINDIFF) {
        indexes += i
      }
      i += 1
    }
    
    maxEntropy(pl, pu, indexes)
  }

  /**
   * Maximum-entropy search algorithm for credal sets defined by intervals.
   * Based on Abellán and Moral (2003).
   */
  private def maxEntropy(pl: Array[Double], pu: Array[Double], 
                         indexes: ArrayBuffer[Int]): Array[Double] = {
    
    val sumPl = pl.sum
    if (math.abs(1.0 - sumPl) < 0.001) {
      return pl.clone()
    }
    
    var i = 0
    while (i < indexes.length) {
      if (math.abs(pl(indexes(i)) - pu(indexes(i))) < MINDIFF) {
        indexes.remove(i)
      } else {
        i += 1
      }
    }
    
    if (indexes.isEmpty) return pl.clone()
    
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

  private def minIndex(p: Array[Double], indexes: ArrayBuffer[Int]): Int = {
    var min = Double.MaxValue
    var index = -1
    for (idx <- indexes) {
      if (p(idx) < min) {
        min = p(idx)
        index = idx
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
    for (idx <- indexes) {
      if (math.abs(p(idx) - p(indexlowest)) < MINDIFF) count += 1
    }
    count
  }

  override def calculate(count: Double, sum: Double, sumSquares: Double): Double =
    throw new UnsupportedOperationException("EContamination does not support regression.")

  def instance: this.type = this
}

/**
 * Factory to facilitate instantiation.
 */
object EContamination {
  def apply(epsilon: Double): EContamination = new EContamination(epsilon)
}
