/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
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
 * Class for calculating imprecise entropy during multiclass classification. 
 *  Using the maximun of entropy for teh exact nonparametric Inference model for
 *  multinomial data (NPI-M).  
 *  Abellán, J., Baker, R. M., & Coolen, F. P. (2011). 
 *  Maximising entropy on the nonparametric predictive inference model for
 *  multinomial data. European Journal of Operational Research, 212(1), 112-122.
 */

@Since("1.0.0")
object NPI extends Impurity {

  private val _log2 = scala.math.log(2)

  // 2. CORRECCIÓN DEL ÁMBITO
  private[tree] def log2(x: Double) = scala.math.log(x) / _log2  
  private val MINDIFF = 0.00001

  // ====================================================================
  // [AQUÍ VA TODO TU CÓDIGO MATEMÁTICO INTACTO]
  // Pega aquí tus métodos: calculateOLD, calculateNEW, calculate, 
  // npiTransformation, maxEntropy2, minIndex2, sigIndex2, nMin2, etc.
  // ====================================================================
  /**
   * information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  @Since("1.1.0")
  /*override*/ def calculateOLD(counts: Array[Double], totalCount: Double): Double = {

    //var before = System.currentTimeMillis

    if (totalCount == 0) {
      return 0
    }

    //compute the variables used in the algorithm
    val k = counts.length //Number of classes
    val n = totalCount    //number of elements
    val m = counts.max.toInt + 1  //max number of classes

    //Compute K(0), K(1), ..., K(i)
    val K = new Array[Int](m)
    for (i <- 0 until m)
      K(i)= counts.count(_ == i)

    //compute K'
    val Kprime=counts.count(_ > 1)

    //pmax is the vector where the probabilities will be stored
    val pmax= new Array[Double](k)

    //Start the Max NPIM algorithm

    //CASE K'==0
    if (Kprime == 0) {
      for (i <- counts.indices)
        pmax(i) = 1.0 / k
    }

    //CASE K(0) > K'
    else if (K(0) > Kprime) {
      val beta = (K(0) + K(1)) / (Kprime + K(1))
      val h    = (K(0) + K(1)) % (Kprime + K(1))
      val indexes = counts.zipWithIndex.sorted.map( _._2 )

      if (h < K(1)+1) {
        var j = 0
        while ( j < (beta * (K(1) + 1) + h) ) {
          pmax(indexes(j)) = (K(1) + 1) / (n * (beta * (K(1) + 1) + h))
          j += 1
        }
        while ( j < (K(0) + K(1)) ) {
          pmax(indexes(j)) = 1.0 / (n * beta)
          j += 1
        }
        while ( j < k ) {
          pmax(indexes(j)) = (counts(indexes(j)) - 1.0) / n
          j += 1
        }
      } else {
        var j = 0
        while ( j < (h * (beta + 1)) ) {
          pmax(indexes(j)) = 1.0 / (n * (beta + 1))
          j += 1
        }
        while ( j < (K(0) + K(1)) ) {
          pmax(indexes(j)) = 1.0 / (n * beta)
          j += 1
        }
        while ( j < k ) {
          pmax(indexes(j)) = (counts(indexes(j)) - 1.0) / n
          j += 1
        }
      }
    } else { //When K0<=K'
      var mass = Kprime - K(0)
      var j = 0
      while ( j < k ) {
        if (counts(j) == 0 || counts(j) == 1) pmax(j) = 1.0 / n
        else pmax(j) = (counts(j) - 1) / n
        j += 1
      }
      //REPARTIMOS LA MASA
      var i = 1
      while ( mass > 0 ) {
        if ((K(i) + K(i+1)) < mass) {
          for ( j <- 0 until k) {
            if (counts(j) == i || counts(j) == (i + 1)) {
              pmax(j) += 1.0 / n
              mass -= 1
            }
          }
        } else {
          val min = Math.min(mass + 1 + K(i), K(i) + K(i+1))
          var G = min
          val index = counts.zipWithIndex.sorted.map( _._2 )
          var j = 0
          while ( (j < k ) && G>0) {
            if (counts(index(j)) == i || counts(index(j)) == (i + 1)) {
              pmax(index(j)) += mass.toDouble / (n * min)
              G -= 1
            }
            j += 1
          }
          mass = 0
        }
        i += 1
      }
    }

    //compute the impurity(entropy)
    var impurity = 0.0
    for (j <- pmax.indices) {  // más idiomático que 0 until pmax.length
      if (pmax(j) > 0) {
        impurity -= pmax(j) * scala.math.log(pmax(j))
      }
    }
    
    /*val totalTime=System.currentTimeMillis-before

    before = System.currentTimeMillis
    val entropyO=calculateOPT(counts, totalCount)
    val totalTimeO=System.currentTimeMillis-before

    if (impurity != entropyO) {
      println("FIFERENTE Ver 0="+impurity+" Ver Optimized="+entropyO)
    } else {
      println("NOW="+totalTime)
      println("OPT="+totalTimeO+"\n")
    }*/

    impurity
  }



  /**
   * Information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  @Since("1.1.0")
  /*override*/ def calculateNEW(counts: Array[Double], totalCount: Double): Double = {

    if (totalCount == 0) {
      return 0
    }

    val k = counts.length // Number of classes
    val n = totalCount    // Number of elements

    // Compute K(0), K(1), ..., K(i) using a single pass over counts
    val countMap = counts.groupBy(identity).view.map { case (key, value) => key -> value.length }.toMap.withDefaultValue(0)
    val K = Array.tabulate(counts.max.toInt + 1)(i => countMap(i.toDouble))

    // Compute K'
    val Kprime = counts.count(_ > 1)

    // pmax is the vector where the probabilities will be stored
    val pmax = new Array[Double](k)

    if (Kprime == 0) { // CASE K'==0
      java.util.Arrays.fill(pmax, 1.0 / k)
    } else if (K(0) > Kprime) { // CASE K(0) > K'
      calculatePmaxForK0GreaterThanKprime(counts, K, Kprime, n, pmax)
    } else { // CASE K(0) <= K'
      calculatePmaxForK0LessOrEqualToKprime(counts, K, Kprime, n, pmax)
    }

    //compute the impurity(entropy)
    var impurity = 0.0
    for (j <-0 until k)
      impurity -= pmax(j) * scala.math.log(pmax(j))

    impurity
  }
  
  private def calculatePmaxForK0GreaterThanKprime(counts: Array[Double], K: Array[Int], Kprime: Int, n: Double, pmax: Array[Double]): Unit = {
    val k = counts.length
    val beta = (K(0) + K(1)).toDouble / (Kprime + K(1))
    val h = (K(0) + K(1)) % (Kprime + K(1))
    val indexes = counts.zipWithIndex.sortBy(_._1).map(_._2)

    if (h < K(1) + 1) {
      var j = 0
      while (j < Math.min(indexes.length, (beta * (K(1) + 1) + h).toInt)) {
        pmax(indexes(j)) = (K(1) + 1).toDouble / (n * (beta * (K(1) + 1) + h))
        j += 1
      }
      while (j < Math.min(indexes.length, (K(0) + K(1)))) {
        pmax(indexes(j)) = 1.0 / (n * beta)
        j += 1
      }
      while (j < k && j < indexes.length) {
        pmax(indexes(j)) = (counts(indexes(j)) - 1.0) / n
        j += 1
      }
    } else {
      var j = 0
      while (j < Math.min(indexes.length, (h * (beta + 1)).toInt)) {
        pmax(indexes(j)) = 1.0 / (n * (beta + 1))
        j += 1
      }
      while (j < Math.min(indexes.length, (K(0) + K(1)))) {
        pmax(indexes(j)) = 1.0 / (n * beta)
        j += 1
      }
      while (j < k && j < indexes.length) {
        pmax(indexes(j)) = (counts(indexes(j)) - 1.0) / n
        j += 1
      }
    }
  }


  private def calculatePmaxForK0GreaterThanKprimeOLD(counts: Array[Double], K: Array[Int], Kprime: Int, n: Double, pmax: Array[Double]): Unit = {
    val k = counts.length
    val beta = (K(0) + K(1)).toDouble / (Kprime + K(1))
    val h = (K(0) + K(1)) % (Kprime + K(1))
    val indexes = counts.zipWithIndex.sortBy(_._1).map(_._2)

    if (h < K(1) + 1) {
      var j = 0
      while (j < (beta * (K(1) + 1) + h).toInt) {
        pmax(indexes(j)) = (K(1) + 1).toDouble / (n * (beta * (K(1) + 1) + h)) //BUG java.lang.ArrayIndexOutOfBoundsException: Index 9 out of bounds for length 9
        j += 1
      }
      while (j < (K(0) + K(1))) {
        pmax(indexes(j)) = 1.0 / (n * beta)
        j += 1
      }
      while (j < k) {
        pmax(indexes(j)) = (counts(indexes(j)) - 1.0) / n
        j += 1
      }
    } else {
      var j = 0
      while (j < (h * (beta + 1)).toInt) {
        pmax(indexes(j)) = 1.0 / (n * (beta + 1))
        j += 1
      }
      while (j < (K(0) + K(1))) {
        pmax(indexes(j)) = 1.0 / (n * beta)
        j += 1
      }
      while (j < k) {
        pmax(indexes(j)) = (counts(indexes(j)) - 1.0) / n
        j += 1
      }
    }
  }

  private def calculatePmaxForK0LessOrEqualToKprime(counts: Array[Double], K: Array[Int], Kprime: Int, n: Double, pmax: Array[Double]): Unit = {
    val k = counts.length
    var mass = Kprime - K(0)
    for (j <- counts.indices) {
      pmax(j) = if (counts(j) <= 1) 1.0 / n else (counts(j) - 1.0) / n
    }

    var i = 1
    while (mass > 0 && i < K.length - 1) {
      if ((K(i) + K(i + 1)) <= mass) {
        for (j <- counts.indices) {
          if (counts(j) == i || counts(j) == i + 1) {
            pmax(j) += 1.0 / n
            mass -= 1
          }
        }
      } else {
        val min = Math.min(mass + 1 + K(i), K(i) + K(i + 1))
        var G = min
        val index = counts.zipWithIndex.sortBy(_._1).map(_._2)
        var j = 0
        while (j < k && G > 0) {
          if (counts(index(j)) == i || counts(index(j)) == i + 1) {
            pmax(index(j)) += mass.toDouble / (n * min)
            G -= 1
          }
          j += 1
        }
        mass = 0
      }
      i += 1
    }
  }



/**
   * Information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  @Since("1.1.0")
  override def calculate(counts: Array[Double], totalCount: Double): Double = {

    if (totalCount == 0) {
      return 0
    }

    val pmax = npiTransformation(counts)

    // Compute the impurity (entropy)
    var impurity = 0.0
    for (j <- pmax.indices) {
      if (pmax(j) > 0) {
        impurity -= pmax(j) * scala.math.log(pmax(j))
      }
    }

    impurity
  }

/**
   * This method transforms the given array into probability intervals and then
   * computes the maximum entropy for these intervals. The intervals are obtained 
   * by applying the NPI (Non Parametric Predictive Inference) approach.
   * 
   * @param array the array with counts
   * @return an array with the maximum entropy probabilities
   */
  private def npiTransformation(array: Array[Double]): Array[Double] = {
    val n = array.sum  // n debe ser la suma, no array.length
    
    val pl = new Array[Double](array.length)
    val pu = new Array[Double](array.length)
    
    for (i <- array.indices) {
      pl(i) = math.max(0.0, (array(i) - 1) / n)
      pu(i) = math.min(1.0, (array(i) + 1) / n)
    }
    
    val indexes = ArrayBuffer.range(0, array.length)
    
    maxEntropy(pl, pu, indexes)
  }

/**
   * This method computes the maximum entropy for intervals of probability 
   * distributions. More info in:
   *   Joaquin Abellan and Serafin Moral. 
   *   Maximum of entropy for credal sets. 
   *   Int. J. Uncertain. Fuzziness Knowl.-Based Syst. 11(5), 587-597, 2003.
   *   doi: 10.1142/S021848850300234X 
   * 
   * @param pl array with the lower values of the probability intervals
   * @param pu array with the upper values of the probability intervals
   * @param indexes list of indexes to use
   * @return array with the maximum entropy for each value
   */
  private def maxEntropy(pl: Array[Double], pu: Array[Double], 
                         indexes: ArrayBuffer[Int]): Array[Double] = {
    
    val sumPl = pl.sum
    
    // Base case: if sum of pl is approximately 1, we're done
    if (math.abs(1 - sumPl) < 0.001) {
      return pl.clone()
    }
    
    // Remove indexes where pl[i] == pu[i] (no room to increase)
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
    
    // Find minimum value index
    val men = minIndex(pl, indexes)
    
    // Find second minimum value index
    val sig = sigIndex(pl, men, indexes)
    
    // Count how many elements have the minimum value
    val nme = nMin(pl, men, indexes)
    
    val minVal = pl(men)
    
    // Increase probabilities for elements with minimum value
    for (i <- pl.indices) {
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
    }
    
    // Recursive call
    maxEntropy(pl, pu, indexes)
  }
  
  /**
   * Find the index of the minimum value in the array p, considering only
   * the given indexes.
   * 
   * @param p array with values
   * @param indexes list of indexes to consider
   * @return index of the minimum value
   */
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
  
  /**
   * Find the index of the second minimum value in the array p.
   * 
   * @param p array with values
   * @param indexlowest index of the minimum value
   * @param indexes list of indexes to consider
   * @return index of the second minimum value, or indexlowest if all values are equal
   */
  private def sigIndex(p: Array[Double], indexlowest: Int, 
                       indexes: ArrayBuffer[Int]): Int = {
    val q = new Array[Double](indexes.length)
    var different = false
    
    for (i <- indexes.indices) {
      val idx = indexes(i)
      if (math.abs(p(idx) - p(indexlowest)) < MINDIFF) {
        q(i) = Double.MaxValue
      } else {
        different = true
        q(i) = p(idx)
      }
    }
    
    if (different) {
      val minIdx = q.zipWithIndex.minBy(_._1)._2
      indexes(minIdx)
    } else {
      indexlowest
    }
  }
  
  /**
   * Count how many indexes have the minimum value.
   * 
   * @param p array with values
   * @param indexlowest index of the minimum value
   * @param indexes list of indexes to consider
   * @return number of indexes with the minimum value
   */
  private def nMin(p: Array[Double], indexlowest: Int, 
                   indexes: ArrayBuffer[Int]): Int = {
    var count = 0
    
    for (i <- indexes) {
      if (math.abs(p(i) - p(indexlowest)) < MINDIFF) {
        count += 1
      }
    }
    
    count
  }
  
  

  @Since("1.0.0")
  override def calculate(count: Double, sum: Double, sumSquares: Double): Double =
    throw new UnsupportedOperationException("NPI.calculate")

  @Since("1.1.0")
  def instance: this.type = this
}

private[spark] class NPIAggregator(numClasses: Int)
  extends ImpurityAggregator(numClasses + 1) with Serializable {

  def update(
      allStats: Array[Double],
      offset: Int,
      label: Double,
      numSamples: Int,
      sampleWeight: Double): Unit = {
    if (label >= numClasses) {
      throw new IllegalArgumentException(s"NPIAggregator given label $label" +
        s" but requires label < numClasses (= ${numClasses}).")
    }
    if (label < 0) {
      throw new IllegalArgumentException(s"NPIAggregator given label $label" +
        s"but requires label is non-negative.")
    }
    allStats(offset + label.toInt) += numSamples * sampleWeight
    allStats(offset + statsSize - 1) += numSamples
  }

  def getCalculator(allStats: Array[Double], offset: Int): NPICalculator = {
    // 3. CORRECCIÓN DE RENDIMIENTO (Eliminado el .view.toArray)
    new NPICalculator(allStats.slice(offset, offset + statsSize - 1),
      allStats(offset + statsSize - 1).toLong)
  }
}

private[spark] class NPICalculator(stats: Array[Double], var rawCount: Long)
  extends ImpurityCalculator(stats) {

  def copy: NPICalculator = new NPICalculator(stats.clone(), rawCount)

  def calculate(): Double = NPI.calculate(stats, stats.sum)

  def count: Double = stats.sum

  def predict: Double = if (count == 0) {
    0
  } else {
    indexOfLargestArrayElement(stats)
  }

  override def prob(label: Double): Double = {
    val lbl = label.toInt
    require(lbl < stats.length,
      s"NPICalculator.prob given invalid label: $lbl (should be < ${stats.length}")
    require(lbl >= 0, "NPI does not support negative labels")
    val cnt = count
    if (cnt == 0) {
      0
    } else {
      stats(lbl) / cnt
    }
  }

  override def toString: String = s"NPICalculator(stats = [${stats.mkString(", ")}])"
}
