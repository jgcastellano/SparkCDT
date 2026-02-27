package org.apache.spark.mllib.tree.impurity

import org.apache.spark.annotation.Since
import scala.collection.mutable.ArrayBuffer

/**
 * Class for calculating imprecise entropy during multiclass classification. 
 * Using the maximum of entropy for epsilon-contamination model for multinomial data.
 */
@Since("1.0.0")
object EContamination extends Impurity {

  private val MINDIFF = 0.00001
  
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
    val epsilon = EContaminationParameter.getE  // Hyperparameter
    val pmax = eContaminationTransformation(counts,epsilon)

    // Compute the impurity (entropy)
    var impurity = 0.0
    for (j <- pmax.indices) {  // más idiomático que 0 until pmax.length
      if (pmax(j) > 0) {
        impurity -= pmax(j) * scala.math.log(pmax(j))
      }
    }

    impurity
  }
  
  /**
   * This method transforms the given array into probability intervals and then
   * computes the maximum entropy for these intervals. The intervals are obtained 
   * by applying the epsilo-contamination approach.
   * 
   * @param array the array with counts
   * @param epsilon the contamination parameter
   * @return an array with the maximum entropy probabilities
   */
  private def eContaminationTransformation(array: Array[Double], epsilon: Double): Array[Double] = {
    val sum = array.sum
    val cfactor = (1.0 - epsilon)/sum // precompute constant factor, for efficiency

    // Calculate probability intervals and only add valid indexes
    val pl = new Array[Double](array.length)
    val pu = new Array[Double](array.length)
    val indexes = ArrayBuffer.empty[Int]
    
    //Compute interval of the form [(1-\epsilon)p_i, (1-\epsilon)p_i + \epsilon]
    for (i <- 0 until array.length) {
      val lower = math.max(0.0, cfactor * array(i))
      val upper = math.min(1.0, lower + epsilon)
      pl(i) = lower
      pu(i) = upper
      
      // Only add index if there's room to grow. Optimization 
      //to avoid unnecessary calculations in maxEntropy.
      if (upper - lower >= MINDIFF) {
        indexes += i
      }
    }
    
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


  /**
   * variance calculation
   * @param count number of instances
   * @param sum sum of labels
   * @param sumSquares summation of squares of the labels
   * @return information value, or 0 if count = 0
   */
  @Since("1.0.0")
  override def calculate(count: Double, sum: Double, sumSquares: Double): Double =
    throw new UnsupportedOperationException("EContamination.calculate")

  /**
   * Get this impurity instance.
   * This is useful for passing impurity parameters to a Strategy in Java.
   */
  @Since("1.1.0")
  def instance: this.type = this


}

/**
 * Class for updating views of a vector of sufficient statistics.
 */
private[spark] class EContaminationAggregator(numClasses: Int)
  extends ImpurityAggregator(numClasses + 1) with Serializable {

  def update(
      allStats: Array[Double],
      offset: Int,
      label: Double,
      numSamples: Int,
      sampleWeight: Double): Unit = {
    if (label >= numClasses) {
      throw new IllegalArgumentException(s"EContaminationAggregator given label $label" +
        s" but requires label < numClasses (= ${numClasses}).")
    }
    if (label < 0) {
      throw new IllegalArgumentException(s"EContaminationAggregator given label $label" +
        s"but requires label is non-negative.")
    }
    
    // Sumamos el conteo ponderado para la clase correspondiente
    allStats(offset + label.toInt) += numSamples * sampleWeight
    
    // En la última posición guardamos el rawCount total
    allStats(offset + statsSize - 1) += numSamples
  }

  def getCalculator(allStats: Array[Double], offset: Int): EContaminationCalculator = {
    new EContaminationCalculator(
      allStats.slice(offset, offset + statsSize - 1),
      allStats(offset + statsSize - 1).toLong
    )
  }
}

/**
 * Stores statistics for one (node, feature, bin) for calculating impurity.
 */
private[spark] class EContaminationCalculator(stats: Array[Double], var rawCount: Long)
  extends ImpurityCalculator(stats) {

  def copy: EContaminationCalculator = new EContaminationCalculator(stats.clone(), rawCount)

  def calculate(): Double = EContamination.calculate(stats, stats.sum)

  def count: Double = stats.sum

  def predict: Double = if (count == 0) {
    0
  } else {
    indexOfLargestArrayElement(stats)
  }

  override def prob(label: Double): Double = {
    val lbl = label.toInt
    require(lbl < stats.length,
      s"EContaminationCalculator.prob given invalid label: $lbl (should be < ${stats.length}")
    require(lbl >= 0, "EContaminationCalculator does not support negative labels")
    val cnt = count
    if (cnt == 0) {
      0
    } else {
      stats(lbl) / cnt
    }
  }

  override def toString: String = s"EContaminationCalculator(stats = ${stats.mkString("[", ", ", "]")})"
}
