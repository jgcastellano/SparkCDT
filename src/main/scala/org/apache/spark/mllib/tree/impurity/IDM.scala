package org.apache.spark.mllib.tree.impurity

import org.apache.spark.annotation.Since

/**
 * Class for calculating imprecise entropy during multiclass classification. Using IDM with s=1, see
 * J. Abellan, S. Moral
 * Upper entropy of credal sets. Applications to credal classification
 * Int. J. Approx. Reason., 39 (2 3) (2005), pp. 235 255
 */
@Since("1.0.0")
object IDM extends Impurity {

  //private val _log2 = scala.math.log(2)

  //private[cdt] def log2(x: Double) = scala.math.log(x) / _log2

  /**
   * This private method searches for an element in a ordered integer vector.
   * @param element Int to look for
   * @param vector Array[Int] in which we have to do the search
   * @param size effective size of the vector
   * @return true if the element is found, false otherwise
   */
  private[tree] def contains(element: Int, vector: Array[Int], size: Int): Boolean = {
    var i = 0

    while (i < size && vector(i) < element) i += 1

    i < size && vector(i) == element
  }

  /**
   * information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  @Since("1.1.0")
  /*override*/ def calculateVER0(counts: Array[Double], totalCount: Double): Double = {

    if (totalCount == 0) {
      return 0
    }

    // Used to store the index values of the elements with min freq
    var minIndexes = new Array[Int](counts.length)
    val s: Double = 1.0 // Hyperparameter
    val total = totalCount + s
    var minValue = total // value of min freq, initialized with a big value
    val numClasses = counts.length
    var impurity = 0.0
    var j = 0 // minIndexes size

    // Search for the min freq indexes
    for(i <- 0 until numClasses) {
      if ((counts(i)-minValue).abs <= 0.01) {
        minIndexes(j) = i
        j += 1
        // DEBUG println(counts(i)+" is the same min, i="+i)
      } else if (counts(i)<minValue) {
        j = 1 // new min freq found
        minValue = counts(i)
        minIndexes(0) = i
        // DEBUG println(counts(i)+" is a new min, i="+i)
      } // DEBUG else println(counts(i)+" is greater than "+minValue+",i="+i)
    }

    /* DEBUG
     for(i <- 0 until minIndexes.length) {
     println(i+"'th element is: " + minIndexes(i));
     }
     for(i <- 0 until j) {
     println(i+"'th element is: " + minIndexes2(i));
     }
      DEBUG */

    var classIndex = 0
    val adjust = s / j

    while (classIndex < numClasses) {
      val classCount = counts(classIndex)
      // if (classCount != 0) { //it's not necessary here!
      var freq = 0.0
      if (minIndexes.indexOf(classIndex) != -1 ) {
        // P* case, the class belongs to the min freq classes, only s=1
        freq = ( classCount + adjust ) / total
      } else {
        // Normal P case
        freq = classCount / total
        // *DEBUG*/println("Class" + classIndex + "/" + numClasses +" P="+freq)
      }
      impurity -= freq * scala.math.log(freq) // log2(freq)
      // }
      classIndex += 1
    }
    impurity
  }

 
  /**
   * information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  @Since("1.1.0")
  /*override*/ def calculateVER3(counts: Array[Double], totalCount: Double): Double = {
    val impurityO=calculate(counts, totalCount)
    val impurityO2=calculateVER4(counts, totalCount)
    
    import java.util.concurrent.TimeUnit
    var startNanoTime = System.nanoTime()
    var impurity = 0.0
    for (i <- 1 to 100000) { 

    if (totalCount == 0) {
      if (impurityO!= 0) {
        println("ERROR: Impurity = "+0+" çImpurityOptimized="+impurityO)
      }
      return 0
    }

    val s: Double = 1.0 // Hyperparameter
    

    //Value with the minimum frequency
    val min=counts.min
    // Store the indexes of the elements with min freq
    val minIndexes = counts.zipWithIndex.filter(pair => pair._1 == min).map(pair => pair._2)
    val total = totalCount + s
    val adjust = s / minIndexes.length
    val numClasses = counts.length
    var classIndex = 0
    //var impurity = 0.0
    impurity = 0.0
    while (classIndex < numClasses) {
      val classCount = counts(classIndex)
      // if (classCount != 0) { //it's not necessary here!
      var freq = 0.0
      if (minIndexes.indexOf(classIndex) != -1 ) {
        // P* case, the class belongs to the min freq classes, only s=1
        freq = ( classCount + adjust ) / total
      } else {
        // Normal P case
        freq = classCount / total
        // *DEBUG*/println("Class" + classIndex + "/" + numClasses +" P="+freq)
      }
      impurity -= freq * scala.math.log(freq) // log2(freq)
      // }
      classIndex += 1
    }

     }
    var endNanoTime = System.nanoTime()
    val executionTime0 = TimeUnit.NANOSECONDS.toMillis(endNanoTime - startNanoTime)


    if (impurityO!= impurity) {
        println("ERROR: Impurity = "+impurity+" ImpurityOptimized1="+impurityO)
    }
    if (impurityO!= impurity) {
        println("ERROR: Impurity = "+impurity+" ImpurityOptimized2="+impurityO)
    }
    import java.util.concurrent.TimeUnit
    startNanoTime = System.nanoTime()
    for (i <- 1 to 100000) { 
      val feo =calculate(counts, totalCount)
    }
    endNanoTime = System.nanoTime()
    val executionTime1 = TimeUnit.NANOSECONDS.toMillis(endNanoTime - startNanoTime)

    startNanoTime = System.nanoTime()
    for (i <- 1 to 100000) { 
      val feo =calculateVER4(counts, totalCount)
    }
    endNanoTime = System.nanoTime()
    val executionTime2 = TimeUnit.NANOSECONDS.toMillis(endNanoTime - startNanoTime)
    println ("OLD0:"+executionTime0)
    println ("OPt1:"+executionTime1)
    println ("Opt2:"+executionTime2+"\n")
    impurity
  }
/**
   * information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  @Since("1.1.0")
  /*override*/ def calculateVER2(counts: Array[Double], totalCount: Double): Double = {
    
    if (totalCount == 0) {
      return 0
    }

    val s: Double = 1.0 // Hyperparameter
    

    //Value with the minimum frequency
    val min=counts.min
    // Store the indexes of the elements with min freq
    val minIndexes = counts.zipWithIndex.filter(pair => pair._1 == min).map(pair => pair._2)
    val total = totalCount + s
    val adjust = s / minIndexes.length
    val numClasses = counts.length
    var classIndex = 0
    var impurity = 0.0
    while (classIndex < numClasses) {
      val classCount = counts(classIndex)
      // if (classCount != 0) { //it's not necessary here!
      var freq = 0.0
      if (minIndexes.indexOf(classIndex) != -1 ) {
        // P* case, the class belongs to the min freq classes, only s=1
        freq = ( classCount + adjust ) / total
      } else {
        // Normal P case
        freq = classCount / total
        // *DEBUG*/println("Class" + classIndex + "/" + numClasses +" P="+freq)
      }
      impurity -= freq * scala.math.log(freq) // log2(freq)
      // }
      classIndex += 1
    }
    impurity
  }





/*------------------------------------------------------------------------*/
/**
   * IDM based information calculation for multiclass classification. 
   *Hyperparameter s form IDM obtained from IDMParameter singleton.
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
 @Since("1.1.0")
 override def calculate(counts: Array[Double], totalCount: Double): Double = {
    if (totalCount == 0) {
      return 0
    }
    val s = IDMParameter.getS  // Hyperparameter
    
    if (s==1.0){
      //Specific case s=1.0
      calculateS1(counts,totalCount)
    } else {
      //General case
      val newCounts= calculateCountsS(counts,s)
      val total = totalCount + s
      var impreciseInfoGain = 0.0
      for (count <- newCounts) {
        var freq=count/total
        impreciseInfoGain -= freq * scala.math.log(freq)
      }
      impreciseInfoGain
    }
  }

/*------------------------------------------------------------------------*/
/**
   * IDM based information calculation for multiclass classification. 
   * Case for parameter s=1.
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
 @Since("1.1.0")
 def calculateS1(counts: Array[Double], totalCount: Double): Double = {
   
    // Store the indexes of the elements with min freq
    val (minIndexes, _ ) = counts.zipWithIndex.foldLeft((List.empty[Int], Double.MaxValue)) {
      case ((indexes, currentMin), (value, index)) =>
        if (value < currentMin) (List(index), value)
        else if (value == currentMin) (index :: indexes, currentMin)
        else (indexes, currentMin)
      }
    
    val s = 1.0 // Hyperparameter
    val total = totalCount + s
    val mass = s / minIndexes.length
    var impreciseInfoGain = 0.0
    var classIndex = 0

    //distribute the mass in the minimum frequency labels
    while (classIndex < counts.length) {
        val classCount = counts(classIndex)
        var freq = 0.0
        if (minIndexes.indexOf(classIndex) != -1 ) {
          // P* case, the class belongs to the min freq classes, only s=1
          freq = ( classCount + mass ) / total
        } else {
          // Normal P case
          freq = classCount / total
        }
        impreciseInfoGain -= freq * scala.math.log(freq) 
        classIndex += 1
    }
    impreciseInfoGain
  }
/*------------------------------------------------------------------------*/
/**
   * IDM based information calculation for multiclass classification. 
   * Case for parameter s=1.
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
 @Since("1.1.0")
 def calculateCountsS(counts: Array[Double], s: Double): Array[Double] = {
   
    // Store the indexes of the elements with min freq
    val (minIndexes, _ ) = counts.zipWithIndex.foldLeft((List.empty[Int], Double.MaxValue)) {
      case ((indexes, currentMin), (value, index)) =>
        if (value < currentMin) (List(index), value)
        else if (value == currentMin) (index :: indexes, currentMin)
        else (indexes, currentMin)
      }

    //compute the mass depending on s and the number of classes with min freq
    var mass = s
    if (s>minIndexes.length)
      mass=minIndexes.length
    var newCounts = Array.fill(counts.length)(0.0)//Array to store the frequencies adjusted

    //distribute the mass in the minimum frequency labels
    var classIndex=0
    while (classIndex < counts.length) {
        val classCount = counts(classIndex)
        if (minIndexes.indexOf(classIndex) != -1 ) {
          // P* case, the class belongs to the min freq classes, only s=1
          newCounts(classIndex) =  classCount + mass 
        } else {
          // Normal P case
          newCounts(classIndex)= classCount 
        }
        classIndex+=1
    }
    //Adjust the mass and check if there is any mass to redistribute
    mass=s-mass
    if (mass>0.001)
      calculateCountsS(newCounts,mass)
    else
      newCounts
  }




/*------------------------------------------------------------------------*/

@Since("1.1.0")
def calculateVER4(counts: Array[Double], totalCount: Double): Double = {
  if (totalCount == 0) {
    return 0
  }

  val s: Double = 1.0 // Hyperparameter

  // Find the minimum value and its indices in a single pass
  val minFreq = counts.min
  val minIndexes = counts.zipWithIndex.collect {
    case (value, index) if value == minFreq => index
  }

  val total = totalCount + s
  val mass = s / minIndexes.length
  val logTotal = scala.math.log(total)

  // Calculate the imprecise information gain
  counts.indices.foldLeft(0.0) { (impreciseInfoGain, classIndex) =>
    val classCount = counts(classIndex)
    val freq = if (minIndexes.contains(classIndex)) {
      (classCount + mass) / total
    } else {
      classCount / total
    }
    impreciseInfoGain - freq * (scala.math.log(freq) - logTotal)
  }
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
    throw new UnsupportedOperationException("IDM.calculate")

  /**
   * Get this impurity instance.
   * This is useful for passing impurity parameters to a Strategy in Java.
   */
  @Since("1.1.0")
  def instance: this.type = this

}



/**
 * Class for updating views of a vector of sufficient statistics,
 * in order to compute impurity from a sample.
 */
private[spark] class IDMAggregator(numClasses: Int)
  extends ImpurityAggregator(numClasses + 1) with Serializable {

  def update(
      allStats: Array[Double],
      offset: Int,
      label: Double,
      numSamples: Int,
      sampleWeight: Double): Unit = {
    if (label >= numClasses) {
      throw new IllegalArgumentException(s"IDMAggregator given label $label" +
        s" but requires label < numClasses (= ${numClasses}).")
    }
    if (label < 0) {
      throw new IllegalArgumentException(s"IDMAggregator given label $label" +
        s"but requires label is non-negative.")
    }
    
    // Sumamos el conteo ponderado para la clase correspondiente
    allStats(offset + label.toInt) += numSamples * sampleWeight
    
    // En la última posición (statsSize - 1) guardamos el rawCount total
    allStats(offset + statsSize - 1) += numSamples
  }

  /**
   * Get an [[IDMCalculator]] for calculating impurity.
   */
  def getCalculator(allStats: Array[Double], offset: Int): IDMCalculator = {
    // Extraemos las estadísticas de las clases y el rawCount del array global
    new IDMCalculator(
      allStats.slice(offset, offset + statsSize - 1),
      allStats(offset + statsSize - 1).toLong
    )
  }
}


/**
 * Stores statistics for one (node, feature, bin) for calculating impurity.
 * Unlike [[IDMAggregator]], this class stores its own data and is for a specific
 * (node, feature, bin).
 * @param stats  Array of sufficient statistics for a (node, feature, bin).
 */
private[spark] class IDMCalculator(stats: Array[Double], var rawCount: Long)
  extends ImpurityCalculator(stats) {

  /**
   * Make a deep copy of this [[IDMCalculator]].
   */
  def copy: IDMCalculator = new IDMCalculator(stats.clone(), rawCount)

  /**
   * Calculate the impurity from the stored sufficient statistics.
   * (CORREGIDO: Llamamos a IDM en lugar de ImpreciseEntropy)
   */
  def calculate(): Double = IDM.calculate(stats, stats.sum)

  /**
   * Weighted number of data points accounted for in the sufficient statistics.
   */
  def count: Double = stats.sum

  /**
   * Prediction which should be made based on the sufficient statistics.
   */
  def predict: Double = if (count == 0) {
    0
  } else {
    indexOfLargestArrayElement(stats)
  }

  /**
   * Probability of the label given by [[predict]].
   */
  override def prob(label: Double): Double = {
    val lbl = label.toInt
    require(lbl < stats.length,
      s"IDMCalculator.prob given invalid label: $lbl (should be < ${stats.length}")
    require(lbl >= 0, "IDM does not support negative labels")
    val cnt = count
    if (cnt == 0) {
      0
    } else {
      stats(lbl) / cnt
    }
  }

  override def toString: String = s"IDMCalculator(stats = ${stats.mkString("[", ", ", "]")})"
}
