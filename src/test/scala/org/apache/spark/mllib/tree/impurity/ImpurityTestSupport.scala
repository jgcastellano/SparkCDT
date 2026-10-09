package org.apache.spark.mllib.tree.impurity

private[impurity] object ImpurityTestSupport {
  val NumericTolerance: Double = 1e-12
  // ANPI's production recursion deliberately stops when the residual mass is < 0.001.
  val ApproximateMassTolerance: Double = 1e-3

  def entropy(probabilities: Array[Double]): Double =
    probabilities.filter(_ > 0.0).map(p => -p * math.log(p)).sum

  def assertFiniteProbabilities(probabilities: Array[Double]): Unit = {
    assert(probabilities.forall(_.isFinite), s"Non-finite probabilities: ${probabilities.toSeq}")
    assert(probabilities.forall(p => p >= -NumericTolerance && p <= 1.0 + NumericTolerance),
      s"Probability outside [0, 1]: ${probabilities.toSeq}")
  }

  def assertDistribution(
      counts: Array[Double],
      distribution: Array[Double],
      bounds: (Array[Double], Array[Double]),
      sumTolerance: Double): Unit = {
    assertFiniteProbabilities(distribution)
    assert(distribution.length == counts.length)
    assert(math.abs(distribution.sum - 1.0) <= sumTolerance,
      s"Distribution sums to ${distribution.sum}, expected 1 ± $sumTolerance: ${distribution.toSeq}")
    distribution.indices.foreach { i =>
      assert(distribution(i) + NumericTolerance >= bounds._1(i),
        s"p[$i]=${distribution(i)} below singleton lower bound ${bounds._1(i)}")
      assert(distribution(i) <= bounds._2(i) + NumericTolerance,
        s"p[$i]=${distribution(i)} above singleton upper bound ${bounds._2(i)}")
    }
  }

  def smallCountVectors: Iterator[Array[Double]] = {
    (2 to 5).iterator.flatMap { categories =>
      val all = (0 until categories).foldLeft(Vector(Vector.empty[Int])) { (prefixes, _) =>
        prefixes.flatMap(prefix => (0 to 5).map(value => prefix :+ value))
      }
      all.iterator.map(_.map(_.toDouble).toArray).filter(_.exists(_ > 0.0))
    }
  }
}
