package org.apache.spark.mllib.tree.impurity

import org.scalatest.funsuite.AnyFunSuite

class ANPISuite extends AnyFunSuite {
  import ImpurityTestSupport._

  test("A-NPI-M handles balanced, imbalanced, singleton and unobserved categories") {
    val cases = Seq(
      Array(1.0) -> 0.0,
      Array(1.0, 1.0) -> math.log(2.0),
      Array(8.0, 2.0) -> entropy(Array(0.7, 0.3)),
      Array(2.0, 2.0, 2.0) -> math.log(3.0),
      Array(5.0, 2.0, 1.0, 0.0) -> entropy(Array(0.5, 0.125, 0.1875, 0.1875)),
      Array(0.0, 0.0, 1.0) -> math.log(3.0),
      Array(0.0, 0.0) -> 0.0
    )

    cases.foreach { case (counts, expectedEntropy) =>
      val actual = ANPI.calculate(counts, counts.sum)
      assert(actual.isFinite && actual >= 0.0)
      assert(math.abs(actual - expectedEntropy) <= NumericTolerance,
        s"counts=${counts.toSeq}: expected $expectedEntropy, got $actual")
      if (counts.sum > 0.0) {
        val distribution = ANPI.maximumEntropyDistribution(counts, counts.sum)
        val bounds = ANPI.singletonBounds(counts, counts.sum)
        assertDistribution(counts, distribution, bounds, ApproximateMassTolerance)
        assert(math.abs(actual - entropy(distribution)) <= NumericTolerance)
      }
    }
  }

  test("A-NPI-M singleton bounds are finite, feasible and match the specified formula") {
    Seq(Array(1.0), Array(4.0, 0.0), Array(3.0, 2.0, 1.0), Array(0.0, 4.0, 0.0, 2.0))
      .foreach { counts =>
        val total = counts.sum
        val (lower, upper) = ANPI.singletonBounds(counts, total)
        val expectedLower = counts.map(c => math.max(0.0, (c - 1.0) / total))
        val expectedUpper = counts.map(c => math.min(1.0, (c + 1.0) / total))
        assert(lower.zip(expectedLower).forall { case (a, b) => math.abs(a - b) <= NumericTolerance })
        assert(upper.zip(expectedUpper).forall { case (a, b) => math.abs(a - b) <= NumericTolerance })
        assertFiniteProbabilities(lower)
        assertFiniteProbabilities(upper)
        counts.indices.foreach(i => assert(lower(i) <= upper(i) + NumericTolerance))
        assert(lower.sum <= 1.0 + NumericTolerance)
        assert(upper.sum >= 1.0 - NumericTolerance)
      }
  }

  test("A-NPI-M remains finite for large counts within floating-point range") {
    val counts = Array(1000000.0, 0.0)
    val distribution = ANPI.maximumEntropyDistribution(counts, counts.sum)
    assertDistribution(counts, distribution, ANPI.singletonBounds(counts, counts.sum),
      ApproximateMassTolerance)
    assert(ANPI.calculate(counts, counts.sum).isFinite)
  }

  test("A-NPI-M distribution is repeatable, permutation invariant and symmetric on ties") {
    val counts = Array(5.0, 2.0, 1.0, 0.0)
    val total = counts.sum
    val first = ANPI.maximumEntropyDistribution(counts, total)
    assert(first.sameElements(ANPI.maximumEntropyDistribution(counts, total)))

    val permutation = Array(2, 0, 3, 1)
    val permuted = ANPI.maximumEntropyDistribution(permutation.map(counts(_)), total)
    permutation.indices.foreach(i =>
      assert(math.abs(permuted(i) - first(permutation(i))) <= ApproximateMassTolerance))

    val symmetricCounts = Array(2.0, 2.0, 2.0)
    val symmetric = ANPI.maximumEntropyDistribution(symmetricCounts, symmetricCounts.sum)
    assert(symmetric.max - symmetric.min <= ApproximateMassTolerance)
  }

  test("A-NPI-M returns zero impurity for zero total and rejects regression use") {
    assert(ANPI.calculate(Array.emptyDoubleArray, 0.0) == 0.0)
    assert(ANPI.calculate(Array(0.0, 0.0), 0.0) == 0.0)
    intercept[UnsupportedOperationException](ANPI.calculate(1.0, 1.0, 1.0))
  }
}
