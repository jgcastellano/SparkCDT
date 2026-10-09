package org.apache.spark.mllib.tree.impurity

import org.scalatest.funsuite.AnyFunSuite

class NPISuite extends AnyFunSuite {
  import ImpurityTestSupport._

  test("exact NPI-M handles hand-checkable and edge frequency vectors") {
    val cases = Seq(
      Array(1.0) -> 0.0,
      Array(1.0, 1.0) -> math.log(2.0),
      Array(9.0, 1.0) -> entropy(Array(0.8, 0.2)),
      Array(2.0, 2.0, 2.0) -> math.log(3.0),
      Array(4.0, 2.0, 0.0) -> entropy(Array(3.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0)),
      Array(0.0, 0.0, 1.0) -> math.log(3.0),
      Array(0.0, 0.0) -> 0.0
    )

    cases.foreach { case (counts, expectedEntropy) =>
      val total = counts.sum
      val actual = NPI.calculate(counts, total)
      assert(math.abs(actual - expectedEntropy) <= 1e-12,
        s"counts=${counts.toSeq}: expected entropy $expectedEntropy, got $actual")
      assert(actual.isFinite && actual >= 0.0)
      if (total > 0.0) {
        val distribution = NPI.maximumEntropyDistribution(counts, total)
        val bounds = NPI.singletonBounds(counts, total)
        assertDistribution(counts, distribution, bounds, NumericTolerance)
      }
    }
  }

  test("exact NPI-M distribution is deterministic, permutation invariant and symmetric") {
    val counts = Array(4.0, 2.0, 0.0, 1.0)
    val total = counts.sum
    val first = NPI.maximumEntropyDistribution(counts, total)
    val second = NPI.maximumEntropyDistribution(counts, total)
    assert(first.sameElements(second))

    val permutation = Array(2, 0, 3, 1)
    val permutedCounts = permutation.map(counts(_))
    val permuted = NPI.maximumEntropyDistribution(permutedCounts, total)
    permutation.indices.foreach(i =>
      assert(math.abs(permuted(i) - first(permutation(i))) <= NumericTolerance))

    val symmetricCounts = Array(3.0, 3.0, 3.0)
    val symmetric = NPI.maximumEntropyDistribution(symmetricCounts, symmetricCounts.sum)
    assert(symmetric.forall(p => math.abs(p - 1.0 / 3.0) <= NumericTolerance))
  }

  test("exact NPI-M bounds satisfy singleton interval and feasibility invariants") {
    Seq(Array(1.0), Array(4.0, 0.0), Array(3.0, 2.0, 1.0), Array(0.0, 4.0, 0.0, 2.0))
      .foreach { counts =>
        val (lower, upper) = NPI.singletonBounds(counts, counts.sum)
        assert(lower.length == counts.length && upper.length == counts.length)
        assertFiniteProbabilities(lower)
        assertFiniteProbabilities(upper)
        counts.indices.foreach(i => assert(lower(i) <= upper(i) + NumericTolerance))
        assert(lower.sum <= 1.0 + NumericTolerance)
        assert(upper.sum >= 1.0 - NumericTolerance)
      }
  }

  test("exact NPI-M remains finite for a large but representable count") {
    val counts = Array(1000000.0, 0.0)
    val distribution = NPI.maximumEntropyDistribution(counts, counts.sum)
    assertDistribution(counts, distribution, NPI.singletonBounds(counts, counts.sum), NumericTolerance)
    assert(NPI.calculate(counts, counts.sum).isFinite)
  }

  test("NPI rejects invalid positive-total count inputs and defines zero-total impurity") {
    assert(NPI.calculate(Array.emptyDoubleArray, 0.0) == 0.0)
    assert(NPI.calculate(Array(0.0, 0.0), 0.0) == 0.0)
    intercept[IllegalArgumentException](NPI.calculate(Array.emptyDoubleArray, 1.0))
    intercept[IllegalArgumentException](NPI.calculate(Array(1.0, -1.0), 1.0))
    intercept[IllegalArgumentException](NPI.calculate(Array(1.5, 0.5), 2.0))
    intercept[IllegalArgumentException](NPI.calculate(Array(1.0, 1.0), 3.0))
    intercept[IllegalArgumentException](NPI.calculate(Array(Double.NaN), Double.NaN))
    intercept[IllegalArgumentException](NPI.calculate(Array(Int.MaxValue.toDouble + 1.0),
      Int.MaxValue.toDouble + 1.0))
  }

  test("NPI regression calculation is unsupported") {
    intercept[UnsupportedOperationException](NPI.calculate(1.0, 1.0, 1.0))
  }
}
