package org.apache.spark.mllib.tree.impurity

import org.scalatest.funsuite.AnyFunSuite

class NPIANPIComparisonSuite extends AnyFunSuite {
  import ImpurityTestSupport._

  private val ManualCases = Seq(
    Array(1.0, 1.0), Array(8.0, 2.0), Array(2.0, 2.0, 2.0),
    Array(5.0, 2.0, 1.0, 0.0), Array(4.0, 2.0, 0.0), Array(0.0, 0.0, 1.0)
  )

  test("NPI-M and A-NPI-M produce identical singleton bounds by construction") {
    ManualCases.foreach { counts =>
      val n = counts.sum
      val (npiLower, npiUpper) = NPI.singletonBounds(counts, n)
      val (anpiLower, anpiUpper) = ANPI.singletonBounds(counts, n)
      counts.indices.foreach { category =>
        assert(math.abs(npiLower(category) - anpiLower(category)) <= NumericTolerance,
          s"counts=${counts.toSeq}, category=$category, lower NPI-M=${npiLower(category)}, " +
            s"lower A-NPI-M=${anpiLower(category)}, difference=${npiLower(category) - anpiLower(category)}")
        assert(math.abs(npiUpper(category) - anpiUpper(category)) <= NumericTolerance,
          s"counts=${counts.toSeq}, category=$category, upper NPI-M=${npiUpper(category)}, " +
            s"upper A-NPI-M=${anpiUpper(category)}, difference=${npiUpper(category) - anpiUpper(category)}")
      }
    }
  }

  test("exhaustive small integer counts preserve singleton-bound equality and model invariants") {
    var evaluated = 0
    var distributionsDiffer = 0
    var entropyDiffers = 0
    var maxBoundDifference = 0.0
    var maxEntropyDifference = 0.0
    var firstDistributionCounterexample = "none"

    smallCountVectors.foreach { counts =>
      val total = counts.sum
      val (npiLower, npiUpper) = NPI.singletonBounds(counts, total)
      val (anpiLower, anpiUpper) = ANPI.singletonBounds(counts, total)

      counts.indices.foreach { category =>
        val lowerDifference = math.abs(npiLower(category) - anpiLower(category))
        val upperDifference = math.abs(npiUpper(category) - anpiUpper(category))
        maxBoundDifference = math.max(maxBoundDifference, math.max(lowerDifference, upperDifference))
        assert(lowerDifference <= NumericTolerance,
          s"counts=${counts.toSeq}, category=$category, lower NPI-M=${npiLower(category)}, " +
            s"lower A-NPI-M=${anpiLower(category)}, difference=$lowerDifference")
        assert(upperDifference <= NumericTolerance,
          s"counts=${counts.toSeq}, category=$category, upper NPI-M=${npiUpper(category)}, " +
            s"upper A-NPI-M=${anpiUpper(category)}, difference=$upperDifference")
      }

      val npiDistribution = NPI.maximumEntropyDistribution(counts, total)
      val anpiDistribution = ANPI.maximumEntropyDistribution(counts, total)
      assertDistribution(counts, npiDistribution, (npiLower, npiUpper), NumericTolerance)
      assertDistribution(counts, anpiDistribution, (anpiLower, anpiUpper), ApproximateMassTolerance)

      val distributionDifference = npiDistribution.zip(anpiDistribution)
        .map { case (npiP, anpiP) => math.abs(npiP - anpiP) }.max
      if (distributionDifference > NumericTolerance) {
        distributionsDiffer += 1
        if (firstDistributionCounterexample == "none") {
          firstDistributionCounterexample =
            s"counts=${counts.toSeq}, NPI-M=${npiDistribution.toSeq}, A-NPI-M=${anpiDistribution.toSeq}"
        }
      }

      val npiEntropy = NPI.calculate(counts, total)
      val anpiEntropy = ANPI.calculate(counts, total)
      val entropyDifference = math.abs(npiEntropy - anpiEntropy)
      maxEntropyDifference = math.max(maxEntropyDifference, entropyDifference)
      if (entropyDifference > NumericTolerance) entropyDiffers += 1
      assert(npiEntropy.isFinite && anpiEntropy.isFinite,
        s"Non-finite entropy for counts=${counts.toSeq}: NPI-M=$npiEntropy, A-NPI-M=$anpiEntropy")
      evaluated += 1
    }

    println(s"NPI/A-NPI exhaustive comparison: cases=$evaluated, " +
      s"maxSingletonBoundDifference=$maxBoundDifference, distributionsDiffer=$distributionsDiffer, " +
      s"entropiesDiffer=$entropyDiffers, maxEntropyDifference=$maxEntropyDifference, " +
      s"firstDistributionCounterexample=$firstDistributionCounterexample")
    assert(evaluated == 9320, s"Unexpected exhaustive vector count: $evaluated")
  }

  test("comparison is invariant to category permutation in representative cases") {
    ManualCases.foreach { counts =>
      val permutation = counts.indices.reverse.toArray
      val permutedCounts = permutation.map(counts(_))
      val total = counts.sum
      val npi = NPI.maximumEntropyDistribution(counts, total)
      val anpi = ANPI.maximumEntropyDistribution(counts, total)
      val npiPermuted = NPI.maximumEntropyDistribution(permutedCounts, total)
      val anpiPermuted = ANPI.maximumEntropyDistribution(permutedCounts, total)

      permutation.indices.foreach { i =>
        assert(math.abs(npiPermuted(i) - npi(permutation(i))) <= NumericTolerance)
        assert(math.abs(anpiPermuted(i) - anpi(permutation(i))) <= ApproximateMassTolerance)
      }
      assert(math.abs(NPI.calculate(counts, total) - NPI.calculate(permutedCounts, total)) <= NumericTolerance)
      assert(math.abs(ANPI.calculate(counts, total) - ANPI.calculate(permutedCounts, total)) <=
        ApproximateMassTolerance)
    }
  }

  test("candidate split rankings are compared empirically using each criterion's actual gains") {
    // Rows are class-count histograms for ordered feature values 0, 1, 2, 3;
    // each threshold creates a distinct binary candidate split.
    val featureValueClassCounts = Seq(
      Array(4.0, 0.0, 0.0),
      Array(2.0, 1.0, 0.0),
      Array(0.0, 0.0, 3.0),
      Array(1.0, 2.0, 0.0)
    )
    val parent = featureValueClassCounts.reduce((left, right) =>
      left.zip(right).map { case (a, b) => a + b })
    val total = parent.sum

    def gain(
        impurity: (Array[Double], Double) => Double,
        left: Array[Double]): Double = {
      val right = parent.zip(left).map { case (p, l) => p - l }
      impurity(parent, total) - left.sum / total * impurity(left, left.sum) -
        right.sum / total * impurity(right, right.sum)
    }

    val candidates = (0 until featureValueClassCounts.length - 1).map { threshold =>
      val left = featureValueClassCounts.take(threshold + 1)
        .reduce((a, b) => a.zip(b).map { case (x, y) => x + y })
      val candidateId = s"feature<=${threshold}"
      (candidateId, gain(NPI.calculate, left), gain(ANPI.calculate, left))
    }
    val npiRanking = candidates.sortBy(row => -row._2).map(_._1)
    val anpiRanking = candidates.sortBy(row => -row._3).map(_._1)

    println(s"Empirical candidate gains (NPI-M, A-NPI-M): ${candidates.mkString("; ")}")
    println(s"Empirical candidate rankings: NPI-M=$npiRanking, A-NPI-M=$anpiRanking")
    assert(candidates.forall { case (_, npiGain, anpiGain) =>
      npiGain.isFinite && anpiGain.isFinite
    })
    assert(npiRanking.length == candidates.length && anpiRanking.length == candidates.length)
  }
}
