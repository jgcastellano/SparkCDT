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
package org.apache.spark.mllib.tree.configuration

import org.apache.spark.annotation.Since
import org.apache.spark.mllib.tree.impurity._

/**
 * Utility to create training configurations compatible with
 * credal impurities.
 */
object CredalStrategy {

  /**
  * Creates a `Strategy` instance configured for credal classification.
  * @param impurity Impurity object (IDM, NPI, ANPI or EContamination).
  * @param maxDepth Maximum tree depth.
  * @param numClasses Number of classes for the problem.
  * @param maxBins Maximum number of bins for discretization.
  * @param categoricalFeaturesInfo Map of categorical features.
  * @return A `Strategy` instance ready to be used by CredalRandomForest.
   */
  def createClassificationStrategy(
      impurity: Impurity,
      maxDepth: Int,
      numClasses: Int,
      maxBins: Int,
      categoricalFeaturesInfo: Map[Int, Int]): Strategy = {
    
    val strategy = new Strategy(
      algo = Algo.Classification,
      impurity = impurity,
      maxDepth = maxDepth,
      numClasses = numClasses,
      maxBins = maxBins,
      categoricalFeaturesInfo = categoricalFeaturesInfo
    )
    
    // Additional validation: ensure regression is not attempted
    // with credal criteria based on imprecise entropy.
    require(strategy.algo == Algo.Classification,
      s"Credal impurities are only supported for Classification.")
      
    strategy
  }
}
