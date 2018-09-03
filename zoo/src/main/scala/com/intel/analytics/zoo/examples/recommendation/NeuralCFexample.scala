/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.examples.recommendation

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, Sample, SampleToBatch, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.recommendation.{NeuralCF, UserItemFeature, Utils}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions._
import scopt.OptionParser
import com.intel.analytics.bigdl.example.recommendation.NeuralCFV2

case class NeuralCFParams(val inputDir: String = "./data/ml-1m",
                          val batchSize: Int = 2048,
                          val nEpochs: Int = 10,
                          val learningRate: Double = 1e-3,
                          val learningRateDecay: Double = 1e-6
                    )

case class Rating(userId: Int, itemId: Int, label: Int)

object NeuralCFexample {

  def main(args: Array[String]): Unit = {

    val defaultParams = NeuralCFParams()

    val parser = new OptionParser[NeuralCFParams]("NCF Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
      opt[Double]('l', "lRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x.toDouble))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: NeuralCFParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("NCFExample").set("spark.sql.crossJoin.enabled", "true")
    conf.set("spark.driver.maxResultSize", "2048")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (ratings, userCount, itemCount) = loadMl20mData(sqlContext, param.inputDir)

    val isImplicit = false
    val ncf = NeuralCFV2[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 5,
      userEmbed = 128,
      itemEmbed = 128,
      hiddenLayers = Array(256, 128, 64),
      includeMF = true,
      mfEmbed = 64
    )
    // val plength = ncf.parameters()._1.map(_.nElements()).reduce(_+_)
    // println(s"parameter length: $plengt")

    val pairFeatureRdds: RDD[UserItemFeature[Float]] =
      assemblyFeature(isImplicit, ratings, userCount, itemCount)

    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      pairFeatureRdds.randomSplit(Array(0.8, 0.2))
    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
    val trainData = DataSet.array(trainRdds.collect()) -> SampleToMiniBatch(param.batchSize)
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)

    val optimizer = new NCFOptimizer(
      model = ncf,
      dataset = trainData.toLocal(),
      criterion = ClassNLLCriterion[Float]())

    val optimMethod = new ParallelAdam[Float](
      learningRate = param.learningRate,
      learningRateDecay = param.learningRateDecay)

    optimizer
      .setOptimMethod(optimMethod)
      .setEndWhen(Trigger.maxEpoch(param.nEpochs))
      .optimize()

    val results = ncf.predict(validationRdds)
    results.take(5).foreach(println)
    val resultsClass = ncf.predictClass(validationRdds)
    resultsClass.take(5).foreach(println)

    /*val userItemPairPrediction = ncf.predictUserItemPair(validationpairFeatureRdds)

    userItemPairPrediction.take(5).foreach(println)

    val userRecs = ncf.recommendForUser(validationpairFeatureRdds, 3)
    val itemRecs = ncf.recommendForItem(validationpairFeatureRdds, 3)

    userRecs.take(10).foreach(println)
    itemRecs.take(10).foreach(println)*/
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String): (DataFrame, Int, Int) = {
    import sqlContext.implicits._
    val ratings = sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::").map(n => n.toInt)
        Rating(line(0), line(1), line(2))
      }).toDF()

    val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
    val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))

    (ratings, userCount, itemCount)
  }

  def loadMl20mData(sqlContext: SQLContext, dataPath: String): (DataFrame, Int, Int) = {
    import sqlContext.implicits._
    val ratings = sqlContext.read
      .option("inferSchema", "true")
      .option("header", "true")
      .option("delimiter", ",")
      .csv(dataPath + "/ratings.csv")
      .toDF()
    println(ratings.schema)

    val minMaxRow = ratings.agg(max("userId")).collect()(0)
    val userCount = minMaxRow.getInt(0)

    val uniqueMovie = ratings.rdd.map(_.getAs[Int]("movieId")).distinct().collect()
    val mapping = uniqueMovie.zip(1 to uniqueMovie.length).toMap

    val bcMovieMapping = sqlContext.sparkContext.broadcast(mapping)

    val mappingUdf = udf((itemId: Int) => {
      bcMovieMapping.value(itemId)
    })
    val mappedItemID = mappingUdf.apply(col("movieId"))
    val mappedRating = ratings//.drop(col("itemId"))
      .withColumn("movieId", mappedItemID)
    mappedRating.show()


    (mappedRating, userCount, uniqueMovie.length)
  }

  def assemblyFeature(isImplicit: Boolean = false,
                      indexed: DataFrame,
                      userCount: Int,
                      itemCount: Int): RDD[UserItemFeature[Float]] = {

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(indexed)
      negativeDF.unionAll(indexed.withColumn("label", lit(2)))
    }
    else indexed

    val rddOfSample: RDD[UserItemFeature[Float]] = unioned
      .select("userId", "movieId", "rating")
      .rdd.map(row => {
      val uid = row.getAs[Int](0)
      val iid = row.getAs[Int](1)

      val label = row.getAs[Double](2).ceil.toInt
      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

      UserItemFeature(uid, iid, Sample(feature, Tensor[Float](T(label))))
    })
    rddOfSample
  }

}
