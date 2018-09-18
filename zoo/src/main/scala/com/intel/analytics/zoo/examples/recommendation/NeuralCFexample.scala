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

import java.util

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.example.recommendation.NeuralCFV2
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.{BCECriterion, ClassNLLCriterion}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, File, T}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.recommendation.{NeuralCF, UserItemFeature, Utils}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions._
import scopt.OptionParser

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.reflect.ClassTag
import scala.util.Random

case class NeuralCFParams(val inputDir: String = "./data/ml-1m",
                          val dataset: String = "ml-1m",
                          val batchSize: Int = 256,
                          val nEpochs: Int = 2,
                          val learningRate: Double = 1e-3,
                          val learningRateDecay: Double = 0.0,
                          val trainNegtiveNum: Int = 4,
                          val valNegtiveNum: Int = 100,
                          val layers: String = "64,32,16,8",
                          val numFactors: Int = 8
                    )

case class Rating(userId: Int, itemId: Int, label: Int, timestamp: Int, train: Boolean)

object NeuralCFexample {

  def main(args: Array[String]): Unit = {

    val defaultParams = NeuralCFParams()

    // run with ml-20m, please use
    val parser = new OptionParser[NeuralCFParams]("NCF Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[String]("dataset")
        .text(s"dataset, ml-20m or ml-1m, default is ml-1m")
        .action((x, c) => c.copy(dataset = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
      opt[Double]('l', "lRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x))
      opt[Double]("lrd")
        .text("learning rate decay")
        .action((x, c) => c.copy(learningRateDecay = x))
      opt[Int]("trainNeg")
        .text("The Number of negative instances to pair with a positive train instance.")
        .action((x, c) => c.copy(trainNegtiveNum = x))
      opt[Int]("valNeg")
        .text("The Number of negative instances to pair with a positive validation instance.")
        .action((x, c) => c.copy(valNegtiveNum = x))
      opt[String]("layers")
        .text("The sizes of hidden layers for MLP. Default is 64,32,16,8")
        .action((x, c) => c.copy(layers = x))
      opt[Int]("numFactors")
        .text("The Embedding size of MF model.")
        .action((x, c) => c.copy(numFactors = x))
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
      .set("spark.driver.maxResultSize", "2048")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)


    val (ratings, userCount, itemCount, itemMapping) =
      loadPublicData(sqlContext, param.inputDir, param.dataset)
    println(s"${userCount} ${itemCount}")
    val hiddenLayers = param.layers.split(",").map(_.toInt)

    val isImplicit = false
    val ncf = NeuralCFV2[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 1,
      userEmbed = hiddenLayers(0) / 2,
      itemEmbed = hiddenLayers(0) / 2,
      hiddenLayers = hiddenLayers.slice(1, hiddenLayers.length),
      mfEmbed = param.numFactors)

    // load pytorch weight
    val pyBigDL = new PythonBigDL[Float]()
    val pytorchW = com.intel.analytics.bigdl.utils.File
      .load[util.HashMap[String, JTensor]]("pytorch_weight.obj")
//      .load[util.HashMap[String, JTensor]]("/home/xin/ncf/i7_weight.obj")
      .asScala.map(v => (v._1, pyBigDL.toTensor(v._2)))
    val embeddingNames = Array("mfUserEmbedding", "mfItemEmbedding",
      "mlpUserEmbedding", "mlpItemEmbedding")
    val fcNames = Array("fc256->256", "fc256->128",
      "fc128->64", "fc128->1")
    embeddingNames.foreach{name =>
      ncf.ncfModel(name).get.setWeightsBias(Array(pytorchW(s"${name}_weight")))
    }
    fcNames.foreach{name =>
      ncf.ncfModel(name).get.setWeightsBias(Array(
        pytorchW(s"${name}_weight"), pytorchW(s"${name}_bias")))
    }

    println(ncf)

    println(s"parameter length: ${ncf.parameters()._1.map(_.nElement()).sum}")


//    val trainData = sc.textFile("/tmp/ncf_recommendation_buffer/")

    println("local from local")

    val optimMethod = Map(
      "linears" -> new ParallelAdam[Float](
        learningRate = param.learningRate,
        learningRateDecay = param.learningRateDecay),
      "embeddings" ->
        new EmbeddingAdam2[Float](
        learningRate = param.learningRate,
        learningRateDecay = param.learningRateDecay,
        userCount = userCount, itemCount = itemCount,
        embedding1 = param.numFactors, embedding2 = hiddenLayers(0) / 2
      ))
    println(s"${param.learningRate}, ${param.learningRateDecay}")

    val validateBatchSize = optimMethod("linears").asInstanceOf[ParallelAdam[Float]].parallelNum

    val trainDataset = (DataSet.array[MiniBatch[Float]](loadPytorchTrain("0.txt", param.batchSize))).toLocal()
    val valDataset = (DataSet.array[Sample[Float]](loadPytorchTest("test-ratings.csv",
      "test-negative.csv")) -> SampleToMiniBatch[Float](validateBatchSize)).toLocal()
//    val result = ncf.evaluate(valDataset, Array(new HitRate(10, 999), new Ndcg[Float](10, 999)))

//    ratings.cache()
//    val (trainDataFrame, valDataFrame) = generateTrainValData(ratings, userCount, itemCount,
//      trainNegNum = param.trainNegtiveNum, valNegNum = param.valNegtiveNum)
//    val trainpairFeatureRdds =
//      assemblyFeature(isImplicit, trainDataFrame, userCount, itemCount)
//    val validationpairFeatureRdds =
//      assemblyValFeature(isImplicit, valDataFrame, userCount, itemCount, param.valNegtiveNum)
//    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
//    val validationRdds = validationpairFeatureRdds.map(x => x.sample).cache()
//    println(s"Train set ${trainRdds.count()} records")
//    println(s"Val set ${validationRdds.count()} records")
//    val valDataset = DataSet.array(validationRdds.collect()) -> SampleToMiniBatch(validateBatchSize)
//    val trainDataset = (DataSet.array[Sample[Float]](trainRdds.collect()) ->
//      SampleToMiniBatch(param.batchSize)).toLocal()
//    trainDataset.shuffle()

    val optimizer = new NCFOptimizer2[Float](ncf,
      trainDataset, BCECriterion[Float]())

//    val optimizer = Optimizer(
//      model = ncf,
//      sampleRDD = trainRdds,
//      criterion = BCECriterion[Float](),
//      batchSize = param.batchSize)

//    val optimMethod = new Adam[Float](
//      learningRate = param.learningRate,
//      learningRateDecay = param.learningRateDecay)

//    val endTrigger = if (param.iteration != 0) {
//      Trigger.maxIteration(param.iteration)
//    } else {
//      Trigger.maxEpoch(param.nEpochs)
//    }

    optimizer
      .setOptimMethods(optimMethod)
        .setValidation(Trigger.everyEpoch, valDataset,
          Array(new HitRate[Float](negNum = param.valNegtiveNum),
          new Ndcg[Float](negNum = param.valNegtiveNum)))
//      .setValidation(Trigger.everyEpoch, validationRdds, Array(new HitRate[Float](),
//      new Ndcg[Float]()), 4)
    val endTrigger = Trigger.maxEpoch(1)
    optimizer
      .setEndWhen(endTrigger)
      .optimize()
    var e = 2
    while(e <= param.nEpochs) {
      println(s"Starting epoch $e/${param.nEpochs}")
      val endTrigger = Trigger.maxEpoch(e)
      val newTrainDataset = (DataSet.array[MiniBatch[Float]](
        loadPytorchTrain(s"${e - 1}.txt", param.batchSize))).toLocal()

      optimizer
        .setTrainData(newTrainDataset)
        .setEndWhen(endTrigger)
        .optimize()

      e += 1
    }
  }

  def loadPytorchTest(posFile: String, negFile: String): Array[Sample[Float]] = {
    val testSet = new ArrayBuffer[Sample[Float]]()
    val positives = Source.fromFile(posFile).getLines()
    val negatives = Source.fromFile(negFile).getLines()
    while(positives.hasNext && negatives.hasNext) {
      val pos = positives.next().split("\t")
      val userId = pos(0).toFloat
      val posItem = pos(1).toFloat
      val neg = negatives.next().split("\t").map(_.toFloat)
      val distinctNegs = neg.distinct
      val testFeature = Tensor[Float](1 + neg.size, 2)
      testFeature.select(2, 1).fill(userId + 1)
      val testLabel = Tensor[Float](1 + neg.size).fill(0)
      var i = 1
      while(i <= distinctNegs.size) {
        testFeature.setValue(i, 2, distinctNegs(i - 1) + 1)
        i += 1
      }
      testFeature.setValue(i, 2, posItem + 1)
      testLabel.setValue(i, 1)
      testFeature.narrow(1, i + 1, neg.size - distinctNegs.size).fill(1)
      testLabel.narrow(1, i + 1, neg.size - distinctNegs.size).fill(-1)

      testSet.append(Sample(testFeature, testLabel))
    }

    testSet.toArray
  }

  def loadPytorchTrain(path: String, batchSize: Int = 2048): Array[MiniBatch[Float]] = {
    val file = Source.fromFile(path)
    val lines = Source.fromFile(path).getLines()
    val miniBatches = new ArrayBuffer[MiniBatch[Float]]()
    while(lines.hasNext) {
      var i = 1
      val input = Tensor(batchSize, 2)
      val target  = Tensor(batchSize, 1)
      while(i <= batchSize && lines.hasNext) {
        val line = lines.next().split(",").map(_.toFloat)
        input.setValue(i, 1, line(0) + 1)
        input.setValue(i, 2, line(1) + 1)
        target.setValue(i, 1, line(2))
        i += 1
      }
      val miniBatch = if (i <= batchSize) {
        input.narrow(1, i, batchSize + 1 - i).copy(
          miniBatches(0).getInput().toTensor.narrow(1, 1, batchSize + 1 - i))
        target.narrow(1, i, batchSize + 1 - i).copy(
          miniBatches(0).getTarget().toTensor.narrow(1, 1, batchSize + 1 - i))
        miniBatches.append(MiniBatch(input, target))
      } else {
        miniBatches.append(MiniBatch(input, target))
      }
    }
    miniBatches.toArray
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String,
                     dataset: String): (DataFrame, Int, Int, Map[Int, Int]) = {
    import sqlContext.implicits._
    val ratings = dataset match {
      case "ml-1m" =>
        loadMl1mData(sqlContext, dataPath)
      case "ml-20m" =>
        loadMl20mData(sqlContext, dataPath)
      case _ =>
        throw new IllegalArgumentException(s"Only support dataset ml-1m and ml-20m, but got ${dataset}")
    }

    val minMaxRow = ratings.agg(max("userId")).collect()(0)
    val userCount = minMaxRow.getInt(0)

    val uniqueMovie = ratings.rdd.map(_.getAs[Int]("itemId")).distinct().collect().sortWith(_ < _)
    val mapping = uniqueMovie.zip(1 to uniqueMovie.length).toMap

    val bcMovieMapping = sqlContext.sparkContext.broadcast(mapping)

    val mappingUdf = udf((itemId: Int) => {
     bcMovieMapping.value(itemId)
    })
    val mappedItemID = mappingUdf.apply(col("itemId"))
    val mappedRating = ratings//.drop(col("itemId"))
      .withColumn("itemId", mappedItemID)
    mappedRating.show()


    (mappedRating, userCount, uniqueMovie.length, mapping)
  }

  def loadMl1mData(sqlContext: SQLContext, dataPath: String): DataFrame = {
    import sqlContext.implicits._
    sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::")
        Rating(line(0).toInt, line(1).toInt, 1, line(3).toInt, true)
      }).toDF()
  }

  def loadMl20mData(sqlContext: SQLContext, dataPath: String): DataFrame = {
    val ratings = sqlContext.read
      .option("inferSchema", "true")
      .option("header", "true")
      .option("delimiter", ",")
      .csv(dataPath + "/ratings.csv")
      .toDF()
    println(ratings.schema)
    val result = ratings.withColumnRenamed("movieId", "itemId").withColumn("rating", lit(1))
      .withColumnRenamed("rating", "label").withColumn("train", lit(true))
    println(result.schema)
    result
  }

  def generateTrainValData(rating: DataFrame, userCount: Int, itemCount: Int,
                           trainNegNum: Int = 4, valNegNum: Int = 100): (DataFrame, DataFrame) = {
    val maxTimeStep = rating.groupBy("userId").max("timestamp").collect().map(r => (r.getInt(0), r.getInt(1))).toMap
    val bcT = rating.sparkSession.sparkContext.broadcast(maxTimeStep)
    val evalPos = rating.filter(r => bcT.value.apply(r.getInt(0)) == r.getInt(3)).dropDuplicates("userId")
      .collect().toSet
    val bcEval = rating.sparkSession.sparkContext.broadcast(evalPos)

    val negDataFrame = rating.sqlContext.createDataFrame(
      rating.rdd.groupBy(_.getAs[Int]("userId")).flatMap{v =>
        val userId = v._1
        val items = scala.collection.mutable.Set(v._2.map(_.getAs[Int]("itemId")).toArray: _*)
        val itemNumOfUser = items.size
        val gen = new Random(userId + System.currentTimeMillis())
        var i = 0
        val totalNegNum = trainNegNum * (itemNumOfUser - 1) + valNegNum

        val negs = new Array[Rating](totalNegNum)
        // gen negative sample to validation
        while(i < valNegNum) {
          val negItem = Random.nextInt(itemCount) + 1
          if (!items.contains(negItem)) {
            negs(i) = Rating(userId, negItem, 0, 0, false)
            i += 1
          }
        }

        // gen negative sample for train
        while(i < totalNegNum) {
          val negItem = gen.nextInt(itemCount) + 1
          if (!items.contains(negItem)) {
            negs(i) = Rating(userId, negItem, 0, 0, true)
            i += 1
          }
        }
        negs.toIterator
    })
//    println("neg train" + negDataFrame.filter(_.getAs[Boolean]("train")).count())
//    println("neg eval" + negDataFrame.filter(!_.getAs[Boolean]("train")).count())

    (negDataFrame.filter(_.getAs[Boolean]("train"))
      .union(rating.filter(r => !bcEval.value.contains(r))),
      negDataFrame.filter(!_.getAs[Boolean]("train"))
        .union(rating.filter(r => bcEval.value.contains(r))))

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
      .select("userId", "itemId", "label")
      .rdd.map(row => {
      val uid = row.getAs[Int](0)
      val iid = row.getAs[Int](1)

      val label = row.getAs[Int](2)
      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

      UserItemFeature(uid, iid, Sample(feature, Tensor[Float](T(label))))
    })
    rddOfSample
  }

  def assemblyValFeature(isImplicit: Boolean = false,
                      indexed: DataFrame,
                      userCount: Int,
                      itemCount: Int,
                      negNum: Int = 100): RDD[UserItemFeature[Float]] = {

    val rddOfSample: RDD[UserItemFeature[Float]] = indexed
      .select("userId", "itemId", "label")
      .rdd.groupBy(_.getAs[Int]("userId")).map(data => {
      val totalNum = 1 + negNum
      val uid = data._1
      val rows = data._2.toIterator
      val feature = Tensor(totalNum, 2).fill(uid)
      val label = Tensor(totalNum)

      var i = 1
      while(rows.hasNext) {
        val current = rows.next()
        val iid = current.getAs[Int]("itemId")
        val l = current.getAs[Int]("label")
        feature.setValue(i, 2, iid)
        label.setValue(i, l)

        i += 1
      }
      require(i == totalNum + 1)

      UserItemFeature(uid, -1, Sample(feature, label))
    })
    rddOfSample
  }

}

class HitRate[T: ClassTag](k: Int = 10, negNum: Int = 100)(
    implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {
  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    val o = output.toTensor[T].resize(1 + negNum)
    val t = target.toTensor[T].resize(1 + negNum)
    var exceptedTarget = 0
    var i = 1
    while(i <= t.nElement()) {
      if (t.valueAt(i) == 1) {
        exceptedTarget = i
      }
      i += 1
    }
    require(exceptedTarget != 0, s"No positive sample")

    val hr = hitRate(exceptedTarget,
      o.narrow(1, 1, exceptedTarget), k)

    new LossResult(hr, 1)
  }

  def hitRate(index: Int, o: Tensor[T], k: Int): Float = {
    var topK = 1
    var i = 1
    val precision = ev.toType[Float](o.valueAt(index))
    while (i < o.nElement() && topK <= k) {
      if (ev.toType[Float](o.valueAt(i)) > precision) {
        topK += 1
      }
      i += 1
    }

    if(topK <= k) {
      1
    } else {
      0
    }
  }

  override def format(): String = "HitRate@10"
}

class Ndcg[T: ClassTag](k: Int = 10, negNum: Int = 100)(
    implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {
  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    val o = output.toTensor[T].resize(1 + negNum)
    val t = target.toTensor[T].resize(1 + negNum)
    var exceptedTarget = 0
    var i = 1
    while(i <= t.nElement()) {
      if (t.valueAt(i) == 1) {
        exceptedTarget = i
      }
      i += 1
    }
    require(exceptedTarget != 0, s"No positive sample")

    val n = ndcg(exceptedTarget, o.narrow(1, 1, exceptedTarget), k)

    new LossResult(n, 1)
  }

  def ndcg(index: Int, o: Tensor[T], k: Int): Float = {
    var ranking = 1
    var i = 1
    val precision = ev.toType[Float](o.valueAt(index))
    while (i < o.nElement() && ranking <= k) {
      if (ev.toType[Float](o.valueAt(i)) > precision) {
        ranking += 1
      }
      i += 1
    }

    if(ranking <= k) {
      (math.log(2) / math.log(ranking + 1)).toFloat
    } else {
      0
    }
  }

  override def format(): String = "NDCG"
}
