/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.optim

import java.util

import com.intel.analytics.bigdl.dataset.{DataSet, LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.image.{BGRImgToBatch, LabeledBGRImage}
import com.intel.analytics.bigdl.example.recommendation.NeuralCFV2
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.mkldnn.HeapData
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL}
import com.intel.analytics.bigdl.tensor.{DnnStorage, Storage, Tensor}
import com.intel.analytics.bigdl.utils.{Engine, File, RandomGenerator, T}
import com.intel.analytics.bigdl.visualization.TrainSummary
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

object NCFDataSet extends LocalDataSet[MiniBatch[Float]] {
  val totalSize = 10
  var isCrossEntropy = true

  def ncfDataSet: LocalDataSet[MiniBatch[Float]] = {
    isCrossEntropy = false
    NCFDataSet
  }

  private val feature1 = Tensor[Float](
    Storage[Float](
      Array[Float](
        1, 2,
        3, 4,
        1, 4,
        2, 3,
        3, 5,
        4, 1,
        6, 3,
        2, 6
      )
    ),
    storageOffset = 1,
    size = Array(8, 2)
  )

  private val feature2 = Tensor[Float](
    Storage[Float](
      Array[Float](
        2, 2,
        3, 3,
        2, 3,
        3, 1,
        3, 2,
        1, 3,
        1, 2,
        1, 3
      )
    ),
    storageOffset = 1,
    size = Array(8, 2)
  )
  private val labelMSE = Tensor[Float](
    Storage[Float](
      Array[Float](
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1
      )
    ),
    storageOffset = 1,
    size = Array(8, 1)
  )

  private val labelCrossEntropy = Tensor[Float](
    Storage[Float](
      Array[Float](
        1,
        2,
        1,
        2
      )
    ),
    storageOffset = 1,
    size = Array(4)
  )

  override def size(): Long = totalSize

  override def shuffle(): Unit = {}

  override def data(train : Boolean): Iterator[MiniBatch[Float]] = {
    new Iterator[MiniBatch[Float]] {
      var i = 0

      override def hasNext: Boolean = train || i < totalSize

      override def next(): MiniBatch[Float] = {
        i += 1
        if (i % 2 == 0) {
          MiniBatch(feature1, if (isCrossEntropy) labelCrossEntropy else labelMSE)
        } else {
          MiniBatch(feature2, if (isCrossEntropy) labelCrossEntropy else labelMSE)
        }
      }
    }
  }
}

class NCFOptimizerSpec extends FlatSpec with Matchers with BeforeAndAfter{
  import NCFDataSet._

  private val nodeNumber = 1
  private val coreNumber = 1

  before {
    System.setProperty("bigdl.localMode", "true")
    Engine.init(nodeNumber, coreNumber, false)
  }

  after {
    System.clearProperty("bigdl.localMode")
  }

  "ncf optimizer" should "works the same with localOptimizer" in {
    val userCount = 6
    val itemCount = 7
    val mlpEmbed = 4
    val mfEmbed = 2

    RandomGenerator.RNG.setSeed(1)

    val ncf = NeuralCFV2[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = 1,
      userEmbed = mlpEmbed,
      itemEmbed = mlpEmbed,
      hiddenLayers = Array(256, 128, 64),
      mfEmbed = mfEmbed).buildModel()

    val itera = 2

    val optimMethod = Map(
      "embeddings" -> new EmbeddingAdam2[Float](
        userCount = userCount, itemCount = itemCount,
        embedding1 = mfEmbed, embedding2 = mlpEmbed),
      "linears" -> new Adam[Float]())

    val ncfOptimizer = new NCFOptimizer2[Float](ncf.cloneModule(),
      ncfDataSet, BCECriterion[Float]()).setOptimMethods(optimMethod)
      .setEndWhen(Trigger.severalIteration(itera))

    val ncfResult = ncfOptimizer.optimize()

    val optimMethod2 = Map(
      "embeddings" -> new Adam[Float](),
      "linears" -> new Adam[Float]())
    val localOptimizer = new NCFOptimizer[Float](ncf.cloneModule(),
      ncfDataSet, BCECriterion[Float]()).setOptimMethods(optimMethod2)
      .setEndWhen(Trigger.severalIteration(itera))

    val localResult = localOptimizer.optimize()

    val buffer10 = optimMethod("embeddings").state("buffer10").asInstanceOf[Tensor[Float]]


    val a = ncfResult.getParameters()._1
    val b = localResult.getParameters()._1

    ncfResult.getParameters()._1 should be (localResult.getParameters()._1)
  }

  val url = "/tmp" // getClass.getClassLoader.getResource("pytorch").toString()
  "ncf model compare with pytorch" should "have the same result" in {
    import scala.collection.JavaConverters._
    val pyBigDL = new PythonBigDL[Float]()

    val i1pytorchW = File.load[util.HashMap[String, JTensor]](url + "/ncf_i1_weight.obj")
      .asScala.map(v => (v._1, pyBigDL.toTensor(v._2)))
    val i2pytorchW = File.load[util.HashMap[String, JTensor]](url + "/ncf_i2_weight.obj")
      .asScala.map(v => (v._1, pyBigDL.toTensor(v._2)))
    val i3pytorchW = File.load[util.HashMap[String, JTensor]](url + "/ncf_i3_weight.obj")
      .asScala.map(v => (v._1, pyBigDL.toTensor(v._2)))
    val userCount = 1384
    val itemCount = 267
    val hiddenLayers = Array(256, 128, 64)
    val mfEmbed = 64
    val numClasses = 1
    val userEmbed = 128
    val itemEmbed = 128
    val ncfModel = NeuralCFV2[Float](
      userCount,
      itemCount,
      numClasses,
      userEmbed,
      itemEmbed,
      hiddenLayers,
      mfEmbed = mfEmbed
    ).ncfModel
    val criterion = BCECriterion[Float]()
    val embeddingNames = Array("mfUserEmbedding", "mfItemEmbedding",
      "mlpUserEmbedding", "mlpItemEmbedding")
    val fcNames = Array("fc256->256", "fc256->128",
      "fc128->64", "fc128->1")
    val adam = new ParallelAdam[Float](learningRate = 0.0005)

    embeddingNames.foreach{name =>
      ncfModel(name).get.setWeightsBias(Array(i1pytorchW(s"${name}_weight")))
    }
    fcNames.foreach{name =>
      ncfModel(name).get.setWeightsBias(Array(
        i1pytorchW(s"${name}_weight"), i1pytorchW(s"${name}_bias")))
    }
    val input1 = Tensor[Float](T(T(userCount, itemCount), T(userCount - 1, itemCount - 1)))
    val target1 = Tensor[Float](T(T(0), T(1)))
    val output1 = ncfModel.forward(input1)
    val loss1 = criterion.forward(output1, target1)
    val gradOutput1 = criterion.backward(output1, target1)
    ncfModel.backward(input1, gradOutput1)

    val (ncfWeight, ncfGrad) = ncfModel.getParameters()
    adam.optimize(_ => (loss1, ncfGrad), ncfWeight)

    embeddingNames.foreach{name =>
      val p = ncfModel(name).get.parameters()
      p._1(0) should be (i2pytorchW(s"${name}_weight"))
    }

    fcNames.foreach{name =>
      val p = ncfModel(name).get.parameters()
      p._1(0) should be (i2pytorchW(s"${name}_weight"))
      p._1(1) should be (i2pytorchW(s"${name}_bias"))
    }

    ncfModel.zeroGradParameters()
    val input2 = Tensor[Float](T(T(userCount - 1, itemCount), T(userCount - 2, itemCount)))
    val target2 = Tensor[Float](T(T(0), T(1)))
    val output2 = ncfModel.forward(input2)
    val loss2 = criterion.forward(output2, target2)
    val gradOutput2 = criterion.backward(output2, target2)
    ncfModel.backward(input2, gradOutput2)
    adam.optimize(_ => (loss2, ncfGrad), ncfWeight)

    val ncfP = ncfModel.getParametersTable()

    embeddingNames.foreach{name =>
      val p = ncfModel(name).get.parameters()
      p._1(0) should be (i3pytorchW(s"${name}_weight"))
    }
    Array(
      "fc128->64", "fc128->1").foreach{name =>
      val p = ncfModel(name).get.parameters()
      p._1(0) should be (i3pytorchW(s"${name}_weight"))
      p._1(1) should be (i3pytorchW(s"${name}_bias"))
    }
    loss2 should be (0.6919203996658325f +- 1e8f)
  }

}

