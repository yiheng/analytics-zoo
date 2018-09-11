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

import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.mkl.hardware.Affinity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, Table}
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.math.{pow, sqrt}
import scala.reflect.ClassTag

class EmbeddingAdam2[@specialized(Float, Double) T: ClassTag](
  var learningRate: Double = 1e-3,
  var learningRateDecay: Double = 0.0,
  var beta1: Double = 0.9,
  var beta2: Double = 0.999,
  var eps: Double = 1e-8)(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {

  val parallelNum = Engine.coreNumber()

  val gradients: Array[Array[(Tensor[T], Tensor[T])]] = new Array[Array[(Tensor[T], Tensor[T])]](4)
  for(i <- 0 until 4) {
    gradients(i) = new Array[(Tensor[T], Tensor[T])](parallelNum)
  }

  val userCount = 138493
  val itemCount = 26744

  val embedding1 = 64
  val embedding2 = 128

  val userTaskSize = userCount / parallelNum
  val extraUserTask = userCount % parallelNum
  val itemTaskSize = itemCount / parallelNum
  val extraItemTask = itemCount % parallelNum

  val times = new Array[Long](parallelNum)

  (0 until parallelNum).foreach{tid =>
    if (state.get[Tensor[T]](s"buffer1$tid").isEmpty) {
      val userLength = userTaskSize + (if (tid < extraUserTask) 1 else 0)
      val itemLength = itemTaskSize + (if (tid < extraItemTask) 1 else 0)
      state(s"buffer1$tid") = Tensor[T](itemLength * embedding1 * 3)
      state(s"buffer2$tid") = Tensor[T](userLength * embedding1 * 3)
      state(s"buffer3$tid") = Tensor[T](itemLength * embedding2 * 3)
      state(s"buffer4$tid") = Tensor[T](userLength * embedding2 * 3)
    }
  }

  @transient
  private var ones: Tensor[T] = null

  /**
   * An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
    parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    MklDnn.isLoaded

    val (fx, dfdx) = feval(parameter)

    var timestep = state.getOrElse[Int]("evalCounter", 0)

    val clr = learningRate / (1 + timestep*learningRateDecay)

    timestep = timestep + 1

    val start = System.nanoTime()

    Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
      Affinity.setAffinity()
      val start = System.nanoTime()
      var offset = 0
      EmbeddingAdam2.updateEmbedding(tid, itemTaskSize, extraItemTask, embedding1, parameter,
        state[Tensor[T]](s"buffer1$tid"), clr, beta1, beta2, timestep, eps, offset, gradients(3))
      offset += itemCount * embedding1
      EmbeddingAdam2.updateEmbedding(tid, userTaskSize, extraUserTask, embedding1, parameter,
        state[Tensor[T]](s"buffer2$tid"), clr, beta1, beta2, timestep, eps, offset, gradients(2))
      offset += userCount * embedding1
      EmbeddingAdam2.updateEmbedding(tid, itemTaskSize, extraItemTask, embedding2, parameter,
        state[Tensor[T]](s"buffer3$tid"), clr, beta1, beta2, timestep, eps, offset, gradients(1))
      offset += itemCount * embedding2
      EmbeddingAdam2.updateEmbedding(tid, userTaskSize, extraUserTask, embedding2, parameter,
        state[Tensor[T]](s"buffer4$tid"), clr, beta1, beta2, timestep, eps, offset, gradients(0))

      times(tid) = (System.nanoTime() - start) / 1000000
    }))

    // EmbeddingAdam2.logger.
    //  info(s"update ${parameter.nElement()} parameters, maximum time is ${times.max} ms")
    //EmbeddingAdam2.logger.info(s"Time is ${times.sortWith((a, b) => a > b).mkString("\t")} ms")
    EmbeddingAdam2.logger.info(s"optim method time is ${(System.nanoTime() - start) / 1e6}ms")

    state("evalCounter") = timestep // A tmp tensor to hold the sqrt(v) + epsilon

    (parameter, Array(fx))
  }

  override def loadFromTable(config: Table): this.type = {
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.learningRateDecay = config.get[Double]("learningRateDecay")
      .getOrElse(this.learningRateDecay)
    this.beta1 = config.get[Double]("beta1").getOrElse(this.beta1)
    this.beta2 = config.get[Double]("beta2").getOrElse(this.beta2)
    this.eps = config.get[Double]("Epsilon").getOrElse(this.eps)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("s")
    state.delete("r")
  }

  override def getLearningRate(): Double = this.learningRate
}

object EmbeddingAdam2 {
  val logger = Logger.getLogger(this.getClass)

  private[optim] def updateEmbedding[T: ClassTag](
    tid: Int,
    taskSize: Int,
    extraTask: Int,
    embedding: Int,
    parameter: Tensor[T],
    buffer: Tensor[T],
    clr: Double,
    beta1: Double,
    beta2: Double,
    timestep: Int,
    eps: Double,
    totalOffset: Int,
    gradient: Array[(Tensor[T], Tensor[T])]
  )(implicit ev: TensorNumeric[T]): Unit = {
    val offset = (tid * taskSize + math.min(tid, extraTask)) * embedding
    val length = (taskSize + (if (tid < extraTask) 1 else 0)) * embedding
    val currentParameter = parameter.narrow(1, totalOffset + offset + 1, length)
    val _s = buffer.narrow(1, 1, length)
    val _r = buffer.narrow(1, length, length)
    val _denom = buffer.narrow(1, 2 * length, length)

    EmbeddingAdam2.updateSparse(_s, _denom, _r, beta1, beta2, offset, length, gradient)
    EmbeddingAdam2.updateFrame(_s, _r, _denom, clr, null, currentParameter,
      beta1, beta2, timestep, null, eps)
  }

  private[optim] def updateSparse[T: ClassTag](_s: Tensor[T], _denom: Tensor[T], _r: Tensor[T],
    beta1: Double, beta2: Double, offset: Int, length: Int,
    gradient: Array[(Tensor[T], Tensor[T])])(implicit ev: TensorNumeric[T]): Unit = {
    val record = new ArrayBuffer[(Int, Int, Int)]()
    var i = 0
    while(i < gradient.length) {
      val indexes = gradient(i)._1
      val values = gradient(i)._2
      val embedding = values.size(2)
      val indexData = indexes.storage().array()
      val indexOffset = indexes.storageOffset() - 1
      var j = 0
      while(j < indexes.size(1)) {
        val curOffset = ev.toType[Int](indexData(indexOffset + 1)) * embedding - offset
        if (curOffset > 0 && curOffset < length) {
          record.append((curOffset, i, j))
        }
        j += 1
      }
      i += 1
    }
    val recordArray = record.toArray.sortWith(_._1 < _._1)
    i = 0
    while(i < recordArray.length) {
      val values = gradient(recordArray(i)._2)._2
      val embedding = values.size(2)
      val _denomTmp = _denom.narrow(1, 1, embedding)
      val dfdx = values.select(1, recordArray(i)._3 + 1)
      _s.narrow(1, recordArray(i)._1, embedding)
        .add(ev.fromType[Double](1 - beta1), dfdx)
      _denomTmp.cmul(dfdx, dfdx)
      _r.narrow(1, recordArray(i)._1, embedding)
        .add(ev.fromType[Double](1 - beta2), _denomTmp)
      i += 1
    }
  }

  private[optim] def updateFrame[T: ClassTag](
    _s: Tensor[T], _r: Tensor[T], _denom: Tensor[T],
    clr: Double, dfdx: Tensor[T], parameter: Tensor[T],
    beta1: Double, beta2: Double, timestep: Int,
    ones: Tensor[T], eps: Double)(implicit ev: TensorNumeric[T]): Unit = {
    /**
     * m_t = beta_1 * m_t-1 + (1 - beta_1) * g_t
     * v_t = beta_2 * v_t-1 + (1 - beta_2) * g_t * g_t
     */
    // 7ms ~ 10ms
    _s.mul(ev.fromType[Double](beta1))// .add(ev.fromType[Double](1-beta1), dfdx)
    // _denom.cmul(dfdx, dfdx)

    // 10ms
    _r.mul(ev.fromType[Double](beta2))// .add(ev.fromType[Double](1-beta2), _denom)
    _denom.sqrt(_r)

    // used as MKL.axpy: 1 * a + y = y, and fill buffer with one
    // 2ms
    _denom.add(ev.fromType(eps))

    // efficiency improved upon by changing the order of computation, at expense of clarity
    // 3ms
    val biasCorrection1 = 1 - pow(beta1, timestep)
    val biasCorrection2 = 1 - pow(beta2, timestep)
    val stepSize = clr * sqrt(biasCorrection2) / biasCorrection1
    _denom.cdiv(_s, _denom)
    // 3ms
    parameter.add(ev.fromType[Double](-stepSize), _denom)
  }


  private[optim] def updateFrameZeroGrad[T: ClassTag](
    currentIteration: Int, lastUpdatedIteration: Int,
    _s: Tensor[T], _r: Tensor[T], _denom: Tensor[T], _buffer: Tensor[T],
    clr: Double, parameter: Tensor[T],
    beta1: Double, beta2: Double,
    ones: Tensor[T], eps: Double)(
    implicit ev: TensorNumeric[T]): Unit = {

    var timestep = lastUpdatedIteration
    while(timestep < currentIteration) {
      val biasCorrection1 = 1 - pow(beta1, timestep)
      val biasCorrection2 = 1 - pow(beta2, timestep)
      val stepSize = clr * sqrt(biasCorrection2) / biasCorrection1
      /**
       * m_t = beta_1 * m_t-1
       * v_t = beta_2 * v_t-1
       */
      _s.mul(ev.fromType[Double](beta1))
      _r.mul(ev.fromType[Double](beta2))
      _denom.sqrt(_r)

      // used as MKL.axpy: 1 * a + y = y
      _denom.add(ev.fromType(eps), ones)

      _denom.cdiv(_s, _denom)
      parameter.add(ev.fromType[Double](-stepSize), _denom)

      timestep += 1
    }
  }
}
