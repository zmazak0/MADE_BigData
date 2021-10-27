import breeze.linalg.{DenseMatrix, DenseVector, csvread, csvwrite, sum}
import breeze.numerics.{pow, sqrt}
import java.io.{File, FileWriter}


class LinearRegression {
  def fit(X: DenseMatrix[Double], y: DenseVector[Double],
          nIterations: Int, learnRate: Double): (DenseVector[Double], Double) = {
    var (weights, bias) = (DenseVector.zeros[Double](X.cols), .0)
    for (_ <- 0 to nIterations) {
      val yHat = (X * weights) + bias
      weights :-= learnRate * 2 * (X.t * (yHat - y))
      weights = weights.map(el => el / X.rows)
      bias -= learnRate * 2 * sum(yHat - y) / X.rows
    }
    (weights, bias)
  }

  def predict(X: DenseMatrix[Double], weights: DenseVector[Double],
              bias: Double): DenseVector[Double] = {
    (X * weights) + bias
  }

  def RMSE(yTrue: DenseVector[Double], yPred: DenseVector[Double]): Double = {
    val error = sum((yTrue - yPred).map(el => pow(el, 2))) / yTrue.length
    sqrt(error)
  }
}

class io {
  def readAndSplit(dataPath: String, targetPath: String): (DenseMatrix[Double],
    DenseVector[Double], DenseMatrix[Double], DenseVector[Double]) = {
    val X = csvread(file=new File(dataPath), separator=',')
    val y = csvread(file=new File(targetPath), separator=',').toDenseVector
    val n = y.length
    val trainFraction = (n * .8).toInt

    val (yTrain, yTest) = (y(0 until trainFraction), y(trainFraction until n))
    val (xTrain, xTest) = (X(0 until trainFraction, ::), X(trainFraction until n, ::))
    (xTrain, yTrain, xTest, yTest)
  }

  def writePredictions(Predict: DenseVector[Double], outPath: String): Unit = {
    csvwrite(new File(outPath), separator = ',', mat = Predict.toDenseMatrix)
  }
}


object Main {
  def main(args: Array[String]): Unit = {
    val writer = new FileWriter("C:\\Users\\Zaven\\Desktop\\HW-3\\data\\logs.txt")

    val (targetPath, dataPath, outPath) = (args(0), args(1), args(2))
    val (nIterations, learningRate) = (args(3).toInt, args(4).toDouble) // 1_000 .01
    val (io, model) = (new io, new LinearRegression)

    val (xTrain, yTrain, xTest, yTest) = io.readAndSplit(dataPath, targetPath)
    println(s"Successfully read and split data, ${xTrain.rows + xTest.rows} rows, ${xTrain.cols} columns.")
    writer.write(s"Successfully read and split data, ${xTrain.rows + xTest.rows} rows, ${xTrain.cols} columns.\n")

    val (weights, bias) = model.fit(xTrain, yTrain, nIterations, learningRate)
    println(s"Linear Regression fitted.")
    writer.write(s"Linear Regression fitted.")

    val yPredict = model.predict(xTest, weights, bias)
    println(s"RMSE on validation dataset: ${model.RMSE(yTest, yPredict)}.")
    writer.write(s"RMSE on validation dataset: ${model.RMSE(yTest, yPredict)}.\n")

    io.writePredictions(yPredict, outPath)
    println(s"Predictions are saved to: ${outPath}")
    writer.write(s"Predictions are saved to: ${outPath}\n")

    writer.close()
  }
}

