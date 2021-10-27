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

  def writePredictions(Ypredict: DenseVector[Double], outPath: String): Unit = {
    csvwrite(new File(outPath), separator = ',', mat = Ypredict.toDenseMatrix)
  }
}


object Main {
  def main(args: Array[String]): Unit = {
    val (targetPath, dataPath, outPath) = (args(0), args(1), args(2))
    val (nIter, learnRate) = (args(3).toInt, args(4).toDouble) // 1_000 .01
    val (io, model) = (new io, new LinearRegression)

    val (xTrain, yTrain, xTest, yTest) = io.readAndSplit(dataPath, targetPath)
    println(s"Successfully read and split data, ${xTrain.rows + xTest.rows} rows, ${xTrain.cols} columns")
    val (weights, bias) = model.fit(xTrain, yTrain, nIter, learnRate)
    println(s"Linear Regression fitted. \nweights: ${weights} \nbias: ${bias}")
    val yPred = model.predict(xTest, weights, bias)
    println(s"RMSE on validation dataset: ${model.RMSE(yTest, yPred)}")
    io.writePredictions(yPred, outPath)
    println(s"Predictions are saved to: ${outPath}")
    val writer = new FileWriter("C:\\Users\\Zaven\\Desktop\\HW-3\\data\\RMSE.txt")

    writer.write(s"RMSE on validation dataset = ${model.RMSE(yTest, yPred)} ")
    writer.close()
  }
}

