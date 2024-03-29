import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.types.DoubleType

object SparkAssignment2 {
    def main(args: Array[String]): Unit = {

        val spark = SparkSession.builder()
          .appName("Titanic Survival Prediction")
          .master("local[*]") // Change this to run on a cluster
          .getOrCreate()


        val trainDF = spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv("src\\main\\resources\\train.csv")

        trainDF.describe().show()

        //Feature Enginerring
        val removeDF=trainDF.drop("cabin")
        val combineDF=removeDF.withColumn("Family", concat(removeDF.col("SibSp")+ removeDF.col("Parch")))
        val finalDF=combineDF.drop("SibSp").drop("Parch").drop("Ticket").drop("Name")


        val analysisData  = finalDF.withColumn("Sex", when(col("Sex") === "male", 0).otherwise(1))
          .withColumn("Embarked", when(col("Embarked") === "Q", 0)
            .when(col("Embarked") === "S", 1)
            .when(col("Embarked") === "C", 2))
          .withColumn("Family", finalDF("Family").cast(DoubleType))
        analysisData.show()

        val cleanedDF = analysisData.na.drop()


        val Array(trainingData, testData) = cleanedDF.randomSplit(Array(0.7, 0.3))

        // Train a Random Forest model
        val rf = new RandomForestClassifier()
          .setLabelCol("Survived")
          .setFeaturesCol("features")
          .setNumTrees(10)
          .setMaxDepth(5)


        val assembler = new VectorAssembler()
          .setInputCols(cleanedDF.columns.filter(_ != "Survived")) // Drop the label column
          .setOutputCol("features")



        val pipeline = new Pipeline().setStages(Array(assembler, rf))
        val model = pipeline.fit(trainingData)


        val predictions = model.transform(testData)


        val evaluator = new BinaryClassificationEvaluator()
          .setLabelCol("Survived")
          .setRawPredictionCol("prediction")
          .setMetricName("areaUnderROC")

        val accuracy = evaluator.evaluate(predictions)
        println(s"Accuracy: $accuracy")




        val testDF = spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv("src\\main\\resources\\test.csv")


        val testRemoveDF = testDF.drop("Cabin")
        val testCombineDF = testRemoveDF.withColumn("Family", col("SibSp") + col("Parch"))
        val testFinalDF = testCombineDF.drop("SibSp", "Parch", "Ticket")

        val testAnalysisData = testFinalDF.withColumn("Sex", when(col("Sex") === "male", 0).otherwise(1))
          .withColumn("Embarked", when(col("Embarked") === "Q", 0)
            .when(col("Embarked") === "S", 1)
            .when(col("Embarked") === "C", 2))
          .withColumn("Family", testFinalDF("Family").cast(DoubleType))

        val testCleanedDF = testAnalysisData.na.drop()


        val testPredictions = model.transform(testCleanedDF)


        testPredictions.select("PassengerId", "Name","prediction").show()


        spark.stop()
    }
}