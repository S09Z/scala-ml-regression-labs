import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.regression.{RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator

object HousePricePrediction {
  def main(args: Array[String]): Unit = {
    // ✅ Create Spark Session
    val spark = SparkSession
      .builder()
      .appName("HousePricePrediction")
      .master("local[*]")
      .getOrCreate()

    // ✅ Load Data
    var df: DataFrame = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/housing.csv")

    df.printSchema()

    // ✅ Convert relevant columns to Double type
    df = df
      .withColumn("bedrooms", col("bedrooms").cast("double"))
      .withColumn("bathrooms", col("bathrooms").cast("double"))
      .withColumn("stories", col("stories").cast("double"))
      .withColumn("parking", col("parking").cast("double"))

    // ✅ Normalize Features
    val featureColumns = Array("bedrooms", "bathrooms", "stories", "parking", "area")
    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val featureDf = assembler.transform(df).select("features", "price")

    // ✅ Split Data into Train & Test
    val Array(trainData, testData) = featureDf.randomSplit(Array(0.8, 0.2), seed = 1234)

    // ✅ Use RandomForestRegressor instead of XGBoost
    val rf = new RandomForestRegressor()
      .setLabelCol("price")
      .setFeaturesCol("features")
      .setNumTrees(100) // Adjust number of trees
      .setMaxDepth(10) // Adjust max depth
      .setSeed(42)

    // ✅ Train Model
    val model = rf.fit(trainData)

    // ✅ Predict on Test Data
    val predictions = model.transform(testData)

    // ✅ Evaluate Model
    val evaluatorRMSE = new RegressionEvaluator()
      .setLabelCol("price")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val evaluatorR2 = new RegressionEvaluator()
      .setLabelCol("price")
      .setPredictionCol("prediction")
      .setMetricName("r2")

    val evaluatorMAE = new RegressionEvaluator()
      .setLabelCol("price")
      .setPredictionCol("prediction")
      .setMetricName("mae")

    val rmse = evaluatorRMSE.evaluate(predictions)
    val r2 = evaluatorR2.evaluate(predictions)
    val mae = evaluatorMAE.evaluate(predictions)

    println(s"Root Mean Squared Error (RMSE): $rmse")
    println(s"R² Score: $r2")
    println(s"Mean Absolute Error (MAE): $mae")

    // ✅ Stop Spark Session
    spark.stop()
  }
}
