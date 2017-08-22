import java.time.LocalDate
import models.Passenger
import transformers.FemaleTransformer
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StringType}
import org.apache.spark.ml.param.ParamMap

object Titanic extends LocalSpark with App {

    import spark.implicits._

    /***********************************************
     *     Load in data with the following features:
     *     - survived (label)
     *     - age
     *     - fare
     *     - pclass
     *     - sibsp
     *     - parch
     *     - embarked
     *
     *     Split 80/20% randomly into training/test sets
     ************************************************/

    val dataPath: String = "src/main/resources/titanic.csv"

    val ds: Dataset[Passenger] = spark.read
      .option("header", "true")
      .option("charset", "UTF8")
      .option("delimiter", ",")
      .csv(dataPath)
      .withColumn("survived", 'survived.cast(LongType))
      .withColumn("age", 'age.cast(DoubleType))
      .withColumn("sex", 'sex.cast(StringType))
      .withColumn("fare", 'fare.cast(DoubleType))
      .withColumn("pclass", 'pclass.cast(IntegerType))
      .withColumn("sibsp", 'sibsp.cast(IntegerType))
      .withColumn("parch", 'parch.cast(IntegerType))
      .withColumn("embarked", 'embarked.cast(StringType))
      .as[Passenger]

    val data: Dataset[Passenger] = ds.filter(_.embarked.isDefined)

    val seed: Int = 1000

    val Array(training, test): Array[DataFrame] = data.withColumnRenamed("survived","label")
      .randomSplit(Array(0.8,0.2),seed)

  /***********************************************
    *   Perform the following transformations:
    *     - "sex" (string) into "female" (boolean)
    *     - One hot encoding for "pclass" & "embarked"
    *     - Assemble features into vector
    *     - Perform standard scaling
    *     - Perform PCA
    *
    *   Implement RandomForestClassifier estimator
    *
    *   Fit these all into a Pipeline
    ************************************************/

    val female: FemaleTransformer = new FemaleTransformer()
      .setInputCol("sex")
      .setOutputCol("female")

    val classEncoder: OneHotEncoder = new OneHotEncoder()
      .setInputCol("pclass")
      .setOutputCol("classVector")

    val embarkedIndexer: StringIndexer = new StringIndexer()
      .setInputCol("embarked")
      .setOutputCol("embarkedIndex")

    val embarkedEncoder: OneHotEncoder = new OneHotEncoder()
      .setInputCol("embarkedIndex")
      .setOutputCol("embarkedVector")

    val imputer: Imputer = new Imputer()
      .setInputCols(Array("age","fare"))
      .setOutputCols(Array("ageImputed","fareImputed"))
      .setStrategy("median")

    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("fareImputed","ageImputed","classVector","embarkedVector","sibsp","parch"))
      .setOutputCol("features")

    val scaler: StandardScaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithMean(true)
      .setWithStd(true)

    val pca: PCA = new PCA()
      .setInputCol("scaledFeatures")
      .setOutputCol("pcaFeatures")

    val rf: RandomForestClassifier = new RandomForestClassifier()
      .setNumTrees(500)

    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(female,classEncoder,embarkedIndexer,embarkedEncoder,imputer,assembler,scaler,pca,rf))

  /***********************************************
    *   Set Parameter Grid for Pipeline stages
    ************************************************/

    val paramGrid: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(pca.k, Array(2,3,4))
      .addGrid(rf.maxDepth, Array(5,10))
      .build()

  /***********************************************
    *   Use Area Under ROC Curve to evaluate models
    ************************************************/

    val evaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")

  /***********************************************
  *     Use Cross Validation to perform
  *       - hyper-parameter tuning
  *       - model selection
  *       - model fitting
  *       - model write
  ************************************************/

    val cv: CrossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val cvModel: CrossValidatorModel = cv.fit(training)

    cvModel.write.overwrite.save(s"trained-models/cv-pipeline-${LocalDate.now}")

  /***********************************************
    *   Evaluate predictions on our test set using
    *     - AUC ROC
    *     - AUC PR
    ************************************************/

    val cvPredictions: DataFrame = cvModel.transform(test)
      .select("label","prediction","probability")

    val aucROC: Double = new BinaryClassificationEvaluator()
      .setRawPredictionCol("probability")
      .setMetricName("areaUnderROC")
      .evaluate(cvPredictions)

    val aucPR: Double = new BinaryClassificationEvaluator()
      .setRawPredictionCol("probability")
      .setMetricName("areaUnderPR")
      .evaluate(cvPredictions)

    println(s"AUC ROC: $aucROC, AUC PR: $aucPR")

    spark.stop

}
