package transformers

import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{BooleanType, StringType, StructField, StructType}
import org.apache.spark.sql.functions.{col, udf}

class FemaleTransformer(override val uid: String) extends Transformer with DefaultParamsWritable {

  final val inputCol = new Param[String](this, "inputCol", "The input column")
  final val outputCol = new Param[String](this, "outputCol", "The output column")

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def this() = this(Identifiable.randomUID("femaleTransformer"))

  def copy(extra: ParamMap): FemaleTransformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {

    val idx = schema.fieldIndex($(inputCol))
    val field = schema.fields(idx)

    if (field.dataType != StringType) {

      throw new Exception(s"Input type ${field.dataType} did not match input type StringType")

    }

    schema.add(StructField($(outputCol),BooleanType, false))
  }

  def transform(ds: Dataset[_]): DataFrame = {

    val isFemale = udf {in: String => in == "female"}

    ds.select(col("*"), isFemale(ds.col($(inputCol))).as($(outputCol)))
  }

}

object FemaleTransformer extends DefaultParamsReadable[FemaleTransformer] {

  override def load(path: String): FemaleTransformer = super.load(path)

}
