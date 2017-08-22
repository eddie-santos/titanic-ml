package models

case class Passenger(
  pclass: String,
  survived: Long,
  name: String,
  sex: String,
  age: Option[Double],
  sibsp: Int,
  parch: Int,
  ticket: String,
  fare: Option[Double],
  cabin: Option[String],
  embarked: Option[String],
  boat: Option[String],
  body: Option[String],
  homeDest: Option[String]
)
