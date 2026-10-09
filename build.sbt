name := "SparkCDT"

version := "1.4"

scalaVersion := "2.13.18"
val sparkVersion = "4.1.1"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion)



//to run an example:
//sbt "runMain example.cdtExample"  
