// Databricks notebook source
// MAGIC %md
// MAGIC # Ethereum Notebook
// MAGIC ### Sanjif Rajaratnam

// COMMAND ----------

// MAGIC %md 
// MAGIC ## Import Libraries

// COMMAND ----------

import java.sql.Timestamp
import org.apache.commons.io.IOUtils
import java.net.URL
import java.nio.charset.Charset
import java.math.BigInteger

// COMMAND ----------

// MAGIC %md
// MAGIC ## Import Data

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from blocks

// COMMAND ----------

val blocksDF = sqlContext.sql("select * from blocks")
val transfersDF = sqlContext.sql("select * from transfers")
val transacationsDF = sqlContext.sql("select * from transactions")

val blocksRDD: RDD[Row] = blocksDF.rdd
val transfersRDD: RDD[Row] = transfersDF.rdd
val transactionsRDD: RDD[Row] = transacationsDF.rdd

// COMMAND ----------

// MAGIC %md
// MAGIC ## Cleaning Functions

// COMMAND ----------

// patching the String class with new functions that have a defualt value if conversion to another type fails.
implicit class StringConversion(val s: String) {
def toTypeOrElse[T](convert: String=>T, defaultVal: T) = try {
    convert(s)
  } catch {
    case _: Throwable => defaultVal
  }
  
  def toIntOrElse(defaultVal: Int = 0) = toTypeOrElse[Int](_.toInt, defaultVal)
  def toDoubleOrElse(defaultVal: Double = 0D) = toTypeOrElse[Double](_.toDouble, defaultVal)
  def toDateOrElse(defaultVal: java.sql.Timestamp = java.sql.Timestamp.valueOf("1970-01-01 00:00:00")) = toTypeOrElse[java.sql.Timestamp](java.sql.Timestamp.valueOf(_), defaultVal)
}

//Fix this date format: 2015-09-27 18:41:18 UTC
def fixDateFormat(orig: String): String = {
    val split_date = orig.split(" ")
    split_date(0) + " " + split_date(1)
}
// Output: 2015-09-01 00:15:00.0

// Convert Hex String to Big Integer
def hexToBigInt(s: String): BigInteger = {
  val defaultVal = "0x0"
  if ( s.size < 2 ){
    val x = new BigInteger(defaultVal.substring(2,defaultVal.size),16)
    x
  }
  
  else if ( s.substring(0,2) == "0x" ){
    val x = new BigInteger(s.substring(2,s.size),16)
    x
  }
  else {
    val x = new BigInteger(defaultVal.substring(2,defaultVal.size),16)
    x
  }
}

def hexToFloat(s: String): Float = {
  val defaultVal = "0x0"
  if ( s.size < 2 ){
    val x = new BigInteger(defaultVal.substring(2,defaultVal.size),16).floatValue
    x
  }
  
  else if ( s.substring(0,2) == "0x" ){
    val x = new BigInteger(s.substring(2,s.size),16).floatValue
    x
  }
  else {
    val x = new BigInteger(defaultVal.substring(2,defaultVal.size),16).floatValue
    x
  }
}

def convertToString(s: Any): String = {
  val defaultVal = ""
  if ( s == null ) {
    defaultVal
  }
  else {
    s.toString()
  }
}

// COMMAND ----------

// MAGIC %md
// MAGIC ## Blocks

// COMMAND ----------

case class Blocks(                     // column index
  difficulty: Float,                     // 0
  extraData: String,                   // 1  
  gasLimit: Int,                       // 2
  gasUsed: Int,                        // 3
  hash: String,                        // 4
  miner: String,                       // 5
  nonce: String,                       // 6
  number: Int,                         // 7
  parentHash: String,                  // 8
  size: Int,                           // 9
  timeStamp: java.sql.Timestamp,       // 10
  totalDifficulty: Float,          // 11
  transactionCount: Int                // 12
  )

def getBlocksCleaned(row: Array[String]):Blocks = {
  return Blocks(
    hexToFloat(row(0)),
    row(1),
    row(2).toIntOrElse(),
    row(3).toIntOrElse(),
    row(4),
    row(5),
    row(6),
    row(7).toIntOrElse(),
    row(8),
    row(9).toIntOrElse(),
    fixDateFormat(row(10)).toDateOrElse(),
    hexToFloat(row(11)),
    row(12).toIntOrElse()
  )
}

// COMMAND ----------

val blocks = blocksRDD.map(r => getBlocksCleaned(r.toString().stripPrefix("[").stripSuffix("]").split(","))).toDF()
blocks.createOrReplaceTempView("cblocks")

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from cblocks

// COMMAND ----------

// MAGIC %md
// MAGIC ## Transfers

// COMMAND ----------

case class Transfers(                     // column index
  blockHash: String,                      // 0
  blockNumber: Int,                       // 1
  depth: Int,                             // 2
  fromAccount: String,                           // 3
  fromBalance: Float,                    // 4
  timeStamp: java.sql.Timestamp,          // 5
  toAccount: String,                             // 6
  toBalance: Float,                      // 7
  transactionHash: String,                // 8
  transferIndex: Int,                     // 9
  transferType: String,                           // 10
  value: Float                         // 11
  )

def getTransfersCleaned(row: Array[String]):Transfers = {
  return Transfers(
    row(0),
    row(1).toIntOrElse(),
    row(2).toIntOrElse(),
    row(3),
    hexToFloat(row(4)),
    fixDateFormat(row(5)).toDateOrElse(),
    row(6),
    hexToFloat(row(7)),
    row(8),
    row(9).toIntOrElse(),
    row(10),
    hexToFloat(row(11))
  )
}

// COMMAND ----------

val transfers = transfersRDD.map(r => getTransfersCleaned(r.toString().stripPrefix("[").stripSuffix("]").split(","))).toDF()
transfers.createOrReplaceTempView("ctransfers")

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from ctransfers

// COMMAND ----------

// MAGIC %md
// MAGIC ## Transactions

// COMMAND ----------

case class Transactions(                      // column index
  blockHash: String,                          // 0
  blockNumber: Int,                           // 1
  contractAddress: String,                    // 2
  error: String,                              // 3
  from: String,                               // 4
  gas: Int,                                   // 5
  gasPrice: Int,                              // 6
  gasUsed: Int,                               // 7
  hash: String,                               // 8
  input: String,                              // 9
  nonce: Int,                                 // 11
  timeStamp: String,              // 12
  to: String,                                 // 13
  value: String                               // 14
  )

def getTransactionsCleaned(row: org.apache.spark.sql.Row):Transactions = {
  return Transactions(
    convertToString(row.apply(0)),
    convertToString(row.apply(1)).toIntOrElse(),
    convertToString(row.apply(2)),
    convertToString(row.apply(3)),
    convertToString(row.apply(4)),
    convertToString(row.apply(5)).toIntOrElse(),
    convertToString(row.apply(6)).toIntOrElse(),
    convertToString(row.apply(7)).toIntOrElse(),
    convertToString(row.apply(8)),
    convertToString(row.apply(9)),
    convertToString(row.apply(10)),
    convertToString(row.apply(11)).toIntOrElse(),
    convertToString(row.apply(12)),
    convertToString(row.apply(13)),
    convertToString(row.apply(14))
  )
}

// COMMAND ----------

case class Transactions(                      // column index
  blockHash: String,                          // 0
  blockNumber: Int,                           // 1
  contractAddress: String,                    // 2
  error: String,                              // 3
  from: String,                               // 4
  gas: Int,                                   // 5
  gasPrice: Int,                              // 6
  gasUsed: Int,                               // 7
  hash: String,                               // 8
  input: String,                              // 9
  nonce: Int,                                 // 11
  timeStamp: java.sql.Timestamp,                          // 12
  to: String,                                 // 13
  value: Float                               // 14
  )

def getTransactionsCleaned(row: org.apache.spark.sql.Row):Transactions = {
  return Transactions(
    convertToString(row.apply(0)),
    convertToString(row.apply(1)).toIntOrElse(),
    convertToString(row.apply(2)),
    convertToString(row.apply(3)),
    convertToString(row.apply(4)),
    convertToString(row.apply(5)).toIntOrElse(),
    convertToString(row.apply(6)).toIntOrElse(),
    convertToString(row.apply(7)).toIntOrElse(),
    convertToString(row.apply(8)),
    convertToString(row.apply(9)),
    convertToString(row.apply(11)).toIntOrElse(),
    fixDateFormat(convertToString(row.apply(12))).toDateOrElse(),
    convertToString(row.apply(13)),
    hexToFloat(convertToString(row.apply(14)))
  )
}


// COMMAND ----------

val transactions = transactionsRDD.map(r => getTransactionsCleaned(r)).toDF()
transactions.createOrReplaceTempView("ctransactions")

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from ctransactions

// COMMAND ----------


