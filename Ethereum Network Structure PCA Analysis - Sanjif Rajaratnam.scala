// Databricks notebook source
// MAGIC %md
// MAGIC ## MIE1512H
// MAGIC # Ethereum Network Structure Analysis with PCA
// MAGIC ### Sanjif Rajaratnam
// MAGIC 
// MAGIC **Paper**: *Kondor, Daniel, et al. “Inferring the Interplay between Network Structure and Market Effects in Bitcoin.” New Journal of Physics, vol. 16, no. 12, Feb. 2014, p. 125003., doi:10.1088/1367-2630/16/12/125003* 

// COMMAND ----------

// MAGIC %md
// MAGIC The work presented in the notebook builds off the work presented in *Inferring the Interplay between Network Structures and Market Effects in Bitcoins*. The same techniques presented in the paper were applied towards the dataset of Ethereum.

// COMMAND ----------

// MAGIC %md 
// MAGIC # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// COMMAND ----------

// MAGIC %md
// MAGIC ## Schedule
// MAGIC The work completed each week will be listed here. The steps are out of order because previous steps were added to in future weeks. This follows the order of the timeline in which parts of the notebook were completed.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Week 1
// MAGIC 
// MAGIC #### 1. Import Data (1/1)h
// MAGIC During this step, the raw data provided by Shahan was imported into Databricks. 
// MAGIC 
// MAGIC #### 3(a-d). Preliminary Cleaning (4/4)h
// MAGIC During this step, the imported data was cleaned in terms of format. Hexadecimals were converted to strings, dates were converted into Java Timestamps, etc.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Week 2
// MAGIC 
// MAGIC #### 2. Data Mining (1/1)h
// MAGIC During this step, Ethereum to USD historical rate data was acquired from Poloniex.
// MAGIC 
// MAGIC #### Presentation (15/15)h
// MAGIC During this step, time was spent thoroughly reading the paper and researching topics presented in the paper in order to get a better understanding of the material. Then the presentation was created with the information learnt in mind.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Week 3
// MAGIC 
// MAGIC #### 5. Data Prepartion (9/7)h
// MAGIC The data was cleaned by adding vital information to the transfers tables identifies contracts, vs. users. Then the useful subgraph was extracted from the raw data. Then the cores were extracted from the subgraph. User ids were mapped to addresses. 

// COMMAND ----------

// MAGIC %md
// MAGIC ### Week 4
// MAGIC 
// MAGIC #### 3e. Preliminary Cleaning: The DAO Data (0.5/0.5)h
// MAGIC During this step, theDAO information was cleaned and imported into the notebook.
// MAGIC 
// MAGIC #### 4. Data Exploration (2/2)h
// MAGIC During this step, the given tables were explored to better understand what happens in an Ethereum transaction. This step was important to understand what exactly was housed inside the database provided.
// MAGIC 
// MAGIC #### 5. Data Preparation (4/4)h
// MAGIC Here most of the work was updated with knowledge derived from data exploration. The additional step of preparing the data to be put into an adjaceny matrix was done.
// MAGIC 
// MAGIC #### 6. Weighted Adjacency Matrix (3/2)h
// MAGIC Construct the daily adjaceny matrix and concatenate all the information into a matrix. Here mainly NetworkX, Pandas, and Numpy are used since its simple to create an adjacency matrix using these. The output is nxn graph per day which when converted into the final matrix is lx(nxn) or 386 x 1 million plus. To make this simpler, the columns that sum to zero in the both networks (e.g. no interactions between user u and user v, ever in both networks), are removed since they're unimportant features. This leaves us with a 356 x 5585 matrix.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Week 5
// MAGIC #### 7. Principal Component Analysis (1/1)h
// MAGIC Here the Principal Components will be found using Singular Value Decomposition (SVD) as per the paper. This is implemented in MLlib Scala so the matrix in Python was exported and reimported. Since both our core networks are Txl = (356 x 5585), there are only T = 356 non-zero singular values and only T = 356 relevant singular vectors (columns of v). This means the full SVD can be truncated from (U: TxT, s: Txl, v: lxl) to (U:TxT, s:TxT, v: lxT) to ignore irrelevant vectors.
// MAGIC 
// MAGIC #### 1. Import Data (2/1)h
// MAGIC Here the data provided by Shahan was read in from the cloud using a bash script. The data is hosted on my dropbox, and is downloaded into a directory on the cluster. This however means that if the cluster dies or is reattached all the existing data is destroyed. 
// MAGIC 
// MAGIC #### 8. Data Validation (4/2)h
// MAGIC Here the same method employed in the paper will be used. If the PCA was performed correctly, the original matrix can be reconstructed by using a subset of the principal components. The greater the number of components used, the lower the error between both matrices. The error will be plotted against the number of components as per the paper. How the error was computed was not mentioned but the standard summed square distance between all elements will be used.  
// MAGIC **Plots**
// MAGIC 1. Singular value vs. components graph
// MAGIC 
// MAGIC #### 9. Data Analysis (4/2)h
// MAGIC Here the singular value weight vs date will be analyzed for both cores. It will also be checked if the Singular Value Weight can be used to predict the Ethereum Exchange rate. The findings will be reported from this analysis.
// MAGIC **Plots**
// MAGIC 1. Singular value weight vs. date for both cores
// MAGIC 2. Singular value weight vs. Ethereum to usd exchange rate

// COMMAND ----------

// MAGIC %md 
// MAGIC # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// COMMAND ----------

// MAGIC %md
// MAGIC #### Clear cache
// MAGIC This line clears any tables that are stored in the memory from previous runs

// COMMAND ----------

// MAGIC %sql
// MAGIC clear cache

// COMMAND ----------

// MAGIC %md
// MAGIC #### Import and install any necessary libraries
// MAGIC This imports any libraries used in the following notebook for both Python and Scala.

// COMMAND ----------

// Scala Libraries
import java.sql.Timestamp
import java.sql.Timestamp
import org.apache.commons.io.IOUtils
import java.net.URL
import java.nio.charset.Charset
import java.math.BigInteger
import scala.math.pow
import spark.implicits._

import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix

// COMMAND ----------

// MAGIC %python
// MAGIC # Import libaries used in Python
// MAGIC !pip install bs4 # Install BeautifulSoup
// MAGIC !pip install networkx # Install Networkx
// MAGIC !pip install pyzmq # Install pyzmq - needed for ipython
// MAGIC 
// MAGIC import urllib2
// MAGIC import pandas as pd
// MAGIC import datetime
// MAGIC from bs4 import BeautifulSoup
// MAGIC import ast
// MAGIC import numpy as np
// MAGIC import json
// MAGIC import io
// MAGIC from ctypes import *
// MAGIC import networkx as nx
// MAGIC import matplotlib.pyplot as plt
// MAGIC from datetime import date, timedelta
// MAGIC from operator import itemgetter
// MAGIC from numpy import linalg as LA

// COMMAND ----------

// MAGIC %md 
// MAGIC # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// COMMAND ----------

// MAGIC %md
// MAGIC ## 1. Import Data
// MAGIC The  data for this project was provided by Shahan Khatchadourian (ConsenSys). The data was compressed using tar and lzma and uploaded to dropbox. "cURL" is used to download it from dropbox and it is extracted and placed in the "sanjif" folder in the root directory. 
// MAGIC 
// MAGIC The size of the files provided was about 8 GB. The data provided was approximately one year's worth of Ethereum related data. The *Transfers* table contained all the main information used in this analysis. The other three tables (DAOContracts, Blocks, Transactions) are only used to identify *miners*, *contracts*, and *DAO contracts*. All these tables contain information beyond what was used but they weren't relevant for this analysis. This will be explained in further detail later on in *Section 3*.

// COMMAND ----------

// MAGIC %md
// MAGIC ### 1a. Read in files from cloud
// MAGIC Here the data was read in from Dropbox, and extracted.

// COMMAND ----------

// MAGIC %md
// MAGIC ** Declare directory paths**
// MAGIC Here, the directory paths are declared so Python and Scala code can use them later as required.

// COMMAND ----------

// Declare diirectories in Scala
val relDir = "/sanjif/"

val absDir = "file:" + relDir
val absResultsDir = absDir + "results/"
val relResultsDir = relDir + "results/"

// COMMAND ----------

// MAGIC %python
// MAGIC # Declare directories in Python
// MAGIC relDir = "/sanjif/"
// MAGIC 
// MAGIC absDir = "file:" + relDir
// MAGIC absResultsDir = absDir + "results/"
// MAGIC relResultsDir = relDir + "results/"

// COMMAND ----------

// MAGIC %sh
// MAGIC # Install lzma to unzip files
// MAGIC apt-get install lzma
// MAGIC 
// MAGIC # Parameters # These files are all saved to a root directory folder called 'sanjif'
// MAGIC startFromScratch=0
// MAGIC dir=/sanjif/
// MAGIC 
// MAGIC # Make the directory and results directory if it doesnt exist
// MAGIC mkdir -p $dir
// MAGIC rm -rf $dir/results/
// MAGIC mkdir -p $dir/results/
// MAGIC 
// MAGIC # Remove Old Files
// MAGIC if [ $startFromScratch == 1 ]; then
// MAGIC   rm -rf $dir*
// MAGIC fi
// MAGIC 
// MAGIC # Change to directory
// MAGIC cd $dir
// MAGIC 
// MAGIC ## Download Data ##
// MAGIC # Dowload the data if it doesn't already exist
// MAGIC if [ ! -f EthereumRawData.tar.lzma ] || [ ! -f EthereumRawData.tar ]; then 
// MAGIC   curl -L -o "EthereumRawData.tar.lzma" https://www.dropbox.com/s/25gcchhh3uvq8c3/EthereumRawData.tar.lzma?dl=1
// MAGIC fi
// MAGIC 
// MAGIC # Extract the file if this hasn't already been done
// MAGIC if [ ! -d transfers ] || [ ! -d transactions ] || [ ! -f blocks.json ] || [ ! -f daoAccounts.csv ]; then 
// MAGIC   unlzma EthereumRawData.tar.lzma
// MAGIC   tar -xvf EthereumRawData.tar
// MAGIC fi

// COMMAND ----------

// MAGIC %md
// MAGIC The file /sanjif should now contain a *transactions* folder with 30 files, a *transfers* folder with 59 files, *blocks.json*, and *daoAccounts.csv*. It should also contain the files they were extracted from: *EthereumRawData.tar* and *EthereumRawData.tar.lzma*.

// COMMAND ----------

// MAGIC %sh
// MAGIC echo "Files in /sanjif:" 
// MAGIC ls /sanjif
// MAGIC echo ""
// MAGIC echo "Number of files in transfers: " $(ls /sanjif/transfers/ | wc -l)
// MAGIC echo "Number of files in transactions: " \
// MAGIC       $(ls /sanjif/transactions/ | wc -l)

// COMMAND ----------

// MAGIC %md
// MAGIC #### 1b. Import the data into Spark RDDs
// MAGIC The JSON files that were imported were converted into Spark dataframes by using Spark's JSON reader. The dataframes were then converted into RDDs so they can be cleaned. The CSV file was read directly into an RDD.

// COMMAND ----------

// Convert the JSON files into Spark DataFrames
val blocksDF = spark.read.json(absDir + "blocks.json")
val transfersDF = spark.read.json(absDir + "transfers")
val transactionsDF = spark.read.json(absDir + "transactions")

// Convert into RDDs so they can be cleaned
val blocksRDD: RDD[Row] = blocksDF.rdd
val transfersRDD: RDD[Row] = transfersDF.rdd
val transactionsRDD: RDD[Row] = transactionsDF.rdd

// Read in the DAO csv
val daoRDD = sc.textFile(absDir + "daoAccounts.csv").map(line => line.split(",", -1).map(_.trim)).filter(line => line(0) != "address")

// COMMAND ----------

// MAGIC %md 
// MAGIC # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2. Data Mining
// MAGIC 
// MAGIC During this step, Ethereum to USD historical rate data was acquired from Poloniex. The BeautifulSoup library was used for web scraping. The data was also cleaned. The data provided was: 'high','low','open','close','weightedAverage','volume', and 'quoteVolume' at several time intervals during the day. The volume information wasn't required so they were dropped. Also information about dates that were out of the dataset range were also dropped. Next the average price of the Ethers for the day were acquired using this procedure:
// MAGIC 
// MAGIC 1. Find the average highest price and lowest price every day
// MAGIC 2. Take the average of these averages to get the average price for a given day.

// COMMAND ----------

// MAGIC %python
// MAGIC # Open url and read it
// MAGIC url = urllib2.urlopen("https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start=1435699200&end=9999999999&period=14400")
// MAGIC content = url.read()
// MAGIC 
// MAGIC # Read the url content using Beautiful Soup
// MAGIC soup = BeautifulSoup(content,'lxml')
// MAGIC url_dict = ast.literal_eval(soup.body.string)
// MAGIC 
// MAGIC # Split the information into a pandas dataframe
// MAGIC url_df = pd.DataFrame(url_dict)
// MAGIC url_df['date'] = url_df['date'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
// MAGIC url_df['date_only'] = url_df['date'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S").date())
// MAGIC 
// MAGIC # Get aggregate price information per day
// MAGIC ethToUSDDF = url_df.groupby('date_only')[['high','low']].mean() # Find the average per day
// MAGIC 
// MAGIC # Make a date column
// MAGIC ethToUSDDF.reset_index(inplace=True) # Reset the index
// MAGIC ethToUSDDF.rename(columns={'date_only': 'date'}, inplace=True) # Rename date only column
// MAGIC 
// MAGIC # Remove dates out of this range [2015-07-30:2016-07-27]
// MAGIC ethToUSDDF = ethToUSDDF[(ethToUSDDF['date'] >= date(2015,7,8)) & (ethToUSDDF['date'] <= date(2016,7,27))]
// MAGIC 
// MAGIC # Compute average price
// MAGIC ethToUSDDF['average'] = 0.5*(ethToUSDDF['high'] + ethToUSDDF['low']) # Find the average of the high and low
// MAGIC 
// MAGIC tempEthToUSDSDF = sqlContext.createDataFrame(ethToUSDDF) # Create spark dataframe from pandas dataframe
// MAGIC tempEthToUSDSDF.write.mode("overwrite").parquet(absResultsDir + "ethToUSD")  # Write it to file
// MAGIC display(tempEthToUSDSDF.orderBy("date")) # show table

// COMMAND ----------

// MAGIC %md 
// MAGIC # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// COMMAND ----------

// MAGIC %md
// MAGIC ## 3. Preliminary Cleaning
// MAGIC 
// MAGIC During this step, the imported data was cleaned in terms of format. Hexadecimals were converted to strings, dates were converted into Java Timestamps, etc. 
// MAGIC 
// MAGIC There were problems directly typecasting strings as ints because the value exceeded the storage capacity of an int. The hex strings had to be converted big ints then into floats.

// COMMAND ----------

// MAGIC %md
// MAGIC ### 3a. Cleaning Helper Functions
// MAGIC 
// MAGIC This Scala functions will be used below to clean the data in the imported data into the correct format. The first class was borrowed from the lab examples provided.

// COMMAND ----------

// From class labs for cleaning ints, doubles, and dates
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

// Convert Hex String to Float
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

// Convert Weis To Ethers
def weiToEther(wei: Float): Double = {
  val x = wei/pow(10,18)
  x
}

// Handle null cases with strings
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
// MAGIC ### 3b. Blocks Table
// MAGIC 
// MAGIC This section is used to read in and clean the block table. The block table contained information regarding the block. The only necessary column on this table was the miner addresses as it will be used to identified addresses that belong to miners.
// MAGIC 
// MAGIC This table contains the following pertinent information:
// MAGIC 1. The miner address(*address*)

// COMMAND ----------

// Define Blocks Table Types
case class Blocks(                     // column index
  address: String                       // 6
  )

// Blocks Cleaning Function
def getBlocksCleaned(row: Array[String]):Blocks = {
  return Blocks(
    row(6)
  )
}

// COMMAND ----------

// Strip and clean
val tempBlocksDF = blocksRDD.map(r => getBlocksCleaned(r.toString().stripPrefix("[").stripSuffix("]").split(","))).toDF()

// Only keep distinct addresses
val blocksDF = tempBlocksDF.select(tempBlocksDF("address")).distinct

blocksDF.createOrReplaceTempView("blocks") // Replace table
//sqlContext.cacheTable("blocks") // Cache table
display(sqlContext.sql("select * from blocks")) // Show table

// COMMAND ----------

// MAGIC %md
// MAGIC ### 3c. Transactions Table
// MAGIC 
// MAGIC This section is used to read in and clean the transactions table. The transaction table contains information about various transactions. Again, from this table, only the information regarding contract addresses were needed so this was the only information pulled from the table. This table did not contain all the contracts but a decent amount of them.
// MAGIC 
// MAGIC This table contains the following pertinent information:
// MAGIC 1. The contract address associated with the transaction if there is any (*address*)

// COMMAND ----------

// Define Transactions table types
case class Transactions(                      // column index
  address: String                    // 2
  )

// Transactions Cleaning Function
def getTransactionsCleaned(row: org.apache.spark.sql.Row):Transactions = {
  return Transactions(
    convertToString(row.apply(2))
  )
}

// COMMAND ----------

// Split and clean Transactions Table
val tempTransactionsDF = transactionsRDD.map(r => getTransactionsCleaned(r)).toDF()
tempTransactionsDF.createOrReplaceTempView("transactions") // Create table

// Drop NaN values
val transactionsDF = sqlContext.sql("select * from transactions where address != ''")

// Create, cache and show table
transactionsDF.createOrReplaceTempView("transactions") 
//sqlContext.cacheTable("transactions") 
display(sqlContext.sql("select * from transactions")) 

// COMMAND ----------

// MAGIC %md
// MAGIC ### 3d. TheDAO Accounts Table
// MAGIC 
// MAGIC During The DAO crowdfunding event on Ethereum, the crowdfunded account was hacked and millions of dollars were stolen. To remedy this, the people of Ethereum voted to have have a hard fork where Ethereum continued as ETC and a new fork was created called ETC. Several contracts were created to refund the accounts that were affected. This table holds information regarding the DAO contracts that they used to remedy this.
// MAGIC 
// MAGIC This table contains the following pertinent information:
// MAGIC 1. The DAO contract addresses (*address*)
// MAGIC 1. The DAO contract addresses where they stored extra money (*extraBalanceAccount*)

// COMMAND ----------

// Define DAO table types
case class DAO(                      // column index
  address: String,                    // 0
  extraBalanceAccount: String         // 2
  )

// DAO table cleaning function
def getDAOCleaned(row: Array[String]):DAO = {
  return DAO(
    row(0),
    row(2)
  )
}

// COMMAND ----------

// Map with class and create spark dataframe
val tempDaoDF = daoRDD.map(r => getDAOCleaned(r)).toDF()

tempDaoDF.createOrReplaceTempView("DAO") // Create SQL table

// Union the dataframes and keep the unique addresses
val daoDF = tempDaoDF.select(tempDaoDF("address")).union(tempDaoDF.select(tempDaoDF("extraBalanceAccount"))).distinct

// Create and cache new table
daoDF.createOrReplaceTempView("DAO")
//sqlContext.cacheTable("DAO")
display(sqlContext.sql("select * from DAO"))

// COMMAND ----------

// MAGIC %md
// MAGIC ### 3e. Transfers Table
// MAGIC 
// MAGIC This section is used to read in and clean the transfers table. The transfers table contains information about any type of transfer between accounts. The unnecessary information was ignored as they weren't needed for the analysis.
// MAGIC 
// MAGIC This table contains the following pertinent information:
// MAGIC 1. The block the interaction was in (*blockNum*)
// MAGIC 1. The depth of the transaction (*depth*)
// MAGIC 1. The sender account (*fromAccount*)
// MAGIC 1. The time stamp associated with the transfer (*timeStamp*)
// MAGIC 1. The receiver account (*toAccount*)
// MAGIC 1. The transaction hash associated with the transaction if there is one (*TID*)
// MAGIC 1. The transfer type: 'MINED', 'FEE', 'TRANSACTION', 'TRANSFER', 'GENESIS', 'CREATION', 'SELFDESTRUCT', 'UNCLE' (*transferType*)
// MAGIC 1. The value sent in the transfer (*value*)
// MAGIC 
// MAGIC The following definitions were found by querying the table and analysis what type of results were associated with each transferType. Examples of this will be shown below.
// MAGIC 
// MAGIC **GENESIS** events only occur in the first block and this contains all the events where the initial accounts were created, and the initial investors were given Ethers. This is one of the only type of events were Ethers not originally in the system as added to the system. 
// MAGIC 
// MAGIC **MINED** events occur when the block chain rewards miners for successfully mining a block. This is also another instance when Ethers are added to the system.
// MAGIC 
// MAGIC **UNCLE** events occur when a miner almost successfully mines a block. They are also partially rewarded. This is also another instance when Ethers are added to the system.
// MAGIC 
// MAGIC **CREATION** events occur when an account creates a smart contract address to execute a contract.
// MAGIC 
// MAGIC **SELFDESTRUCT** events occur because a contract was constructed with insufficient gas funds. Users must pay the computational cost of the contract when they construct it but if there isn't enough funds to run the contract. The contract is destroyed and any left over money is returned to the original creater.
// MAGIC 
// MAGIC **TRANSACTION** events include events where a user sends money to another user, events where a user sends money to a contract to initiate it, and when a contract pays a user.
// MAGIC 
// MAGIC **TRANSFER** tend to include events where contracts send money to other contrats and vice versa. There are levels associated with this and they are stored in depth. The activites of contracts seem to be a bit convoluted. When a person sends money to a contract, the contract can pass the money between contracts randomly prior to giving the money to another person. This could be hide the identity of those involved. Some contracts also don't pay out to anyone, as in the contract was not specfically designed to transfer funds. 

// COMMAND ----------

// Define Transfers table class
case class Transfers(                     // column index
  blockNum: Int,                       // 1
  depth: Int,                             // 2
  fromAccount: String,                           // 3
  timeStamp: java.sql.Timestamp,          // 5
  toAccount: String,                             // 6
  TID: String,                // 8
  transferType: String,                           // 10
  value: Double                         // 11
  )

// transfers cleaning function
def getTransfersCleaned(row: Array[String]):Transfers = {
  return Transfers(
    row(1).toIntOrElse(),
    row(2).toIntOrElse(),
    row(3),
    fixDateFormat(row(5)).toDateOrElse(),
    row(6),
    row(8),
    row(10),
    weiToEther(hexToFloat(row(11)))
  )
}

// COMMAND ----------

// Split and clean
val transfersDF = transfersRDD.map(r => getTransfersCleaned(r.toString().stripPrefix("[").stripSuffix("]").split(","))).toDF()

// Create and cache table
transfersDF.createOrReplaceTempView("transfers")
//sqlContext.cacheTable("transfers")
sqlContext.sql("cache table transfers")
display(sqlContext.sql("select * from transfers"))

// COMMAND ----------

// MAGIC %md 
// MAGIC # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// COMMAND ----------

// MAGIC %md
// MAGIC ## 4. Data Exploration
// MAGIC 
// MAGIC The purpose of this section is to understand the information that is in the provided database. SQL queries are used to investigate the system. Also the limits of the data are found to see if the results at the end make sense and fit inside the system. The implications of contracts was also investigated.
// MAGIC 
// MAGIC The first transaction occured on August 7, 2015, and this data ends at July 27, 2016. The GENESIS events were stamped with January 1st, 1970. This lines up with when Ethereum was launched (July 30, 2015).
// MAGIC 
// MAGIC The highest block number in the system is 1,961,326, and there are 7,557,615 total unique transactions in the database.
// MAGIC 
// MAGIC Ethers are only added to the Ethereum system when one of the following events occur: GENESIS (initial investors getting Ethers), MINED (miners getting rewarded), and UNCLE (miners that were close to successfully mining a block getting rewarded). They are paid from a special account (0x0000000000000000000000000000000000000000). So far a total of 78.5 million Ethers are circulated in the Ethereum system.
// MAGIC 
// MAGIC There are a total of 383,305 unique addresses that could belong to either a user or a contract. Users can be further subdivided into uncles, miners, and regular users. Contracts can be subdivided into DAO contracts, the Ethereum countract, and regular contracts.
// MAGIC 
// MAGIC 
// MAGIC Ref: https://en.wikipedia.org/wiki/Ethereum

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Find general stats about the system
// MAGIC select cast(min(timeStamp) as date) as firstDate, cast(max(timeStamp) as date) as lastDate
// MAGIC from transfers
// MAGIC where blockNum > 0

// COMMAND ----------

// Count number of distinct transactions 
  // Note: subtract one to ignore null count
val numUniqueTransactions = transfersDF.select(transfersDF("TID")).distinct.count - 1

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Find out how much Ethereum has been introduced in the system so far and who paid them
// MAGIC select fromAccount as etheremAccount, sum(value) as totalEthersInSystem
// MAGIC from transfers
// MAGIC where transferType == 'GENESIS' or transferType == 'UNCLE' or transferType == 'MINED'
// MAGIC group by fromAccount

// COMMAND ----------

val numUniqueAccounts = transfersDF.select(transfersDF("fromAccount")).union(transfersDF.select(transfersDF("toAccount"))).distinct.count

// COMMAND ----------

// MAGIC %md
// MAGIC ### 4b. Investigate specific transferTypes
// MAGIC 
// MAGIC 
// MAGIC #### Transactions
// MAGIC These are typical transactions but there are also some transactions that occur between users and contracts. This is the second most common type of event.
// MAGIC 
// MAGIC #### Fee
// MAGIC Fees are paid to the person who mined the block for adding their transaction. One transaction can have multiple fees paid to different miners because it can get split up across a couple blocks (Ethereum blocks are mined much quicker than Bitcoin -- around 6 mins). The fee can range from 0 to 14.37 Ethers. This is the most common type of event.
// MAGIC 
// MAGIC #### Genesis
// MAGIC Genesis events only occured during the first block and in these events the Ethereum main account would send money to a user. This is probably the initial investors getting paid some Ethers at the start of Ethereum. 
// MAGIC 
// MAGIC #### Mined
// MAGIC The mining reward averages around 5 Ethers per block usually, and there are 1,222,510 blocks where miners have received rewards. Oddly this is not equivalent to the number of total blocks in the system. It could be that prior miners joined the network it was mined by the people that created Ethereum. Mined events have one unique sender (The Ethereum main account).
// MAGIC 
// MAGIC #### Uncle
// MAGIC The uncle reward averages around 1.25 to 4.4 Ethers and there are 101,076 uncles thus far so most blocks don't have uncles. They also had one unique sender (The Ethereum main account).
// MAGIC 
// MAGIC #### Creation
// MAGIC Creation and Transfer events have values that range anywhere from 0 to infinity. During creation events, users usually create contracts but this is not the case when the depth of the event is greater than 0. When this occurs, a conract address (found by searching addresses on etherscan.io) pays the Ethereum account some Ethers.
// MAGIC 
// MAGIC #### Transfer 
// MAGIC Transfer events have values that range anywhere from 0 to infinity and at least one of the addresses involved is a contract address. Contract addresses can pass money between each other. Get money from a user, or send money to a user. 
// MAGIC 
// MAGIC #### SelfDestruct
// MAGIC These occur when a contract is created with insufficient funds to cover the gas. Users, when creating contracts, need to pay enough Ethers to cover the computational cost of the contract at the current gas price. Failure to supply enough funds result in the contract self destructing and the funds being returned the user that created the contract. 

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Investigate MINED events
// MAGIC select transferType, count(value) as occurances, count(distinct fromAccount) as numberOfUniqueSenders, max(value) as maxValue, min(value) as minValue
// MAGIC from transfers
// MAGIC group by transferType
// MAGIC order by occurances desc

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Investigate CREATION events
// MAGIC select depth, max(value), min(value), count(value), count(distinct fromAccount), count(distinct toAccount) from transfers where transferType == 'CREATION'
// MAGIC group by depth
// MAGIC order by depth asc

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Investigate CREATION events
// MAGIC select toAccount, count(*) as countEvents from transfers where transferType == 'CREATION' and depth > 0
// MAGIC group by toAccount

// COMMAND ----------

// MAGIC %md
// MAGIC ### 4c. Investigate contract addresses behaviour
// MAGIC 
// MAGIC In this section, some contract account's activities were investigated.
// MAGIC 
// MAGIC From 0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae's activities, it is seen how convoluted the system can get in transferring money from one person to another. The same contract spans several blocks and uses several transaction hashes.
// MAGIC 
// MAGIC From 0x9741ccd8f741f91e9cfde267cba49cb1bb834cf1's activities, it is seen the contract being created during a CREATION event, and then destroyed during a SELFDESTRUCT event.
// MAGIC 
// MAGIC From 0xba95c0e91e06cf60197f247f4bcff03fc32300db's activities, a contract is created with 0 Ethers and that contract does nothing else.
// MAGIC 
// MAGIC From 0x1719ebdc0646fbf268b4feeb57af842ffbe4d45f's activites, a user creates a contract, that contract passes money to other various contracts. Some of the contracts that the contract gets passed to might belong to other users. They could be getting paid through those contracts.

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Contract Address 0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae's activities
// MAGIC select * from transfers
// MAGIC where fromAccount == '0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae' 
// MAGIC    or toAccount == '0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae'
// MAGIC order by blockNum

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Contract Address 0x9741ccd8f741f91e9cfde267cba49cb1bb834cf1's activities
// MAGIC select * from transfers
// MAGIC where fromAccount == '0x9741ccd8f741f91e9cfde267cba49cb1bb834cf1'
// MAGIC    or toAccount == '0x9741ccd8f741f91e9cfde267cba49cb1bb834cf1'
// MAGIC order by blockNum

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Contract Address 0xba95c0e91e06cf60197f247f4bcff03fc32300db's activities
// MAGIC select * from transfers
// MAGIC where fromAccount == '0xba95c0e91e06cf60197f247f4bcff03fc32300db'
// MAGIC    or toAccount == '0xba95c0e91e06cf60197f247f4bcff03fc32300db'
// MAGIC order by blockNum

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Contract Address 0x1719ebdc0646fbf268b4feeb57af842ffbe4d45f's activities
// MAGIC select * from transfers
// MAGIC where fromAccount == '0x1719ebdc0646fbf268b4feeb57af842ffbe4d45f'
// MAGIC    or toAccount == '0x1719ebdc0646fbf268b4feeb57af842ffbe4d45f'
// MAGIC order by blockNum

// COMMAND ----------

// MAGIC %md 
// MAGIC # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5. Data Preparation
// MAGIC 
// MAGIC During this step, the types of users are going to be identified and added to the transfers table.

// COMMAND ----------

// MAGIC %md 
// MAGIC ### 5a. Identify user types
// MAGIC 
// MAGIC During this step, the types of users are going to be identified and added to the transfers table. This step is important to identify user-to-user transactions.
// MAGIC 
// MAGIC The following precedence below is followed so the minimum guess is returned.
// MAGIC 
// MAGIC {0:'ethereum', 1:'DAOcontract', 2:'contract', 3:'miner', 4:'uncle', 5:'user'}
// MAGIC 
// MAGIC Ranks will be appeneded below, and then the lowest rank will be chosen. Then this heirarchy table will be joined to identify users by text instead of ranks.

// COMMAND ----------

// Schema for Heirarchy Table
case class Heirarchy(userType: String, rank: Int)

// Create dataframe
val heirarchyDF = Seq(
  Heirarchy("ethereum",0),
  Heirarchy("dao",1),
  Heirarchy("contract",2),
  Heirarchy("miner",3),
  Heirarchy("uncle",4),
  Heirarchy("user",5)
).toDF()

// Create Table
heirarchyDF.createOrReplaceTempView("heirarchy") // Create SQL table
//sqlContext.cacheTable("heirarchy")
display(sqlContext.sql("select * from heirarchy"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Identify the main Ethereum address
// MAGIC Here the main Ethereum account that adds Ethers to the system is identified. It is the account used to pay miners, uncles, and those in the genesis block.
// MAGIC It can be identified in a number of ways. It was just grabbed as the unique sender in the MINER transferType case.

// COMMAND ----------

// Grab the Ethereum account
val tempEthAccDF = sqlContext.sql("select first(fromAccount) as address, 0 as rank from transfers where transferType == 'MINED'")

tempEthAccDF.createOrReplaceTempView("tempEthAcc")
//sqlContext.cacheTable("tempEthAcc")
display(sqlContext.sql("select * from tempEthAcc"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Identify the DAO contract addresses
// MAGIC Several contract addresses were created to remedy the DAO theft. They were invovled in several refund transactions with users. Some of these addresses can be identified from the DAO table. They can be found in the *address* and *extraBalanceAccount* columns.

// COMMAND ----------

val DAOcontractAccDF = sqlContext.sql("select address, 1 as rank from DAO")


DAOcontractAccDF.createOrReplaceTempView("tempDAOContractAcc")
//sqlContext.cacheTable("tempDAOContractAcc")
display(sqlContext.sql("select * from tempDAOContractAcc"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Identify contract addresses
// MAGIC Here contract addresses were identified using the following methods:
// MAGIC 1. In the transactions table, some contract addresses were explicitly listed in *contractAddress* column
// MAGIC 2. In the transfers table, two types of events link back to contract addresses:
// MAGIC 
// MAGIC   a. In a *CREATION* event w depth = 0, a user sends money to a *contractAddress* to initiate it: The contract can be identified in the *toAccount* column
// MAGIC   
// MAGIC   b. In a *CREATION* event w depth > 0, a contract address sends money to the ethereum account: The contract can be identified in the *fromAccount* column
// MAGIC   
// MAGIC   c. In a *SELFDESTRUCT* event, a contract is terminated and gas/money are refunded to a user: The contract can be identified in the *fromAccount* column

// COMMAND ----------

// Grab the contract addresses via the above method
val contractAccDF = sqlContext.sql("""
select address, 2 as rank from transactions
union
select toAccount as address, 2 as rank from transfers
where transferType == 'CREATION' and depth == 0
union
select fromAccount as address, 2 as rank from transfers
where transferType == 'CREATION' and depth > 0
union
select fromAccount as address, 2 as rank from transfers
where transferType == 'SELFDESTRUCT'
"""
)

// Create and cache table
contractAccDF.createOrReplaceTempView("tempContractAcc")
//sqlContext.cacheTable("tempContractAcc")
display(sqlContext.sql("select * from tempContractAcc"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Identify miner addresses
// MAGIC Here the miner address were identified using the following methods:
// MAGIC 1. In the blocks table, some miner addresses are explicitly listed in the *miner* column
// MAGIC 2. In the transfers table, miners are sent money from users during a *FEE* event. Fees accompany transaction/contract inititations and are paid to the person that mined the block. They can be identified in the *toAccount* column

// COMMAND ----------

// Grab the miners using the above method
val minerAccDF = sqlContext.sql("""
select address, 3 as rank from blocks
union
select toAccount as address, 3 as rank from transfers
where transferType == 'FEE' and toAccount is not null 
"""
)

// Create and cache table
minerAccDF.createOrReplaceTempView("tempMinerAcc")
//sqlContext.cacheTable("tempMinerAcc")
display(sqlContext.sql("select * from tempMinerAcc"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Identify uncle addresses
// MAGIC 
// MAGIC Here the uncle address were identified using the following methods:
// MAGIC 1. In the transfers table, uncles are paid some money for almost mining a block. They are paid through an UNCLE event as the recepient of some of the block reward. They can be identified in the *toAccount* column

// COMMAND ----------

// Grab the uncle's addresses
val uncleAccDF = sqlContext.sql("""
select distinct(toAccount) as address, 4 as rank from transfers
where transferType == 'UNCLE'
"""
)

// Create and cache table
uncleAccDF.createOrReplaceTempView("tempUncleAcc")
//sqlContext.cacheTable("tempUncleAcc")
display(sqlContext.sql("select * from tempUncleAcc"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Identifying user addresses
// MAGIC 
// MAGIC Here the users address were identified using the following methods:
// MAGIC 
// MAGIC In the transfers table,
// MAGIC 
// MAGIC 1. Only users can create contracts so in CREATION events they can be identified in the *fromAccount* column when depth is equal to 0
// MAGIC 2. Users typically receive money from self-destructed contract (return to sender function), so they can be identified in the *toAccount* column in a *SELFDESTRUCT* event
// MAGIC 3. Users pay fees to miners during a *FEE* event so they can identified in the *fromAccount* column

// COMMAND ----------

// Grab list of user addresses
val userAccDF = sqlContext.sql("""
select fromAccount as address, 5 as userType from transfers
where transferType == 'CREATION' and depth == 0
union
select toAccount as address, 5 as userType from transfers
where transferType == 'SELFDESTRUCT'
union
select fromAccount as address, 5 as userType from transfers
where transferType == 'FEE'
"""
)

// Create and cache table
userAccDF.createOrReplaceTempView("tempUserAcc")
//sqlContext.cacheTable("tempUserAcc")
display(sqlContext.sql("select * from tempUserAcc"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Combine all user types
// MAGIC 
// MAGIC Keep the lowest type. The precedence was set since a user can be an uncle and a miner, and an uncle could be a miner, and so on. The following precedence is followed: ethereum > DAOcontract > contract > miner > uncle > user. Join the heirarchy table to get types instead of rank.

// COMMAND ----------

// Combine all the above userType tables
val userTypesDF = sqlContext.sql("""
select address, userType from (
  (select address, min(rank) as rank 
   from (
     select * from tempEthAcc
     union
     select * from tempDAOContractAcc
     union
     select * from tempContractAcc
     union
     select * from tempMinerAcc
     union
     select * from tempUncleAcc
     union
     select * from tempUserAcc
   )
   group by address ) U
   join
   (select userType, rank from heirarchy) h
   on U.rank = h.rank
)
""")

// Save and load from file
userTypesDF.write.mode("overwrite").parquet(absResultsDir + "userTypes")
val newUserTypesDF = sqlContext.read.parquet(absResultsDir + "userTypes")

// Create and cache table
newUserTypesDF.createOrReplaceTempView("userTypes")
//sqlContext.cacheTable("userTypes")
sqlContext.sql("cache table userTypes")
display(sqlContext.sql("select * from userTypes"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Get stats on the identified users
// MAGIC 
// MAGIC The following set of queries is to get information on the identified set of user types and to see how much of the database is now labeled.
// MAGIC 
// MAGIC There are a total of 380,872 identified user types in the database and this is about 88% of all the addresses in the database. The main majority of the identified addresses belonged to users. Contract addresses were much less significant than users.

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Find percentage of Identified Users
// MAGIC select (select count(*) from userTypes) as numUsersIdentified, 
// MAGIC        (select count(*) from userTypes)/count(*) * 100 
// MAGIC           as percentageOfUserIdentified
// MAGIC from (
// MAGIC   select fromAccount from transfers
// MAGIC   union
// MAGIC   select toAccount from transfers
// MAGIC )

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Count type of users identified
// MAGIC select userType, count(*) as count from userTypes
// MAGIC group by userType
// MAGIC order by count

// COMMAND ----------

// MAGIC %md
// MAGIC #### Drop tables that are no longer required
// MAGIC 
// MAGIC Drop some tables to clear up space in memory.

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Drop temporary tables below if they exist
// MAGIC drop table if exists heirarchy;  
// MAGIC drop table if exists blocks; 
// MAGIC drop table if exists transactions; 
// MAGIC drop table if exists dao; 
// MAGIC drop table if exists tempethacc; 
// MAGIC drop table if exists tempdaocontractacc; 
// MAGIC drop table if exists tempcontractacc; 
// MAGIC drop table if exists tempmineracc; 
// MAGIC drop table if exists tempuncleacc; 
// MAGIC drop table if exists tempuseracc; 

// COMMAND ----------

// MAGIC %md
// MAGIC ### 5b. Update transfers table with user types
// MAGIC 
// MAGIC This step is add all the usertype information to the transfers table.

// COMMAND ----------

// MAGIC %md
// MAGIC #### Add fromType to transfers table
// MAGIC 
// MAGIC This section of code joins the *user_types* table and *transfers* table to create a column called *fromType* to identify *fromAccount* users.

// COMMAND ----------

// Join userTypes to fromAccount on transfers table
val transfers2DF = sqlContext.sql("""
select cast(timeStamp as date) as date, blockNum, TID, transferType, userType as fromType, fromAccount, toAccount, value from 
(
  (select * from transfers) t
  left join
  (select * from userTypes) u
  on t.fromAccount = u.address
)
""")

// Create new table with fromType
transfers2DF.createOrReplaceTempView("transfers2")

// COMMAND ----------

// MAGIC %md
// MAGIC #### Add toType to transfers table
// MAGIC 
// MAGIC This section of code joins the *user_types* table and *transfers* table to create a column called *toType* to identify *toAccount* users.

// COMMAND ----------

// Join userTypes to toAccount on transfers table
val transfers3DF = sqlContext.sql("""
select date, blockNum, TID, transferType, fromType, fromAccount, toAccount, userType as toType, value from 
(
  (select * from transfers2) t
  left join
  (select * from userTypes) u
  on t.toAccount = u.address
)
""")

// Write to file
transfers3DF.write.mode("overwrite").parquet(absResultsDir + "transfersWTypes")

// COMMAND ----------

// Read in and create and cache table
val transfersWTypesDF = sqlContext.read.parquet(absResultsDir + "transfersWTypes")
transfersWTypesDF.createOrReplaceTempView("transfersWTypes")
sqlContext.sql("uncache table userTypes")
sqlContext.sql("cache table transfersWTypes")
//sqlContext.cacheTable("transfersWTypes")

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Show transfers table with both fromType and toType
// MAGIC select * from transfersWTypes

// COMMAND ----------

// MAGIC %md
// MAGIC ### 5c. Extract the subgraph of relevant users
// MAGIC 
// MAGIC It is hard to accurately and reliably join users who exchange funds via contract addresses as contract addresses can be re-used and some contracts aren't meant for exchanging money. There is also the large outlier of the DAO contract addresses which would be an outlier because it does not follow the normal operation of the network. For these reasons, the contract accounts will be dropped for the dataset and the remaining accounts will be analyzed. Miner fees will also be ignored because users don't choose who gets the miner fees and a miners will collect fees regardless. The percentage of data loss will also be examined. 

// COMMAND ----------

// MAGIC %md
// MAGIC #### Get list of transaction hashes without contracts involved
// MAGIC The following transfer types: TRANSFER, SELFDESTRUCT, and CREATION events, are used by contract addresses. To find the list of TID that doesn't have these the types in them. 

// COMMAND ----------

// Get list of TID that don't contain TRANSFER, SELFDESTRUCT, or CREATION events
val TIDDF = sqlContext.sql("""
select distinct(TID) as TID from transfersWTypes
minus
select distinct(TID) from transfersWTypes 
where transferType == 'TRANSFER' or transferType == 'SELFDESTRUCT' 
   or transferType == 'CREATION'
""")

// Create, cache, and display table
TIDDF.createOrReplaceTempView("TID_wocontracts")
//sqlContext.cacheTable("TID_wocontracts")
sqlContext.sql("cache table TID_wocontracts")
display(sqlContext.sql("select * from TID_wocontracts"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Update transfers table
// MAGIC 
// MAGIC Remove all events that don't meet these conditions:
// MAGIC - TID must be in the TID_wocontracts table
// MAGIC - TID must not be null
// MAGIC - Value exchanged must be real (>0 and <infinity)
// MAGIC - Involve real entities (No ethereum, dao, or contract accounts)
// MAGIC - TransferType must be TRANSACTION

// COMMAND ----------

// Remove from transfers any event that doesn't meet the above conditions
val transfers4DF = sqlContext.sql("""
select * from transfersWTypes
where TID in (select * from TID_wocontracts)
  and transferType != 'FEE'
  and value > pow(10,-17)
  and value < pow(10,17)
  and toType != 'contract'
  and toType != 'daocontract'
  and toType != 'ethereum'
  and TID != 'null'
""")

// Replace view and show table
transfers4DF.createOrReplaceTempView("transfersWTypes")
display(sqlContext.sql("select * from transfersWTypes"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Combine transactions that are split over blocks
// MAGIC 
// MAGIC Some transactions can be split across several blocks. This step is to combine those into a single transaction. The minimum date will be taken, the value will be summed, and the blocknum, transferType, and userTypes will be dropped.

// COMMAND ----------

// Sanity Check: Check if there is more than one distinct from account or to account involved in a transaction
sqlContext.sql("""
select TID, count(distinct fromAccount), count(distinct toAccount) from transfersWTypes
group by TID
having count(distinct fromAccount) > 1 or count(distinct toAccount) > 1
""")

// COMMAND ----------

// Take the first date, fromAccount, toAccount, and sum the total value and put that into a table
val transfers5DF = sqlContext.sql("""
select min(date) as date, TID, first(fromAccount) as fromAccount, first(toAccount) as toAccount, sum(value) as value 
from transfersWTypes
group by TID
""")

// Write this new table to file
transfers5DF.write.mode("overwrite").parquet(absResultsDir + "subgraph")

// COMMAND ----------

// Read in the subgraph table, create a view, and cache it
val subGraphDF = sqlContext.read.parquet(absResultsDir + "subgraph")
subGraphDF.createOrReplaceTempView("subgraph")

sqlContext.sql("uncache table transfersWTypes")
sqlContext.sql("drop table if exists transfersWTypes")
sqlContext.sql("uncache table TID_wocontracts")
sqlContext.sql("drop table if exists TID_wocontracts")
sqlContext.sql("uncache table transfers")
sqlContext.sql("drop table if exists transfers")

sqlContext.sql("cache table subgraph")
//sqlContext.cacheTable("subgraph")

// COMMAND ----------

// MAGIC %sql
// MAGIC -- show subgraph table
// MAGIC select * from subgraph

// COMMAND ----------

// MAGIC %md
// MAGIC #### Find loss in information
// MAGIC 
// MAGIC Excluding all the contract related interactions and accounts still leaves us with 78.88% of the transactions and 73.83% of the accounts.

// COMMAND ----------

val percentageTotalUniqueTIDS = subGraphDF.select(subGraphDF("TID")).distinct.count.toDouble / numUniqueTransactions * 100.0

// COMMAND ----------

val percentageOfTotalUniqueAccounts = subGraphDF.select(subGraphDF("fromAccount")).union(subGraphDF.select(subGraphDF("toAccount"))).distinct.count.toDouble / numUniqueAccounts * 100.0

// COMMAND ----------

// MAGIC %md
// MAGIC #### Drop unnecessary tables

// COMMAND ----------

// MAGIC %sql
// MAGIC -- drop tables if they exist to clear some memory
// MAGIC -- drop table if exists transfers;
// MAGIC -- drop table if exists tid_wocontracts;
// MAGIC -- drop table if exists transferswtypes;

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5d. Extract both cores
// MAGIC 
// MAGIC The paper defines two cores: a Long-term (LT) core and a All-Users (AU) core. These cores are what they call the active users of the network. This is the definitions they used:
// MAGIC 
// MAGIC All-Users Core
// MAGIC - top 2000 most active users
// MAGIC 
// MAGIC Long-Term Core
// MAGIC - involved in 100+ individual transactions
// MAGIC - active over 600+ consecutive days
// MAGIC 
// MAGIC This definition cannot be used directly with our database because we have only a years worth of data. The days will be halved to 300 days to maintain the same ratio per year provided.

// COMMAND ----------

// MAGIC %md
// MAGIC #### Get account stats
// MAGIC 
// MAGIC Get the number of transactions each account was involved in, the first date they were active in the system, the last date, the number of days they were active, and the amount of Ethers they have sent over time.

// COMMAND ----------

// Get aggregate stats on the fromAccounts - the number of transctions they were involved in, the first date and last date they appeared in the system, the number of days elapsed, and the amount of ethers sent by them.
val subgraphStatsDF = sqlContext.sql("""
select fromAccount as account, count(distinct TID) as numTransactions, min(date) as firstDate, max(date) as lastDate, datediff(max(date),min(date)) as daysElapsed, sum(value) as ethersSent from subgraph
group by fromAccount
sort by daysElapsed desc
""")

// Write to file
subgraphStatsDF.write.mode("overwrite").parquet(absResultsDir + "subgraphStats")

// COMMAND ----------

// Read the table back in, create a view and cache it
val statsDF = sqlContext.read.parquet(absResultsDir + "subgraphStats")
statsDF.createOrReplaceTempView("subgraphstats")
//sqlContext.cacheTable("subgraphstats")
sqlContext.sql("cache table subgraphstats")

// COMMAND ----------

// MAGIC %sql
// MAGIC -- show table
// MAGIC select * from subgraphstats

// COMMAND ----------

// MAGIC %md
// MAGIC #### Get LT core
// MAGIC Here the LT core will be extracted from the subgraph created earlier (transfers3). The LT core will consist of users who participated in greater than 100 transactions and were active for greater than 250 days. The core with 300 days failed to represent the tail end of market change later in the year and captured mainly the first couple months. The core was expanded to include all users who were active for > 250 days to see if this performance can be improved. 

// COMMAND ----------

// Find the set of users that are in the LTcore
val LTusersDF = sqlContext.sql("""
select account from subgraphstats
where numTransactions > 100
  and daysElapsed > 250
""")

// Create and cache view
LTusersDF.createOrReplaceTempView("LTusers")
display(sqlContext.sql("select * from LTusers"))

// COMMAND ----------

// Grab the above users from the subgraph
val LTcoreDF = sqlContext.sql("""
select * from subgraph
where fromAccount in (select * from LTusers)
  and toAccount in (select * from LTusers)
""")

// Create and cache view
LTcoreDF.createOrReplaceTempView("LTcore")
sqlContext.sql("cache table LTcore")
display(sqlContext.sql("select * from LTcore"))

// COMMAND ----------

// MAGIC %md
// MAGIC **LT core Stats**
// MAGIC 
// MAGIC The LT core consists of 869 nodes (users) and 3309 unique edges. These users participated in 1,156,946 transactions.

// COMMAND ----------

// Get number of transactions and nodes
val numLTtransactions = LTcoreDF.count
val numLTNodes =  LTcoreDF.select(LTcoreDF("fromAccount")).union(LTcoreDF.select(LTcoreDF("toAccount"))).distinct.count

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Get number of edges
// MAGIC select count(*) as numLTEdges from (
// MAGIC select fromAccount, toAccount from LTcore
// MAGIC group by fromAccount, toAccount
// MAGIC order by fromAccount, toAccount)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Get AU core
// MAGIC Here the AU core will be extracted from the subgraph created earlier (transfers3). The AU core will consist of top 2000 users. 

// COMMAND ----------

// Find the set of users that are in the AUcore
val AUusersDF = sqlContext.sql("""
select account
  from ( select
    account,
    row_number() OVER (ORDER BY numTransactions DESC) AS rank
    from subgraphstats
  ) tmp
  where rank <= 1000
""")

// Create and cache view
AUusersDF.createOrReplaceTempView("AUusers")
display(sqlContext.sql("select * from AUusers"))

// COMMAND ----------

// Grab the users in the AUusers table from the subgraph table
val AUcoreDF = sqlContext.sql("""
select * from subgraph
where fromAccount in (select * from AUusers)
  and toAccount in (select * from AUusers)
""")

// Create and cache view
AUcoreDF.createOrReplaceTempView("AUcore")
sqlContext.sql("cache table AUcore")
display(sqlContext.sql("select * from AUcore"))

// COMMAND ----------

// MAGIC %md
// MAGIC **AU core Stats**
// MAGIC 
// MAGIC The AU core consists of 986 nodes (users) and 5,050 unique edges (transactions). These users participated in 1,603,324 transactions.

// COMMAND ----------

// Get number of transactions and nodes
val numAUtransactions = AUcoreDF.count
val numAUNodes =  AUcoreDF.select(AUcoreDF("fromAccount")).union(AUcoreDF.select(AUcoreDF("toAccount"))).distinct.count

// COMMAND ----------

// MAGIC %sql
// MAGIC select count(*) as numAUEdges from (
// MAGIC select fromAccount, toAccount from AUcore
// MAGIC group by fromAccount, toAccount
// MAGIC order by fromAccount, toAccount)

// COMMAND ----------

// MAGIC %md
// MAGIC ### 5e. Assign IDs to users
// MAGIC This step is to assign integers to the users in the cores. This is so they can used as row and column elements in the adjaceny matrix later on.

// COMMAND ----------

// Assign an integer to each user based on row number
val coreUsersDF = sqlContext.sql("""
select * from (
  select *,
    row_number() over(ORDER by address DESC) -1 as idx
    from (
      select fromAccount as address from LTcore
      union
      select toAccount as address from LTcore
      union
      select fromAccount as address from AUcore
      union
      select toAccount as address from AUcore
    )
)
""")

// Write mapping to file
coreUsersDF.write.mode("overwrite").parquet(absResultsDir + "usersIndex")

// COMMAND ----------

// Read back in and create and cache table
val usersIndexDF = sqlContext.read.parquet(absResultsDir + "usersIndex")
usersIndexDF.createOrReplaceTempView("usersIndex")
//sqlContext.cacheTable("usersIndex")
sqlContext.sql("cache table usersIndex")

// COMMAND ----------

// MAGIC %sql
// MAGIC -- Show table
// MAGIC select * from usersIndex

// COMMAND ----------

// MAGIC %md
// MAGIC #### Assign ids to LTcore

// COMMAND ----------

// Assigned id to fromAccount
val LTcoreDF = sqlContext.sql("""
select date, TID, idx as fromID, toAccount, value
from (
  (select * from LTcore) LT
  left join
  (select * from usersIndex) U
  on LT.fromAccount = U.address
)
""")

// Create and cache table
LTcoreDF.createOrReplaceTempView("LTcore")
//sqlContext.cacheTable("LTcore")
//sqlContext.sql("cache table LTcore")

// COMMAND ----------

// Assigned id to toAccount
val LTcoreDF = sqlContext.sql("""
select date, TID, fromID, idx as toID, value
from (
  (select * from LTcore) LT
  left join
  (select * from usersIndex) U
  on LT.toAccount = U.address
)
""")

// Create and cache table
LTcoreDF.createOrReplaceTempView("LTcore")
//sqlContext.cacheTable("LTcore")
display(sqlContext.sql("select * from LTcore"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Assign ids to AUcore

// COMMAND ----------

// Assign id to fromAccount
val AUcoreDF = sqlContext.sql("""
select date, TID, idx as fromID, toAccount, value
from (
  (select * from AUcore) LT
  left join
  (select * from usersIndex) U
  on LT.fromAccount = U.address
)
""")

// Create and cache table
AUcoreDF.createOrReplaceTempView("AUcore")
//sqlContext.cacheTable("AUcore")

// COMMAND ----------

// Assign id to toAccount
val AUcoreDF = sqlContext.sql("""
select date, TID, fromID, idx as toID, value
from (
  (select * from AUcore) LT
  left join
  (select * from usersIndex) U
  on LT.toAccount = U.address
)
""")

// Create and cache table
AUcoreDF.createOrReplaceTempView("AUcore")
//sqlContext.cacheTable("AUcore")
display(sqlContext.sql("select * from AUcore"))

// COMMAND ----------

// MAGIC %md
// MAGIC ### 5f. Get number of daily transactions
// MAGIC 
// MAGIC For the weighted adjacency matrix that is to be constructed, the weight is the number of transactions that occurred between two users. 

// COMMAND ----------

// MAGIC %md
// MAGIC #### LT core
// MAGIC Group by date, fromID, and toID and count transactions to get number of transactions between users daily.

// COMMAND ----------

// Group by date, fromID, and toID and count rows to get number of transactions 
val LTcoreDF = sqlContext.sql("""
select date, fromID, toID, count(*) as numTransactions from LTcore
group by date, fromID, toID
order by date, numTransactions desc, fromID, toID
""")

// Show and save table
display(LTcoreDF)
LTcoreDF.write.mode("overwrite").parquet(absResultsDir + "LTcore") // Write to file


// COMMAND ----------

// MAGIC %md
// MAGIC #### AU core
// MAGIC Group by date, fromID, and toID and count transactions to get number of transactions between users daily.

// COMMAND ----------

// Group by date, fromID, and toID and count rows to get number of transactions 
val AUcoreDF = sqlContext.sql("""
select date, fromID, toID, count(*) as numTransactions from AUcore
group by date, fromID, toID
order by date, numTransactions desc, fromID, toID
""")

// Show and save table
display(AUcoreDF)
AUcoreDF.write.mode("overwrite").parquet(absResultsDir + "AUcore") // Write to file

// COMMAND ----------

// MAGIC %md
// MAGIC #### Clear the cache and only leave the cores and the Ethereumtousd graph

// COMMAND ----------

sqlContext.sql("clear cache")
// val AUcoreDF = sqlContext.read.parquet(absResultsDir + "AUcore")
// AUcoreDF.createOrReplaceTempView("AUcore")
// sqlContext.sql("AUcore")
// val LTcoreDF = sqlContext.read.parquet(absResultsDir + "LTcore")
// LTcoreDF.createOrReplaceTempView("LTcore")
// sqlContext.cacheTable("LTcore")
val ethToUSDDF = sqlContext.read.parquet(absResultsDir + "ethToUSD")
ethToUSDDF.createOrReplaceTempView("ethToUSD")
sqlContext.cacheTable("ethToUSD")

// COMMAND ----------

// MAGIC %md 
// MAGIC # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// COMMAND ----------

// MAGIC %md
// MAGIC ## 6. Weighted Adjacency Matrix
// MAGIC 
// MAGIC Here the weighted adjacency matrix will be constructed daily and converted into a vector. Then all the daily weighted adjacency vectors will be concatenated to create a matrix. This matrix is what will be passed through PCA.

// COMMAND ----------

// MAGIC %md
// MAGIC #### Convert to Pandas dataframe

// COMMAND ----------

// MAGIC %python
// MAGIC # Convert into a pandas dataframe from spark - to use with NetworkX
// MAGIC LT_core = sqlContext.read.parquet(absResultsDir + "AUcore").toPandas()
// MAGIC AU_core = sqlContext.read.parquet(absResultsDir + "LTcore").toPandas()
// MAGIC #LT_core = sqlContext.sql("select * from LTcore").toPandas()
// MAGIC #AU_core = sqlContext.sql("select * from AUcore").toPandas()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Get core details
// MAGIC 
// MAGIC These steps are to find out how large the final matrix needs to be. The maximum nodes in the database are found, and the date range is also found.

// COMMAND ----------

// MAGIC %python
// MAGIC # Find the max node ID in the cores
// MAGIC maxID = max(LT_core['fromID'].max(),
// MAGIC             LT_core['toID'].max(),
// MAGIC             AU_core['fromID'].max(),
// MAGIC             AU_core['toID'].max())
// MAGIC 
// MAGIC # Find the first date in the cores
// MAGIC first_date = min(LT_core['date'].min(), AU_core['date'].min())
// MAGIC 
// MAGIC # Find the last date in the cores
// MAGIC last_date = max(LT_core['date'].max(), AU_core['date'].max())
// MAGIC 
// MAGIC # Find the number of days in the cores
// MAGIC num_days = (last_date-first_date).days+1
// MAGIC 
// MAGIC # Find the daily vector size
// MAGIC daily_vect_sz = (maxID+1)*(maxID+1)
// MAGIC 
// MAGIC print ("There are a total of {} unique users in the cores.".format(maxID))
// MAGIC print ("The data spans from {} to {}.".format(first_date,last_date))
// MAGIC print ("This is a total of {} days.".format(num_days-1))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Initialize matrix to store values
// MAGIC 
// MAGIC Create a matrices of 0s to hold the output of the concatenated daily vectors for bot the AU and LT core

// COMMAND ----------

// MAGIC %python
// MAGIC # Initialize some arrays
// MAGIC AU_matrix = np.zeros((num_days, daily_vect_sz))
// MAGIC LT_matrix = np.zeros((num_days, daily_vect_sz))
// MAGIC #dates_arr = [0]*num_days

// COMMAND ----------

// MAGIC %md
// MAGIC #### Populate matrix
// MAGIC 
// MAGIC Here each day will be looped through, and the daily adjacency matrix will be constructed using NetworkX. The matrix will then be rearranged into a vector and added to the corresponding main matrix initialized in the cell above. 

// COMMAND ----------

// MAGIC %python
// MAGIC matrix_idx = 0 # Matrix iterator
// MAGIC print_check = 9
// MAGIC date_arr = [ first_date + datetime.timedelta(i) for i in range(num_days) ]
// MAGIC for i in range(num_days): # Loop through each day
// MAGIC   date = first_date + datetime.timedelta(i) # Get new date
// MAGIC   #dates_arr[i] = date # Add date to a list
// MAGIC   
// MAGIC   print 'progress:',matrix_idx/float(num_days)*100
// MAGIC 
// MAGIC   # Initialize lists to store tuples in the form of (u,v,weight) for both cores
// MAGIC   AU_daily_tuples = [0] * len(AU_core[AU_core['date'] == date])
// MAGIC   LT_daily_tuples = [0] * len(LT_core[LT_core['date'] == date])
// MAGIC   AU_idx = 0 # iterator for above tuple
// MAGIC   LT_idx = 0 # iterator for above tuple
// MAGIC 
// MAGIC   # Find all the weighted edges in the AU_core for the given date and populate the above list of tuples
// MAGIC   for edge in AU_core[AU_core['date'] == date].itertuples(index=False):
// MAGIC       AU_daily_tuples[AU_idx] = edge[1:4]
// MAGIC       AU_idx += 1
// MAGIC 
// MAGIC   # Find all the weighted edges in the LT_core for the given date and populate the above list of tuples
// MAGIC   for edge in LT_core[LT_core['date'] == date].itertuples(index=False):
// MAGIC       LT_daily_tuples[LT_idx] = edge[1:4]
// MAGIC       LT_idx += 1
// MAGIC 
// MAGIC   # Form the AU adjancency matrix using NetworkX
// MAGIC   AU_DG = nx.DiGraph() # Create directed graph
// MAGIC   AU_DG.add_nodes_from(range(0, maxID + 1)) # add nodes
// MAGIC   AU_DG.add_weighted_edges_from(AU_daily_tuples) # add weighted edges
// MAGIC   AU_adjacency_matrix = nx.to_numpy_matrix(AU_DG) # create adjacency matrix
// MAGIC   
// MAGIC   # Convert into a vector
// MAGIC   AU_adjacency_vector = AU_adjacency_matrix.reshape(1,daily_vect_sz)
// MAGIC 
// MAGIC   # Add to main matrix
// MAGIC   AU_matrix[matrix_idx,:] = AU_adjacency_vector
// MAGIC 
// MAGIC   # Form the LT adjacency matrix using NetworkX
// MAGIC   LT_DG = nx.DiGraph() # Create directed graph
// MAGIC   LT_DG.add_nodes_from(range(0,maxID+1)) # add nodes
// MAGIC   LT_DG.add_weighted_edges_from(LT_daily_tuples) # add weighted edges
// MAGIC   LT_adjacency_matrix = nx.to_numpy_matrix(LT_DG) # create adjacency matrix
// MAGIC 
// MAGIC   # Convert into vector
// MAGIC   LT_adjacency_vector = LT_adjacency_matrix.reshape(1,daily_vect_sz)
// MAGIC 
// MAGIC   # Add to main matrix
// MAGIC   LT_matrix[matrix_idx,:] = LT_adjacency_vector
// MAGIC 
// MAGIC   # Increment matrix index
// MAGIC   matrix_idx += 1 

// COMMAND ----------

// MAGIC %md
// MAGIC #### Remove all common non-zeros columns to make matrix smaller
// MAGIC 
// MAGIC Right now there are over 1 million columns since the adjacency matrix is nxn and and when it is rearranged to a vector it is (nxn) x 1. This is too much and there are several links that don't normally exist (e.g. all users in the network don't send all other users money). The links that are commonly zero for both matrices is removed to reduce unnecessary features from the matrix.

// COMMAND ----------

// MAGIC %python
// MAGIC AU_cols_sum = np.sum(AU_matrix, axis=0) # Sum by column for AU_matrix
// MAGIC LT_cols_sum = np.sum(LT_matrix, axis=0) # Sum by column for LT_matrix
// MAGIC 
// MAGIC # Create a mask that basically hides any columns that equal to 0 for both matrices
// MAGIC mask = (AU_cols_sum != 0) | (LT_cols_sum != 0)
// MAGIC AU_matrix_compressed = AU_matrix.compress(mask, axis=1) # Apply the mask to the AU_matrix
// MAGIC LT_matrix_compressed = LT_matrix.compress(mask, axis=1) # Apply the mask to the LT-matrix

// COMMAND ----------

// MAGIC %md 
// MAGIC #### Normalize the matrix to prepare it for PCA
// MAGIC Per the paper,
// MAGIC 1. Normalize each row so they sum to 1
// MAGIC 2. Remove the average from each column

// COMMAND ----------

// MAGIC %python
// MAGIC ## Normalize so each row sums to 1
// MAGIC for i in range(AU_matrix_compressed.shape[0]):
// MAGIC   AU_matrix_compressed[i,:] = AU_matrix_compressed[i,:] / AU_matrix_compressed[i,:].sum()
// MAGIC 
// MAGIC ## Preprocess so average is removed from each column
// MAGIC for i in range(AU_matrix_compressed.shape[1]):
// MAGIC   AU_matrix_compressed[:,i] = AU_matrix_compressed[:,i] - AU_matrix_compressed[:,i].mean()
// MAGIC 
// MAGIC ## Normalize so each row sums to 1
// MAGIC for i in range(LT_matrix_compressed.shape[0]):
// MAGIC   LT_matrix_compressed[i,:] = LT_matrix_compressed[i,:] / LT_matrix_compressed[i,:].sum()
// MAGIC 
// MAGIC ## Preprocess so average is removed from each column
// MAGIC for i in range(LT_matrix_compressed.shape[1]):
// MAGIC   LT_matrix_compressed[:,i] = LT_matrix_compressed[:,i] - LT_matrix_compressed[:,i].mean()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Export the data to file
// MAGIC Convert the numpy array back into a pandas dataframe and write it to file so that it can be read into Scala (MLlib PCA is only in Scala).
// MAGIC Show the created matricies.

// COMMAND ----------

// MAGIC %python
// MAGIC # Get indicies and columns from LT_matrix_compressed (same for both matrices)
// MAGIC index = range(0,LT_matrix_compressed.shape[0])
// MAGIC columns = range(0,LT_matrix_compressed.shape[1])
// MAGIC 
// MAGIC # Convert numpy matrix into Pandas dataframe
// MAGIC LT_pdDF = pd.DataFrame(data=LT_matrix_compressed, index=index, columns=columns)
// MAGIC AU_pdDF = pd.DataFrame(data=AU_matrix_compressed, index=index, columns=columns)
// MAGIC 
// MAGIC # Save to file
// MAGIC LT_pdDF.to_csv(relResultsDir + 'LTmatrix.csv',index=False, header=False)
// MAGIC AU_pdDF.to_csv(relResultsDir + 'AUmatrix.csv',index=False, header=False)
// MAGIC 
// MAGIC # Print
// MAGIC print "LT core: ", LT_pdDF.shape
// MAGIC print LT_matrix_compressed
// MAGIC print "\nAU core: ", AU_pdDF.shape
// MAGIC print AU_matrix_compressed

// COMMAND ----------

// MAGIC %md 
// MAGIC # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// COMMAND ----------

// MAGIC %md
// MAGIC ## 7. Principal Component Analysis (PCA)
// MAGIC 
// MAGIC In this section, Spark MLlib will be used to compute the Singular Value Decomposition (SVD) of the matrix. SVD is a method of getting the Prinicipal components which allows us to project higher dimensional data into fewer dimensions. The new features that make this possible are linear combinations of existing features.
// MAGIC 
// MAGIC Mainly the sample code provided from https://spark.apache.org/docs/1.2.1/mllib-dimensionality-reduction.html was used
// MAGIC 
// MAGIC Since both matrices are Txl = (356 x 5585), there are only T = 356 non-zero singular values and only T = 356 relevant singular vectors (columns of v). This means the full SVD can be truncated from (U: TxT, s: Txl, v: lxl) to (U:TxT, s:TxT, v: lxT) to ignore irrelevant vectors.

// COMMAND ----------

// MAGIC %md
// MAGIC ### 7a. Read in matricies, clean, and split each row into a MLlib vector

// COMMAND ----------

// Read in matrices from Python
val LTrdd = sc.textFile(absResultsDir + "LTmatrix.csv").map(row => Vectors.dense(row.split(',').map(_.toDouble)))
val AUrdd = sc.textFile(absResultsDir + "AUmatrix.csv").map(row => Vectors.dense(row.split(',').map(_.toDouble)))

// COMMAND ----------

// MAGIC %md
// MAGIC ### 7c. Compute SVD for LT core

// COMMAND ----------

val LTmat: RowMatrix = new RowMatrix(LTrdd)

// Compute the top 5 singular values and corresponding singular vectors.
val LTsvd: SingularValueDecomposition[RowMatrix, Matrix] = LTmat.computeSVD(356, computeU = true)
val LT_U: RowMatrix = LTsvd.U  // The U factor is a RowMatrix.
val LT_s: Vector = LTsvd.s  // The singular values are stored in a local dense vector.
val LT_V: Matrix = LTsvd.V  // The V factor is a local dense matrix.

// COMMAND ----------

// MAGIC %md
// MAGIC ### 7d. Compute SVD for AU core

// COMMAND ----------

val AUmat: RowMatrix = new RowMatrix(AUrdd)

// Compute the top 5 singular values and corresponding singular vectors.
val AUsvd: SingularValueDecomposition[RowMatrix, Matrix] = AUmat.computeSVD(356, computeU = true)
val AU_U: RowMatrix = AUsvd.U  // The U factor is a RowMatrix.
val AU_s: Vector = AUsvd.s  // The singular values are stored in a local dense vector.
val AU_V: Matrix = AUsvd.V  // The V factor is a local dense matrix.

// COMMAND ----------

// MAGIC %md
// MAGIC ### 7e. Check matrices sizes
// MAGIC 
// MAGIC This is just a sanity check to ensure the matrices are all the same size. U should be 356x356. s should be (356x1) since the singular vectors are shown in a vector. This can be expanded to the full 356x356 by creating a 356x356 matrix with s along the diagonal and zero elsewhere. v should be 5585x356. 

// COMMAND ----------

print ("U size: (", AU_U.numRows, AU_U.numCols,")")
print ("\ns size: (", AU_s.size, 1, ")")
print ("\nV size: (", AU_V.numRows, AU_V.numCols,")")

// COMMAND ----------

// MAGIC %md 
// MAGIC # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// COMMAND ----------

// MAGIC %md
// MAGIC ## 8. Data Validation
// MAGIC 
// MAGIC Here the PCA will be validated. As previously mentioned, PCA is a dimensionality reduction technique. The goal of PCA is to find new features that are linear combinatios of existing features that explain as much variance as possible. Each subsequent principle component explains as much ofthe variance left as possible. This mean the greater the number of components used, the greater the ability to recreate the original matrix. So in order to verify that the PCA technique performed as expected the error curve will be drawn from the approximated matrix to the actual matrix for both cores. The error should reach near zero eventually if PCA was performed correctly. The error will be calculated using the Frobenius norm (numpy norm function): the square root of the sum of squared elements of the matrix.

// COMMAND ----------

// MAGIC %md
// MAGIC ### 8a. Write resultant matrices to file

// COMMAND ----------

// MAGIC %md
// MAGIC #### Functon to write a local dense matrix to an RDD of strings (for V)

// COMMAND ----------

// Convert Local Dense Matrix to RDD of Strings (for V)
def toRDD(m: Matrix): RDD[String] = {
  val columns = m.toArray.grouped(m.numRows)
  val rows = columns.toSeq.transpose
  val vectors = rows.map(row => row.toArray.mkString(","))
  sc.parallelize(vectors)
}

// COMMAND ----------

// MAGIC %md
// MAGIC #### Write LT (U,s,V) matrices to file

// COMMAND ----------

// Write LT_U to file
val LTU = LT_U.rows.map(x => x.toArray.mkString(","))
dbutils.fs.rm(absResultsDir + "LT_U", true) // Remove saved file if it exists
LTU.saveAsTextFile(absResultsDir + "LT_U") // save file

// Write LT_s to file
val LTs = sc.parallelize(LT_s.toArray)
dbutils.fs.rm(absResultsDir + "LT_s", true) // Remove saved file if it exists
LTs.coalesce(1).saveAsTextFile(absResultsDir + "LT_s") // Save file

// Write LT_V to file
val LTV = toRDD(LT_V)
dbutils.fs.rm(absResultsDir + "LT_V", true) // Remove saved file if it exists
LTV.coalesce(1).saveAsTextFile(absResultsDir + "LT_V") // save file

// COMMAND ----------

// MAGIC %md
// MAGIC #### Write AU (U,s,V) matrices to file

// COMMAND ----------

// Write AU_U to file
val AUU = AU_U.rows.map(x => x.toArray.mkString(","))
dbutils.fs.rm(absResultsDir + "AU_U", true) // Remove saved file if it exists
AUU.saveAsTextFile(absResultsDir + "AU_U") // save file

// Write AU_s to file
val AUs = sc.parallelize(AU_s.toArray)
dbutils.fs.rm(absResultsDir + "AU_s", true) // Remove saved file if it exists
AUs.coalesce(1).saveAsTextFile(absResultsDir + "AU_s") // Save file

// Write AU_V to file
val AUV = toRDD(AU_V)
dbutils.fs.rm(absResultsDir + "AU_V", true) // Remove saved file if it exists
AUV.coalesce(1).saveAsTextFile(absResultsDir + "AU_V") // save file

// COMMAND ----------

// MAGIC %md
// MAGIC ### 8b. Read files into Python

// COMMAND ----------

// MAGIC %md
// MAGIC #### Read in LT Files

// COMMAND ----------

// MAGIC %python
// MAGIC # Read in LT_U files
// MAGIC LT_U = sqlContext.read.option("inferSchema","true").csv(absResultsDir + "LT_U").toPandas().as_matrix()[:355,:355]
// MAGIC 
// MAGIC # Read in LT_s files
// MAGIC LT_s = pd.read_csv(relResultsDir + "LT_s/part-00000",header=None).as_matrix().T
// MAGIC LT_s = np.diag(LT_s[0])[:355,:355] # diagonlize the s vector
// MAGIC 
// MAGIC # Read in LT_V files
// MAGIC LT_V = pd.read_csv(relResultsDir + "LT_V/part-00000",header=None).as_matrix()[:,:355]

// COMMAND ----------

// MAGIC %md
// MAGIC #### Read in AU Files

// COMMAND ----------

// MAGIC %python
// MAGIC # Read in AU_U files
// MAGIC AU_U = sqlContext.read.option("inferSchema","true").csv(absResultsDir + "AU_U").toPandas().as_matrix()[:355,:355]
// MAGIC 
// MAGIC # Read in AU_s files
// MAGIC AU_s = pd.read_csv(relResultsDir + "AU_s/part-00000",header=None).as_matrix().T
// MAGIC AU_s = np.diag(AU_s[0])[:355,:355] # diagonlize the s vector
// MAGIC 
// MAGIC # Read in AU_V files
// MAGIC AU_V = pd.read_csv(relResultsDir + "AU_V/part-00000",header=None).as_matrix()[:,:355]

// COMMAND ----------

// MAGIC %md
// MAGIC ### 8c. Compute n_components vs. error
// MAGIC 
// MAGIC Calculate the error from adding more components to validate. The expected result is the greater the components, the less the error. The error should eventually reach near zero.
// MAGIC 
// MAGIC The estimated matrix is calculated using A = U \* s \* v^T. The inner indices were modified depending on how many componenents were to be included.

// COMMAND ----------

// MAGIC %python
// MAGIC # The max possible features is equivalent the number of eigenvalues in s
// MAGIC numPossibleFeatures = LT_s.shape[0]
// MAGIC 
// MAGIC # Initialize the error arrays to store values
// MAGIC LT_error = np.zeros(numPossibleFeatures)
// MAGIC AU_error = np.zeros(numPossibleFeatures)
// MAGIC 
// MAGIC # Loop through number of features used from 0 to the maximum number of features possible
// MAGIC for numFeatures in range(0,numPossibleFeatures):
// MAGIC   # Compute LT error and store to array
// MAGIC   LT_error[numFeatures] = LA.norm(np.dot(np.dot(LT_U[:,0:numFeatures],LT_s[0:numFeatures,0:numFeatures]),LT_V[:,0:numFeatures].T) - LT_matrix_compressed[:355,:])
// MAGIC   
// MAGIC   # Compute AU error and store to array
// MAGIC   AU_error[numFeatures] = LA.norm(np.dot(np.dot(AU_U[:,0:numFeatures],AU_s[0:numFeatures,0:numFeatures]),AU_V[:,0:numFeatures].T) - AU_matrix_compressed[:355,:])

// COMMAND ----------

// MAGIC %md
// MAGIC ### 8c. Plot the error curves for both cores on the same plot
// MAGIC 
// MAGIC As it is seen here, the error does approach zero with relatively few features. This means that a few principal components are required to explain a majority of the variance in this dataset, and that it can accurately be represented with a few features. It is also likely that only the first couple eigenvalues have any significant magnitude and the remaining will be near zero and have very little impact on the data set. This will be graphed next to check.

// COMMAND ----------

// MAGIC %python
// MAGIC # Plot the error curve
// MAGIC errorDF = pd.DataFrame(
// MAGIC   {
// MAGIC     'Number of Principal Components':range(1,numPossibleFeatures+1),
// MAGIC     'LT Core Error':LT_error,
// MAGIC     'AU Core Error':AU_error
// MAGIC   }
// MAGIC )
// MAGIC errorSDF = sqlContext.createDataFrame(errorDF)
// MAGIC display(errorSDF.orderBy('Number of Principal Components'))

// COMMAND ----------

// MAGIC %md
// MAGIC #### 8d. Plot the eigenvalues vs. component # for both cores on the same plot
// MAGIC Eigenvalues represent the overall significance of a basis vector. The eigenvalues are sorted in descending order along the diagonal of s in SVD. The first being the most significant and the last being the least significant. Each eigenvalue correspond to one principal component. As such this curve has to be decreasing. This curve is similar to the one above. This is because they both are related and show that only the first few basis componenets are important, and have the only significant eigenvalues.

// COMMAND ----------

// MAGIC %python
// MAGIC # plot the eigenvalues vs component #
// MAGIC LT_singular_values = np.diag(LT_s)
// MAGIC AU_singular_values = np.diag(AU_s)
// MAGIC 
// MAGIC print LT_singular_values.shape
// MAGIC print AU_singular_values.shape
// MAGIC 
// MAGIC coreComparisonDF = pd.DataFrame(
// MAGIC   {
// MAGIC     'Component #':range(1,numPossibleFeatures+1),
// MAGIC     'LT Singular Values':LT_singular_values,
// MAGIC     'AU Singular Values':AU_singular_values
// MAGIC   }
// MAGIC )
// MAGIC 
// MAGIC coreComparisonSDF = sqlContext.createDataFrame(coreComparisonDF)
// MAGIC display(coreComparisonSDF.orderBy('Component #'))

// COMMAND ----------

// MAGIC %md 
// MAGIC # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// COMMAND ----------

// MAGIC %md
// MAGIC ## 9. Data Analysis
// MAGIC The goal here was the analyize the singular value weights to see if it can be related back to the Ethereum Exchange Rate. The singular value weight is represented as columns in the U matrix of SVD. It represents the significance of each basis vector (columns in V) impact over time (i.e. weight). E.g. the first singular value weight vector will show the impact of the first principle component over time. The procedure of analysis used in the paper will be used here.

// COMMAND ----------

// MAGIC %md
// MAGIC ### 9a. Top 6 singular value weights
// MAGIC 
// MAGIC Here the first 6 singular value weight vectors will be shown against time. It is clearly seen that that different basis vectors play a role at different times. Some of the basis vectors have 0 weight most of the time put have very high contributions during short intervals.

// COMMAND ----------

// MAGIC %md
// MAGIC #### LT Core
// MAGIC Here, the first 6 singular vector weights is shown for the LT core. The first core is the most active of the 6 after Febrauary 2016. Every other core's activity is mainly focused in the early months. It looks likes the 5th core is modelling only some activity in the first month and the 6th core is modelling some activity in November. 
// MAGIC This makes seens when comparing to loss curves seen above. The loss curve is very steep so the first few vectors accurately represent the entire space. This likely means that the basis vectors will focus on the outliers and try to explain the variance there. The vectors past this will have less impact that the 6th component.

// COMMAND ----------

// MAGIC %python
// MAGIC # Plot first 6 singular value weights for the LT core
// MAGIC LTSingularVectorWeightsDF = pd.DataFrame(
// MAGIC   {
// MAGIC     'date': date_arr[0:355],
// MAGIC     'LT_1': LT_U[:,0:1].T[0],
// MAGIC     'LT_2': LT_U[:,1:2].T[0],
// MAGIC     'LT_3': LT_U[:,2:3].T[0],
// MAGIC     'LT_4': LT_U[:,3:4].T[0],
// MAGIC     'LT_5': LT_U[:,4:5].T[0],
// MAGIC     'LT_6': LT_U[:,5:6].T[0]
// MAGIC   }
// MAGIC )
// MAGIC LTSingularVectorWeightsSDF = sqlContext.createDataFrame(LTSingularVectorWeightsDF)
// MAGIC display(LTSingularVectorWeightsSDF.orderBy('date'))
// MAGIC LTSingularVectorWeightsSDF.createOrReplaceTempView("LTSingularVectorWeights")

// COMMAND ----------

// MAGIC %md
// MAGIC #### AU Core
// MAGIC The activity of the AU core is similar to the LT core. The first component mainly has weights in the
// MAGIC months after February 2016. The other components mainly focus on the early months. The 3rd and 4th
// MAGIC weights are also grabbing some feature near the end of the data.

// COMMAND ----------

// MAGIC %python
// MAGIC AUSingularVectorWeightsDF = pd.DataFrame(
// MAGIC   {
// MAGIC     'date': date_arr[0:355],
// MAGIC     'AU_1': AU_U[:,0:1].T[0],
// MAGIC     'AU_2': AU_U[:,1:2].T[0],
// MAGIC     'AU_3': AU_U[:,2:3].T[0],
// MAGIC     'AU_4': AU_U[:,3:4].T[0],
// MAGIC     'AU_5': AU_U[:,4:5].T[0],
// MAGIC     'AU_6': AU_U[:,5:6].T[0]
// MAGIC   }
// MAGIC )
// MAGIC AUSingularVectorWeightsSDF = sqlContext.createDataFrame(AUSingularVectorWeightsDF)
// MAGIC display(AUSingularVectorWeightsSDF.orderBy('date'))
// MAGIC AUSingularVectorWeightsSDF.createOrReplaceTempView("AUSingularVectorWeights")

// COMMAND ----------

// MAGIC %md
// MAGIC ### 9b. First singular value weight vs. price
// MAGIC Here the first singular value weight for both cores will be shown with a scaled price to compare the shapes. 

// COMMAND ----------

// MAGIC %md
// MAGIC #### LT core
// MAGIC The LT core first singular weight has some activity in the early months while the price series doesn't change.
// MAGIC In the later months both start rising at the same time. The declined at first at the same time. 
// MAGIC Then they started to increase at the same time again. Then the activity dropped off while the price
// MAGIC increased. This could mean another singular vector is needed to match the activity here. 

// COMMAND ----------

// MAGIC %sql
// MAGIC select LT.date, LT_1, average/100 as avg
// MAGIC from LTSingularVectorWeights LT, ethToUSD xc
// MAGIC where LT.date = xc.date
// MAGIC order by LT.date

// COMMAND ----------

// MAGIC %md
// MAGIC #### AU core
// MAGIC The AU core was similarly inactive until February 2016. Then it started having a large negative impact until July 2016 when it went back to 0.

// COMMAND ----------

// MAGIC %sql
// MAGIC select AU.date, AU_1, AU_2, AU_3, AU_4, AU_5, AU_6, average/100 as avg
// MAGIC from AUSingularVectorWeights AU, ethToUSD xc
// MAGIC where AU.date = xc.date
// MAGIC order by AU.date

// COMMAND ----------

// MAGIC %md
// MAGIC ### 9c. Estimating the price with singular value weights
// MAGIC Here the first singular value weight for both cores will be shown with a scaled price to compare the shapes. Similar to the observations found in the paper, the first singular vector seemed to have a similar shape to the price time series. Both the singular value weight and the price aren't very active at the beginning because thats when the network was starting off. The network structure picks up with the price.
// MAGIC 
// MAGIC The estimated price was estimated by removing the mean of the price index and estimating it as linear combination of singular vectors. The correlation factor was calculated by taking the dot product between the time price series and the Etheruem exchange rate. 

// COMMAND ----------

// MAGIC %md
// MAGIC #### Get the average stock price for days in the database
// MAGIC 
// MAGIC Join the singular weights dataframe with the price time series and drop everything but date and time. There was no price for the very first day so it was set to 1.0

// COMMAND ----------

// MAGIC %python
// MAGIC xRateDF = pd.merge(left=AUSingularVectorWeightsDF,right=ethToUSDDF, how="left") # Merge the exchange rate and any of the above tables to get the dates
// MAGIC xRateDF = xRateDF[['date','average']].reset_index() # Keep the date and the rate only and reset the index
// MAGIC xRateDF.ix[0,2] = 1.0 # Set NaN to 1 (No data for first day)
// MAGIC  

// COMMAND ----------

// MAGIC %md
// MAGIC #### Remove the mean from the price time series

// COMMAND ----------

// MAGIC %python
// MAGIC xRateDF['meanRemovedPrice'] = xRateDF['average'] - xRateDF['average'].mean()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Calculate the correlation factor for both cores
// MAGIC 
// MAGIC Compute the dot product between the price matrix and the singular vector weights

// COMMAND ----------

// MAGIC %python
// MAGIC coreComparisonDF['LTxRateCorr'] = np.dot(xRateDF['average'].as_matrix().reshape(1,355), LT_U)[0]
// MAGIC coreComparisonDF['AUxRateCorr'] = np.dot(xRateDF['average'].as_matrix().reshape(1,355), AU_U)[0]

// COMMAND ----------

// MAGIC %md
// MAGIC #### Find the top 4 basis vectors by correlation to the price matrix

// COMMAND ----------

// MAGIC %python
// MAGIC LTcorrOrder = coreComparisonDF.sort_values('LTxRateCorr', ascending=False)['Component #'].values
// MAGIC AUcorrOrder = coreComparisonDF.sort_values('AUxRateCorr', ascending=False)['Component #'].values

// COMMAND ----------

// MAGIC %md
// MAGIC #### Find approximated price using the top 4 by correlation and covariance for both cores
// MAGIC The top 4 by correlation can be found from the above vector. The top 4 by covariance is the natural order acquired through SVD so its the first basis vectors in the U matrix. The estimated price can be found by using a formula from the paper. That formula is implemented here.

// COMMAND ----------

// MAGIC %python
// MAGIC price_woMean = xRateDF['average'] - xRateDF['average'].mean()
// MAGIC 
// MAGIC LTprice_w4Cov = price_woMean
// MAGIC AUprice_w4Cov = price_woMean
// MAGIC LTprice_w4Corr = price_woMean 
// MAGIC AUprice_w4Corr = price_woMean
// MAGIC 
// MAGIC for i in range(0,4):
// MAGIC   LT_indx = LTcorrOrder[i]
// MAGIC   AU_indx = AUcorrOrder[i]
// MAGIC   LT_corrCi = coreComparisonDF[coreComparisonDF['Component #'] == LT_indx]['LTxRateCorr'].values[0]
// MAGIC   AU_corrCi = coreComparisonDF[coreComparisonDF['Component #'] == AU_indx]['LTxRateCorr'].values[0]
// MAGIC   LT_covCi = coreComparisonDF[coreComparisonDF['Component #'] == i+1]['LTxRateCorr'].values[0]
// MAGIC   AU_covCi = coreComparisonDF[coreComparisonDF['Component #'] == i+1]['LTxRateCorr'].values[0]
// MAGIC   
// MAGIC   LTprice_w4Cov = np.add(LTprice_w4Cov,LT_covCi * LT_U[:,LT_indx-1:LT_indx][:,0])
// MAGIC   AUprice_w4Cov = np.add(AUprice_w4Cov,AU_covCi * AU_U[:,AU_indx-1:AU_indx][:,0])
// MAGIC   LTprice_w4Corr = np.add(LTprice_w4Corr,LT_corrCi * LT_U[:,i:i+1][:,0])
// MAGIC   AUprice_w4Corr = np.add(AUprice_w4Corr,AU_corrCi * AU_U[:,i:i+1][:,0])

// COMMAND ----------

// MAGIC %md
// MAGIC #### Plot the top 4 for each against the mean removed price
// MAGIC Lots of the data dropped off at the end. This could be due to boundary errors. Related activities that happen over periods of times could have got cut off. Also these dates were near the date the DAO occurred. The network was modified as such to refund people that were robbed. These mass refunds could effect the core netowrk strucutre. To avoid this, the graph is cut to dates before July 20th. This removes 7 days of data.

// COMMAND ----------

// MAGIC %python
// MAGIC # Create a spark dataframe and plot the price vs the top 4 by covariance and correlation
// MAGIC pricePredictionDF = pd.DataFrame(
// MAGIC   {
// MAGIC     'Date' : xRateDF['date'],
// MAGIC     'Price' : xRateDF['average'],
// MAGIC     'Price wo Mean' : price_woMean,
// MAGIC     'LT, Fitted, 4 Components (SVD Order)' : LTprice_w4Cov,
// MAGIC     'LT, Fitted, 4 Components (Correlation)' : LTprice_w4Corr,
// MAGIC     'AU, Fitted, 4 Components (SVD Order)' : AUprice_w4Cov,
// MAGIC     'AU, Fitted, 4 Components (Correlation)' : AUprice_w4Corr,
// MAGIC   }
// MAGIC )
// MAGIC pricePredictionSDF = sqlContext.createDataFrame(pricePredictionDF)
// MAGIC display(pricePredictionSDF.orderBy('date'))
// MAGIC pricePredictionSDF.createOrReplaceTempView("pricePrediction")

// COMMAND ----------

// MAGIC %md
// MAGIC #### Ignore the last couple dates
// MAGIC The last few dates could be bad because some transactions could have been cut off. The end date was also close to when the DAO event occurred. This is the ignored to see how the curves look to each other.
// MAGIC 
// MAGIC The LT core graphs seemed to do a bad job fitting the data. Both the LT graphs were very similar.
// MAGIC The AU core graphs had a much better fit. The AU core fitted by covariance captured some of the 
// MAGIC peaks but it missed the end part. It may fit better with more features. The AU core fit with the
// MAGIC top 4 by correlation to the price did the best. It had a very similar shape to the price graph
// MAGIC and it seemed to capture a majority of the activity. 

// COMMAND ----------

display(sqlContext.sql("select * from pricePrediction where date < cast('2016-07-20' as date) order by date asc"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Best fit: AU core, Fitted, 4 Components (Correlation)
// MAGIC The plot that had the best fit was the AU core fitted by top 4 by correlation as seen below. There are several distinct areas where both plots have very similar shapes. The graphs increase and decrease together. 

// COMMAND ----------

display(sqlContext.sql("select * from pricePrediction where date < cast('2016-07-20' as date) order by Date asc"))

// COMMAND ----------

// MAGIC %md
// MAGIC # -------------------------------------------------------------------------------------------------------------------------------------------------------------

// COMMAND ----------


