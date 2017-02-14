/* Lab 3 Solutions */

/* Download zipped dataset and extract it */

%sh

curl -o /resources/data/babs_open_data_year_3.zip https://s3.amazonaws.com/babs-open-data/babs_open_data_year_3.zip

unzip -o -d /resources/data/babs/ /resources/data/babs_open_data_year_3.zip

// Look at readme file
%sh

cat /resources/data/babs/README.txt

// Import libraries
import java.sql.Timestamp

import org.apache.commons.io.IOUtils

import java.net.URL

import java.nio.charset.Charset

// Write into RDD type
val bikeshareRDD = sc.textFile("/resources/data/babs/201608_trip_data.csv").map(line => line.split(",", -1).map(_.trim)).filter(line => line(0) != "Trip ID")

// Read first line of RDD type
bikeshareRDD.first

// Create class

case class Trip(                            // column index
    
	tripID: Int,                            // 0
    
	tripDuration: Int,                      // 1
    
	startTime: java.sql.Timestamp,          // 2
    
	startStation: String,                   // 3  
    
	startTerminal: Int,                     // 4  
    
	endTime: java.sql.Timestamp,            // 5
    
	endStation: String,                     // 6 
    
	endTerminal: Int,                       // 7  
    
	bikeID: Int,                            // 8
    
	subscriberType: String,                 // 9
    
	zipCode: Int                            // 10
  
)


// Create implicit classes to convert strings
implicit class StringConversion(val s: String) {

	def toTypeOrElse[T](convert: String=>T, defaultVal: T) = try {
    
		convert(s)
  
	} catch {
    
		case _: Throwable => defaultVal
  
	}
  
  

	def toIntOrElse(defaultVal: Int = 0) = toTypeOrElse[Int](_.toInt, defaultVal)
  
	def toDoubleOrElse(defaultVal: Double = 0D) = toTypeOrElse[Double](_.toDouble, defaultVal)
  
	def toDateOrElse(defaultVal: java.sql.Timestamp = java.sql.Timestamp.valueOf("1970-01-01 00:00:00")) = 		toTypeOrElse[java.sql.Timestamp](java.sql.Timestamp.valueOf(_), defaultVal)

}


// Fix date format in this dataset
def fixDateFormat(orig: String): String = {
    
	val splited_date = orig.split(" ")
    
	val fixed_date_parts = splited_date(0).split("/").map(part => if (part.size == 1) "0" + part else part)
    	val fixed_date = List(fixed_date_parts(2), fixed_date_parts(0), fixed_date_parts(1)).mkString("-")
    
	val fixed_time = splited_date(1).split(":").map(part => if (part.size == 1) "0" + part else part).mkString(":")
    
	fixed_date + " " + fixed_time + ":00"
 // return value
}

// clean up Ride fields and convert them to proper formats

def getTripCleaned(row:Array[String]):Trip = {
  
	return Trip(
    
		row(0).toIntOrElse(),
    
		row(1).toIntOrElse(),
    
		fixDateFormat(row(2)).toDateOrElse(),
    
		row(3),
    
		row(4).toIntOrElse(),
    
		fixDateFormat(row(5)).toDateOrElse(),
    
		row(6),
    
		row(7).toIntOrElse(),
    
		row(8).toIntOrElse(),
    
		row(9),
    
		row(10).toIntOrElse()
  
	)

}

// Load the data table into a Dataframe used for SparkSQL

val trip = bikeshareRDD.map(r => getTripCleaned(r)).toDF()


// register this data as an SQL table

trip.createOrReplaceTempView("trip")


/* Queries */

/* 1. Show for every hour of the day (0,1,...,23) the number of rides started at this hour, and the average trip
      duration of the rides starting in this hour. Are they correlated? */

%sql
select hour(startTime) hour, count(*), avg(tripDuration)
from trip
group by hour(startTime)
order by hour(startTime)

/* 2. Compare the endStationID of a ride to the startStationID of the next ride of the bike. Look at top 20. */

%sql
SELECT t1.bikeID, t1.endstation, t2.startStation
FROM (
	SELECT bikeID, startStation, endStation, row_number() OVER (PARTITION BY bikeID ORDER BY startTime ASC) 		rownum
	FROM trip
      ) t1 
INNER JOIN (
	SELECT bikeID, startStation, endStation, row_number() OVER (PARTITION BY bikeID ORDER BY startTime ASC) 		rownum
	FROM trip
      ) t2
ON t1.bikeID = t2.bikeID AND t2.rownum = t1.rownum+1
LIMIT 20

/* 2.1 Calculate the percent of cases in which bikes were moved between rides (i.e., the percent of rides that started at a station that is different from the station the bike were last left)

%sql

SELECT (t1.countMoved / t2.countALL) ratioMoved

FROM
(	SELECT COUNT(*) as countMoved

	FROM (
	select bikeID, startStation, endStation, 
		row_number() OVER (PARTITION BY bikeID ORDER BY startTime ASC) rownum
	
	from trip
	
) t1 
	INNER JOIN (	
select bikeID, startStation, endStation, 
			row_number() OVER (PARTITION BY bikeID ORDER BY startTime ASC) rownum
			
from trip
			
) t2

	ON t1.bikeID = t2.bikeID AND t2.rownum = t1.rownum+1

	WHERE t1.endStation <> t2.startStation
) t1 JOIN ( SELECT COUNT(*) countAll from trip) t2

/* 2.2 Calculate this percentage on an hour by hour basis. Are there hours in which it is more likely to be moved? */

%sql

SELECT tt1.hour, (tt1.countMoved / tt2.countALL) ratioMoved

FROM

(SELECT HOUR(t1.startTime) hour, COUNT(*) as countMoved

FROM (

select bikeID, startTime, startStation, endStation, row_number() OVER (PARTITION BY bikeID ORDER BY startTime ASC) rownum

from trip

) t1 INNER JOIN (

select bikeID, startTime, startStation, endStation, row_number() OVER (PARTITION BY bikeID ORDER BY startTime ASC) rownum

from trip
) t2

ON t1.bikeID = t2.bikeID AND t2.rownum = t1.rownum+1

WHERE t1.endStation <> t2.startStation

GROUP BY HOUR(t1.startTime)) tt1 INNER JOIN (SELECT HOUR(startTime) hour, COUNT(*) countAll from trip group by HOUR(startTime)) tt2

ON tt1.hour = tt2.hour

order by tt1.hour ASC

/* 2.3 Are there some bikes are being moved more often that others? Analyze the frequency of ves per bike and find out. */

%sql

SELECT t1.bikeID, COUNT((*)) countMoved

FROM (

select bikeID, startStation, endStation, row_number() OVER (PARTITION BY bikeID ORDER BY startTime ASC) rownum

from trip

) t1 INNER JOIN (

select bikeID, startStation, endStation, row_number() OVER (PARTITION BY bikeID ORDER BY startTime ASC) rownum

from trip
) t2

ON t1.bikeID = t2.bikeID AND t2.rownum = t1.rownum+1

WHERE t1.endStation <> t2.startStation

GROUP BY t1.bikeID

ORDER BY countMoved DESC





/* Part 2 */

// Read in station file
val stationRDD = sc.textFile("/resources/data/babs/201608_station_data.csv").map(line => line.split(",", -1).map(_.trim)).filter(line => line(0) != "station_id")


// Create station class
// Station class
case class Station(                     // column index
    
	stationID: Int,                     // 0
    
	stationName: String,                // 1
    
	latitude: Double,                   // 2
    
	longitude: Double,                  // 3  
    
	dockCount: Int,                     // 4  
    	
	landmark: String                    // 5
  
	)

// Clean Station Class and define types
def getStationCleaned(row:Array[String]):Station = {
  
	return Station(
    
		row(0).toIntOrElse(),
    
		row(1),
    
		row(2).toDoubleOrElse(),
    
		row(3).toDoubleOrElse(),
    
		row(4).toIntOrElse(),
    
		row(5)
  
	)

}

// Load into SparkSQL DataFrame
val station = stationRDD.map(r => getStationCleaned(r)).toDF()


// register this data as an SQL table

station.createOrReplaceTempView("station")


/* 3. Create a query that includes all the fields of "trip" as well as "startStationLat", "startStationLong", "endStationLat", "endStationLong" that correspond to the latitude and longitutude of the start and end station ( this information can be found in station table )
*/

%sql
SELECT trip.*, s1.latitude startStationLat, s1.longitude startStationLong, s2.latitude endStationLat, s2.longitude endStationLong

FROM (trip INNER JOIN station s1 ON trip.startStation == s1.stationName) INNER JOIN station s2 ON trip.endStation == s2.stationName


LIMIT 10

// Save this query as a new table called "trip_station"
val tripStationDF = sqlContext.sql("SELECT trip.*, s1.latitude startStationLat, s1.longitude startStationLong, s2.latitude endStationLat, s2.longitude endStationLong FROM (trip INNER JOIN station s1 ON trip.startStation == s1.stationName) INNER JOIN station s2 ON trip.endStation == s2.stationName")


tripStationDF.createOrReplaceTempView("trip_station")

// Define a function called greatCircleDistance
val greatCircleDistance = (latitude1: Double, longitude1: Double, latitude2: Double, longitude2: Double) => {
  
	val radLat1 = math.toRadians(latitude1)
  
	val radLong1 = math.toRadians(longitude1)
  
	val radLat2 = math.toRadians(latitude2)
  
	val radLong2 = math.toRadians(longitude2)
  
	val deltaLatitude = radLat2 - radLat1
  
	val deltaLongitude = radLong2 - radLong1

  
	val deltaR = 2.0 * math.asin(math.sqrt(math.pow(math.sin(deltaLatitude/2), 2) + (math.cos(radLat1) * 		math.cos(radLat2) * math.pow(math.sin(deltaLongitude/2), 2))))
  
	val sphereRadius = 6371
  
	val distanceKM = deltaR * sphereRadius
  
  

	distanceKM
 // Return value
}

// Log this function as a SQL UDF Function
sqlContext.udf.register("greatCircleDistance", greatCircleDistance)

/* 4.1. Use the great circle distance, find the average distance per bikeID */

%sql

SELECT bikeID, AVG(greatCircleDistance(startStationLat, startStationLong, endStationLat, endStationLong)) avgGcDistance

FROM trip_station

GROUP BY bikeID

ORDER BY avgGcDistance DESC

/* 4.2. For every station, find the 5 longest rides (based on great circle distance) that started at this station
HINT: you probably want to consider using window functions:
	Web: https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html
*/

%sql

SELECT * FROM
(
SELECT startStation, greatCircleDistance(startStationLat, startStationLong, endStationLat, endStationLong) gcDistance,  ROW_NUMBER() OVER (PARTITION BY startStation ORDER BY greatCircleDistance(startStationLat, startStationLong, endStationLat, endStationLong) DESC) distRank

FROM trip_station
)

WHERE distRank <= 5

ORDER BY startStation, distRank ASC

/* 4.3. First, lets calcuate the station activity for each station. In order to do these query we need to get a list of (tripDuration, stationID) tuples, where stationID is either startStationID or endStationID. To do this, we will use UNION. We create a relation of (tripDuration, stationID) where stationID is startStationID, and a relation of (tripDuration, stationID) where stationID is endStationID and we UNION the two relations. Then we can sum the tripDuration for each station. */

%sql


SELECT station, SUM(tripDuration) stationActivity

FROM

(
   
	SELECT tripDuration, startStation as station
    
	FROM trip_station
    
	UNION
    
	SELECT tripDuration, endStation as station
    
	FROM trip_station

)

GROUP BY station

ORDER BY stationActivity DESC

/* 5. For each station we want to find the longest-distance (based on great circle distance) that has *started or ended* at this station */

%sql


SELECT station, MAX(gcDistance) longestTrip

FROM

(
    
	SELECT greatCircleDistance(startStationLat, startStationLong, endStationLat, endStationLong) gcDistance, 		startStation as station
    
	FROM trip_station
    
	UNION
    
	SELECT greatCircleDistance(startStationLat, startStationLong, endStationLat, endStationLong) gcDistance, 		endStation as station
    
	FROM trip_station

)

GROUP BY station

ORDER BY longestTrip DESC

/* 6. Use the additional dataset "201608_weather_data.csv", perform an exploratory data analysis and look for interesting patterns. Specifically, you can look into the relation between good/bad weather and the trip distance or trip length */



















