/*SQL Queries*/

/* Part 1 - loading Github's Python repositories CSV */

/* Read in Repositories CSV File */
// URL of CSV file

val pythonRepositoriesPath = "https://drive.google.com/uc?export=download&id=0B4-QnHc0OvzxOERPbkdaWnpYY00"



// load Github's python repositories csv

val reposCSV = sc.parallelize(IOUtils.toString(new URL(pythonRepositoriesPath)).split("\n"))



// convert text file into an Resilient Distributed Dataset (RDD)

val reposRDD = reposCSV.map(line => line.split(",").map(elem => elem.trim))

// Look at first record
val look = reposRDD.take(1)

// Create class
case class Repository(
    
	repositoryName: String,
    
	ownerLogin: String,
    
	createdAt: java.sql.Timestamp,
    
	updatedAt: java.sql.Timestamp,
    
	stargazersCount: Int,
    
	forksCount: Int
    
	)

// Clean data and convert them to proper formats
def getRepository(row:Array[String]):Repository = {
    
	return Repository(
        
		row(0),
        
		row(1),
        
		Timestamp.valueOf(row(2).replace("T", " ").replace("Z", "")), 
		// Timestamp.valueOf expects dates in the format "yyyy-mm-dd hh:mm"
        
		Timestamp.valueOf(row(3).replace("T", " ").replace("Z", "")),
        
		row(4).toInt,
        
		row(5).toInt
    
	)

// Load into SparkSQL

val reposDF = reposRDD.map(row => getRepository(row)).toDF()



// register this data as an SQL table

reposDF.registerTempTable("repositories")



// Let's take a look at the schema and data

reposDF.printSchema

// Count number of records
reposDF.count

// Look at data
reposDF.show()

// Use describe to get stats of data like count/mean/stddev/min/max
reposDF.describe().show()


// Queries

/* 1. List 10 newest queries */

%sql
select * from repositories
order by createdat desc
limit 10

/* 2. List all unique logins */

%sql
select distinct ownerLogin
from repositories

/* 3. List all the users with more than 1 repository */

%sql

select ownerLogin
, count(repositoryName) as repositoriesCount
from repositories

group by ownerLogin

having repositoriesCount > 1
order by repositoriesCount desc

/* 4. List the number of repositories created per year */

%sql
select year(createdAt) createdYear, count(repositoryName) repositoriesCount
from repositories
group by year(createdAt)
order by createdYear asc





/* Part 2 - Github Contributions */



// Url of contributions CSV file
val pythonContributionsPath = "https://drive.google.com/uc?export=download&id=0B4-QnHc0OvzxTnJjcktIbktZOTg"

// Load github's python repositories csv
val contrCSV = sc.parallelize(IOUtils.toString(new URL(pythonContributionsPath)).split("\n"))

// Convert text file into RDD
val contrRDD = contrCSV.map(line => line.split(",").map(elem => elem.trim))

// View first line
contrRDD.take(1)

// Create class
case class Contribution(
    
	repository: String,
    
	login: String,
    
	contributions: Int
    
	)

// Patch bad records with this:
implicit class StringConversion(val s: String) {
	
def toTypeOrElse[T](convert: String=>T, defaultVal: T) = try {
    
		convert(s)
  	
	} catch {
    
		case _: NumberFormatException => defaultVal
  
        }
  
  
	
	def toIntOrElse(defaultVal: Int = 0) = toTypeOrElse[Int](_.toInt, defaultVal)
  
	def toDoubleOrElse(defaultVal: Double = 0D) = toTypeOrElse[Double](_.toDouble, defaultVal)

}

// Define contributions
def getContribution(row:Array[String]):Contribution = {
  	
	return Contribution(
    
		row(0),
    
		row(1),
    
		row(2).toIntOrElse()
  
	)

}

// Load into SparkSQL
val contrDF = contrRDD.map(row => getContribution(row)).toDF()

// Register as SparkSQL Table
contrDF.registerTempTable("contributions")

// Look at schema and data
contrDF.printSchema

/* Queries */

/* 1. List all unique logins in reverse alphabetic order (starting from z) */

%sql
select distinct login
from contributions
order by login desc

/* 2. List contributors that contributed to more than 20 repositories */

select login
from contributions
group by login
having count(repository) > 20

/* 3. Calculate for every repository the number of contributing users, and the average number of contributions per year */

%sql
select repository, count(login), avg(contributions)
from contributions
group by repository
order by repository asc


/* Part 3 - Combining the repositories */

/* Example query: All the repositories that have more stargazers than total number of contributions */

%sql

select repositoryName, max(stargazersCount) as starCount, SUM(contributions) as numContributions

from repositories r, contributions c

where r.repositoryName = c.repository

group by r.repositoryName

having starCount > numContributions

order by (starCount - numContributions) desc

/* 1. All the repositories that have more forks than total number of contributions */

%sql
SELECT r.repositoryName, r.forksCount, SUM(c.contributions) as numContributions

FROM repositories r 
JOIN contributions c 
ON r.repositoryName = c.repository

GROUP BY r.repositoryName, r.forksCount

HAVING r.forksCount > numContributions

/* 2. All the users that the sum of the stars for the repositories they have contributed to is larger than 50,000 */

%sql

SELECT c.login, SUM(r.stargazersCount) as sumStargazers

FROM repositories r 
JOIN contributions c 
ON r.repositoryName = c.repository

GROUP BY c.login

HAVING sumStargazers > 50000

ORDER BY sumStargazers DESC

/* 3. All the repositories with their number of contributors, forks count, and stargazers count */

%sql
SELECT r.repositoryName, r.stargazersCount, r.forksCount, COUNT(c.login)
FROM repositories r
JOIN contributions c
ON r.repositoryName = c.repository
GROUP BY r.repositoryName, r.stargazersCount, r.forksCount

/* 4. For every year, sum the contributions made to repositories in this year */

%sql
SELECT YEAR(r.createdAt), SUM(contributions)
FROM repositories r
JOIN contributions c
ON r.repositoryName = c.repository
GROUP BY YEAR(r.createdAt)
ORDER BY YEAR(r.createdAt) ASC

















