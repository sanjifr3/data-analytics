Sys.setenv(SPARK_HOME='/Users/liang/Downloads/spark-1.4.1-bin-hadoop2.6/')
.libPaths(c(file.path(Sys.getenv('SPARK_HOME'), 'R', 'lib'), .libPaths()))
library(SparkR)
sc <- sparkR.init(master='local')
sqlContext <- sparkRSQL.init(sc)
# Create the DataFrame
df <- createDataFrame(sqlContext, faithful)
# Select only the "eruptions" column
collect(select(df, df$eruptions))
# Select only the "eruptions" column
collect(select(df, "eruptions"))
# Filter the DataFrame to only retain rows with wait times shorter than 50 mins
collect(filter(df, df$waiting < 50))
# We use the `n` operator to count the number of times each waiting time appears
collect(summarize(groupBy(df, df$waiting), count = n(df$waiting)))
# We can also sort the output from the aggregation to get the most common waiting times
waiting_counts <- summarize(groupBy(df, df$waiting), count = n(df$waiting))
collect(arrange(waiting_counts, desc(waiting_counts$count)))
# Convert waiting time from hours to seconds.
df$waiting_secs <- df$waiting * 60
collect(df)