import sys

from pyspark import StorageLevel
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

def get_spark_session() -> SparkSession:
    """getting spark session"""
    spark = SparkSession.builder.appName("parking_violations_statistics").getOrCreate()
    #spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
    #spark.conf.set("spark.sql.join.preferSortMergeJoin", False)
    #spark.conf.set("spark.sql.inMemoryColumnarStorage.compressed", True)
    #spark.conf.set("spark.sql.inMemoryColumnarStorage.batchSize", 50000)
    return spark


def read_csv_file(spark: SparkSession, path: str) -> DataFrame:
    """read csv files from path"""
    return spark.read.csv(path, header=True, inferSchema=True)


def read_parking_violation_data(spark: SparkSession, path: str) -> DataFrame:
    """read parking violation data"""
    return read_csv_file(spark, path)


def transform_parking_violation_data(df: DataFrame, column: str = "Violation County") -> DataFrame:
    """Transforming parking vialation data to make it joinable, below are the things steps in high level

    1. Added Borocode
    2. Converted house number in case it is separated by '-'
    3. Converted 'Street Name' to upper case
    4. Removed any data having no house number
    """

    df = (
        df.select("Violation County", "House Number", "Street Name", "Summons Number", "Issue Date")
        .distinct()
        .withColumn("year", F.year(F.to_date(F.col("Issue Date"), "MM/dd/yyyy"))).orderBy("Violation County", "House Number", "Street Name", "year")
        .coalesce(100)
        .groupBy("Violation County", "House Number", "Street Name", "year")
        .agg({"Summons Number": "count"})
        .withColumnRenamed("count(Summons Number)", "total_cnt")
        .withColumn(
            "BOROCODE",
            F.when(F.col(column).isin(["MAN", "MH", "MN", "NEWY", "NEW Y", "NY"]), 1)
            .when(F.col(column).isin(["BRONX", "BX"]), 2)
            .when(F.col(column).isin(["BK", "K", "KING", "KINGS"]), 3)
            .when(F.col(column).isin(["Q", "QN", "QNS", "QU", "QUEEN"]), 4)
            .when(F.col(column).isin(["R", "RICHMOND"]), 5)
            .otherwise(0),
        )
    )

    df = (
        df.filter(F.col("House Number").isNotNull())
        .withColumn("temp", F.split("House Number", "-"))
        .withColumn(
            "House Number",
            F.col("temp").getItem(0).cast("int")
            + F.when(F.col("temp").getItem(1).isNull(), "0").otherwise(F.col("temp").getItem(1)).cast("int") / 1000,
        ).withColumn("temp",F.col("temp").getItem(0).cast("int"))
        .withColumn("Street Name", F.upper(F.col("Street Name")))
    )
    return df


def read_centerline_data(spark: SparkSession, path: str) -> DataFrame:
    """reading center line data"""
    return read_csv_file(spark, path)


def transform_read_centerline_data(df: DataFrame) -> DataFrame:
    """Transforming centerline data to make it joinable, below are the things steps in high level

    1. Converted ST_LABEL & FULL_STREE to upper case
    2. Converted L_LOW_HN & L_HIGH_HN  separated by '-' for odd house number
    3. Converted R_LOW_HN & R_HIGH_HN  separated by '-' for even house number
    4. Removed any data having no house number in L_LOW_HN and R_LOW_HN
    """
    df = (
        df.select("PHYSICALID", "BOROCODE", "FULL_STREE", "ST_NAME", "L_LOW_HN", "L_HIGH_HN", "R_LOW_HN", "R_HIGH_HN").orderBy("PHYSICALID", "BOROCODE", "FULL_STREE", "ST_NAME", "L_LOW_HN", "L_HIGH_HN", "R_LOW_HN", "R_HIGH_HN").coalesce(200)
        .withColumn("ST_NAME", F.upper(F.col("ST_NAME")))
        .withColumn("FULL_STREE", F.upper(F.col("FULL_STREE")))
        .filter((F.col("L_LOW_HN").isNotNull()) | (F.col("R_LOW_HN").isNotNull()))
    )
    df = df.withColumn("L_TEMP_ODD", F.split("L_LOW_HN", "-")).withColumn(
        "L_LOW_HN",
        F.col("L_TEMP_ODD").getItem(0).cast("int")
        + F.when(F.col("L_TEMP_ODD").getItem(1).isNull(), "0").otherwise(F.col("L_TEMP_ODD").getItem(1)).cast("int")
        / 1000,
    )

    df = df.withColumn("L_TEMP_ODD", F.split("L_HIGH_HN", "-")).withColumn(
        "L_HIGH_HN",
        F.col("L_TEMP_ODD").getItem(0).cast("int")
        + F.when(F.col("L_TEMP_ODD").getItem(1).isNull(), "0").otherwise(F.col("L_TEMP_ODD").getItem(1)).cast("int")
        / 1000,
    )

    df = df.withColumn("L_TEMP_ODD", F.split("R_LOW_HN", "-")).withColumn(
        "R_LOW_HN",
        F.col("L_TEMP_ODD").getItem(0).cast("int")
        + F.when(F.col("L_TEMP_ODD").getItem(1).isNull(), "0").otherwise(F.col("L_TEMP_ODD").getItem(1)).cast("int")
        / 1000,
    )

    df = df.withColumn("L_TEMP_ODD", F.split("R_HIGH_HN", "-")).withColumn(
        "R_HIGH_HN",
        F.col("L_TEMP_ODD").getItem(0).cast("int")
        + F.when(F.col("L_TEMP_ODD").getItem(1).isNull(), "0").otherwise(F.col("L_TEMP_ODD").getItem(1)).cast("int")
        / 1000,
    )

    return df


def join_park_violation_with_centerline(df_park_violation: DataFrame, df_centerline: DataFrame) -> DataFrame:
    """
    Joining park_violation dataframe and centerline datafrmae based on borocode, street name and house number

    Basic steps:
    1. joined odd house numbers with L_LOW_HN & L_HIGH_HN of centerline data
    2. joined even house numbers with R_LOW_HN & R_HIGH_HN of centerline data
    3. Also other criteria was borocode and street name to join the data

    :param df_park_violation:
    :param df_centerline:
    :return:
    """
    # df_park_violation = df_park_violation.repartition("BOROCODE", "Street Name", "House Number")
    # df_centerline.cache()

    """below steps for even house number"""

    """below steps for odd house number"""
    df_park_violation.cache()
    df_centerline.cache()

    df_park_violation_odd = df_park_violation.filter(F.col("temp") % 2 != 0)
    df_park_violation_even = df_park_violation.filter(F.col("temp") % 2 == 0)
    df_centerline.count()

    df_joined_1 = (
        df_park_violation_even
        .alias("park")
        .join(
            df_centerline.alias("centerline").hint("broadcast"),
            ((F.col("Street Name") == F.col("ST_NAME")) | (F.col("Street Name") == F.col("FULL_STREE")))
            & (F.col("park.BOROCODE") == F.col("centerline.BOROCODE"))
            & (
                (F.col("park.House Number") >= F.col("centerline.R_LOW_HN"))
                & (F.col("park.House Number") <= F.col("centerline.R_HIGH_HN"))
            ),
        )
        .select("total_cnt", "year", "PHYSICALID")
    )

    df_joined_2 = (
        df_park_violation_odd
        .alias("park")
        .join(
            df_centerline.alias("centerline").hint("broadcast"),
            ((F.col("Street Name") == F.col("ST_NAME")) | (F.col("Street Name") == F.col("FULL_STREE")))
            & (F.col("park.BOROCODE") == F.col("centerline.BOROCODE"))
            & (
                (F.col("park.House Number") >= F.col("centerline.L_LOW_HN"))
                & (F.col("park.House Number") <= F.col("centerline.L_LOW_HN"))
            ),
        )
        .select("total_cnt", "year", "PHYSICALID")
    )

    """returing union of 2 dataframes"""
    return df_joined_1.unionAll(df_joined_2)


def aggregate_dataset_by_year(joined_df: DataFrame) -> DataFrame:
    """aggregating the data based on 'PHYSICALID' and 'Year' of issue date and sorting based on 'PHYSICALID'"""

    # df = joined_df.withColumn("year", F.year(F.to_date(F.col("Issue Date"), "MM/dd/yyyy")))
    df = (
        joined_df.repartition(5, "year")
        .groupBy("PHYSICALID")
        .pivot("year", [2015, 2016, 2017, 2018, 2019])
        .sum("total_cnt")
    )
    df = (
        df.withColumn("2015", F.when(F.col("2015").isNull(), 0).otherwise(F.col("2015")))
        .withColumnRenamed("2015", "COUNT_2015")
        .withColumn("2016", F.when(F.col("2016").isNull(), 0).otherwise(F.col("2016")))
        .withColumnRenamed("2016", "COUNT_2016")
        .withColumn("2017", F.when(F.col("2017").isNull(), 0).otherwise(F.col("2017")))
        .withColumnRenamed("2017", "COUNT_2017")
        .withColumn("2018", F.when(F.col("2018").isNull(), 0).otherwise(F.col("2018")))
        .withColumnRenamed("2018", "COUNT_2018")
        .withColumn("2019", F.when(F.col("2019").isNull(), 0).otherwise(F.col("2019")))
        .withColumnRenamed("2019", "COUNT_2019")
        .sort("PHYSICALID")
    )
    return df


def write_output(df: DataFrame, target_path: str):
    """writing data as csv to target path without header"""
    df.write.mode("overwrite").csv(target_path, header=False)


if __name__ == "__main__":
    """Main entrypoint"""
    spark = get_spark_session()
    centerline_date_path = "hdfs:///data/share/bdm/nyc_cscl.csv"  # TODO: Provide correct hdfs path for centerline
    parking_violation_data_path = "hdfs:///data/share/bdm/nyc_parking_violation/"  # TODO: Provide correct hdfs path for parking violation
    """read output path from commandline"""
    output_path = sys.argv[1]
    df_park_violation = read_parking_violation_data(spark, parking_violation_data_path)
    df_park_violation = transform_parking_violation_data(df_park_violation)
    # df_park_violation.show()
    centerline = read_centerline_data(spark, centerline_date_path)
    centerline = transform_read_centerline_data(centerline)
    # df_park_violation.show()
    df = join_park_violation_with_centerline(df_park_violation, centerline)
    df = aggregate_dataset_by_year(df)
    write_output(df, output_path)
    # df.filter(F.col("PHYSICALID").isin([166776, 89620, 79863, 48068])).show()  # TODO: remove later
