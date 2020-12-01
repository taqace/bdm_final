import sys

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F


def get_spark_session() -> SparkSession:
    """getting spark session"""
    spark = SparkSession.builder.appName("parking_violations_statistics").getOrCreate()
    return spark


@udf(returnType=IntegerType())
def map_county_to_borough_number(county: str) -> str:
    """
    Function to convert county information to get borough code.
    :param county:
    :return:
    """
    _borough_name_number_map = {"MANHATTAN": 1, "BRONX": 2, "BROOKLYN": 3, "QUEENS": 4, "STATEN ISLAND": 5}
    _county_borough_map = {
        "MAN": _borough_name_number_map["MANHATTAN"],
        "MH": _borough_name_number_map["MANHATTAN"],
        "MN": _borough_name_number_map["MANHATTAN"],
        "NEWY": _borough_name_number_map["MANHATTAN"],
        "NEW Y": _borough_name_number_map["MANHATTAN"],
        "NY": _borough_name_number_map["MANHATTAN"],
        "BRONX": _borough_name_number_map["BRONX"],
        "BX": _borough_name_number_map["BRONX"],
        "BK": _borough_name_number_map["BROOKLYN"],
        "K": _borough_name_number_map["BROOKLYN"],
        "KING": _borough_name_number_map["BROOKLYN"],
        "KINGS": _borough_name_number_map["BROOKLYN"],
        "Q": _borough_name_number_map["QUEENS"],
        "QN": _borough_name_number_map["QUEENS"],
        "QNS": _borough_name_number_map["QUEENS"],
        "QU": _borough_name_number_map["QUEENS"],
        "QUEEN": _borough_name_number_map["QUEENS"],
        "R": _borough_name_number_map["STATEN ISLAND"],
        "RICHMOND": _borough_name_number_map["STATEN ISLAND"],
    }

    return _county_borough_map.get(county, -99)


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

    df = df.withColumn("BOROCODE", map_county_to_borough_number(column))
    df = (
        df.withColumn("temp", F.split("House Number", "-"))
        .withColumn("L_HOUSE_NO", F.col("temp").getItem(0).cast("int"))
        .withColumn(
            "H_HOUSE_NO",
            F.when(F.col("temp").getItem(1).isNull(), "9999").otherwise(F.col("temp").getItem(1)).cast("int"),
        )
        .drop("temp")
        .filter(F.col("L_HOUSE_NO").isNotNull())
        .withColumn("Street Name", F.upper(F.col("Street Name")))
        .distinct()
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
        df.withColumn("ST_LABEL", F.upper(F.col("ST_LABEL")))
        .withColumn("FULL_STREE", F.upper(F.col("FULL_STREE")))
        .filter((F.col("L_LOW_HN").isNotNull()) | (F.col("R_LOW_HN").isNotNull()))
    )
    df = (
        df.withColumn("L_TEMP_ODD", F.split("L_LOW_HN", "-"))
        .withColumn("LOWER_LOW_HOUSE_NO_ODD", F.col("L_TEMP_ODD").getItem(0).cast("int"))
        .withColumn(
            "UPPER_LOW_HOUSE_NO_ODD",
            F.when(F.col("L_TEMP_ODD").getItem(1).isNull(), "9000")
            .otherwise(F.col("L_TEMP_ODD").getItem(1))
            .cast("int"),
        )
        .drop("L_TEMP_ODD")
    )
    df = (
        df.withColumn("L_TEMP_ODD", F.split("L_HIGH_HN", "-"))
        .withColumn("LOWER_HIGH_HOUSE_NO_ODD", F.col("L_TEMP_ODD").getItem(0).cast("int"))
        .withColumn(
            "UPPER_HIGH_HOUSE_NO_ODD",
            F.when(F.col("L_TEMP_ODD").getItem(1).isNull(), "9999")
            .otherwise(F.col("L_TEMP_ODD").getItem(1))
            .cast("int"),
        )
        .drop("L_TEMP_ODD")
    )

    df = (
        df.withColumn("R_TEMP_EVEN", F.split("R_LOW_HN", "-"))
        .withColumn("LOWER_LOW_HOUSE_NO_EVEN", F.col("R_TEMP_EVEN").getItem(0).cast("int"))
        .withColumn(
            "UPPER_LOW_HOUSE_NO_EVEN",
            F.when(F.col("R_TEMP_EVEN").getItem(1).isNull(), "9000")
            .otherwise(F.col("R_TEMP_EVEN").getItem(1))
            .cast("int"),
        )
        .drop("R_TEMP_EVEN")
    )
    df = (
        df.withColumn("R_TEMP_EVEN", F.split("R_HIGH_HN", "-"))
        .withColumn("LOWER_HIGH_HOUSE_NO_EVEN", F.col("R_TEMP_EVEN").getItem(0).cast("int"))
        .withColumn(
            "UPPER_HIGH_HOUSE_NO_EVEN",
            F.when(F.col("R_TEMP_EVEN").getItem(1).isNull(), "9999")
            .otherwise(F.col("R_TEMP_EVEN").getItem(1))
            .cast("int"),
        )
        .drop("R_TEMP_EVEN")
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
    df_park_violation.cache()

    """below steps for even house number"""
    df_joined_even = (
        df_park_violation.filter(F.col("House Number") % 2 == 0)
        .alias("park")
        .join(
            F.broadcast(df_centerline).alias("centerline"),
            ((F.col("Street Name") == F.col("ST_NAME")) | (F.col("Street Name") == F.col("FULL_STREE")))
            & (F.col("park.BOROCODE") == F.col("centerline.BOROCODE"))
            & (
                F.col("park.L_HOUSE_NO").between(
                    F.col("centerline.LOWER_LOW_HOUSE_NO_EVEN"), F.col("centerline.LOWER_HIGH_HOUSE_NO_EVEN")
                )
            )
            & (
                F.col("park.H_HOUSE_NO").between(
                    F.col("centerline.UPPER_LOW_HOUSE_NO_EVEN"), F.col("centerline.UPPER_HIGH_HOUSE_NO_EVEN")
                )
            ),
        )
        .select("Summons Number", "Issue Date", "PHYSICALID")
    )

    """below steps for odd house number"""

    df_joined_odd = (
        df_park_violation.filter(F.col("House Number") % 2 != 0)
        .alias("park")
        .join(
            F.broadcast(df_centerline).alias("centerline"),
            ((F.col("Street Name") == F.col("ST_NAME")) | (F.col("Street Name") == F.col("FULL_STREE")))
            & (F.col("park.BOROCODE") == F.col("centerline.BOROCODE"))
            & (
                F.col("park.L_HOUSE_NO").between(
                    F.col("centerline.LOWER_LOW_HOUSE_NO_ODD"), F.col("centerline.LOWER_HIGH_HOUSE_NO_ODD")
                )
            )
            & (
                F.col("park.H_HOUSE_NO").between(
                    F.col("centerline.UPPER_LOW_HOUSE_NO_ODD"), F.col("centerline.UPPER_HIGH_HOUSE_NO_ODD")
                )
            ),
        )
        .select("Summons Number", "Issue Date", "PHYSICALID")
    )

    """returing union of 2 dataframes"""
    return df_joined_even.union(df_joined_odd)


def aggregate_dataset_by_year(joined_df: DataFrame) -> DataFrame:
    """aggregating the data based on 'PHYSICALID' and 'Year' of issue date and sorting based on 'PHYSICALID'"""

    df = joined_df.withColumn("year", F.year(F.to_date(F.col("Issue Date"), "MM/dd/yyyy")))
    df = df.groupBy("PHYSICALID").pivot("year", [2015, 2016, 2017, 2018, 2019]).count()
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
    centerline_date_path = "/data/share/bdm/nyc_cscl.csv"  # TODO: Provide correct hdfs path for centerline
    parking_violation_data_path = "/data/share/bdm/nyc_parking_violation"  # TODO: Provide correct hdfs path for parking violation
    """read output path from commandline"""
    output_path = sys.argv[1]
    df_park_violation = read_parking_violation_data(spark, parking_violation_data_path)
    df_park_violation = transform_parking_violation_data(df_park_violation)
    centerline = read_centerline_data(spark, centerline_date_path)
    centerline = transform_read_centerline_data(centerline)
    df = join_park_violation_with_centerline(df_park_violation, centerline)
    df = aggregate_dataset_by_year(df)
    write_output(df, output_path)
    
