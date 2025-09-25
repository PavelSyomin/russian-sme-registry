from pyspark.sql.types import (ByteType, DateType, FloatType, IntegerType, ShortType,
    StringType, StructField, StructType)


sme_schema = StructType([
    StructField("kind", ByteType(), False),
    StructField("category", ByteType(), False),
    StructField("reestr_date", DateType(), False),
    StructField("data_date", DateType(), False),
    StructField("ind_tin", StringType(), True),
    StructField("ind_number", StringType(), True),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("patronymic", StringType(), True),
    StructField("org_name", StringType(), True),
    StructField("org_short_name", StringType(), True),
    StructField("org_tin", StringType(), True),
    StructField("org_number", StringType(), True),
    StructField("region_code", ByteType(), True),
    StructField("region_name", StringType(), True),
    StructField("region_type", StringType(), True),
    StructField("district_name", StringType(), True),
    StructField("district_type", StringType(), True),
    StructField("city_name", StringType(), True),
    StructField("city_type", StringType(), True),
    StructField("settlement_name", StringType(), True),
    StructField("settlement_type", StringType(), True),
    StructField("activity_code_main", StringType(), True),
    StructField("file_id", StringType(), True),
    StructField("doc_cnt", ShortType(), True),
])

sme_aggregated_schema = StructType([
    StructField("kind", ByteType(), False),
    StructField("category", ByteType(), False),
    StructField("tin", StringType(), True),
    StructField("reg_number", StringType(), True),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("patronymic", StringType(), True),
    StructField("org_name", StringType(), True),
    StructField("org_short_name", StringType(), True),
    StructField("region_code", ByteType(), True),
    StructField("region_name", StringType(), True),
    StructField("region_type", StringType(), True),
    StructField("district_name", StringType(), True),
    StructField("district_type", StringType(), True),
    StructField("city_name", StringType(), True),
    StructField("city_type", StringType(), True),
    StructField("settlement_name", StringType(), True),
    StructField("settlement_type", StringType(), True),
    StructField("activity_code_main", StringType(), True),
    StructField("start_date", DateType(), True),
    StructField("end_date", DateType(), True),
])

sme_geocoded_schema = StructType([
    StructField("tin", StringType(), True),
    StructField("reg_number", StringType(), True),
    StructField("kind", ByteType(), False),
    StructField("category", ByteType(), False),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("patronymic", StringType(), True),
    StructField("org_name", StringType(), True),
    StructField("org_short_name", StringType(), True),
    StructField("activity_code_main", StringType(), True),
    StructField("region_iso_code", StringType(), True),
    StructField("region_code", StringType(), True),
    StructField("region", StringType(), True),
    StructField("area", StringType(), True),
    StructField("settlement", StringType(), True),
    StructField("settlement_type", StringType(), True),
    StructField("oktmo", StringType(), True),
    StructField("lat", FloatType(), True),
    StructField("lon", FloatType(), True),
    StructField("start_date", DateType(), False),
    StructField("end_date", DateType(), True),
])

revexp_schema = StructType([
    StructField("org_tin", StringType(), False),
    StructField("revenue", FloatType(), True),
    StructField("expenditure", FloatType(), True),
    StructField("data_date", DateType(), True),
    StructField("doc_date", DateType(), True),
])

revexp_agg_schema = StructType([
    StructField("tin", StringType(), False),
    StructField("year", ShortType(), True),
    StructField("revenue", FloatType(), True),
    StructField("expenditure", FloatType(), True),
])

empl_schema = StructType([
    StructField("org_tin", StringType(), False),
    StructField("employees_count", IntegerType(), True),
    StructField("data_date", DateType(), True),
    StructField("doc_date", DateType(), True),
])

empl_agg_schema = StructType([
    StructField("tin", StringType(), False),
    StructField("year", ShortType(), True),
    StructField("employees_count", IntegerType(), True),
])
