import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
from pyspark.sql.functions import explode
from pyspark.sql.functions import avg,count
# availiable on AWS EMR

INPUT_FORMAT = 'dataframe' 

if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics

def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    grouped = review_data.select(asin_column, overall_column).groupBy(asin_column) \
    .agg(F.count(overall_column).alias(count_rating_column), \
         F.avg(overall_column).alias(mean_rating_column))
   
    product_join = product_data[[asin_column]].join(grouped, on=asin_column, how="left")
   
    count_total = product_join.count()
    mean_countRating = product_join.select(F.avg(F.col(count_rating_column))).head()[0]
    variance_countRating = product_join.select(F.variance(F.col(count_rating_column))).head()[0]
    mean_meanRating = product_join.select(F.avg(F.col(mean_rating_column))).head()[0]
    variance_meanRating = product_join.select(F.variance(F.col(mean_rating_column))).head()[0]
   
    numNulls_countRating = product_join.select([F.count(F.when(F.col(count_rating_column).isNull(),
                                                               count_rating_column))]).head()[0]
    numNulls_meanRating = product_join.select([F.count(F.when(F.col(mean_rating_column).isNull(), mean_rating_column))
                                               ]).head()[0]

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:
    res['count_total'] = count_total
    res['mean_meanRating'] = mean_meanRating
    res['variance_meanRating'] = variance_meanRating
    res['numNulls_meanRating'] = numNulls_countRating
    res['mean_countRating'] = mean_countRating
    res['variance_countRating'] = variance_countRating
    res['numNulls_countRating'] = numNulls_countRating

    # -------------------------------------------------------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------
