from cProfile import run
from dask.distributed import Client, LocalCluster
import time
import json
import pandas as pd
import dask.dataframe as dd
import numpy as np
import regex as re

def main(user_reviews_csv,products_csv):
    start = time.time()
    client = Client('insert')
    client = client.restart()
    print(client)
        
    user_reviews = dd.read_csv(user_reviews_csv)
    product_release = dd.read_csv(products_csv, dtype={'asin': 'object'})

    #part 1
    output1_reviews = round((user_reviews.isna().mean().compute() * 100), 2)
    output1_products = round((product_release.isna().mean().compute() * 100), 2)

    #part 2
    user_reviews_rating = user_reviews[["overall", "asin"]]
    product_release_price = product_release[["price", "asin"]]
    merged = dd.merge(left=user_reviews_rating, right=product_release_price, left_on='asin', right_on='asin')[["price", "overall"]]
    merged = merged.compute()
    output2 = round((merged['price'].corr(merged['overall'])), 2)

    #part 3
    output3 = round((product_release["price"].describe().compute()),2)[['mean', 'std', '50%', 'min', 'max']]

    #part 4
    reduced = product_release["categories"].fillna("")
    stripped = reduced.str.split(']').str[0].str.strip('[["').str.split("',").str[0].str.split('",').str[0].str.strip("\'").compute()

    stripped_df = pd.DataFrame(stripped).reset_index()
    stripped_df_grouped = stripped_df.groupby("categories").count()
    output4 = stripped_df_grouped.sort_values(["index", "categories"], ascending = [False, False])
    
    #part 5
    pr_asin_computed = product_release["asin"].compute()
    output5 = int(merged.shape[0] != len(pr_asin_computed))

    #part 6
    unique_id = product_release[["asin"]].dropna()
    related_df = product_release[["related"]].dropna()

    output6 = 0
    is_looping = True

    for i, j in related_df.iterrows():
        dictionary = eval(j["related"])
        for m in dictionary.keys():
            for k in dictionary[m]:
                if k not in unique_id:
                    output6 = 1
                    is_looping = False
                    break

        if is_looping == False:
            break


    q1_reviews = output1_reviews
    q1_products = output1_products
    q2 = output2
    q3 = output3
    q4 = output4
    q5 = output5
    q6 = output6
    end = time.time()
    runtime = end-start

    # writing results to "results.json" 
    with open('OutputSchema.json','r') as json_file:
        data = json.load(json_file)
        #print(data)

        data["q1"]["products"] = json.loads(q1_products.to_json())
        data["q1"]["reviews"] = json.loads(q1_reviews.to_json())
        data["q2"] = q2
        data["q3"] = json.loads(q3.to_json())
        data["q4"] = json.loads(q4.to_json())["index"]
        data["q5"] = q5
        data["q6"] = q6
    
    print(data)
    with open('results.json', 'w') as outfile: json.dump(data, outfile)

    return runtime
