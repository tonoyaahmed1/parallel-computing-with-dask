# parallel-computing-with-dask

This project is a group assignment for Systems for Scalable Analytics. The purpose of this project is to use a Dask library to explore task parallelism on multiple cores on a cluster of machines. It includes feature exploration, data consistency checks, and computing several descriptive statistics about the data. 

## Dataset 

The datasets are no longer available. It was an Amazon Reviews dataset with the reviews and products tables as CSV files. 

## Outline

- Part 1: get percentage of missing values for all columns in the reviews table and the products table
- Part 2: find pearson correlation value between the price and ratings
- Part 3: find mean, standard deviation, median, min, and max for the price column in the products table
- Part 4: get number of of products for each super-category
- Part 5: check if there are any dangling references to the product ids from the reviews table to products and return as binary variable
- Part 6: check if there is any dangling reference between product ids in the related column and “asin” of the product table and return as binary variable
