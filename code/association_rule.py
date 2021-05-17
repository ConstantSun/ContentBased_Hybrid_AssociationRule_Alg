import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from apyori import apriori
import logging
logging.basicConfig(filename="../log/result_association.txt",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

def load_data(filename='../input/Groceries_dataset.csv'):
    """
    This function for: 
        load_data
    Args:
        filename : path to file csv.
    Return:    
        df: data frame
    """     
    df = pd.read_csv(filename, parse_dates=['Date'])
    logging.info(df.head())
    logging.info(f"any null value:\n{df.isnull().any()}") 
    return df 



def get_total_product(df):
    """
    This function for: 
        get total_products
    Args:
        df: data frame
    Return:    
        all_products: data frame contains all products infor.
    """       
    all_products = df['itemDescription'].unique()
    print("Total products: {}".format(len(all_products)))   
    return all_products


def ditribution_plot(x,y,name=None,xaxis=None,yaxis=None):
    """
    This function for: 
        visualise by plotting distribution
    Args:
        x, y: axis to be plotted.
    Return:    
        showing figure.
    """     
    fig = go.Figure([
        go.Bar(x=x, y=y)
    ])

    fig.update_layout(
        title_text=name,
        xaxis_title=xaxis,
        yaxis_title=yaxis
    )
    fig.show()


def get_top10_frequently_sold(df):
    """
    This function for: 
        get top 10 frequently sold products
    Args:
        df: data frame
    Return:    
        x: list of 10 most recently sold products.
    """        
    x = df['itemDescription'].value_counts()
    x = x.sort_values(ascending = False) 
    x = x[:10]
    return x


def get_onehot_presentation(df):
    """
    This function for: 
        One-hot representation of products purchased
    Args:
        df: data frame
    Return:    
        df : data frame after one_hot-ize.
    """          
    one_hot = pd.get_dummies(df['itemDescription'])
    df.drop('itemDescription', inplace=True, axis=1)
    df = df.join(one_hot)
    logging.info(f"One-hot representation of products purchased:\n{df.head()}")
    return df 


def get_Pnames(x):
    """
    This function for: 
        Replacing non-zero values with product names
    Args:
        x : data frame
    Return:    
        x : data frame
    """         
    for product in all_products:
        if x[product] > 0:
            x[product] = product
    return x


def remove_zeros(records):
    """
    This function for: 
        Removing zeros for records
    Args:
        records: data frame
    Return:    
        transactions: records after removing zeros
    """          
    x = records.values
    x = [sub[~(sub == 0)].tolist() for sub in x if sub[sub != 0].tolist()]
    transactions = x

    logging.info(f"Example transactions:\n{transactions[0:10]}")
    return transactions
 

def association_rules(transactions):
    """
    This function for: 
        implementing Association Rules Algorithm
    Args:
        transactions: data frame
    Return:    
        List of association rules
    """       
    rules = apriori(transactions,min_support=0.00030,min_confidance=0.05,min_lift=3,min_length=2,target="rules")
    association_results = list(rules)

    for item in association_results:
        pair = item[0] 
        items = [x for x in pair]
        logging.info(f"Rule: {items[0]}  ->  {items[1]}")

        logging.info(f"Support:  {str(item[1])}")
        logging.info(f"Confidence:  {str(item[2][0][2])}")
        logging.info(f"Lift:  {str(item[2][0][3])}")
        logging.info(f"=====================================")


if __name__ == "__main__":
    # Load the dataset
    df = load_data()
    # Get all products
    all_products = get_total_product(df)     

    # Get top 10 products that sold frequently
    x = get_top10_frequently_sold(df)
    # Visualize them
    ditribution_plot(x=x.index, y=x.values, yaxis="Count", xaxis="Products")

    # Turn into one hot representation
    df = get_onehot_presentation(df)

    # Get all records 
    records = df.groupby(["Member_number","Date"])[all_products[:]].apply(sum)
    records = records.reset_index()[all_products]

    records = records.apply(get_Pnames, axis=1)
    records.head()

    logging.info(f"total transactions: {len(records)}")
    # Remove error transactions
    transactions = remove_zeros(records)
    # Appy association_rules
    association_rules(transactions)