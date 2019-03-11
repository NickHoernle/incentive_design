import mysql.connector # pip install mysql-connector-python-rf
from mysql.connector import errorcode
import xmltodict
import time
import os
import pandas as pd
import numpy as np

# Obtain connection string information from the portal
config = {
  'host': 'stack-overflow-db.mysql.database.azure.com',
  'user': 'stack-overflow@stack-overflow-db',
  'password': 'password1!',
  'database': 'stackoverflow'
}

date_from = '2017-01-01'
date_to = '2017-06-01'

def load_data_from_azure(file_name, mysql_string):
    if os.path.exists(file_name):
        df = pd.read_pickle(file_name)
    else:
        try:
            conn = mysql.connector.connect(**config)
            df = pd.read_sql_query(
                mysql_string, con=conn)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with the user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)
        conn.close()
        # cache the result
        df.to_pickle(file_name)
    return df

def load_silver_users():
    my_sql = f'''SELECT b1.UserId, b1.Name, b1.Date FROM badges as b1
            WHERE b1.Name = 'Strunk & White' AND
            b1.Date BETWEEN '{date_from}' AND '{date_to}';'''
    file_name = '../data/df_silver_editers.pkl'
    df = load_data_from_azure(file_name, my_sql)
    return df

def load_gold_users():
    my_sql = f'''SELECT b1.UserId, b1.Name, b1.Date FROM badges as b1
            WHERE b1.Name = 'Copy Editor' AND
            b1.Date >= '{date_from}';'''
    file_name = '../data/df_gold_editers.pkl'
    df = load_data_from_azure(file_name, my_sql)
    return df

def load_silver_editor_posts():
    my_sql = f'''SELECT p.Id, p.PostTypeId, p.Score, p.OwnerUserId, p.Tags, p.CreationDate
            FROM posts as p
            INNER JOIN badges as b
            ON (
                b.Name = 'Strunk & White' AND
                b.UserId = p.OwnerUserId AND
                b.Date BETWEEN '{date_from}' AND '{date_to}'
            );'''
    file_name = '../data/df_silver_editor_posts.pkl'
    df = load_data_from_azure(file_name, my_sql)
    return df

def load_silver_editor_edits():
    my_sql = f'''SELECT ph.Id, ph.PostHistoryTypeId, ph.PostId, ph.UserId, ph.CreationDate
            FROM posthistory as ph
            INNER JOIN badges as b
            ON (
                b.Name = 'Strunk & White' AND
                b.UserId = ph.UserId AND
                b.Date BETWEEN '{date_from}' AND '{date_to}'
            );'''
    file_name = '../data/df_silver_editor_edits.pkl'
    df = load_data_from_azure(file_name, my_sql)
    return df

def load_silver_editor_comments():
    my_sql = f'''SELECT c.Id, c.UserId, c.CreationDate FROM comments as c
            INNER JOIN badges as b
            ON (
                b.Name = 'Strunk & White' AND
                b.UserId = c.UserId AND
                b.Date BETWEEN '{date_from}' AND '{date_to}'
            );'''
    file_name = '../data/df_silver_editor_comments.pkl'
    df = load_data_from_azure(file_name, my_sql)
    return df
