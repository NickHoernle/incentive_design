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

date_from = '2016-01-01'
date_to = '2017-01-01'
date_from2 = '2015-06-01'
date_to2 = '2017-06-01'

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
                b.Date BETWEEN '{date_from}' AND '{date_to}' AND
                ph.CreationDate BETWEEN '{date_from2}' AND '{date_to2}'
            );'''
    file_name = '../data/df_silver_editor_edits.pkl'
    df = load_data_from_azure(file_name, my_sql)
    return df

def load_silver_editor_comments():
    my_sql = f'''SELECT c.Id, c.PostId, c.UserId, c.CreationDate FROM comments as c
            INNER JOIN badges as b
            ON (
                b.Name = 'Strunk & White' AND
                b.UserId = c.UserId AND
                b.Date BETWEEN '{date_from}' AND '{date_to}' AND
                c.CreationDate BETWEEN '{date_from2}' AND '{date_to2}'
            );'''
    file_name = '../data/df_silver_editor_comments.pkl'
    df = load_data_from_azure(file_name, my_sql)
    return df

def load_not_silver_editers():
    my_sql = f'''
    SELECT ph2.UserId, COUNT(*) as count
    FROM posthistory as ph2
    WHERE ph2.UserId not in (SELECT b.UserId FROM Badges as b WHERE b.Name = 'Strunk & White') AND ph2.CreationDate < '2017-06-01'
    GROUP BY UserId
    HAVING COUNT(*) > 40;
    LIMIT 5000;
    '''
    file_name = '../data/df_not_silver_editers.pkl'
    df = load_data_from_azure(file_name, my_sql)
    return df

def load_not_silver_editors_edits(user_ids):
    my_sql = f'''
    SELECT ph.Id, ph.PostHistoryTypeId, ph.PostId, ph.UserId, ph.CreationDate
    FROM posthistory as ph
    WHERE ph.UserId in ({user_ids});
    '''
    file_name = '../data/df_not_silver_editors_edits.pkl'
    df = load_data_from_azure(file_name, my_sql)
    return df


# ALTER TABLE badges ADD FOREIGN KEY (UserId) REFERENCES users(Id);
# ALTER TABLE posthistory ADD FOREIGN KEY (UserId) REFERENCES users(Id);
# ALTER TABLE posthistory ADD FOREIGN KEY (UserId) REFERENCES badges(UserId);
# ALTER TABLE posthistory ADD FOREIGN KEY (PostId) REFERENCES posts(Id);
# ALTER TABLE posthistory ADD FOREIGN KEY (PostId) REFERENCES posts(Id);

{
  "body": "{   \n    \"type\": \"notification_event\",\n    \"app_id\": \"anAppId\",\n    \"data\": {\n      \"type\": \"notification_event_data\",\n      \"item\": {\n        \"type\": \"conversation\",\n        \"id\": \"111111111\",\n        \"created_at\": 111111111,\n        \"updated_at\": 111111111,\n        \"user\": {\n          \"type\": \"user\",\n          \"id\": \"111111111\",\n          \"user_id\": \"111111111\",\n          \"name\": \"a name\",\n          \"email\": \"email@email.com\",\n          \"do_not_track\": null\n        },\n        \"assignee\": {\n          \"type\": \"nobody_admin\",\n          \"id\": null\n        },\n        \"conversation_message\": {\n          \"type\": \"conversation_message\",\n          \"id\": \"111111111\",\n          \"url\": null,\n          \"subject\": \"\",\n          \"body\": \"<p>I forgot my password</p>\",\n          \"author\": {\n            \"type\": \"user\",\n            \"id\": \"111111111\"\n          },\n          \"attachments\": []\n        },\n        \"conversation_parts\": {\n          \"type\": \"conversation_part.list\",\n          \"conversation_parts\": [],\n          \"total_count\": 0\n        },\n        \"open\": true,\n        \"state\": \"open\",\n        \"read\": true,\n        \"metadata\": {},\n        \"tags\": {\n          \"type\": \"tag.list\",\n          \"tags\": []\n        },\n        \"tags_added\": {\n          \"type\": \"tag.list\",\n          \"tags\": []\n        },\n        \"links\": {\n          \"conversation_web\": \"https://app.intercom.io/\"\n        }\n      }\n    },\n    \"links\": {},\n    \"id\": \"notif_something\",\n    \"topic\": \"conversation.user.created\",\n    \"delivery_status\": \"pending\",\n    \"delivery_attempts\": 1,\n    \"delivered_at\": 0,\n    \"first_sent_at\": 111111111,\n    \"created_at\": 111111111,\n    \"self\": null\n    }"
}
