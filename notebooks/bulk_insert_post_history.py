import mysql.connector
from mysql.connector import errorcode
import xmltodict
import time

# Obtain connection string information from the portal
config = {
  'host': 'stack-overflow-db.mysql.database.azure.com',
  'user': 'stack-overflow@stack-overflow-db',
  'password': 'password1!',
  'database': 'stackoverflow'
}

count = 0
fields = [
    'Id',
    'PostHistoryTypeId',
    'PostId',
    'RevisionGUID',
    'CreationDate',
    'UserId',
    'Comment',
    'Text']

SQL = f"INSERT INTO posthistory ({','.join(fields)}) VALUES ({','.join(['%s' for f in fields])});"
done_to = 72500001
limit = 250000
with open('/home/nick/Downloads/stack_overflow/PostHistory.xml', 'r') as f:
    line = f.readline()

    to_insert = []
    now = time.time()
    bulk = False

    while line:

        count += 1
        line = f.readline()

        if count < done_to:
            continue

        elif count == done_to:
            print('made last point of contact')

            # Construct connection string
            try:
                conn = mysql.connector.connect(**config)
                print("Connection established")
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                    print("Something is wrong with the user name or password")
                elif err.errno == errorcode.ER_BAD_DB_ERROR:
                    print("Database does not exist")
                else:
                    print(err)
                break

            cursor = conn.cursor()

            to_insert = []
            continue

        try:
            xml = xmltodict.parse(line)['row']
        except:
            print('Cannot parse,', line)
            continue

        FIELDS = tuple([xml[f'@{field}'] if f'@{field}' in xml else '' for field in fields])
        to_insert.append(FIELDS)

        if len(to_insert) == limit:

            print(count, time.time()-now)
            now = time.time()

            cursor.executemany(SQL, sorted(to_insert, key=lambda x: int(x[0])))
            conn.commit()

            to_insert = []

            print(time.time()-now)
            now = time.time()

            SQL = f"INSERT INTO posthistory ({','.join(fields)}) VALUES ({','.join(['%s' for f in fields])});"
            limit = 250000

    # Cleanup
    conn.commit()
    cursor.close()
    conn.close()

    print("Done.")
