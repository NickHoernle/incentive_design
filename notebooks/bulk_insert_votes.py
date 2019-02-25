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

# Construct connection string
count = 0
fields = [
    'Id',
    'PostId',
    'VoteTypeId',
    'CreationDate',
    'UserId',
    'BountyAmount']

SQL = f"INSERT INTO votes ({','.join(fields)}) VALUES ({','.join(['%s' for f in fields])});"
done_to = 59901448
with open('/home/nick/Downloads/stack_overflow/Votes.xml', 'r') as f:
    line = f.readline()

    to_insert = []
    now = time.time()
    while line:

        try:
            xml = xmltodict.parse(line)['row']
        except:
            line = f.readline()
            continue

        FIELDS = tuple([xml[f'@{field}'] if f'@{field}' in xml else '' for field in fields])
        to_insert.append(FIELDS)

        line = f.readline()
        count += 1

        if count < done_to:
            del to_insert[0]
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

        if len(to_insert) == 1000:

            print(count, time.time()-now)
            now = time.time()

            cursor.executemany(SQL, sorted(to_insert, key=lambda x: int(x[0])))
            conn.commit()

            to_insert = []

            print(time.time()-now)
            now = time.time()

    # Cleanup
    conn.commit()
    cursor.close()
    conn.close()

    print("Done.")