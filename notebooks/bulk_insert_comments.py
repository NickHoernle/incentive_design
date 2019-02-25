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
else:
    cursor = conn.cursor()
    count = 0
    fields = ['Id',
            'PostId',
            'Score',
            'Text',
            'CreationDate',
            'UserId']
    SQL = f"INSERT INTO comments ({','.join(fields)}) VALUES ({','.join(['%s' for f in fields])});"

    with open('/home/nick/Downloads/stack_overflow/Comments.xml', 'r') as f:
        line = f.readline()

        to_insert = []
        now = time.time()
        while line:

            try:
                FIELDS = tuple([xmltodict.parse(line)['row'][f'@{field}'] for field in fields])
                to_insert.append(FIELDS)
            except:
                line = f.readline()
                continue

            line = f.readline()
            count += 1

            if len(to_insert) == 500000:

                print(count, time.time()-now)
                now = time.time()

                cursor.executemany(SQL, sorted(to_insert, key=lambda x: int(x[0])))
                to_insert = []

                print(time.time()-now)
                now = time.time()

                conn.commit()

    # Cleanup
    conn.commit()
    cursor.close()
    conn.close()

    print("Done.")
