import pymongo
import certifi

connection=pymongo.MongoClient("Bulut Bağlantı Adresi",tlsCAFile=certifi.where())
db=connection["kitap"]
BooksCollection=db["books"]
UsersCollection=db["users"]

import openpyxl
workbook = openpyxl.load_workbook('Kitap1.xlsx')

sheet = workbook.active

# Iterate over all rows in the sheet
for row in sheet.iter_rows(min_row=2,values_only=True):
    # Print the values of columns 0 and 1

    schema={
        "title":row[0],
        "author":row[1],
        "publisher":row[2],
        "type":row[3],
        "rating":row[4],
        "link":row[5],
        "image":row[6]

    }
    print(schema)
    BooksCollection.insert_one(schema)
# Close the workbook
workbook.close()
