import pymongo
import certifi
connection=pymongo.MongoClient("Bulut Bağlantı Adresi",tlsCAFile=certifi.where())
db=connection["kitap"]
BooksCollection=db["books"]
UsersCollection=db["users"]
WillReadedCollection=db["willreaded"]


#x = WillReadedCollection.delete_many({})

#print(x.deleted_count, " documents deleted.")

x=WillReadedCollection.find()
print(list(x))