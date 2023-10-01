#împort libraries 
from bson import ObjectId
from flask import Flask, render_template, request, jsonify, redirect, url_for,session
import pymongo
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import certifi

app = Flask(__name__)
app.static_folder = 'static'

#Database Connection Operations
connection=pymongo.MongoClient("Bulut Bağlantı Adresi",tlsCAFile=certifi.where())
db=connection["kitap"]
BooksCollection=db["books"]
UsersCollection=db["users"]
WillReadedCollection=db["willreaded"]

##Fetch all book titles
book_title=BooksCollection.find({},{"_id":0,"title":1})
# Dummy user data for demonstration purposes

#Bring all the books
books=list(BooksCollection.find())
data = pd.DataFrame(books)

# Convert data types to string
data['author'] = data['author'].astype(str)
data['title'] = data['title'].astype(str)
data['publisher'] = data['publisher'].astype(str)

# Preprocess the dataset
book_features = data['title'] + ' ' + data['author'] + ' ' + data['publisher']
# Load the stop words list
stop_words = set(stopwords.words('turkish'))

# Create a stemmer for stemming words
stemmer = PorterStemmer()


# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Stem words
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    return text


# Apply preprocessing to the text
book_features = book_features.apply(preprocess_text)

# Create TfidfVectorizer
vectorizer = TfidfVectorizer()
book_feature_matrix = vectorizer.fit_transform(book_features)

# Genetic Algorithm Parameters
population_size = 100
max_generations = 50
mutation_rate = 0.1

# Initialize similarity matrix
similarity_matrix = None


# Create initial population
def create_initial_population(size):
    population = []
    for _ in range(size):
        chromosome = [random.choice([0, 1]) for _ in range(len(data))]
        population.append(chromosome)
    return population


# Fitness function
def fitness(chromosome):
    global similarity_matrix
    selected_books = [book_features[i] for i in range(1) if chromosome[i] == 1]
    if len(selected_books) == 0:
        return 0

    selected_features = vectorizer.transform(selected_books)

    if similarity_matrix is None:
        similarity_matrix = cosine_similarity(selected_features, book_feature_matrix)
    else:
        similarity_matrix += cosine_similarity(selected_features, book_feature_matrix)

    similarity_scores = np.sum(similarity_matrix, axis=0)

    return np.max(similarity_scores)


# Selection process
#def selection(population, fitness_values):
#    sorted_population = [x for _, x in sorted(zip(fitness_values, population), reverse=True)]
#    return sorted_population[:int(0.5 * len(sorted_population))]

def selection(population, fitness_values):
    selected_population = []
    tournament_size = 2

    while len(selected_population) < int(0.3 * len(population)):
        participants = random.sample(population, tournament_size)
        tournament_fitness = [fitness_values[population.index(participant)] for participant in participants]
        winner = participants[tournament_fitness.index(max(tournament_fitness))]
        selected_population.append(winner)

    return selected_population

# Crossover operation
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


# Mutation process
def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome


# Generating book recommendations with genetic algorithm
def generate_book_recommendations(data, query_book):
    population = create_initial_population(population_size)

    for generation in range(max_generations):
        fitness_values = [fitness(chromosome) for chromosome in population]
        selected_population = selection(population, fitness_values)

        # Elitism step
        elite_size = int(0.1 * len(selected_population))
        elite = selected_population[:elite_size]
        # Add elite individuals to the new population
        new_population = elite.copy()

        while len(new_population) < population_size:
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            child = crossover(parent1, parent2)
            child = mutation(child)
            new_population.append(child)

        population = new_population

    best_chromosome = max(population, key=fitness)
    selected_books = [data.iloc[i] for i in range(len(best_chromosome)) if best_chromosome[i] == 1]
    recommended_books = random.sample(selected_books, 3)

    return recommended_books


#If the user is logged in, redirect to main_page, if not, redirect to login page
@app.route('/')
def index():
    if 'logged_in' in session:

        return redirect(url_for('main_page'))
    else:
        return redirect(url_for('login'))


#login procedures and controls
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user=UsersCollection.find_one({"username":username},{"_id":1,"username":1,"password":1})
        if user:
            if user["password"]==password:
                session['logged_in'] = True
                session['user_id']=str(user['_id'])
                return redirect(url_for('main_page'))
            else:
                return render_template('login.html', error='Invalid password.')

        else:
            return render_template('login.html', error='Invalid username or password.')
    return render_template('login.html')

#Registration procedures and controls
@app.route('/signup', methods=['GET', 'POST'])
def signup():

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirmation_password= request.form['confirm_password']
        user=UsersCollection.find_one({"username":username},{"_id":0,"username":1,"password":1})
        if user:
            return render_template('signup.html', error='Username already exists.')
        if password!=confirmation_password:
            return render_template('signup.html', error='Passwords do not match.')
        if username=="" or password=="":
            return render_template('signup.html', error='Please fill in all the fields.')


        else:
           UsersCollection.insert_one({"username":username,"password":password})
           return render_template('login.html', success='Sign up successful. You can now log in.')

        # Add new user to the users list (in a real application, you would store the user in a database)
    return render_template('signup.html')

#Main page
@app.route('/main')
def main_page():

    if 'logged_in' in session:
        book_title = BooksCollection.find({}, {"_id": 0, "title": 1})

        return render_template('main.html', book_title=list(book_title))
    else:
        return redirect(url_for('login'))
    
#Checkout process
@app.route('/logout')
def logout():
    session.pop('logged_in', None)  # Remove session variable on logout
    return redirect(url_for('login'))

#Book recommender system
@app.route('/recommend', methods=['POST'])
def recommend():
    book_title=BooksCollection.find()
    book_list=[]
    for book in book_title:
        book_id = str(book['_id'])
        book_title = book['title']
        book_list.append({'id': book_id, 'title': book_title})

    selected_data = request.form.get('selected_data')
    recommendations = generate_book_recommendations(data, selected_data)
    recommendations = sorted(recommendations, key=lambda x: x['rating'], reverse=True)
    print(list(book_title))
    return render_template('main.html', book_title=book_list, selected_data=selected_data, books=recommendations)

@app.route('/willreaded',methods=['GET','POST'])
def willreaded():
    if request.method=="GET":
        book_ids=[]
        booksreaded=WillReadedCollection.find({"user_id":str(session["user_id"])})
        for i in booksreaded:
            book_ids.append(i["book_id"])
        bookslist = BooksCollection.find({'_id': {'$in': book_ids}},{"_id":0})
        return render_template('willreaded.html',books=list(bookslist))
    if request.method=="POST":
        book_id = request.form.get('book_id')
        schema={
            "book_id":ObjectId(book_id),
            "user_id":str(session["user_id"])
        }
        WillReadedCollection.insert_one(schema)
        return redirect(url_for('willreaded'))

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(port=5004)
