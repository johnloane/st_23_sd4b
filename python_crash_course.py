from sys import getsizeof
print("Hello World!")

print(13//3)
print(6%2)
print(6**2)

age = 20
print(type(age))
print(getsizeof(age))

blood_type = "AB"

print(int(7/3.5))

dialogue = 'John said "This is some dialogue"'
print(dialogue)

sd4a_and_sd4b_are_the_best = True

print(not(1<2) and 2 < 4)

print(dialogue*10)

print(len(dialogue))

movie_title = "Barbie"
print(movie_title.upper())

# List is mutable and ordered
names = ["Chia", "Daniel", "Marek", "Elisa"]
print(names[0])
#print(names[100])

random_variables = [True, False, "Hello", 1, 1.2]
print(len(random_variables))
length = len(random_variables)
print(random_variables[-1])

ordered_numbers = list(range(0, 10))
print(ordered_numbers)

# Slicing

print(ordered_numbers[0:3])
print(ordered_numbers[0:10:2])
print(ordered_numbers[:10])
print(ordered_numbers[2:])

show_titles = "Breaking Bad"
print(show_titles[2])
print(show_titles[9:])

months = ["January", "February", "March", "April", "May", "June", 
"July", "August", "September", "October"]
print("January" in months)
print("January2" in months)

grocery_list = ["Milk", "Eggs", "Bread", "Butter", "Cheese"]
grocery_list[2] = "Yogurt"

misspelled_vegetable = "Carot"

name = "Nathan"
other_name = name
name = "Michael"
print(other_name)

books = ["The collection of Sherlock Holmes", "Coding for dummies", 
"Data structure and algorithms"]
more_books = books
books[0] = "Song of fire and ice"
print(books[0])
print(more_books)

numbers = list(range(100))
print(len(numbers))
print(max(numbers))
print(min(numbers))
names = ["Chia", "Daniel", "Marek", "Elisa"]
print(sorted(names))
print(names)

name = "Nathan"
class_rank = "the best"

print(f"{name} is {class_rank}")

# Tuples are immutable and ordered
traits = ("Tall", "Handsome", "Smart")
print(traits[0])
height, looks, intelligence = traits

# Sets mutable and unordered and unique
numbers = [1, 2, 3, 4, 5, 5, 5, 5, 5]
unique_numbers = set(numbers)
unique_numbers.add(6)
unique_numbers.add(1)
print(unique_numbers)
print(7 in unique_numbers)

# Dictionaries are mutable and unordered
# keys should be unique and immutable

inventory = {"apples": 430, "bananas": 312, "oranges": 525, "pears": 217}
print(inventory["apples"])
inventory["pears"] = 0
inventory["grapes"] = 5000
print(inventory)

strawberries_amount = inventory.get("strawberries")
print(strawberries_amount)

print("strawberries" in inventory)




