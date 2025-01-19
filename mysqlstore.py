import mysql.connector

# Connect to MySQL without specifying a database
conn = mysql.connector.connect(
    host="localhost",
    user="root",  # Replace with your MySQL username
    password="devaki@123456"  # Replace with your MySQL password
)

cursor = conn.cursor()

# Create the database if it doesn't exist
cursor.execute("CREATE DATABASE IF NOT EXISTS plat;")
cursor.execute("USE plat;")  # Switch to the newly created database

# Drop the table if it already exists
cursor.execute("DROP TABLE IF EXISTS disease_details;")
cursor.execute("DROP TABLE IF EXISTS contact_details;")

# Create the disease_details table without name, email, and message
cursor.execute("""
CREATE TABLE IF NOT EXISTS disease_details (
    id INT AUTO_INCREMENT PRIMARY KEY,
    label VARCHAR(50) NOT NULL,
    symptoms TEXT NOT NULL,
    medicine TEXT NOT NULL,
    cure TEXT NOT NULL
);
""")

# Create the contact_details table with name, email, and message


# Disease details data
disease_details = {
    'eggplantspot': {'Symptoms': 'Leaf spot with purple edges', 'Medicine': 'Fungicide A', 'Cure': 'Prune affected leaves'},
    'healthy': {'Symptoms': 'No symptoms', 'Medicine': 'None', 'Cure': 'Maintain healthy conditions'},
    'potatospot': {'Symptoms': 'Brown spots on leaves', 'Medicine': 'Copper-based fungicide', 'Cure': 'Remove affected leaves'},
    'powdery': {'Symptoms': 'White powdery coating on leaves', 'Medicine': 'Fungicide B', 'Cure': 'Increase air circulation'},
    'rust': {'Symptoms': 'Rust-colored spots on leaves', 'Medicine': 'Rust fungicide', 'Cure': 'Prune infected branches'},
    'tomatospot': {'Symptoms': 'Dark spots on tomato leaves', 'Medicine': 'Tomato fungicide', 'Cure': 'Remove infected leaves'},
    'bananaspot': {'Symptoms': 'Spots on banana leaves', 'Medicine': 'Banana fungicide', 'Cure': 'Prune infected areas'},
    'chilispot': {'Symptoms': 'Spots on chili plant leaves', 'Medicine': 'Chili fungicide', 'Cure': 'Increase sunlight exposure'},
    'cottonspot': {'Symptoms': 'Spots on cotton leaves', 'Medicine': 'Cotton fungicide', 'Cure': 'Remove affected leaves'},
    'sugarcanespot': {'Symptoms': 'Yellow spots on sugarcane leaves', 'Medicine': 'Sugarcane fungicide', 'Cure': 'Remove infected leaves'}
}

# Insert data into the disease_details table
for label, details in disease_details.items():
    cursor.execute("""
    INSERT INTO disease_details (label, symptoms, medicine, cure)
    VALUES (%s, %s, %s, %s);
    """, (label, details['Symptoms'], details['Medicine'], details['Cure']))

# Commit the transaction
conn.commit()

# Insert data into the contact_details table (with empty values for name, email, and message)
for label in disease_details.keys():
    cursor.execute("""
    INSERT INTO contact_details (disease_id, name, email, message)
    VALUES ((SELECT id FROM disease_details WHERE label = %s), '', '', '');
    """, (label,))

# Commit the transaction
conn.commit()

print("Data has been successfully inserted into the MySQL database!")

# Retrieve and display data from both tables
cursor.execute("""
SELECT dd.label, dd.symptoms, dd.medicine, dd.cure, cd.name, cd.email, cd.message
FROM disease_details dd
LEFT JOIN contact_details cd ON dd.id = cd.disease_id;
""")
rows = cursor.fetchall()

print("\nData in 'disease_details' and 'contact_details' tables:")
for row in rows:
    print(row)

# Close the connection
cursor.close()
conn.close()
