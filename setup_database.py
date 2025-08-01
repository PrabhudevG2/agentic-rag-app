import sqlite3
import os

DB_FILE = "company.db"

def create_database():
    """Creates and populates the SQLite database."""
    # Delete the old database file if it exists
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    print("Creating tables...")
    # Create employees table
    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER
        );
    """)

    # Create products table
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL
        );
    """)

    # Create sales table with foreign keys
    cursor.execute("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            product_id INTEGER,
            employee_id INTEGER,
            quantity INTEGER,
            FOREIGN KEY(product_id) REFERENCES products(id),
            FOREIGN KEY(employee_id) REFERENCES employees(id)
        );
    """)

    print("Inserting data...")
    # Insert employee data
    employees = [
        (1, 'Alice', 'Engineering', 90000),
        (2, 'Bob', 'Sales', 75000),
        (3, 'Charlie', 'Engineering', 110000),
        (4, 'Diana', 'Sales', 82000)
    ]
    cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?)", employees)

    # Insert product data
    products = [
        (1, 'Laptop', 1200.00),
        (2, 'Mouse', 25.50),
        (3, 'Keyboard', 75.00)
    ]
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?)", products)

    # Insert sales data
    sales = [
        (1, 1, 2, 5),  # Bob sold 5 Laptops
        (2, 3, 4, 10), # Diana sold 10 Keyboards
        (3, 2, 2, 8)   # Bob sold 8 Mice
    ]
    cursor.executemany("INSERT INTO sales VALUES (?, ?, ?, ?)", sales)

    conn.commit()
    conn.close()
    print(f"Database '{DB_FILE}' created successfully.")

if __name__ == "__main__":
    create_database()