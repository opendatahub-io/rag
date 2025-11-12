-- Create bank_statements database
CREATE DATABASE bank_statements;

-- Connect to bank_statements database
\c bank_statements;

-- Drop tables in correct order (child tables first)
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS statements CASCADE;
DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE users (
  user_id           SERIAL PRIMARY KEY,
  name              text NOT NULL,
  date_of_birth     date NOT NULL,
  address           text NOT NULL,
  phone_number      text NOT NULL UNIQUE
);

CREATE TABLE statements (
  id                SERIAL PRIMARY KEY,
  user_id           integer NOT NULL,
  date              date NOT NULL,
  total             NUMERIC(10,2) NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE transactions (
  id                SERIAL PRIMARY KEY,
  statement_id      integer NOT NULL,
  description       text NOT NULL,
  price             NUMERIC(10,2) NOT NULL,
  date              date NOT NULL,
  FOREIGN KEY (statement_id) REFERENCES statements(id)
);

-- Insert sample users
INSERT INTO users (name, date_of_birth, address, phone_number) VALUES
('John Smith', '1985-03-15', '123 Main St, New York, NY 10001', '+1-555-123-4567'),
('Sarah Johnson', '1990-07-22', '456 Oak Ave, Los Angeles, CA 90210', '+1-555-987-6543'),
('Michael Brown', '1978-11-08', '789 Pine Rd, Chicago, IL 60601', '+1-555-456-7890'),
('Emily Davis', '1992-05-30', '321 Elm St, Houston, TX 77001', '+1-555-321-0987'),
('Ignas Baranauskas', '1988-06-12', '15 Grafton Street, Dublin 2, Ireland', '+353 85 148 0072');

-- Insert sample statements
INSERT INTO statements (user_id, date, total) VALUES
(1, '2024-01-31', 2450.75),
(1, '2024-02-29', 1890.25),
(2, '2024-01-31', 3200.50),
(2, '2024-02-29', 2750.80),
(3, '2024-01-31', 1650.00),
(4, '2024-01-31', 4100.25),
-- Ignas Baranauskas statements (2025)
(5, '2025-01-31', 4250.80),
(5, '2025-02-28', 3890.45),
(5, '2025-03-31', 4120.30),
(5, '2025-04-30', 3750.60),
(5, '2025-05-31', 4380.25),
(5, '2025-06-30', 3950.90),
(5, '2025-07-31', 4560.15),
(5, '2025-08-31', 4200.75),
(5, '2025-09-24', 3825.40);

-- Insert sample transactions
INSERT INTO transactions (statement_id, description, price, date) VALUES
-- John Smith's January statement transactions
(1, 'Salary Deposit', 3000.00, '2024-01-01'),
(1, 'Grocery Store', -125.50, '2024-01-03'),
(1, 'Gas Station', -45.75, '2024-01-05'),
(1, 'Electric Bill', -89.50, '2024-01-10'),
(1, 'Restaurant', -67.25, '2024-01-12'),
(1, 'ATM Withdrawal', -200.00, '2024-01-15'),
(1, 'Online Purchase', -21.25, '2024-01-20'),

-- John Smith's February statement transactions
(2, 'Salary Deposit', 3000.00, '2024-02-01'),
(2, 'Grocery Store', -145.30, '2024-02-02'),
(2, 'Phone Bill', -75.00, '2024-02-05'),
(2, 'Insurance', -250.00, '2024-02-08'),
(2, 'Restaurant', -89.45, '2024-02-14'),
(2, 'Gas Station', -50.00, '2024-02-18'),
(2, 'Online Shopping', -500.00, '2024-02-25'),

-- Sarah Johnson's January statement transactions
(3, 'Salary Deposit', 4500.00, '2024-01-01'),
(3, 'Rent Payment', -1200.00, '2024-01-01'),
(3, 'Grocery Store', -89.50, '2024-01-04'),
(3, 'Coffee Shop', -15.75, '2024-01-06'),
(3, 'Gym Membership', -45.00, '2024-01-10'),
(3, 'Utilities', -125.25, '2024-01-15'),

-- Sarah Johnson's February statement transactions
(4, 'Salary Deposit', 4500.00, '2024-02-01'),
(4, 'Rent Payment', -1200.00, '2024-02-01'),
(4, 'Car Payment', -350.00, '2024-02-05'),
(4, 'Grocery Store', -110.20, '2024-02-07'),
(4, 'Medical Bill', -89.00, '2024-02-12'),

-- Michael Brown's January statement transactions
(5, 'Freelance Payment', 2000.00, '2024-01-15'),
(5, 'Mortgage Payment', -800.00, '2024-01-01'),
(5, 'Utilities', -150.00, '2024-01-05'),
(5, 'Grocery Store', -75.50, '2024-01-08'),
(5, 'Gas Station', -40.25, '2024-01-12'),
(5, 'Internet Bill', -65.75, '2024-01-20'),

-- Emily Davis's January statement transactions
(6, 'Salary Deposit', 5000.00, '2024-01-01'),
(6, 'Investment Deposit', 500.00, '2024-01-05'),
(6, 'Rent Payment', -1500.00, '2024-01-01'),
(6, 'Car Lease', -400.00, '2024-01-03'),
(6, 'Grocery Store', -200.25, '2024-01-07'),
(6, 'Shopping', -299.50, '2024-01-15'),

-- Ignas Baranauskas's 2025 transactions
-- January 2025 (statement_id 7)
(7, 'Salary Deposit', 5500.00, '2025-01-01'),
(7, 'Rent Payment', -1200.00, '2025-01-01'),
(7, 'Tesco Grocery', -89.45, '2025-01-03'),
(7, 'Eir Internet', -65.00, '2025-01-05'),
(7, 'ESB Electricity', -125.30, '2025-01-08'),
(7, 'Dublin Bus', -25.00, '2025-01-10'),
(7, 'Penneys Shopping', -78.95, '2025-01-12'),
(7, 'Centra Coffee', -4.50, '2025-01-15'),
(7, 'Luas Transport', -15.60, '2025-01-18'),
(7, 'SuperValu', -95.20, '2025-01-20'),

-- February 2025 (statement_id 8)
(8, 'Salary Deposit', 5500.00, '2025-02-01'),
(8, 'Rent Payment', -1200.00, '2025-02-01'),
(8, 'Dunnes Stores', -112.35, '2025-02-02'),
(8, 'Vodafone Mobile', -45.00, '2025-02-05'),
(8, 'Gas Company', -89.50, '2025-02-07'),
(8, 'Temple Bar Pub', -35.80, '2025-02-14'),
(8, 'Grafton Street Shopping', -156.70, '2025-02-16'),
(8, 'Costa Coffee', -8.20, '2025-02-18'),
(8, 'Pharmacy', -23.45, '2025-02-20'),
(8, 'Lidl Grocery', -67.45, '2025-02-25'),

-- March 2025 (statement_id 9)
(9, 'Salary Deposit', 5500.00, '2025-03-01'),
(9, 'Rent Payment', -1200.00, '2025-03-01'),
(9, 'St. Patrick''s Day Celebration', -85.60, '2025-03-17'),
(9, 'Aldi Grocery', -78.90, '2025-03-05'),
(9, 'Dublin Gym Membership', -55.00, '2025-03-10'),
(9, 'Brown Thomas', -245.80, '2025-03-15'),
(9, 'Insomnia Coffee', -6.75, '2025-03-20'),
(9, 'Trinity College Bookstore', -89.50, '2025-03-22'),
(9, 'Spar Convenience', -34.25, '2025-03-25'),
(9, 'Dublin Airport Parking', -15.00, '2025-03-28'),

-- April 2025 (statement_id 10)
(10, 'Salary Deposit', 5500.00, '2025-04-01'),
(10, 'Rent Payment', -1200.00, '2025-04-01'),
(10, 'Easter Weekend Trip', -320.50, '2025-04-20'),
(10, 'Tesco Express', -45.80, '2025-04-03'),
(10, 'Health Insurance', -125.00, '2025-04-05'),
(10, 'Guinness Storehouse', -28.50, '2025-04-12'),
(10, 'Zara Shopping', -189.90, '2025-04-15'),
(10, 'Starbucks', -12.40, '2025-04-18'),
(10, 'Boots Pharmacy', -67.30, '2025-04-22'),
(10, 'Fresh Market', -89.60, '2025-04-25'),

-- May 2025 (statement_id 11)
(11, 'Salary Deposit', 5500.00, '2025-05-01'),
(11, 'Rent Payment', -1200.00, '2025-05-01'),
(11, 'Bank Holiday Weekend', -145.75, '2025-05-05'),
(11, 'Marks & Spencer', -98.40, '2025-05-08'),
(11, 'Car Insurance', -180.00, '2025-05-10'),
(11, 'Kilmainham Gaol Tour', -15.00, '2025-05-12'),
(11, 'H&M Shopping', -67.85, '2025-05-15'),
(11, 'Bewley''s Cafe', -9.60, '2025-05-18'),
(11, 'Phoenix Park Parking', -8.00, '2025-05-20'),
(11, 'SuperValu Weekly Shop', -134.65, '2025-05-25'),

-- June 2025 (statement_id 12)
(12, 'Salary Deposit', 5500.00, '2025-06-01'),
(12, 'Rent Payment', -1200.00, '2025-06-01'),
(12, 'June Bank Holiday', -89.30, '2025-06-02'),
(12, 'Arnotts Department Store', -156.90, '2025-06-05'),
(12, 'Dublin Zoo', -22.50, '2025-06-08'),
(12, 'Spar Grocery', -56.75, '2025-06-10'),
(12, 'Trinity College Library', -12.00, '2025-06-12'),
(12, 'Nespresso Coffee', -7.80, '2025-06-15'),
(12, 'Grafton Barber', -35.00, '2025-06-18'),
(12, 'Tesco Online Delivery', -78.95, '2025-06-22'),

-- July 2025 (statement_id 13)
(13, 'Salary Deposit', 5500.00, '2025-07-01'),
(13, 'Rent Payment', -1200.00, '2025-07-01'),
(13, 'Summer Festival', -125.60, '2025-07-15'),
(13, 'Jervis Shopping Centre', -189.40, '2025-07-05'),
(13, 'Dublin Castle Tour', -18.00, '2025-07-08'),
(13, 'Centra Lunch', -8.95, '2025-07-10'),
(13, 'Poolbeg Towers Hike Parking', -5.00, '2025-07-12'),
(13, 'Insomnia Coffee Chain', -15.60, '2025-07-15'),
(13, 'Howth Market', -45.80, '2025-07-18'),
(13, 'Dunnes Stores Weekly', -167.80, '2025-07-25'),

-- August 2025 (statement_id 14)
(14, 'Salary Deposit', 5500.00, '2025-08-01'),
(14, 'Rent Payment', -1200.00, '2025-08-01'),
(14, 'August Bank Holiday', -95.75, '2025-08-05'),
(14, 'Brown Thomas Sale', -234.60, '2025-08-08'),
(14, 'Cliffs of Moher Trip', -85.00, '2025-08-10'),
(14, 'Spar Express', -34.50, '2025-08-12'),
(14, 'Dublin Bike Rental', -12.00, '2025-08-15'),
(14, 'Butler''s Chocolate', -18.90, '2025-08-18'),
(14, 'Avoca Handweavers', -67.45, '2025-08-20'),
(14, 'Aldi Weekly Shop', -89.80, '2025-08-25'),

-- September 2025 (statement_id 15 - up to Sept 24)
(15, 'Salary Deposit', 5500.00, '2025-09-01'),
(15, 'Rent Payment', -1200.00, '2025-09-01'),
(15, 'Back to School Shopping', -145.90, '2025-09-03'),
(15, 'Kilkenny Shop', -78.40, '2025-09-05'),
(15, 'National Gallery', -12.00, '2025-09-08'),
(15, 'Spar Morning Coffee', -4.20, '2025-09-10'),
(15, 'Dublin Mountains Hike', -8.50, '2025-09-12'),
(15, 'Fallon & Byrne', -89.60, '2025-09-15'),
(15, 'Temple Bar Gallery', -15.00, '2025-09-18'),
(15, 'SuperValu Organic', -123.80, '2025-09-20'),
(15, 'Dublin Writers Museum', -9.50, '2025-09-22'),
(15, 'Centra Snacks', -6.40, '2025-09-24');
