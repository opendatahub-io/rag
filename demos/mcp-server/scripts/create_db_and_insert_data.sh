#!/bin/bash
# Copyright 2025 IBM, Red Hat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Set PostgreSQL connection parameters for Docker
export PGHOST=postgres
export PGUSER=postgres
export PGPASSWORD=postgres

# Terminate active connections to the database
psql -d postgres -c "SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = 'bank_statements' AND pid <> pg_backend_pid();"

# Drop the database
psql -d postgres -c "DROP DATABASE IF EXISTS bank_statements;"
psql -d postgres -c "CREATE DATABASE bank_statements;"

psql -d bank_statements <<'SQL'
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
('Emily Davis', '1992-05-30', '321 Elm St, Houston, TX 77001', '+1-555-321-0987');

-- Insert sample statements
INSERT INTO statements (user_id, date, total) VALUES
(1, '2024-01-31', 2450.75),
(1, '2024-02-29', 1890.25),
(2, '2024-01-31', 3200.50),
(2, '2024-02-29', 2750.80),
(3, '2024-01-31', 1650.00),
(4, '2024-01-31', 4100.25);

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
(6, 'Shopping', -299.50, '2024-01-15');
SQL
