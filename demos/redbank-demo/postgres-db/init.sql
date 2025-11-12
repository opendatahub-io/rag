
CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(50),
    address TEXT,
    account_type VARCHAR(50),
    date_of_birth DATE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS statements (
    statement_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    statement_period_start DATE NOT NULL,
    statement_period_end DATE NOT NULL,
    balance DECIMAL(15, 2) NOT NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id SERIAL PRIMARY KEY,
    statement_id INTEGER NOT NULL,
    transaction_date TIMESTAMP NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,
    description TEXT,
    transaction_type VARCHAR(50) NOT NULL,
    merchant VARCHAR(255),
    FOREIGN KEY (statement_id) REFERENCES statements(statement_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_statements_customer_id ON statements(customer_id);
CREATE INDEX IF NOT EXISTS idx_statements_period ON statements(statement_period_start, statement_period_end);
CREATE INDEX IF NOT EXISTS idx_transactions_statement_id ON transactions(statement_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(transaction_type);

INSERT INTO customers (name, email, phone, address, account_type, date_of_birth) VALUES
    ('Alice Johnson', 'alice.johnson@email.com', '555-0101', '123 Oak Street, Springfield, IL', 'CHECKING', '1990-03-15'),
    ('Bob Smith', 'bob.smith@email.com', '555-0102', '456 Maple Ave, Boston, MA', 'SAVINGS', '1985-07-22'),
    ('Carol Williams', 'carol.williams@email.com', '555-0103', '789 Pine Road, Seattle, WA', 'CHECKING', '1992-11-08'),
    ('David Brown', 'david.brown@email.com', '555-0104', '321 Elm Drive, Austin, TX', 'BUSINESS', '1980-05-30')
ON CONFLICT (email) DO NOTHING;

INSERT INTO statements (customer_id, statement_period_start, statement_period_end, balance) VALUES
    -- Alice Johnson's statements
    (1, '2025-01-01', '2025-01-31', 5420.50),
    (1, '2025-02-01', '2025-02-28', 6780.25),
    (1, '2025-03-01', '2025-03-31', 3210.75),
    -- Bob Smith's statements
    (2, '2025-01-01', '2025-01-31', 12500.00),
    (2, '2025-02-01', '2025-02-28', 13200.50),
    -- Carol Williams' statements
    (3, '2025-01-01', '2025-01-31', 875.25),
    (3, '2025-02-01', '2025-02-28', 2100.00),
    (3, '2025-03-01', '2025-03-31', 3500.75),
    (3, '2025-04-01', '2025-04-30', 4200.00),
    -- David Brown's statements
    (4, '2025-01-01', '2025-01-31', 25000.00),
    (4, '2025-02-01', '2025-02-28', 27500.00)
ON CONFLICT DO NOTHING;

INSERT INTO transactions (statement_id, transaction_date, amount, description, transaction_type, merchant) VALUES
    -- January transactions for Alice (statement_id 1)
    (1, '2025-01-05 10:30:00', -45.99, 'Grocery shopping', 'DEBIT', 'Fresh Market'),
    (1, '2025-01-10 14:22:00', -15.50, 'Coffee', 'DEBIT', 'Starbucks'),
    (1, '2025-01-15 09:15:00', 3000.00, 'Salary deposit', 'CREDIT', 'Acme Corp'),
    (1, '2025-01-20 16:45:00', -120.00, 'Electric bill', 'DEBIT', 'Power Company'),

    -- February transactions for Alice (statement_id 2)
    (2, '2025-02-03 11:20:00', -89.99, 'Online purchase', 'DEBIT', 'Amazon'),
    (2, '2025-02-08 13:30:00', -35.00, 'Lunch', 'DEBIT', 'Cafe Delight'),
    (2, '2025-02-15 09:15:00', 3000.00, 'Salary deposit', 'CREDIT', 'Acme Corp'),
    (2, '2025-02-22 10:00:00', -450.00, 'Rent payment', 'DEBIT', 'Landlord Inc'),

    -- January transactions for Bob (statement_id 4)
    (4, '2025-01-07 08:45:00', 5000.00, 'Initial deposit', 'CREDIT', 'Transfer'),
    (4, '2025-01-10 12:30:00', -250.00, 'Monthly savings', 'DEBIT', 'Savings Transfer'),
    (4, '2025-01-25 10:15:00', -150.00, 'Insurance payment', 'DEBIT', 'Insurance Co'),

    -- January transactions for Carol (statement_id 6)
    (6, '2025-01-04 15:20:00', -25.00, 'Gas station', 'DEBIT', 'Shell Station'),
    (6, '2025-01-12 09:30:00', -80.00, 'Pharmacy', 'DEBIT', 'CVS Pharmacy'),
    (6, '2025-01-18 11:00:00', 2000.00, 'Freelance payment', 'CREDIT', 'Client ABC'),
    (6, '2025-01-25 14:45:00', -150.00, 'Internet bill', 'DEBIT', 'Comcast'),

    -- January transactions for David (statement_id 10)
    (10, '2025-01-02 09:00:00', -1500.00, 'Office supplies', 'DEBIT', 'Office Depot'),
    (10, '2025-01-05 13:30:00', -500.00, 'Client lunch', 'DEBIT', 'Restaurant XYZ'),
    (10, '2025-01-15 09:15:00', 30000.00, 'Business deposit', 'CREDIT', 'Client Payment'),
    (10, '2025-01-22 10:30:00', -2500.00, 'Vendor payment', 'DEBIT', 'Supplier Co')
ON CONFLICT DO NOTHING;


ALTER TABLE customers OWNER TO "$POSTGRESQL_USER";
ALTER TABLE statements OWNER TO "$POSTGRESQL_USER";
ALTER TABLE transactions OWNER TO "$POSTGRESQL_USER";

GRANT ALL PRIVILEGES ON TABLE customers TO "$POSTGRESQL_USER";
GRANT ALL PRIVILEGES ON TABLE statements TO "$POSTGRESQL_USER";
GRANT ALL PRIVILEGES ON TABLE transactions TO "$POSTGRESQL_USER";

GRANT ALL PRIVILEGES ON SEQUENCE customers_customer_id_seq TO "$POSTGRESQL_USER";
GRANT ALL PRIVILEGES ON SEQUENCE statements_statement_id_seq TO "$POSTGRESQL_USER";
GRANT ALL PRIVILEGES ON SEQUENCE transactions_transaction_id_seq TO "$POSTGRESQL_USER";
