CREATE TABLE IF NOT EXISTS core_politician (
  id SERIAL PRIMARY KEY,
  name TEXT,
  party_name TEXT,
  district TEXT,
  email TEXT,
  phone TEXT
);
INSERT INTO core_politician (name, party_name, district, email, phone)
VALUES ('Jane Doe','Independent','Central','jane@example.com','555-0100')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS bills_bill (
  id SERIAL PRIMARY KEY,
  title TEXT,
  classification TEXT,
  session TEXT
);
INSERT INTO bills_bill (title, classification, session)
VALUES ('Bill A','public','43-1')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS core_organization (
  id SERIAL PRIMARY KEY,
  name TEXT,
  classification TEXT
);
INSERT INTO core_organization (name, classification)
VALUES ('Finance Committee','committee')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS bills_membervote (
  id SERIAL PRIMARY KEY,
  bill_id INT,
  member_name TEXT,
  vote TEXT
);
INSERT INTO bills_membervote (bill_id, member_name, vote)
VALUES (1,'Jane Doe','yes')
ON CONFLICT DO NOTHING;
