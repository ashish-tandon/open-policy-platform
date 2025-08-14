-- Initialize multiple databases for the Open Policy Platform
CREATE DATABASE openpolicy_app;
CREATE DATABASE openpolicy_scrapers;
CREATE DATABASE openpolicy_auth;
GRANT ALL PRIVILEGES ON DATABASE openpolicy_app TO openpolicy;
GRANT ALL PRIVILEGES ON DATABASE openpolicy_scrapers TO openpolicy;
GRANT ALL PRIVILEGES ON DATABASE openpolicy_auth TO openpolicy;
