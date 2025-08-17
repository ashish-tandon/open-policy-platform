# 🗄️ Database Quick Reference Cards - Open Policy Platform

## 🎯 **QUICK ACCESS OVERVIEW**

This reference provides instant access to database schema, queries, and optimization patterns. Designed for **5-second developer experience** - find what you need instantly.

---

## 🏗️ **DATABASE SCHEMA QUICK REFERENCE**

### **Core Tables Overview**
| Table | Purpose | Key Fields | Relationships |
|-------|---------|------------|---------------|
| `users` | User accounts and authentication | `id`, `username`, `email`, `role` | Policies, files, activities |
| `policies` | Policy documents and metadata | `id`, `title`, `content`, `status`, `user_id` | Users, categories, tags |
| `categories` | Policy categories | `id`, `name`, `description` | Policies |
| `tags` | Policy tags | `id`, `name` | Policies (many-to-many) |
| `files` | File attachments | `id`, `filename`, `path`, `user_id` | Users, policies |
| `activities` | User activity logs | `id`, `user_id`, `action`, `timestamp` | Users |

### **Database Schema Diagram**
```sql
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Policies table
CREATE TABLE policies (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'draft',
    user_id INTEGER REFERENCES users(id),
    category_id INTEGER REFERENCES categories(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Categories table
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tags table
CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Policy tags (many-to-many)
CREATE TABLE policy_tags (
    policy_id INTEGER REFERENCES policies(id),
    tag_id INTEGER REFERENCES tags(id),
    PRIMARY KEY (policy_id, tag_id)
);

-- Files table
CREATE TABLE files (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100),
    user_id INTEGER REFERENCES users(id),
    policy_id INTEGER REFERENCES policies(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Activities table
CREATE TABLE activities (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id INTEGER,
    details JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🔍 **COMMON QUERIES QUICK REFERENCE**

### **User Management Queries**
```sql
-- Get user by username
SELECT * FROM users WHERE username = 'john_doe';

-- Get user with role
SELECT id, username, email, role FROM users WHERE role = 'admin';

-- Create new user
INSERT INTO users (username, email, password_hash, role) 
VALUES ('newuser', 'user@example.com', 'hashed_password', 'user');

-- Update user role
UPDATE users SET role = 'admin' WHERE username = 'john_doe';

-- Delete user (cascade delete related data)
DELETE FROM users WHERE username = 'john_doe';
```

### **Policy Management Queries**
```sql
-- Get all policies with user info
SELECT p.*, u.username, c.name as category_name
FROM policies p
JOIN users u ON p.user_id = u.id
LEFT JOIN categories c ON p.category_id = c.id
ORDER BY p.created_at DESC;

-- Get policies by category
SELECT p.*, u.username 
FROM policies p
JOIN users u ON p.user_id = u.id
WHERE p.category_id = (SELECT id FROM categories WHERE name = 'Health');

-- Get policies by status
SELECT * FROM policies WHERE status = 'published';

-- Search policies by title/content
SELECT * FROM policies 
WHERE title ILIKE '%health%' OR content ILIKE '%health%';

-- Get policy with tags
SELECT p.*, array_agg(t.name) as tags
FROM policies p
LEFT JOIN policy_tags pt ON p.id = pt.policy_id
LEFT JOIN tags t ON pt.tag_id = t.id
WHERE p.id = 123
GROUP BY p.id;
```

### **File Management Queries**
```sql
-- Get files by user
SELECT * FROM files WHERE user_id = 123;

-- Get files by policy
SELECT * FROM files WHERE policy_id = 456;

-- Get large files
SELECT * FROM files WHERE file_size > 10485760; -- > 10MB

-- Get files by type
SELECT * FROM files WHERE mime_type LIKE 'application/pdf%';
```

### **Analytics and Reporting Queries**
```sql
-- User activity summary
SELECT 
    u.username,
    COUNT(a.id) as activity_count,
    MAX(a.timestamp) as last_activity
FROM users u
LEFT JOIN activities a ON u.id = a.user_id
GROUP BY u.id, u.username
ORDER BY activity_count DESC;

-- Policy creation trends
SELECT 
    DATE(created_at) as date,
    COUNT(*) as policies_created
FROM policies
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY date;

-- Category usage statistics
SELECT 
    c.name as category,
    COUNT(p.id) as policy_count,
    COUNT(DISTINCT p.user_id) as unique_users
FROM categories c
LEFT JOIN policies p ON c.id = p.category_id
GROUP BY c.id, c.name
ORDER BY policy_count DESC;
```

---

## ⚡ **PERFORMANCE OPTIMIZATION QUERIES**

### **Indexing Strategies**
```sql
-- Create indexes for common queries
CREATE INDEX idx_policies_user_id ON policies(user_id);
CREATE INDEX idx_policies_category_id ON policies(category_id);
CREATE INDEX idx_policies_status ON policies(status);
CREATE INDEX idx_policies_created_at ON policies(created_at);
CREATE INDEX idx_policies_title_content ON policies USING gin(to_tsvector('english', title || ' ' || content));

-- Create composite indexes
CREATE INDEX idx_policies_user_status ON policies(user_id, status);
CREATE INDEX idx_policies_category_status ON policies(category_id, status);

-- Create partial indexes
CREATE INDEX idx_policies_published ON policies(created_at) WHERE status = 'published';
CREATE INDEX idx_files_recent ON files(created_at) WHERE created_at >= CURRENT_DATE - INTERVAL '30 days';
```

### **Query Optimization Examples**
```sql
-- Use EXISTS instead of IN for large datasets
SELECT * FROM policies p
WHERE EXISTS (
    SELECT 1 FROM policy_tags pt 
    WHERE pt.policy_id = p.id AND pt.tag_id = 123
);

-- Use LIMIT with ORDER BY for pagination
SELECT * FROM policies 
ORDER BY created_at DESC 
LIMIT 20 OFFSET 40;

-- Use CTEs for complex queries
WITH user_stats AS (
    SELECT user_id, COUNT(*) as policy_count
    FROM policies
    GROUP BY user_id
)
SELECT u.username, us.policy_count
FROM users u
JOIN user_stats us ON u.id = us.user_id
WHERE us.policy_count > 10;
```

---

## 🔧 **DATABASE MAINTENANCE QUERIES**

### **Health Check Queries**
```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Check table statistics
SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
```

### **Cleanup and Maintenance**
```sql
-- Clean up old activities
DELETE FROM activities 
WHERE timestamp < CURRENT_DATE - INTERVAL '90 days';

-- Clean up orphaned files
DELETE FROM files 
WHERE policy_id IS NULL 
AND created_at < CURRENT_DATE - INTERVAL '30 days';

-- Update table statistics
ANALYZE users;
ANALYZE policies;
ANALYZE files;
ANALYZE activities;

-- Vacuum tables
VACUUM ANALYZE policies;
VACUUM ANALYZE files;
```

---

## 🚨 **TROUBLESHOOTING QUERIES**

### **Common Issues and Solutions**
```sql
-- Find duplicate usernames
SELECT username, COUNT(*) 
FROM users 
GROUP BY username 
HAVING COUNT(*) > 1;

-- Find orphaned policies
SELECT p.* FROM policies p
LEFT JOIN users u ON p.user_id = u.id
WHERE u.id IS NULL;

-- Find policies without categories
SELECT * FROM policies WHERE category_id IS NULL;

-- Check for data integrity issues
SELECT 
    'policies' as table_name,
    COUNT(*) as total_records,
    COUNT(user_id) as with_user,
    COUNT(category_id) as with_category
FROM policies
UNION ALL
SELECT 
    'files' as table_name,
    COUNT(*) as total_records,
    COUNT(user_id) as with_user,
    COUNT(policy_id) as with_policy
FROM files;
```

### **Performance Diagnostics**
```sql
-- Find slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check connection status
SELECT 
    datname,
    usename,
    application_name,
    client_addr,
    state,
    query_start
FROM pg_stat_activity
WHERE state = 'active';

-- Check locks
SELECT 
    l.pid,
    l.mode,
    l.granted,
    t.relname,
    a.usename
FROM pg_locks l
JOIN pg_class t ON l.relation = t.oid
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE NOT l.granted;
```

---

## 📊 **MONITORING AND METRICS QUERIES**

### **Real-time Monitoring**
```sql
-- Active users in last hour
SELECT COUNT(DISTINCT user_id) as active_users
FROM activities
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 hour';

-- Recent policy changes
SELECT 
    p.title,
    u.username,
    p.updated_at,
    'updated' as action
FROM policies p
JOIN users u ON p.user_id = u.id
WHERE p.updated_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
ORDER BY p.updated_at DESC;

-- File upload activity
SELECT 
    COUNT(*) as files_uploaded,
    SUM(file_size) as total_size_bytes
FROM files
WHERE created_at >= CURRENT_DATE;
```

---

## 🔗 **RELATED REFERENCE CARDS**

- **API Reference**: [API Quick Reference](../api/quick-reference.md)
- **Deployment Reference**: [Deployment Commands](../deployment/quick-reference.md)
- **Troubleshooting**: [Common Issues](../troubleshooting/quick-reference.md)
- **Development**: [Development Workflows](../../processes/development/README.md)

---

## 📚 **ADDITIONAL RESOURCES**

- **Database Schema**: [Full Schema Documentation](../../database/schema.md)
- **Migration Scripts**: `./migrations/` directory
- **Database Config**: [Database Configuration](../../config/database.md)
- **Full Documentation**: [Database Documentation](../../database/README.md)

---

**🎯 This database reference card provides instant access to all database operations and optimization patterns.**

**💡 Pro Tip**: Bookmark this page for quick access during database operations. Use the query examples as templates for your database queries.**
