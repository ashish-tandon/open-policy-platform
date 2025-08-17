# 🚀 API Quick Reference Cards - Open Policy Platform

## 🎯 **QUICK ACCESS OVERVIEW**

This reference provides instant access to all API endpoints, authentication, and usage patterns. Designed for **5-second developer experience** - find what you need instantly.

---

## 🔐 **AUTHENTICATION QUICK REFERENCE**

### **JWT Token Authentication**
```bash
# Get token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use token
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/auth/me
```

### **API Key Authentication**
```bash
# Service-to-service communication
curl -H "X-API-Key: YOUR_API_KEY" \
  http://localhost:8000/api/v1/health
```

---

## 📡 **CORE API ENDPOINTS**

### **1. Authentication Endpoints**
| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/api/v1/auth/login` | POST | User login | No |
| `/api/v1/auth/register` | POST | User registration | No |
| `/api/v1/auth/me` | GET | Current user info | Yes |
| `/api/v1/auth/refresh` | POST | Refresh token | Yes |
| `/api/v1/auth/logout` | POST | User logout | Yes |

### **2. Health & Monitoring Endpoints**
| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/health` | GET | Basic health check | No |
| `/health/database` | GET | Database health | No |
| `/health/dependencies` | GET | Service dependencies | No |
| `/metrics` | GET | Prometheus metrics | No |
| `/metrics/performance` | GET | Performance metrics | No |

### **3. Policy Management Endpoints**
| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/api/v1/policies` | GET | List policies | Yes |
| `/api/v1/policies/{id}` | GET | Get policy details | Yes |
| `/api/v1/policies` | POST | Create policy | Yes |
| `/api/v1/policies/{id}` | PUT | Update policy | Yes |
| `/api/v1/policies/{id}` | DELETE | Delete policy | Yes |

### **4. Search & Analytics Endpoints**
| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/api/v1/search` | GET | Full-text search | Yes |
| `/api/v1/analytics/overview` | GET | Analytics overview | Yes |
| `/api/v1/analytics/trends` | GET | Trend analysis | Yes |
| `/api/v1/analytics/reports` | GET | Generate reports | Yes |

---

## 🎯 **COMMON API PATTERNS**

### **Standard Response Format**
```json
{
  "success": true,
  "data": {
    // Response data
  },
  "message": "Operation successful",
  "timestamp": "2025-01-20T10:30:00Z"
}
```

### **Error Response Format**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "Field validation failed"
    }
  },
  "timestamp": "2025-01-20T10:30:00Z"
}
```

### **Pagination Format**
```json
{
  "success": true,
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "pages": 8
  }
}
```

---

## 🔍 **SEARCH API QUICK REFERENCE**

### **Basic Search**
```bash
# Simple search
curl "http://localhost:8000/api/v1/search?q=policy" \
  -H "Authorization: Bearer $TOKEN"

# Advanced search with filters
curl "http://localhost:8000/api/v1/search?q=policy&category=health&date_from=2024-01-01" \
  -H "Authorization: Bearer $TOKEN"
```

### **Search Parameters**
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `q` | string | Search query | `q=health policy` |
| `category` | string | Filter by category | `category=health` |
| `date_from` | date | Start date filter | `date_from=2024-01-01` |
| `date_to` | date | End date filter | `date_to=2024-12-31` |
| `status` | string | Filter by status | `status=active` |
| `page` | integer | Page number | `page=2` |
| `per_page` | integer | Items per page | `per_page=50` |

---

## 📊 **ANALYTICS API QUICK REFERENCE**

### **Analytics Overview**
```bash
# Get analytics overview
curl "http://localhost:8000/api/v1/analytics/overview" \
  -H "Authorization: Bearer $TOKEN"

# Get trend data
curl "http://localhost:8000/api/v1/analytics/trends?metric=user_activity&period=30d" \
  -H "Authorization: Bearer $TOKEN"
```

### **Analytics Parameters**
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `metric` | string | Metric to analyze | `metric=user_activity` |
| `period` | string | Time period | `period=7d`, `period=30d` |
| `group_by` | string | Grouping field | `group_by=category` |
| `filters` | object | Additional filters | `filters={"status":"active"}` |

---

## 🗄️ **DATA MANAGEMENT API**

### **File Upload**
```bash
# Upload file
curl -X POST http://localhost:8000/api/v1/files/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf" \
  -F "category=policies" \
  -F "description=Policy document"
```

### **File Management**
| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/api/v1/files/upload` | POST | Upload file | Yes |
| `/api/v1/files/{id}` | GET | Download file | Yes |
| `/api/v1/files/{id}` | DELETE | Delete file | Yes |
| `/api/v1/files/search` | GET | Search files | Yes |

---

## 🔧 **ADMIN API QUICK REFERENCE**

### **User Management**
```bash
# List users
curl "http://localhost:8000/api/v1/admin/users" \
  -H "Authorization: Bearer $TOKEN"

# Create user
curl -X POST http://localhost:8000/api/v1/admin/users \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"username": "newuser", "email": "user@example.com", "role": "user"}'
```

### **System Management**
```bash
# System status
curl "http://localhost:8000/api/v1/admin/system/status" \
  -H "Authorization: Bearer $TOKEN"

# System configuration
curl "http://localhost:8000/api/v1/admin/system/config" \
  -H "Authorization: Bearer $TOKEN"
```

---

## 📝 **REQUEST EXAMPLES BY LANGUAGE**

### **Python Examples**
```python
import requests

# Authentication
response = requests.post('http://localhost:8000/api/v1/auth/login', 
                        json={'username': 'user', 'password': 'pass'})
token = response.json()['data']['access_token']

# API calls
headers = {'Authorization': f'Bearer {token}'}
response = requests.get('http://localhost:8000/api/v1/policies', headers=headers)
policies = response.json()['data']
```

### **JavaScript Examples**
```javascript
// Authentication
const response = await fetch('http://localhost:8000/api/v1/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username: 'user', password: 'pass' })
});
const { access_token } = await response.json();

// API calls
const policiesResponse = await fetch('http://localhost:8000/api/v1/policies', {
  headers: { 'Authorization': `Bearer ${access_token}` }
});
const policies = await policiesResponse.json();
```

### **cURL Examples**
```bash
# Get all policies
curl "http://localhost:8000/api/v1/policies" \
  -H "Authorization: Bearer $TOKEN"

# Create new policy
curl -X POST "http://localhost:8000/api/v1/policies" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "New Policy", "content": "Policy content"}'

# Update policy
curl -X PUT "http://localhost:8000/api/v1/policies/123" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Updated Policy"}'
```

---

## 🚨 **ERROR HANDLING QUICK REFERENCE**

### **Common HTTP Status Codes**
| Status | Meaning | Common Causes |
|--------|---------|---------------|
| `200` | Success | Request completed successfully |
| `201` | Created | Resource created successfully |
| `400` | Bad Request | Invalid input data |
| `401` | Unauthorized | Missing or invalid authentication |
| `403` | Forbidden | Insufficient permissions |
| `404` | Not Found | Resource doesn't exist |
| `422` | Validation Error | Data validation failed |
| `500` | Internal Error | Server-side error |

### **Error Response Codes**
| Code | Meaning | Action Required |
|------|---------|-----------------|
| `VALIDATION_ERROR` | Input validation failed | Check input data format |
| `AUTHENTICATION_ERROR` | Invalid credentials | Re-authenticate user |
| `AUTHORIZATION_ERROR` | Insufficient permissions | Check user role |
| `RESOURCE_NOT_FOUND` | Requested resource missing | Verify resource ID |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry |
| `SERVICE_UNAVAILABLE` | Service temporarily down | Retry later |

---

## ⚡ **PERFORMANCE TIPS**

### **Optimization Strategies**
```bash
# Use pagination for large datasets
curl "http://localhost:8000/api/v1/policies?page=1&per_page=20"

# Use specific fields to reduce data transfer
curl "http://localhost:8000/api/v1/policies?fields=id,title,status"

# Use caching headers
curl -H "Cache-Control: max-age=300" \
  "http://localhost:8000/api/v1/policies"
```

### **Rate Limiting**
- **Standard Users**: 100 requests/minute
- **Premium Users**: 500 requests/minute
- **Admin Users**: 1000 requests/minute
- **Service Accounts**: 2000 requests/minute

---

## 🔗 **RELATED REFERENCE CARDS**

- **Database Reference**: [Database Quick Reference](../database/quick-reference.md)
- **Deployment Reference**: [Deployment Commands](../deployment/quick-reference.md)
- **Troubleshooting**: [Common Issues](../troubleshooting/quick-reference.md)
- **Development**: [Development Workflows](../../processes/development/README.md)

---

## 📚 **ADDITIONAL RESOURCES**

- **OpenAPI Specification**: `/api/v1/docs` (Swagger UI)
- **ReDoc Documentation**: `/api/v1/redoc`
- **API Status**: `/health` endpoint
- **Full Documentation**: [API Documentation](../../api/README.md)

---

**🎯 This API reference card provides instant access to all API endpoints and usage patterns.**

**💡 Pro Tip**: Bookmark this page for quick access during development. Use the examples as templates for your API calls.**
