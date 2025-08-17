# 🚨 Troubleshooting Quick Reference Cards - Open Policy Platform

## 🎯 **QUICK ACCESS OVERVIEW**

This reference provides instant access to common issues, error messages, and solutions. Designed for **5-second developer experience** - find what you need instantly.

---

## 🔍 **COMMON ISSUES QUICK REFERENCE**

### **Issue Categories**
| Category | Common Problems | Quick Fix |
|----------|----------------|-----------|
| **Authentication** | Login failures, token issues | Check credentials, refresh token |
| **Database** | Connection errors, query failures | Check connection, restart service |
| **Deployment** | Build failures, deployment errors | Check logs, validate config |
| **Performance** | Slow responses, high latency | Check resources, optimize queries |
| **Monitoring** | Metrics missing, alerts not working | Check endpoints, validate config |

---

## 🔐 **AUTHENTICATION ISSUES**

### **Login Failures**
```bash
# Error: "Invalid credentials"
# Solution: Check username/password
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "correct_user", "password": "correct_pass"}'

# Error: "User not found"
# Solution: Verify user exists in database
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "SELECT username FROM users;"
```

### **Token Issues**
```bash
# Error: "Token expired"
# Solution: Refresh token
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Authorization: Bearer YOUR_REFRESH_TOKEN"

# Error: "Invalid token"
# Solution: Get new token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'
```

### **Permission Issues**
```bash
# Error: "Insufficient permissions"
# Solution: Check user role
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "SELECT username, role FROM users WHERE username = 'username';"

# Update user role
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "UPDATE users SET role = 'admin' WHERE username = 'username';"
```

---

## 🗄️ **DATABASE ISSUES**

### **Connection Errors**
```bash
# Error: "Connection refused"
# Solution: Check database service
kubectl get pods -n openpolicy | grep postgres
kubectl logs <postgres-pod> -n openpolicy

# Restart database service
kubectl delete pod <postgres-pod> -n openpolicy

# Error: "Authentication failed"
# Solution: Check credentials
kubectl get secret database-secret -n openpolicy -o yaml
kubectl describe secret database-secret -n openpolicy
```

### **Query Failures**
```bash
# Error: "Table does not exist"
# Solution: Check schema
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "\dt"

# Error: "Column does not exist"
# Solution: Check table structure
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "\d table_name"

# Error: "Foreign key constraint violation"
# Solution: Check related data
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "SELECT * FROM referenced_table WHERE id = referenced_id;"
```

### **Performance Issues**
```bash
# Slow queries
# Solution: Check query execution plan
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "EXPLAIN ANALYZE SELECT * FROM policies WHERE title ILIKE '%search%';"

# High CPU usage
# Solution: Check active queries
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "SELECT pid, query, query_start FROM pg_stat_activity WHERE state = 'active';"

# Memory issues
# Solution: Check connection pool
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "SELECT count(*) FROM pg_stat_activity;"
```

---

## 🚀 **DEPLOYMENT ISSUES**

### **Build Failures**
```bash
# Error: "Build failed"
# Solution: Check Docker build
docker build -t openpolicy/api-service:latest ./backend
docker build -t openpolicy/web-interface:latest ./web

# Error: "Image not found"
# Solution: Check available images
docker images | grep openpolicy
docker pull openpolicy/api-service:latest

# Error: "Port already in use"
# Solution: Check port usage
lsof -i :8000
lsof -i :3000
```

### **Kubernetes Deployment Issues**
```bash
# Error: "Pod not running"
# Solution: Check pod status
kubectl get pods -n openpolicy
kubectl describe pod <pod-name> -n openpolicy
kubectl logs <pod-name> -n openpolicy

# Error: "Image pull failed"
# Solution: Check image availability
kubectl describe pod <pod-name> -n openpolicy | grep -A 10 Events
docker images | grep <image-name>

# Error: "Service not accessible"
# Solution: Check service configuration
kubectl get services -n openpolicy
kubectl describe service <service-name> -n openpolicy
kubectl get endpoints -n openpolicy
```

### **Helm Deployment Issues**
```bash
# Error: "Chart installation failed"
# Solution: Check chart syntax
helm lint ./helm/charts/openpolicy-platform

# Error: "Values validation failed"
# Solution: Check values file
helm template openpolicy-platform ./helm/charts/openpolicy-platform \
  --values ./helm/values/production.yaml

# Error: "Release not found"
# Solution: Check installed releases
helm list -n openpolicy
helm status openpolicy-platform -n openpolicy
```

---

## 📊 **PERFORMANCE ISSUES**

### **Slow API Responses**
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/api/v1/policies"

# Check database performance
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Check resource usage
kubectl top pods -n openpolicy
kubectl top nodes
```

### **High Resource Usage**
```bash
# High CPU usage
# Solution: Check resource limits
kubectl describe pod <pod-name> -n openpolicy | grep -A 5 Resources

# High memory usage
# Solution: Check memory limits
kubectl describe pod <pod-name> -n openpolicy | grep -A 5 Resources

# High disk usage
# Solution: Check persistent volumes
kubectl get pv
kubectl get pvc -n openpolicy
```

### **Scaling Issues**
```bash
# Auto-scaling not working
# Solution: Check HPA configuration
kubectl get hpa -n openpolicy
kubectl describe hpa <hpa-name> -n openpolicy

# Manual scaling issues
# Solution: Check deployment configuration
kubectl get deployments -n openpolicy
kubectl describe deployment <deployment-name> -n openpolicy
```

---

## 📈 **MONITORING ISSUES**

### **Metrics Not Available**
```bash
# Check Prometheus endpoints
curl http://localhost:8000/metrics
curl http://localhost:8000/health

# Check Prometheus configuration
kubectl get configmap prometheus-config -n monitoring -o yaml

# Check Prometheus targets
kubectl port-forward service/prometheus 9090:9090 -n monitoring
# Then visit http://localhost:9090/targets
```

### **Grafana Issues**
```bash
# Dashboard not loading
# Solution: Check Grafana service
kubectl get pods -n monitoring | grep grafana
kubectl logs <grafana-pod> -n monitoring

# Data source connection failed
# Solution: Check Prometheus connection
kubectl exec -it <grafana-pod> -n monitoring -- curl http://prometheus:9090/api/v1/query?query=up
```

### **Alerting Issues**
```bash
# Alerts not firing
# Solution: Check AlertManager
kubectl get pods -n monitoring | grep alertmanager
kubectl logs <alertmanager-pod> -n monitoring

# Check alert rules
kubectl get configmap alert-rules -n monitoring -o yaml
```

---

## 🔧 **SERVICE ISSUES**

### **API Service Problems**
```bash
# Service not responding
# Solution: Check service health
curl -f http://localhost:8000/health
kubectl get pods -n openpolicy | grep api-service

# Check service logs
kubectl logs -f deployment/api-service -n openpolicy

# Restart service
kubectl rollout restart deployment/api-service -n openpolicy
```

### **Web Interface Issues**
```bash
# Page not loading
# Solution: Check frontend service
curl -f http://localhost:3000/health
kubectl get pods -n openpolicy | grep web-interface

# Check frontend logs
kubectl logs -f deployment/web-interface -n openpolicy

# Check build status
cd web && npm run build
```

### **Database Service Issues**
```bash
# Database connection failed
# Solution: Check database service
kubectl get pods -n openpolicy | grep postgres
kubectl logs <postgres-pod> -n openpolicy

# Check database health
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "SELECT 1;"
```

---

## 🚨 **EMERGENCY PROCEDURES**

### **Service Outage**
```bash
# Quick service restart
kubectl rollout restart deployment/api-service -n openpolicy
kubectl rollout restart deployment/web-interface -n openpolicy

# Check service status
kubectl get pods -n openpolicy
kubectl get services -n openpolicy

# Verify health endpoints
curl -f http://localhost:8000/health
curl -f http://localhost:3000/health
```

### **Database Recovery**
```bash
# Check database status
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "SELECT version();"

# Restart database
kubectl delete pod <postgres-pod> -n openpolicy

# Check data integrity
kubectl exec -it <pod-name> -n openpolicy -- psql -U postgres -d openpolicy -c "SELECT COUNT(*) FROM users;"
```

### **Rollback Deployment**
```bash
# Rollback to previous version
kubectl rollout undo deployment/api-service -n openpolicy
kubectl rollout undo deployment/web-interface -n openpolicy

# Check rollback status
kubectl rollout status deployment/api-service -n openpolicy
kubectl rollout status deployment/web-interface -n openpolicy
```

---

## 📝 **DEBUGGING COMMANDS**

### **General Debugging**
```bash
# Check all resources
kubectl get all -n openpolicy

# Check events
kubectl get events -n openpolicy --sort-by='.lastTimestamp'

# Check logs for all pods
kubectl logs -f -l app=api-service -n openpolicy
kubectl logs -f -l app=web-interface -n openpolicy

# Check resource usage
kubectl top pods -n openpolicy
kubectl top nodes
```

### **Network Debugging**
```bash
# Check network policies
kubectl get networkpolicies -n openpolicy

# Check ingress configuration
kubectl get ingress -n openpolicy
kubectl describe ingress <ingress-name> -n openpolicy

# Check service endpoints
kubectl get endpoints -n openpolicy
kubectl describe endpoints <endpoint-name> -n openpolicy
```

### **Security Debugging**
```bash
# Check RBAC configuration
kubectl get roles -n openpolicy
kubectl get rolebindings -n openpolicy

# Check service accounts
kubectl get serviceaccounts -n openpolicy

# Check secrets
kubectl get secrets -n openpolicy
kubectl describe secret <secret-name> -n openpolicy
```

---

## 🔗 **RELATED REFERENCE CARDS**

- **API Reference**: [API Quick Reference](../api/quick-reference.md)
- **Database Reference**: [Database Quick Reference](../database/quick-reference.md)
- **Deployment Reference**: [Deployment Commands](../deployment/quick-reference.md)
- **Development**: [Development Workflows](../../processes/development/README.md)

---

## 📚 **ADDITIONAL RESOURCES**

- **Logs Directory**: `./logs/` directory
- **Monitoring**: [Monitoring Setup](../../monitoring/README.md)
- **Operations**: [Operations Manuals](../../processes/operations/README.md)
- **Full Documentation**: [Troubleshooting Documentation](../../processes/operations/README.md)

---

**🎯 This troubleshooting reference card provides instant access to common issues and solutions.**

**💡 Pro Tip**: Bookmark this page for quick access during troubleshooting. Use the command examples as templates for your debugging operations.**
