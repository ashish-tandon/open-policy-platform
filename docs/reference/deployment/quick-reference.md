# 🚀 Deployment Quick Reference Cards - Open Policy Platform

## 🎯 **QUICK ACCESS OVERVIEW**

This reference provides instant access to all deployment commands, Docker operations, Kubernetes management, and Helm deployments. Designed for **5-second developer experience** - find what you need instantly.

---

## 🐳 **DOCKER QUICK REFERENCE**

### **Container Management Commands**
```bash
# Build images
docker build -t openpolicy/api-service:latest ./backend
docker build -t openpolicy/web-interface:latest ./web

# Run containers
docker run -d -p 8000:8000 --name api-service openpolicy/api-service:latest
docker run -d -p 3000:3000 --name web-interface openpolicy/web-interface:latest

# Stop containers
docker stop api-service web-interface
docker rm api-service web-interface

# View running containers
docker ps
docker ps -a  # Show all containers including stopped

# View container logs
docker logs api-service
docker logs -f api-service  # Follow logs in real-time
```

### **Docker Compose Commands**
```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d postgres redis

# Stop all services
docker-compose down

# View service logs
docker-compose logs -f api-service
docker-compose logs -f  # All services

# Scale services
docker-compose up -d --scale api-service=3

# Rebuild and restart
docker-compose up -d --build

# View service status
docker-compose ps
```

### **Image Management**
```bash
# List images
docker images
docker image ls

# Remove images
docker rmi openpolicy/api-service:latest
docker rmi $(docker images -q)  # Remove all images

# Tag images
docker tag openpolicy/api-service:latest openpolicy/api-service:v1.2.3

# Push to registry
docker push openpolicy/api-service:latest

# Pull from registry
docker pull openpolicy/api-service:latest
```

---

## ☸️ **KUBERNETES QUICK REFERENCE**

### **Cluster Management**
```bash
# Check cluster status
kubectl cluster-info
kubectl get nodes
kubectl get namespaces

# Switch context
kubectl config current-context
kubectl config use-context openpolicy-cluster

# View cluster resources
kubectl get all --all-namespaces
kubectl get all -n openpolicy
```

### **Deployment Management**
```bash
# Deploy from YAML files
kubectl apply -f infrastructure/k8s/
kubectl apply -f infrastructure/k8s/api-deployment.yaml

# View deployments
kubectl get deployments -n openpolicy
kubectl get deployments -n openpolicy -o wide

# Scale deployments
kubectl scale deployment api-service --replicas=5 -n openpolicy

# Update deployment
kubectl set image deployment/api-service api-service=openpolicy/api-service:v1.2.3 -n openpolicy

# Rollback deployment
kubectl rollout undo deployment/api-service -n openpolicy
kubectl rollout history deployment/api-service -n openpolicy
```

### **Pod Management**
```bash
# View pods
kubectl get pods -n openpolicy
kubectl get pods -n openpolicy -o wide

# View pod details
kubectl describe pod api-service-abc123 -n openpolicy

# View pod logs
kubectl logs api-service-abc123 -n openpolicy
kubectl logs -f api-service-abc123 -n openpolicy

# Execute commands in pod
kubectl exec -it api-service-abc123 -n openpolicy -- /bin/bash
kubectl exec api-service-abc123 -n openpolicy -- curl localhost:8000/health
```

### **Service Management**
```bash
# View services
kubectl get services -n openpolicy
kubectl get svc -n openpolicy

# View service details
kubectl describe service api-service -n openpolicy

# Port forward to service
kubectl port-forward service/api-service 8000:8000 -n openpolicy
```

### **ConfigMap and Secrets**
```bash
# View configmaps
kubectl get configmaps -n openpolicy
kubectl describe configmap app-config -n openpolicy

# View secrets
kubectl get secrets -n openpolicy
kubectl describe secret database-secret -n openpolicy

# Create configmap from file
kubectl create configmap app-config --from-file=config.yaml -n openpolicy

# Create secret
kubectl create secret generic database-secret \
  --from-literal=username=admin \
  --from-literal=password=secret123 \
  -n openpolicy
```

---

## 🎯 **HELM QUICK REFERENCE**

### **Chart Management**
```bash
# Install platform
helm install openpolicy-platform ./helm/charts/openpolicy-platform \
  --values ./helm/values/production.yaml \
  --namespace openpolicy \
  --create-namespace

# Upgrade platform
helm upgrade openpolicy-platform ./helm/charts/openpolicy-platform \
  --values ./helm/values/production.yaml \
  --namespace openpolicy

# Rollback platform
helm rollback openpolicy-platform 1 --namespace openpolicy

# Uninstall platform
helm uninstall openpolicy-platform -n openpolicy
```

### **Chart Operations**
```bash
# List releases
helm list -n openpolicy
helm list --all-namespaces

# View release status
helm status openpolicy-platform -n openpolicy

# View release history
helm history openpolicy-platform -n openpolicy

# Get release values
helm get values openpolicy-platform -n openpolicy

# Test chart
helm test openpolicy-platform -n openpolicy
```

### **Chart Development**
```bash
# Lint chart
helm lint ./helm/charts/openpolicy-platform

# Package chart
helm package ./helm/charts/openpolicy-platform

# Template chart
helm template openpolicy-platform ./helm/charts/openpolicy-platform \
  --values ./helm/values/production.yaml

# Dry run install
helm install openpolicy-platform ./helm/charts/openpolicy-platform \
  --values ./helm/values/production.yaml \
  --dry-run \
  --namespace openpolicy
```

---

## 🔧 **DEPLOYMENT SCRIPTS QUICK REFERENCE**

### **Environment Setup**
```bash
# Setup development environment
./scripts/dev-setup.sh

# Setup staging environment
./scripts/setup-staging.sh

# Setup production environment
./scripts/setup-production.sh

# Validate environment
./scripts/validate-environment.sh
```

### **Deployment Execution**
```bash
# Deploy to development
./scripts/deploy-dev.sh

# Deploy to staging
./scripts/deploy-staging.sh

# Deploy to production
./scripts/deploy-production.sh

# Rollback deployment
./scripts/rollback.sh --environment production
```

### **Validation and Testing**
```bash
# Run health checks
./scripts/health-check.sh

# Run smoke tests
./scripts/smoke-test.sh

# Run deployment validation
./scripts/validate-deployment.sh

# Run performance tests
./scripts/performance-test.sh
```

---

## 📊 **MONITORING AND LOGS**

### **Health Check Commands**
```bash
# Check API health
curl -f http://localhost:8000/health
curl -f http://localhost:8000/health/database

# Check web interface health
curl -f http://localhost:3000/health

# Check Kubernetes health
kubectl get pods -n openpolicy | grep Running
kubectl get services -n openpolicy | grep ClusterIP
```

### **Log Access Commands**
```bash
# View application logs
kubectl logs -f deployment/api-service -n openpolicy
kubectl logs -f deployment/web-interface -n openpolicy

# View system logs
kubectl logs -f deployment/monitoring -n openpolicy
kubectl logs -f deployment/logging -n openpolicy

# View centralized logs
tail -f logs/application/app.log
tail -f logs/services/api-service.log
```

---

## 🚨 **TROUBLESHOOTING COMMANDS**

### **Common Issues**
```bash
# Check pod status
kubectl get pods -n openpolicy | grep -v Running

# Check pod events
kubectl describe pod <pod-name> -n openpolicy

# Check service endpoints
kubectl get endpoints -n openpolicy

# Check ingress status
kubectl get ingress -n openpolicy
kubectl describe ingress <ingress-name> -n openpolicy

# Check persistent volumes
kubectl get pv
kubectl get pvc -n openpolicy
```

### **Debug Commands**
```bash
# Check resource usage
kubectl top pods -n openpolicy
kubectl top nodes

# Check network policies
kubectl get networkpolicies -n openpolicy

# Check RBAC
kubectl get roles -n openpolicy
kubectl get rolebindings -n openpolicy

# Check events
kubectl get events -n openpolicy --sort-by='.lastTimestamp'
```

---

## ⚡ **PERFORMANCE OPTIMIZATION**

### **Resource Management**
```bash
# Set resource limits
kubectl patch deployment api-service -n openpolicy -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "api-service",
          "resources": {
            "limits": {"cpu": "1000m", "memory": "1Gi"},
            "requests": {"cpu": "500m", "memory": "512Mi"}
          }
        }]
      }
    }
  }
}'

# Enable horizontal pod autoscaling
kubectl autoscale deployment api-service -n openpolicy \
  --min=2 --max=10 --cpu-percent=80
```

### **Scaling Commands**
```bash
# Scale horizontally
kubectl scale deployment api-service --replicas=5 -n openpolicy

# Scale vertically (update resources)
kubectl set resources deployment/api-service -n openpolicy \
  --limits=cpu=2000m,memory=2Gi \
  --requests=cpu=1000m,memory=1Gi
```

---

## 🔐 **SECURITY COMMANDS**

### **Access Control**
```bash
# Check service account
kubectl get serviceaccounts -n openpolicy

# Check secrets
kubectl get secrets -n openpolicy

# Check RBAC
kubectl get clusterroles
kubectl get clusterrolebindings

# Check network policies
kubectl get networkpolicies -n openpolicy
```

### **Security Scanning**
```bash
# Scan images for vulnerabilities
docker scan openpolicy/api-service:latest

# Check pod security
kubectl get pods -n openpolicy -o yaml | grep securityContext

# Validate security policies
kubectl get validatingwebhookconfigurations
```

---

## 🔗 **RELATED REFERENCE CARDS**

- **API Reference**: [API Quick Reference](../api/quick-reference.md)
- **Database Reference**: [Database Quick Reference](../database/quick-reference.md)
- **Troubleshooting**: [Common Issues](../troubleshooting/quick-reference.md)
- **Development**: [Development Workflows](../../processes/development/README.md)

---

## 📚 **ADDITIONAL RESOURCES**

- **Infrastructure Docs**: [Infrastructure Documentation](../../components/infrastructure/README.md)
- **Kubernetes Configs**: `./infrastructure/k8s/` directory
- **Helm Charts**: `./helm/charts/` directory
- **Deployment Scripts**: `./scripts/deployment/` directory
- **Full Documentation**: [Deployment Documentation](../../processes/deployment/README.md)

---

**🎯 This deployment reference card provides instant access to all deployment operations and commands.**

**💡 Pro Tip**: Bookmark this page for quick access during deployments. Use the command examples as templates for your deployment operations.**
