# 📚 Open Policy Platform - Documentation

Welcome to the Open Policy Platform documentation. This comprehensive guide will help you understand, develop, and deploy the platform.

## 📖 **Documentation Sections**

### **🏗️ Architecture**
- [Reorganization Plan](architecture/reorganization-plan.md) - Complete reorganization strategy
- [Platform Summary](architecture/platform-summary.md) - Unified platform overview
- [Merge Documentation](architecture/merge-documentation.md) - Repository merge details
- [Final Merge Report](architecture/final-merge-report.md) - Complete merge summary
- [Documentation Consolidation Plan](architecture/documentation-consolidation-plan.md) - Single source of truth rules
- [Master Execution Plan](architecture/master-execution-plan.md) - Coordinated plan across services

### **🔌 API Documentation**
- [Overview](api/overview.md) - API architecture, networking, versioning
- [Authentication](api/authentication.md) - JWT flow and endpoints
- [Endpoints](api/endpoints.md) - Canonical endpoint reference
- [Schemas](api/schemas.md) - Request/response examples for key routes
- [Quick Reference](api/quick-reference.md) - Fast lookup for common calls

### **⚙️ Operations**
- [Environment Variables](operations/environment-variables.md)
- [Dependencies and Requirements](operations/dependencies.md)
- [Scripts Inventory](operations/scripts.md)
- [Health Checks](operations/health-checks.md)
- [Services Overview](operations/services-overview.md)

### **🚀 Deployment**
- [Production Deployment](deployment/production.md) - Production deployment guide
- [Docker Setup](deployment/docker.md) - Containerized deployment
- [Environment Configuration](deployment/environment.md) - Environment setup
- [Monitoring](deployment/monitoring.md) - System monitoring

### **👨‍💻 Development**
- [Development Setup](development/setup.md) - Local development environment
- [Code Guidelines](development/guidelines.md) - Coding standards and practices
- [Testing](development/testing.md) - Testing procedures
- [Contributing](development/contributing.md) - Contribution workflow

### **👥 User Guides**
- [User Guide](user-guides/user-guide.md) - End user documentation
- [Admin Guide](user-guides/admin-guide.md) - Administrator documentation
- [API Usage](user-guides/api-usage.md) - API usage examples

## 🎯 **Quick Start**

1. **Setup**: Run `../scripts/setup-unified.sh`
2. **Start**: Run `../scripts/start-all.sh`
3. **Access**: 
   - Web Interface: http://localhost:5173
   - Admin Interface: http://localhost:5173/admin
   - API Documentation: http://localhost:8000/docs

## 📋 **Platform Overview**

The Open Policy Platform is a unified system for:
- **Policy Analysis**: Comprehensive policy data analysis
- **Data Collection**: Automated data scraping and collection
- **User Interface**: Web-based policy browsing and search
- **Administration**: Complete system management interface

## 🔧 **Technology Stack**

- **Backend**: FastAPI + PostgreSQL + Redis
- **Frontend**: React + TypeScript + Tailwind CSS
- **Database**: PostgreSQL with 6.5GB parliamentary data
- **Infrastructure**: Docker + Nginx + Monitoring

## 📞 **Support**

For questions or issues:
1. Check the relevant documentation section
2. Review the [Development Guide](development/setup.md)
3. Check the [API Documentation](api/overview.md)

---

**Last Updated**: August 8, 2024
**Version**: 1.0.0
