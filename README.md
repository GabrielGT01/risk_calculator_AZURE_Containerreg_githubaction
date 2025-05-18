
## 🛠️ CI/CD Pipeline

This project leverages a fully automated **CI/CD pipeline** to ensure efficient development, testing, and deployment. The pipeline is designed to maintain high code quality and streamline the release process through the following stages:

1. ✅ **Automated Testing**  
   On every push to the repository, unit and integration tests are automatically triggered to validate the integrity and functionality of the codebase.

2. 🐳 **Docker Image Build**  
   After successful tests, a Docker image is built to package the application in a consistent and portable environment.

3. 📦 **Push to Azure Container Registry (ACR)**  
   The newly built image is securely pushed to **Azure Container Registry**, enabling centralized image management and version control.

4. 🚀 **Deployment to Azure Web App**  
   The latest Docker image is then deployed to an **Azure Web App for Containers**, ensuring smooth delivery and continuous availability of the application.

This CI/CD workflow enhances development velocity, improves code reliability, and ensures seamless, production-grade deployment with minimal manual intervention.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE), granting permission to use, modify, and distribute the software with minimal restrictions. Please review the license file for full terms and conditions.

