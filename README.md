# NU Hackathon 2025 | Qazcode Challenge

## Medical Diagnosis Assistant: Symptoms → ICD-10

An AI-powered clinical decision support system that converts patient symptoms into structured diagnoses with ICD-10 codes, built on Kazakhstan clinical protocols.

---

## Challenge Overview

Participants will build an MVP product where users input symptoms as free text and receive:

- **Top-N probable diagnoses** ranked by likelihood
- **ICD-10 codes** for each diagnosis
- **Brief clinical explanations** based on official Kazakhstan protocols

The solution must run **using GPT-OSS** — no external LLM API calls allowed.

---

Data Format

```json
#####TODO add here format
```

---

## Evaluation

### Metrics
- **Primary metric:** Accuracy@1
- **Holdout set:** Private test cases(not included in this repository)

### Product Validation
Working demo interface: user inputs symptoms → system returns diagnoses with ICD-10 codes

---
## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/qazcode/nu-hackathon-2025.git
cd nu-hackathon-2025
```

### 2. Set up the environment
```bash
docker build -t hackathon-env ./docker
docker run -it hackathon-env
```

### 3. 
#########TODO add steps for validation



Submission instructions · MD
Copy

### Submission Checklist

- [ ] Everything packed into a single `Dockerfile` (application, models, vector DB, indexes)
- [ ] Container builds successfully: `docker build -t submission .`
- [ ] Container starts and serves on port 8080: `docker run -p 8080:8080 submission`
- [ ] Web UI or API accepts free-text symptoms input
- [ ] Returns top-N diagnoses with ICD-10 codes
- [ ] No external network calls during inference
- [ ] README with build and run instructions

### How to Submit

1. Push your Docker image to Docker Hub or provide a Git repository with `Dockerfile`
2. Submit the link via [submission form] #######TODO add google form
3. We will pull, build, and run your container on the private holdout set
---
