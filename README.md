# Dental AI Assistant

AI-powered decision-support web app for dentists: upload a dental X-ray, get AI-detected findings, an LLM-generated explanation and treatment guidance, and a chatbot for follow-up questions.

## Product definition

Dentist uploads a tooth X-ray, AI highlights findings, LLM explains the condition and treatment options, and a chatbot answers follow-up questions.

## MVP Scope

### In scope (v1)
- Web-based upload of a dental X-ray (JPG/PNG)
- AI analysis: tooth segmentation plus detection of 3-4 conditions: dental caries, periapical lesion, impacted tooth, missing tooth
- Visual overlay of findings on the X-ray with confidence scores
- LLM-generated report: explanation, severity, treatment options, follow-up advice
- Dental-only chatbot that knows the current patient's findings
- Downloadable PDF report
- Disclaimer: decision support only, not a replacement for clinical judgment

### Out of scope (future work)
- 3D/CBCT scans
- EHR integration
- Clinical validation and regulatory approval
- Storing patient identity

## Target users
Dentists and dental staff in hospitals and clinics.

## Planned architecture
- ML: U-Net segmentation + YOLOv8 detection (PyTorch)
- Backend: FastAPI
- LLM: Claude API with retrieval over trusted dental guidelines (RAG)
- Frontend: Streamlit or React
- Deployment: Docker, Render/Vercel/Hugging Face Spaces

## Dataset
- Dental Radiography Segmentation (Kaggle): https://www.kaggle.com/datasets/abbasseifossadat/dental-radiography-segmentation
- A disease-labeled dataset for detection (to be added in Step 2)

## Limitations
Hackathon prototype trained on limited public data, not clinically validated. All outputs must be reviewed by a qualified dentist.

## Status
- [x] Step 1: MVP scope defined
- [ ] Step 2: Data collection
