#this .py is to test the working of jooble api
import requests

# Replace 'YOUR_API_KEY' with your actual Jooble API key
API_KEY = "ee033cd6-069b-4bb8-99f6-0855bcc45b5a"
API_URL = f"https://jooble.org/api/{API_KEY}"

# Define search parameters
payload = {
    "keywords": "Data Science Internship OR Intern Entry-Level Data Scientist",
    "location": "Remote",
    "page": 1
}

# Send a POST request to the Jooble API
response = requests.post(API_URL, json=payload)

# Check response status
if response.status_code == 200:
    jobs = response.json().get("jobs", [])  # Extract jobs list from response
    
    # Print only the first 5 jobs
    for i, job in enumerate(jobs[:5], start=1):
        print(f"\nJob {i}:")
        print(f"Title: {job.get('title', 'N/A')}")
        print(f"Company: {job.get('company', 'N/A')}")
        print(f"Location: {job.get('location', 'N/A')}")
        print(f"Salary: {job.get('salary', 'N/A')}")
        print(f"Link: {job.get('link', 'N/A')}")
else:
    print("Error:", response.status_code, response.text)