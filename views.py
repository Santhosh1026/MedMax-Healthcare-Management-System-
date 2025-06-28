from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .models import UserProfile
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


# Create the model
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Safety settings for content moderation
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the model with safety and generation configurations
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    safety_settings=safety_settings,
    generation_config=generation_config,
    system_instruction=[
        "You are a medical assistant chatbot designed to analyze given symptoms and suggest possible diseases with appropriate remedies."

            "======================"
            "SYMPTOM-BASED DISEASE AND REMEDY RESPONSE FORMAT"
            "======================"

            "1. IDENTIFIED DISEASE"
            "   - Based on the given symptoms, list the most probable disease."
            "   - If multiple diseases are possible, rank them by likelihood."
            "   - Mention a short cause and which body systems are affected."

            "----------------------"
            "2. SYMPTOM ANALYSIS"
            "   - Reiterate the user’s reported symptoms."
            "   - For each symptom, provide a brief medical explanation."
            "   - Highlight any urgent or severe symptoms requiring quick action."

            "----------------------"
            "3. REMEDIES"
            "   Divide the treatment suggestions into three parts:"

            "   • Allopathic Medicine:"
            "       - List common medicines used for the condition."
            "       - Mention if it is Over-The-Counter (OTC) or requires prescription."
            "       - Give dosage range (approximate, do not advise exact)."
            "       - Mention any common side effects."

            "   • Ayurvedic Medicine:"
            "       - Recommend herbs, decoctions, or traditional practices."
            "       - Describe how to prepare or use them."
            "       - Highlight the traditional benefit or use case."

            "   • Other Suggestions:"
            "       - Lifestyle improvements (rest, hydration, hygiene, etc.)"
            "       - Home remedies like steam, turmeric milk, etc."
            "       - Any physical activity restrictions or precautions."

            "----------------------"
            "4. DIET RECOMMENDATION"
            "   - List suitable food items to eat during recovery."
            "   - List food items to avoid that may worsen the condition."
            "   - Mention the reason behind each recommendation."

            "----------------------"
            "5. SIMILAR CONDITIONS TO CONSIDER"
            "   - List other diseases with similar symptoms."
            "   - Provide 1-line differences to help users understand variations."
            "   - Suggest common diagnostic tests if needed."

            "----------------------"
            "6. DOCTOR CONSULTATION ADVICE"
            "   - Mention if doctor visit is needed (yes/no)."
            "   - If yes, suggest the specialist type (e.g., General Physician, ENT)."
            "   - Add severity level: Routine, Urgent, or Emergency."

            "----------------------"
            "GENERAL INSTRUCTIONS"
            "   - Do not provide any clickable links or URLs."
            "   - Never promote self-medication; always advise consulting a doctor."
            "   - Response should be medically structured, clean, and understandable."
            "   - Keep the format section-wise and avoid mixing information."


    ]
)


@csrf_exempt
def chat_view(request):
    """
    Handles POST requests for chatbot responses.
    If GET, just render 'chat.html'.
    """
    if request.method == "POST":
        try:
            # Parse the incoming JSON data
            data = json.loads(request.body)
            user_input = data.get("message", "")

            if not user_input:
                return JsonResponse({"error": "No message provided"}, status=400)

            # Start a new chat session with the model
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(user_input)
            model_response = response.text

            # Split the response into lines and format them
            response_lines = model_response.split("\n")
            formatted_response = []

            for line in response_lines:
                if line.strip():  # Skip empty lines
                    topic_details = line.split(":")
                    if len(topic_details) > 1:
                        topic = topic_details[0].strip()
                        details = ":".join(topic_details[1:]).strip()
                        formatted_response.append({"topic": topic, "details": details})
                    else:
                        formatted_response.append({"topic": line.strip(), "details": ""})

            return JsonResponse({"response": formatted_response})

        except Exception as e:
            # Return error message for any exception during processing
            return JsonResponse({"error": str(e)}, status=500)

    # If GET request, render the chat.html page
    return render(request, 'chat.html')

def register_view(request):
    """
    Handles user registration by saving details to the database and a UserProfile model.
    """
    if request.method == 'POST':
        # Get form data
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        full_name = request.POST.get('full_name')
        phone = request.POST.get('phone')
        address = request.POST.get('address')

        # Validate passwords match
        if password != confirm_password:
            return render(request, 'register.html', {'error': 'Passwords do not match'})

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            return render(request, 'register.html', {'error': 'Username already exists'})

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            return render(request, 'register.html', {'error': 'Email already exists'})

        try:
            # Create user and profile
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password
            )
            profile = UserProfile.objects.create(
                user=user,
                full_name=full_name,
                phone=phone,
                address=address
            )

            # Log the user in
            login(request, user)
            return redirect('home')

        except Exception as e:
            # If user was partially created and an error occurs, clean up
            if 'user' in locals():
                user.delete()
            return render(request, 'register.html', {'error': str(e)})

    return render(request, 'register.html')

def login_view(request):
    """
    Handles user login.
    """
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if not username or not password:
            return render(request, 'login.html', {'error': 'Please fill in all fields'})

        # Authenticate user
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            return render(request, 'login.html', {'error': 'Invalid username or password'})

    return render(request, 'login.html')

def logout_view(request):
    """
    Logs out the user and redirects to the dashboard page.
    """
    logout(request)
    return redirect('dashboard')

def dashboard_view(request):
    """
    Renders the dashboard page.
    """
    return render(request, 'dashboard.html')

def home_view(request):
    """
    Renders the home page. Requires login.
    """
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'home.html')



import os
import numpy as np
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from .skindisease import skincancer

@csrf_exempt
def skin_view(request):
    return render(request, 'skin.html')

@csrf_exempt
def predict(request):
    if request.method == "POST" and request.FILES.get("file"):
        image_file = request.FILES["file"]
        classifier = skincancer()
        result = classifier.SkinCancerPrediction(image_file)
        return JsonResponse(result, safe=False)
    return JsonResponse({"error": "Invalid request"}, status=400)
