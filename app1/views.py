from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import get_object_or_404
from .models import User, Image
from .serializers import UserSerializer, ImageSerializer
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from django.core.files.base import ContentFile

import numpy as np
from PIL import Image as PILImage
from rest_framework import generics, permissions
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from .models import Image
from .serializers import ImageSerializer
from django.contrib.auth.models import User
from rest_framework.authentication import TokenAuthentication
import io
from django.core.files.uploadedfile import InMemoryUploadedFile
import logging
from django.contrib.auth import get_user_model
import pandas as pd
from io import BytesIO
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth import authenticate, login
from .serializers import UserSerializer
from rest_framework.parsers import MultiPartParser, FormParser
from exif import Image as ExifImage
from io import BytesIO
from .utils import convert_to_decimal_degrees
from .utils import reverse_geocode
from geopy.distance import geodesic
logger = logging.getLogger(__name__)
from rest_framework.permissions import IsAuthenticated

from PIL import Image as PILImage
from rest_framework.views import APIView


import openai
from openai import OpenAI
import boto3
from django.conf import settings
from django.db import connection
import os
from openai import OpenAI
from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image as PILImage
from PIL.ExifTags import TAGS, GPSTAGS
from django.http import JsonResponse
from django.core.files.storage import default_storage


#openai.api_key = settings.OPENAI_API_KEY


#client = OpenAI(api_key=settings.OPENAI_API_KEY)
OpenAI.api_key ="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"
openai.api_key="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"

#client = OpenAI()

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/simpl/Downloads/sowji-447920-108c16dc20ac.json"
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sowji-447920-108c16dc20ac.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/app/sowji-447920-108c16dc20ac.json")

vision_client = vision.ImageAnnotatorClient()


OpenAI.api_key ="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"
#client = OpenAI()
client = OpenAI(api_key="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")





def fetch_location_details(image_id):
    """Retrieve location details from the database"""
    with connection.cursor() as cursor:
        cursor.execute("SELECT location_name, latitude, longitude,user_id FROM app1_image WHERE id = %s", [image_id])
        result = cursor.fetchone()
    if not result:
        return None, None, None,None
    return result
def fetch_username_details(user_id):
    """Retrieve location details from the database"""
    with connection.cursor() as cursor:
        cursor.execute("SELECT username FROM app1_user WHERE id = %s", [user_id])
        result = cursor.fetchone()
    if not result:
        return None
    return result

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chatbot(request):
    """Django API endpoint for chatbot"""
    user_input = request.data.get('message')
    image_id = request.data.get('image_id')
    print(user_input)
    print(image_id)
    if not user_input or not image_id:
        return Response({"error": "Missing required parameters."}, status=400)

    location_data = fetch_location_details(image_id)
    print(location_data)
    if not location_data:
        return Response({"error": "Location not found."}, status=404)

    location_name, latitude, longitude,user_id = location_data
    print(location_name)
    print(latitude)
    print(longitude)
    print(user_id)

    username=fetch_username_details(user_id)
    try:
        print("entered")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a chatbot providing information about locations. Address the user as {username}."},
                {"role": "user", "content": f"Hi {username},The location is {location_name} (Latitude: {latitude}, Longitude: {longitude}). {user_input}"},
            ],
            max_tokens=200
            
        )
        print("response:")
        print(response)
        chatbot_reply=response.choices[0].message.content
        #chatbot_reply = response['choices'][0]['message']['content']
        print(chatbot_reply)
        return Response({"response": chatbot_reply}, status=200)

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return Response({"error": str(e)}, status=500)

class UserListCreateView(APIView):
    def get(self, request):
        users = User.objects.all()
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            

class UserDetailView(APIView):
    def get_object(self, pk):
        try:
            return User.objects.get(pk=pk)
        except User.DoesNotExist:
            return None

    def get(self, request, pk):
        user = self.get_object(pk)
        if user:
            serializer = UserSerializer(user)
            return Response(serializer.data)
        return Response(status=status.HTTP_404_NOT_FOUND)

    def put(self, request, pk):
        user = self.get_object(pk)
        if user:
            serializer = UserSerializer(user, data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return Response(status=status.HTTP_404_NOT_FOUND)

    def delete(self, request, pk):
        user = self.get_object(pk)
        if user:
            user.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        return Response(status=status.HTTP_404_NOT_FOUND)

User = get_user_model()








@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_users(request):
    """
    Uploads a CSV/Excel file to create user accounts dynamically.
    Only Super Admins and Admins can perform this action.
    """
    if not (request.user.is_super_admin or request.user.is_admin):
        return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)

    file = request.FILES.get('users')
    if not file:
        return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

    # Read file using pandas
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return Response({"error": "Invalid file format. Please upload a CSV or Excel file."}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({"error": f"Error reading file: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

    created_users = []
    for _, row in df.iterrows():
        username = row.get('username')
        email = row.get('email')
        password = row.get('password', 'defaultpassword')  # Default password if missing
        role = row.get('role', '').lower()
        first_name = row.get('first_name', '')
        last_name = row.get('last_name', '')

        if not username or not email:
            continue  # Skip invalid rows

        # Check if user already exists
        if User.objects.filter(username=username).exists():
            continue

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name
        )

        # Assign roles based on CSV data
        if role == "super_admin":
            user.is_super_admin = True
            user.is_staff = True
            user.is_superuser = True
        elif role == "admin":
            user.is_admin = True
            user.is_staff = True
        elif role == "user":
            user.is_user = True

        user.save()
        created_users.append(username)

    return Response({"message": f"Users created: {', '.join(created_users)}"}, status=status.HTTP_201_CREATED)

class ImageListView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]  # Ensure only authenticated users access
    queryset = Image.objects.all()
    serializer_class = ImageSerializer

class ImageListCreateView(generics.ListCreateAPIView):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser) 

    @staticmethod
    def convert_to_decimal_degrees(gps_coords, ref=None):
        """
        Convert GPS coordinates from tuple (degrees, minutes, seconds) to decimal degrees.
        ref: 'N', 'S', 'E', 'W' (optional)
        """
        degrees, minutes, seconds = gps_coords
        decimal_degrees = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ['S', 'W']:
            decimal_degrees = -decimal_degrees
        return decimal_degrees
    

    def create(self, request, *args, **kwargs):
        print("Authenticated User:", request.user)  # Debugging
        print("Request Data:", request.data)  # Debugging
        print("Request Files:", request.FILES)  # Debugging
        #if request.user.is_user:
         #   return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)
        image_file = request.FILES['image']
        image_data = image_file.read()
        image_file.seek(0) 

         # Extract EXIF metadata
        exif_image = ExifImage(image_data)
        latitude, longitude, location_name = None, None, None
        print(latitude,location_name,longitude)
        try:
            if exif_image.has_exif:
                if hasattr(exif_image, 'gps_latitude') and hasattr(exif_image, 'gps_longitude'):
                    latitude_tuple = exif_image.gps_latitude
                    longitude_tuple = exif_image.gps_longitude
                    latitude_ref = exif_image.gps_latitude_ref  # 'N' or 'S'
                    longitude_ref = exif_image.gps_longitude_ref  # 'E' or 'W'

                    # Convert tuple-based GPS coordinates to decimal degrees
                    latitude = self.convert_to_decimal_degrees(latitude_tuple, latitude_ref)
                    longitude = self.convert_to_decimal_degrees(longitude_tuple, longitude_ref)

                    if latitude and longitude:
                        location_name = self.reverse_geocode1(latitude, longitude)
                        print("Location Name:", location_name)
                        
                    
                   
                else:
                  if latitude is None or longitude is None:
            
                    logger.info("EXIF data not found. Using Google Cloud Vision API for location detection.")
                    latitude, longitude, location_name = self.detect_location_with_vision(image_file)
                  
            else:
               logger.warning("Image does not have EXIF data.")
               latitude, longitude, location_name = self.detect_location_with_vision(image_file)
        except Exception as e:
            logger.error("Error extracting EXIF data: %s", str(e))
            #latitude, longitude, location_name = self.detect_location_with_vision(image_file)
        # If EXIF data is not available, use the pretrained model
        
        
        

        if latitude is None or longitude is None:
            print("entered")
            latitude, longitude, location_name = 0.0, 0.0, "Unknown Location"
            print(latitude)
            print(longitude)
            print(location_name)
       
     # Save the image with metadata
        try:
            image_content = ContentFile(image_data, name=image_file.name)
            image_instance = Image(
                user=request.user,
                image=image_content,
                latitude=latitude,
                longitude=longitude,
                location_name=location_name
                
            )
            print(image_instance)
            print(request.user)
            print(image_file)
            print(latitude)
            print(location_name)
            print(longitude)
            image_instance.save()
            print("Image saved successfully:", image_instance.id)
        except Exception as e:
           logger.error(f"Error saving image to database: {e}")
           return Response({"error": "Failed to save image to database."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        serializer = self.get_serializer(image_instance)
        return Response( 
            serializer.data  , status=status.HTTP_201_CREATED)
    def detect_location_with_vision(self, image_file):
        """Detects location using Google Cloud Vision API if EXIF data is missing."""
        try:
            with image_file.open('rb') as image:
                content = image.read()

            image = types.Image(content=content)
            response = vision_client.landmark_detection(image=image)
            landmarks = response.landmark_annotations

            if landmarks:
                landmark = landmarks[0]
                lat, lng = landmark.locations[0].lat_lng.latitude, landmark.locations[0].lat_lng.longitude
                location_name = landmark.description
                print(f"Detected Landmark: {location_name}, Latitude: {lat}, Longitude: {lng}")
                return lat, lng, location_name

        except Exception as e:
            logger.error("Error detecting location using Google Vision API: %s", str(e))

            return None, None, None
    
    def reverse_geocode1(self, latitude, longitude):
        """
        Reverse geocode latitude and longitude to get location name.
        """
        geolocator = Nominatim(user_agent="app1")
        try:
            location = geolocator.reverse((latitude, longitude), exactly_one=True)
            if location:
                return location.address
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.error("Geocoding error: %s", str(e))
            return None
        return None
class ImageRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer
    permission_classes = [permissions.IsAuthenticated]
    print("enteredupdate")
    def get_queryset(self):
        print("enteredupdate1")
        user = self.request.user
        if user.is_superuser:
            return Image.objects.all()
        return Image.objects.filter(user=user)
    
    def update(self, request, *args, **kwargs):
        print("enteredupdate2")
        print(request.user.is_user)
        print(request.user.is_admin)
        if not (request.user.is_admin or request.user.is_super_admin):
            print(request.user.is_super_admin)
            return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)

        try:
            image = self.get_object()  # Get the image instance to update
            print("Image found:", image.id)  # Debugging
        except Image.DoesNotExist:
            print("Image not found")  # Debugging
            return Response({"detail": "No Image matches the given query."}, status=status.HTTP_404_NOT_FOUND)
        print("Updating image:", image.id)  # Debugging
        print("Request data:", request.data)  # Debugging
        print("Request files:", request.FILES)  # Debugging

        # Handle image file update
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            image_data = image_file.read()
            #image_file.seek(0) 
            
            # Extract EXIF metadata
            exif_image = ExifImage(image_data)
            latitude, longitude, location_name = None, None, None

            try:
                if exif_image.has_exif:
                    try:
                        latitude_tuple = exif_image.gps_latitude
                        longitude_tuple = exif_image.gps_longitude
                        latitude_ref = exif_image.gps_latitude_ref  # 'N' or 'S'
                        longitude_ref = exif_image.gps_longitude_ref  # 'E' or 'W'

                        # Convert tuple-based GPS coordinates to decimal degrees
                        latitude = convert_to_decimal_degrees(latitude_tuple, latitude_ref)
                        longitude = convert_to_decimal_degrees(longitude_tuple, longitude_ref)

                        if latitude and longitude:
                            location_name = self.reverse_geocode1(latitude, longitude)
                    except AttributeError as e:
                
                        logger.error(f"AttributeError while extracting EXIF data: {e}")
                else:
                  if latitude is None or longitude is None:
            
                    logger.info("EXIF data not found. Using Google Cloud Vision API for location detection.")
                    latitude, longitude, location_name = self.detect_location_with_vision(image_file)
              
            except Exception as e:
                logger.error("Error extracting EXIF data: %s", str(e))

                
            
                if latitude is None or longitude is None:
                    print("entered")
                    logger.info("EXIF data not found. Using pretrained model to predict location.")
            
                    latitude, longitude, location_name = 0.0, 0.0, "Unknown Location"

            
            try:
                image_file.seek(0)  
                image_data = image_file.read()
                image_content = ContentFile(image_data, name=image_file.name)
            # Update image fields
                image.image.save(image_file.name, image_content)
                image.latitude = latitude
                image.longitude = longitude
                image.location_name = location_name
                image.save()
            except Exception as e:
                logger.error(f"Error saving image to database: {e}")
                return Response({"error": "Failed to save image to database."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
            serializer = self.get_serializer(image)
            return Response(serializer.data)

        # If no image file is provided, update other fields
        serializer = self.get_serializer(image, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        logger.error(f"Failed to update image. Errors: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def reverse_geocode1(self, latitude, longitude):
        """
        Reverse geocode latitude and longitude to get location name.
        """
        geolocator = Nominatim(user_agent="app1")
        try:
            location = geolocator.reverse((latitude, longitude), exactly_one=True)
            if location:
                return location.address
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.error("Geocoding error: %s", str(e))
            return None
        return None

        
    def detect_location_with_vision(self, image_file):
        """Detects location using Google Cloud Vision API if EXIF data is missing."""
        try:
            with image_file.open('rb') as image:
                content = image.read()

            image = types.Image(content=content)
            response = vision_client.landmark_detection(image=image)
            landmarks = response.landmark_annotations

            if landmarks:
                landmark = landmarks[0]
                lat, lng = landmark.locations[0].lat_lng.latitude, landmark.locations[0].lat_lng.longitude
                location_name = landmark.description
                print(f"Detected Landmark: {location_name}, Latitude: {lat}, Longitude: {lng}")
                return lat, lng, location_name

        except Exception as e:
            logger.error("Error detecting location using Google Vision API: %s", str(e))

            return None, None, None

@api_view(['POST'])
def user_login(request):
    username = request.data.get('username')
    password = request.data.get('password')
    user = authenticate(username=username, password=password)
    if user:
        token, _ = Token.objects.get_or_create(user=user)  # Generate or retrieve token
        role = "Super Admin" if user.is_super_admin else "Admin" if user.is_admin else "User"
        return Response({
            'token': token.key,  # Return the token
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_super_admin': user.is_super_admin,
                'is_admin': user.is_admin,
                'is_user': user.is_user,
                'role': role 
            }
        })
    return Response({'error': 'Invalid credentials'}, status=400)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_role(request):
    """
    Returns the role of the currently authenticated user.
    """
    user = request.user
    role = "Super Admin" if user.is_super_admin else "Admin" if user.is_admin else "User"
    
    return Response({"role": role})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def user_logout(request):
    logout(request)
    return Response({'message': 'Logged out successfully'})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_image(request):
    images = request.FILES.getlist('images')
    for image in images:
        image_data = image.read()
        exif_image = ExifImage(image_data)
        latitude, longitude , location_name= None, None,None
        if exif_image.has_exif:
            try:
                latitude_tuple = exif_image.gps_latitude
                longitude_tuple = exif_image.gps_longitude
                latitude_ref = exif_image.gps_latitude_ref  # 'N' or 'S'
                longitude_ref = exif_image.gps_longitude_ref  # 'E' or 'W'
                # Convert tuple-based GPS coordinates to decimal degrees
                
                latitude = ImageListCreateView.convert_to_decimal_degrees(latitude_tuple, latitude_ref)
                longitude = ImageListCreateView.convert_to_decimal_degrees(longitude_tuple, longitude_ref)
                print(longitude)
                print(latitude)
                if latitude and longitude:
                    location_name = reverse_geocode(latitude, longitude)
                    print(location_name)
            except AttributeError:
                pass
        Image.objects.create(user=request.user, image=image, latitude=latitude, longitude=longitude,location_name=location_name)
    return Response({'message': 'Images uploaded successfully'})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def compare_images(request):
    image1 = request.FILES['image1']
    image2 = request.FILES['image2']

    img1 = PILImage.open(image1).resize((224, 224))
    img2 = PILImage.open(image2).resize((224, 224))

    img1_array = np.array(img1) / 255.0
    img2_array = np.array(img2) / 255.0

    model = tf.keras.applications.MobileNetV2(include_top=False, pooling='avg')
    features1 = model.predict(np.expand_dims(img1_array, axis=0))
    features2 = model.predict(np.expand_dims(img2_array, axis=0))

    similarity = np.dot(features1, features2.T) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return Response({'similarity': similarity[0][0]})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_coordinates(request):
    location_name = request.data.get('location')
    geolocator = Nominatim(user_agent="app1")
    location = geolocator.geocode(location_name)
    if location:
        Location.objects.create(name=location_name, latitude=location.latitude, longitude=location.longitude)
        return Response({'latitude': location.latitude, 'longitude': location.longitude})
    return Response({'error': 'Location not found'}, status=status.HTTP_404_NOT_FOUND)




@api_view(['POST'])
@permission_classes([IsAuthenticated])
def calculate_transport_cost(request):
    """
    Calculate the cost of traveling to a location based on distance and suggest the best mode of transport.
    """
    # Your place's coordinates (reference location)
    target_coords = (33.7490, -84.3880)  
    image_id = request.data.get('image_id')
    if not image_id:
        return Response({"error": "No image ID provided."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        image = Image.objects.get(id=image_id)
    except Image.DoesNotExist:
        return Response({"error": "Image not found."}, status=status.HTTP_404_NOT_FOUND)

    # Get coordinates
    target_coords = (33.7490, -84.3880)
    image_coords = (image.latitude, image.longitude)
    print(image_coords)

    

    # Calculate distance using geopy
    distance_km =  geodesic(target_coords, image_coords).km 

    # Define transport rates
    cab_rate_per_km = 1.5  # $1.5 per km for cab
    airplane_rate_per_km = 0.5  # $0.5 per km for airplane
    airplane_fixed_cost = 100  # $100 fixed cost for airplane

    # Calculate costs
    cab_cost = distance_km * cab_rate_per_km
    airplane_cost = (distance_km * airplane_rate_per_km) + airplane_fixed_cost

    # Decide the best mode of transport
    if distance_km > 500:
        suggested_transport = "airplane"
        suggested_cost = airplane_cost
    else:
        suggested_transport = "cab"
        suggested_cost = cab_cost

    # Return the results as a JSON response
    return Response({
        'distance_km': distance_km,
        'cab_cost': cab_cost,
        'airplane_cost': airplane_cost,
        'suggested_transport': suggested_transport,
        'suggested_cost': suggested_cost,
    }, status=status.HTTP_200_OK)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_target_image(request):
    """
    Upload a target image and extract its location metadata.
    """
    if 'image' not in request.FILES:
        return Response({"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST)

    image_file = request.FILES['image']
    image_data = image_file.read()

    # Extract EXIF metadata
    exif_image = ExifImage(image_data)
    latitude, longitude, location_name = None, None, None

    try:
        if exif_image.has_exif:
            try:
                latitude_tuple = exif_image.gps_latitude
                longitude_tuple = exif_image.gps_longitude
                latitude_ref = exif_image.gps_latitude_ref  # 'N' or 'S'
                longitude_ref = exif_image.gps_longitude_ref  # 'E' or 'W'

                # Convert tuple-based GPS coordinates to decimal degrees
                latitude = convert_to_decimal_degrees(latitude_tuple, latitude_ref)
                longitude = convert_to_decimal_degrees(longitude_tuple, longitude_ref)

                if latitude and longitude:
                    location_name = reverse_geocode(latitude, longitude)
            except AttributeError as e:
                logger.error(f"AttributeError while extracting EXIF data: {e}")
    except Exception as e:
        logger.error("Error extracting EXIF data: %s", str(e))

    if not latitude or not longitude:
        return Response({"error": "Could not extract location data from the image."}, status=status.HTTP_400_BAD_REQUEST)

    # Save the target location details in the session or database (optional)
    request.session['target_location'] = {
        'latitude': latitude,
        'longitude': longitude,
        'location_name': location_name
    }

    return Response({
        'latitude': latitude,
        'longitude': longitude,
        'location_name': location_name
    }, status=status.HTTP_201_CREATED)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def calculate_distance(request):
    """
    Calculate the distance between the target location and a selected image's location.
    """
    target_location = "Atlanta"
    if not target_location:
        return Response({"error": "No target location found. Upload a target image first."}, status=status.HTTP_400_BAD_REQUEST)

    image_id = request.data.get('image_id')
    if not image_id:
        return Response({"error": "No image ID provided."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        image = Image.objects.get(id=image_id)
    except Image.DoesNotExist:
        return Response({"error": "Image not found."}, status=status.HTTP_404_NOT_FOUND)

    # Get coordinates
    target_coords = (33.7490, -84.3880)
    image_coords = (image.latitude, image.longitude)
    print(image_coords)

    # Calculate distance using geopy
    distance = geodesic(target_coords, image_coords).km  # Distance in kilometers

    return Response({
        'distance_km': distance,
        'target_location': target_location,
        'image_location': {
            'latitude': image.latitude,
            'longitude': image.longitude,
            'location_name': image.location_name
        }
    }, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def image_information(request):
    """
    generate Opne Ai response.
    """
    image_id = request.data.get('image_id')
    if not image_id:
        return Response({"error": "No image ID provided."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        image = Image.objects.get(id=image_id)
    except Image.DoesNotExist:
        return Response({"error": "Image not found."}, status=status.HTTP_404_NOT_FOUND)

    #client = OpenAI(api_key="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")
    try:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",  
            prompt=f"Provide a detailed description of the location {image.location_name}.",
            max_tokens=500,
            temperature=0

        )
        
        description = response.choices[0].text
        print(description)
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return Response({"error": "Failed to generate description."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    #return Response('location_name': image.location_name, status=status.HTTP_200_OK)
    return Response({
    'Description': description
}, status=status.HTTP_200_OK)