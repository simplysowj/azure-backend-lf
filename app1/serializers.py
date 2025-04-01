# myapp/serializers.py

from rest_framework import serializers
from .models import User, Image
from django.contrib.auth import get_user_model
User = get_user_model()



class UserSerializer(serializers.ModelSerializer):
    role = serializers.CharField(write_only=True)  # Make role writable
    computed_role = serializers.SerializerMethodField(read_only=True)  # Add computed role field

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'is_super_admin', 'is_admin', 'is_user', 'role', 'computed_role', 'password']
        extra_kwargs = {
            'password': {'write_only': True},  # Ensure password is write-only
        }

    def get_computed_role(self, obj):
        if obj.is_super_admin:
            return "Super Admin"
        elif obj.is_admin:
            return "Admin"
        elif obj.is_user:
            return "User"
        return "Unknown"

    def create(self, validated_data):
        role = validated_data.pop('role', '').lower()  # Extract role from validated data
        password = validated_data.pop('password')  # Extract password

        # Create the user with the hashed password
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=password,  # Password is hashed here
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', '')
        )
        # Assign roles based on the role field
        if role == "super_admin":
            user.is_super_admin = True
            user.is_staff = True
            user.is_superuser = True
        elif role == "admin":
            user.is_admin = True
            user.is_staff = True
        elif role == "user":
            user.is_user = True

        user.save()  # Save the user with updated roles
        return user

    def update(self, instance, validated_data):
        role = validated_data.pop('role', '').lower()  # Extract role from validated data
        instance = super().update(instance, validated_data)  # Update other fields

        # Assign roles based on the role field
        if role == "super_admin":
            instance.is_super_admin = True
            instance.is_staff = True
            instance.is_superuser = True
        elif role == "admin":
            instance.is_admin = True
            instance.is_staff = True
        elif role == "user":
            instance.is_user = True

        instance.save()  # Save the user with updated roles
        return instance
class ImageSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source='user.username')  # Read-only to prevent manual input

    class Meta:
        model = Image
        fields = ['id', 'user', 'image', 'uploaded_at', 'latitude', 'longitude', 'location_name']

    def create(self, validated_data):
        return Image.objects.create(**validated_data)

    
    
        
        
