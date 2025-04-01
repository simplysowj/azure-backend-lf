from django.db import models
from django.contrib.auth.models import AbstractUser
from storages.backends.s3boto3 import S3Boto3Storage

from django.db import models
from django.dispatch import receiver
from django.db.models.signals import pre_delete, pre_save
from storages.backends.s3boto3 import S3Boto3Storage

class User(AbstractUser):
    ROLE_CHOICES = [
        ('super_admin', 'Super Admin'),
        ('admin', 'Admin'),
        ('user', 'User'),
    ]

    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')
    
    is_super_admin = models.BooleanField(default=False)
    is_admin = models.BooleanField(default=False)
    is_user = models.BooleanField(default=False)

    def __str__(self):
        return self.username  # Return the username for the User model

class Image(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="images")
    #image = models.ImageField(upload_to='images/', storage=S3Boto3Storage(),max_length=500 )  # Files will be uploaded to MEDIA_ROOT/images/
    uploaded_at = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='images/',max_length=500 )
    #image = models.CharField(max_length=1000) 
   # image = models.CharField(max_length=500)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    location_name = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f"Image uploaded by {self.user.username} at {self.uploaded_at}"

    # **1. Automatically delete the old image from S3 when updating**
#@receiver(pre_save, sender=Image)
#def delete_old_image(sender, instance, **kwargs):
 #   """Delete the old image from S3 before saving a new one."""
  #  try:
   #     old_instance = Image.objects.get(id=instance.id)
    #    if old_instance.image and old_instance.image != instance.image:
        #    old_instance.image.delete(save=False)  # Delete from S3
    #except Image.DoesNotExist:
     #   pass  # If the instance is new, nothing to delete

# **2. Automatically delete image from S3 when the record is deleted**
#@receiver(pre_delete, sender=Image)
#def delete_image_on_delete(sender, instance, **kwargs):
 #   """Delete the image from S3 when the record is deleted."""
  #  if instance.image:
   #     instance.image.delete(save=False)  # Delete from S3

#class Location(models.Model):
 #   name = models.CharField(max_length=255, unique=True)
  #  latitude = models.FloatField()
   # longitude = models.FloatField()

    #def __str__(self):
     #   return self.name  # Return the name of the location