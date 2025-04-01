from django.db import migrations

def create_initial_users(apps, schema_editor):
    User = apps.get_model('app1', 'User')  # Update with your actual app name if different

    users_data = [
        {
            'username': 'superadmin1',
            'email': 'superadmin@example.com',
            'password': 'superadmin123',
            'first_name': 'Super',
            'last_name': 'Admin',
            'role': 'super_admin',
        },
        {
            'username': 'admin1',
            'email': 'admin@example.com',
            'password': 'admin123',
            'first_name': 'Admin',
            'last_name': 'User',
            'role': 'admin',
        },
        {
            'username': 'user1',
            'email': 'user@example.com',
            'password': 'user123',
            'first_name': 'Regular',
            'last_name': 'User',
            'role': 'user',
        },
    ]

    for user_data in users_data:
        if not User.objects.filter(username=user_data['username']).exists():
            user = User.objects.create_user(
                username=user_data['username'],
                email=user_data['email'],
                password=user_data['password'],
                first_name=user_data['first_name'],
                last_name=user_data['last_name'],
            )
            
            if user_data['role'] == "super_admin":
                user.is_super_admin = True
                user.is_staff = True
                user.is_superuser = True
            elif user_data['role'] == "admin":
                user.is_admin = True
                user.is_staff = True
            elif user_data['role'] == "user":
                user.is_user = True

            user.save()
            print(f"✅ User {user_data['username']} created successfully")

class Migration(migrations.Migration):
    dependencies = [
        ('app1', '0001_initial'),  # Ensure this points to the last applied migration
    ]

    operations = [
        migrations.RunPython(create_initial_users),
    ]
