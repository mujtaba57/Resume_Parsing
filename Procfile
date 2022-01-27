web: gunicorn resume_parser.wsgi
python manage.py collectstatic --noinput
python manage.py makemigratations
python manage.py migrate
python manage.py runserver
