from django.urls import path
from . import views

urlpatterns = [
	path('', views.list, name='list'),
	path('notjpg', views.notjpg, name='notjpg'),
	path('loading', views.loading, name='loading')
]