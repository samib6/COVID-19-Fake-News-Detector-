from django import forms

class NewsForm(forms.Form):
    message = forms.CharField(label="Enter the message to check ",max_length=1000)
