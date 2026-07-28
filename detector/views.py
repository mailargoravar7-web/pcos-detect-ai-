import os

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from .forms import UploadImageForm
from .predict import predict_pcos


def home(request):
    return render(request, "home.html")


def upload(request):

    result = None
    confidence = None
    image_url = None

    if request.method == "POST":

        form = UploadImageForm(request.POST, request.FILES)

        if form.is_valid():

            image = form.cleaned_data["image"]

            fs = FileSystemStorage()

            filename = fs.save(image.name, image)

            image_path = fs.path(filename)

            image_url = fs.url(filename)

            try:
                result, confidence = predict_pcos(image_path)
                print("Result:", result)
                print("Confidence:", confidence)

            except Exception as e:
                import traceback
                traceback.print_exc()

                return render(
                    request,
                    "upload.html",
                    {
                        "form": form,
                        "error": str(e),
                    },
                )

    else:

        form = UploadImageForm()

    return render(
        request,
        "upload.html",
        {
            "form": form,
            "result": result,
            "confidence": confidence,
            "image_url": image_url,
        },
    )