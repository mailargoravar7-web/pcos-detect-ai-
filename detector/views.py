from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from .forms import UploadImageForm
from .predict import predict_pcos


def home(request):
    return render(request, "home.html")


def upload(request):

    form = UploadImageForm()

    result = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":

        form = UploadImageForm(request.POST, request.FILES)

        if form.is_valid():

            fs = FileSystemStorage()

            filename = None

            try:

                print("=" * 60)
                print("Upload Request Received")
                print("=" * 60)

                image = form.cleaned_data["image"]

                filename = fs.save(image.name, image)

                image_path = fs.path(filename)

                image_url = fs.url(filename)

                print("Saved Image :", image_path)

                result, confidence = predict_pcos(image_path)

                print("=" * 60)
                print("Prediction Successful")
                print("=" * 60)

            except Exception as e:

                import traceback
                traceback.print_exc()

                error = str(e)

            finally:

                # Optional:
                # Delete uploaded image after prediction.
                # Comment these two lines if you want to keep uploads.

                # if filename and fs.exists(filename):
                #     fs.delete(filename)

                pass

    return render(
        request,
        "upload.html",
        {
            "form": form,
            "result": result,
            "confidence": confidence,
            "image_url": image_url,
            "error": error,
        },
    )