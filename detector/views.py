def upload(request):

    result = None
    confidence = None
    image_url = None

    if request.method == "POST":

        form = UploadImageForm(request.POST, request.FILES)

        if form.is_valid():

            try:
                print("=" * 50)
                print("Upload request received")

                image = form.cleaned_data["image"]

                fs = FileSystemStorage()

                filename = fs.save(image.name, image)

                image_path = fs.path(filename)

                image_url = fs.url(filename)

                print("Saved Image:", image_path)

                result, confidence = predict_pcos(image_path)

                print("Prediction Success")

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