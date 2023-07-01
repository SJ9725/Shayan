import face_recognition

# Load the ID card image and the selfie image
id_card_image = face_recognition.load_image_file(r"C:\Users\shaya\OneDrive\Pictures\BRP.jpg")
selfie_image = face_recognition.load_image_file(r"C:\Users\shaya\OneDrive\Pictures\Shayan.jpg")

# Detect faces in the ID card image and the selfie image
id_card_face_locations = face_recognition.face_locations(id_card_image)
selfie_face_locations = face_recognition.face_locations(selfie_image)

# Encode the face in the ID card image
id_card_face_encoding = face_recognition.face_encodings(id_card_image, id_card_face_locations)[0]

# Encode the faces in the selfie image
selfie_face_encodings = face_recognition.face_encodings(selfie_image, selfie_face_locations)

# Compare the face in the ID card image with the faces in the selfie image
for selfie_face_encoding in selfie_face_encodings:
    matches = face_recognition.compare_faces([id_card_face_encoding], selfie_face_encoding)
    if matches[0]:
        print("Face match found!")
    else:
        print("Face match not found!")
