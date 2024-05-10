import keras
import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow Model Prediction
def model_prediction(test_image):
    """Returns the prediction value of the image passed"""
    model = keras.models.load_model("plant_disease.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Remedies or Information for Diseases
remedies_info = {
    'Apple___Apple_scab': "Apply fungicide containing sulfur or copper. Prune infected leaves and destroy them. Improve air circulation around the plant by proper spacing. Use drip irrigation to avoid wetting the leaves.",
    'Apple___Black_rot': "Prune infected twigs and branches. Remove mummified fruits from the tree and the ground. Apply fungicides containing captan, mancozeb, or thiophanate-methyl.",
    'Apple___Cedar_apple_rust': "Prune infected branches. Remove nearby cedar trees if present. Apply fungicides containing myclobutanil, triadimefon, or tebuconazole during the growing season.",
    'Apple___healthy': "Keep the tree well-watered and fertilized. Prune dead or diseased branches regularly. Monitor for signs of pests or diseases.",
    'Blueberry___healthy': "Blueberries generally require little care if planted in the right conditions. Ensure they have well-draining soil, sufficient sunlight, and regular watering.",
    'Cherry_(including_sour)___Powdery_mildew': "Apply fungicides containing sulfur, potassium bicarbonate, or neem oil. Prune infected branches for better air circulation. Remove fallen leaves from around the tree.",
    'Cherry_(including_sour)___healthy': "Cherry trees are relatively low maintenance if planted in well-draining soil with good sunlight. Prune dead or diseased branches as needed.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Rotate crops to prevent disease buildup. Apply fungicides containing chlorothalonil or mancozeb. Remove and destroy infected plant debris.",
    'Corn_(maize)___Common_rust_': "Plant resistant corn varieties if available. Apply fungicides containing triazoles or strobilurins. Remove and destroy infected plant debris.",
    'Corn_(maize)___Northern_Leaf_Blight': "Rotate crops to prevent disease buildup. Apply fungicides containing chlorothalonil or mancozeb. Remove and destroy infected plant debris.",
    'Corn_(maize)___healthy': "Corn plants usually thrive with proper watering and fertilization. Practice crop rotation to prevent disease.",
    'Grape___Black_rot': "Prune infected canes and destroy them. Apply fungicides containing mancozeb or captan during the growing season. Remove mummified berries from the vine.",
    'Grape___Esca_(Black_Measles)': "Prune infected wood back to healthy tissue. Apply fungicides containing thiophanate-methyl or fosetyl-al. Keep vines well-watered and avoid stress.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Apply fungicides containing chlorothalonil or mancozeb. Prune infected leaves and destroy them. Improve air circulation around the vines.",
    'Grape___healthy': "Grapevines require regular pruning and training. Ensure they have proper support and sunlight exposure. Monitor for pests and diseases.",
    'Orange___Haunglongbing_(Citrus_greening)': "Control psyllid insects with insecticides or beneficial insects. Remove and destroy infected trees to prevent spread. Plant resistant citrus varieties if available.",
    'Peach___Bacterial_spot': "Apply copper-based fungicides during the growing season. Prune infected branches and destroy them. Avoid overhead irrigation to reduce leaf wetness.",
    'Peach___healthy': "Peach trees need regular pruning and thinning to maintain air circulation. Apply appropriate fertilizers and water deeply during dry periods.",
    'Pepper,_bell___Bacterial_spot': "Apply copper-based fungicides during the growing season. Rotate crops to prevent disease buildup. Avoid overhead irrigation to reduce leaf wetness.",
    'Pepper,_bell___healthy': "Pepper plants usually thrive with proper watering and fertilization. Practice crop rotation to prevent disease.",
    'Potato___Early_blight': "Rotate crops to prevent disease buildup. Apply fungicides containing chlorothalonil or mancozeb. Remove and destroy infected plant debris.",
    'Potato___Late_blight': "Rotate crops to prevent disease buildup. Apply fungicides containing chlorothalonil or mancozeb. Remove and destroy infected plant debris.",
    'Potato___healthy': "Potato plants need regular watering and fertilization. Hill up soil around the plants to prevent tubers from becoming green.",
    'Raspberry___healthy': "Raspberry plants require regular pruning and thinning to maintain airflow. Mulch around plants to suppress weeds and retain moisture.",
    'Soybean___healthy': "Soybeans usually thrive with proper planting density and weed control. Monitor for pests such as aphids and bean leaf beetles.",
    'Squash___Powdery_mildew': "Apply fungicides containing sulfur or potassium bicarbonate. Prune infected leaves and destroy them. Improve air circulation around the plants.",
    'Strawberry___Leaf_scorch': "Apply fungicides containing chlorothalonil or mancozeb. Prune infected leaves and destroy them. Improve air circulation around the plants.",
    'Strawberry___healthy': "Strawberry plants require regular watering and fertilization. Mulch around plants to suppress weeds and retain moisture.",
    'Tomato___Bacterial_spot': "Apply copper-based fungicides during the growing season. Rotate crops to prevent disease buildup. Avoid overhead irrigation to reduce leaf wetness.",
    'Tomato___Early_blight': "Rotate crops to prevent disease buildup. Apply fungicides containing chlorothalonil or mancozeb. Remove and destroy infected plant debris.",
    'Tomato___Late_blight': "Rotate crops to prevent disease buildup. Apply fungicides containing chlorothalonil or mancozeb. Remove and destroy infected plant debris.",
    'Tomato___Leaf_Mold': "Improve air circulation around the plants. Avoid overhead irrigation to reduce leaf wetness. Apply fungicides containing chlorothalonil or mancozeb.",
    'Tomato___Septoria_leaf_spot': "Rotate crops to prevent disease buildup. Apply fungicides containing chlorothalonil or mancozeb. Remove and destroy infected plant debris.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Apply insecticidal soap or neem oil to control spider mites. Prune heavily infested leaves and destroy them. Introduce predatory mites for biological control.",
    'Tomato___Target_Spot': "Rotate crops to prevent disease buildup. Apply fungicides containing chlorothalonil or mancozeb. Remove and destroy infected plant debris.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Control whiteflies with insecticides or sticky traps. Plant resistant tomato varieties if available. Remove and destroy infected plants to prevent spread.",
    'Tomato___Tomato_mosaic_virus': "Control aphids, thrips, and whiteflies with insecticides or beneficial insects. Plant resistant tomato varieties if available. Remove and destroy infected plants to prevent spread.",
    'Tomato___healthy': "Tomato plants need regular watering and fertilization. Prune and stake plants for better air circulation. Monitor for pests such as aphids and tomato hornworms.",
}


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "appimage.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves which are categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        predicted_disease = class_name[result_index]
        if predicted_disease in remedies_info:
            st.success("Model predicts it's a {}".format(predicted_disease))
            st.write("Remedies/Information:")
            st.info(remedies_info[predicted_disease])
        else:
            st.error("Disease not found in remedies information. Please consult an expert.")
