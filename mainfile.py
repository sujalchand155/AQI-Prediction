import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load trained model and scaler
def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    return model

# Predict function
def predict_aqi(model, features):

    prediction = model.predict(features)
    return prediction[0]

# Streamlit UI
st.title("Air Quality Index (AQI) Classification")

st.sidebar.header("Input Parameters")

# User input fields
state = st.sidebar.text_input("State", "Maharashtra")
city = st.sidebar.text_input("City", "Mumbai")
PM2_5 = st.sidebar.number_input("PM2.5", min_value=0.0, value=50.0)
PM10 = st.sidebar.number_input("PM10", min_value=0.0, value=80.0)
O3 = st.sidebar.number_input("O3", min_value=0.0, value=30.0)
SO2 = st.sidebar.number_input("SO2", min_value=0.0, value=10.0)
CO = st.sidebar.number_input("CO", min_value=0.0, value=0.8)
wind_speed = st.sidebar.number_input("Wind Speed", min_value=0.0, value=3.0)
humidity = st.sidebar.number_input("Humidity", min_value=0.0, value=60.0)
temp = st.sidebar.number_input("Temperature", min_value=-10.0, value=25.0)
aqi = st.sidebar.number_input("AQI", min_value=-10.0, value=300.0)

statename=['Andaman and Nicobar Islands', 'Andhra Pradesh', 'Assam',
       'Aurangabad', 'Bihar', 'Chandigarh', 'Chhattisgarh',
       'Churachandpur', 'Cuddalore', 'Dadra and Nagar Haveli',
       'Daman and Diu', 'Delhi', 'Goa', 'Gujarat', 'Haryana',
       'Himachal Pradesh', 'Hisar', 'India', 'Jharkhand', 'Jind',
       'Karnataka', 'Kerala', 'Koshi', 'Lakshadweep', 'Madhya Pradesh',
       'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland',
       'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim',
       'South Goa', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh',
       'Uttarakhand', 'West Bengal', 'West Medinipur']
cityname=['Addanki', 'Aizawl', 'Ajmer', 'Alappakkam', 'Amarpur', 'Ambad',
       'Ambassa', 'Amguri', 'Amli', 'Angul', 'Annigeri', 'Arambag',
       'Arang', 'Ariyalur', 'Arki', 'Athagarh', 'Athmallik', 'Aurangabad',
       'Ayan Kurinjipadi', 'Azhikkal', 'Babra', 'Baddi', 'Baghpat',
       'Bagnan', 'Bahadurgarh', 'Bakal', 'Baloda Bazar', 'Bandoda',
       'Banki', 'Banur', 'Barabanki', 'Bardhaman', 'Barkakana',
       'Barpathar', 'Barwala', 'Basni', 'Begumganj', 'Bemetra',
       'Bezarchuk Bongali', 'Bhag Kohalia', 'Bhanjipur', 'Bhatapara',
       'Bhilai', 'Bhiwani', 'Bhuban', 'Bhuvanagiri', 'Bihar Sharif',
       'Bilimora', 'Birpur', 'Bishnupur', 'Bokajan', 'Bommayapalayam',
       'Bordubi No.1', 'Botad', 'Bundu', 'Chandel', 'Chandigarh',
       'Charoda', 'Chatapur', 'Chelakkara', 'Cherpulassery', 'Chichli',
       'Chikhli', 'Chilakaluripet', 'Chinchinim', 'Chistipur', 'Chotila',
       'Cuddalore', 'Cuncolim', 'Curchorem', 'Dadra', 'Dagshai', 'Dahanu',
       'Dakshin Odlabari', 'Daman', 'Darjeeling', 'Darlawn', 'Darsi',
       'Davorlim', 'Delhi', 'Deogarh', 'Deori', 'Deulgaon Raja',
       'Devprayag', 'Dewa', 'Dhamtari', 'Dhandri Talli', 'Dharampur',
       'Dharmanagar', 'Dhekiajuli', 'Dhenkanal', 'Dhing Town',
       'Dibrugarh', 'Didwana', 'Diphu', 'Doom Dooma', 'Dugadda',
       'Dungariya', 'Durg', 'Erumapatti', 'Ferozpur Bangar',
       'Gadag-Betigeri', 'Gadarwara', 'Gajia', 'Gangtok', 'Gangwa',
       'Garehagi', 'Garmari', 'Georai', 'Geyzing', 'Ghaglori', 'Ghatal',
       'Goalpara', 'Golaghat', 'Gosainganj', 'Govindapuram', 'Gudur',
       'Gumia', 'Gurugram', 'Guruvayur', 'Guwahati', 'Handique Gaon',
       'Hansi', 'Harrai', 'Hatimuria', 'Hazaribagh', 'Hisar', 'Hisua',
       'Hojai', 'Huvina Hadagali', 'Islampur', 'Itaunja', 'Jahanabad',
       'Jalna', 'Jandiala Manjki', 'Jangaon', 'Jasdan', 'Jawhar', 'Jind',
       'Jiribam', 'Jogbani', 'Jorhat', 'Julana', 'Kailashahar',
       'Kakching', 'Kakori', 'Kalanaur', 'Kalimpong', 'Kamakhyanagar',
       'Kamalpur', 'Kanhangad', 'Kanigiri', 'Kanke', 'Kannur',
       'Kapurthala', 'Karimganj', 'Karimnagar', 'Kartarpur', 'Kasauli',
       'Kavaratti', 'Kawaloor', 'Kesapuri Tanda', 'Kharar', 'Kharkhoda',
       'Khawhai', 'Khekra', 'Khetri Hardia', 'Khewra', 'Khliehriāt',
       'Khowai', 'Khunti', 'Khusrupur', 'Killa-pardi', 'Kohima',
       'Kolasib', 'Kot Ise Khan', 'Kotdwar', 'Kotkhai', 'Kottapalli',
       'Kuchaman City', 'Kuchera', 'Kuju', 'Kumarpatty', 'Kunnamkulam',
       'Lahraud', 'Laipham Siphai', 'Lakhipur', 'Lalgudi', 'Lansdowne',
       'Laxmeshwar', 'Lohardaga', 'Longleng', 'Loni Dehat', 'Lucknow',
       'Lumding', 'Lunglei', 'Mahé', 'Mairang', 'Majhitar',
       'Makhdoom Pura', 'Makhu', 'Makrana', 'Makum', 'Malappuram',
       'Malihabad', 'Mamit', 'Manjeri', 'Mannarkkad', 'Marakkanam',
       'Margao', 'Markapur', 'Masaurhi', 'Mayang Imphal', 'Meham',
       'Memari', 'Merta', 'Mettupalayam', 'Mirik', 'Mohan', 'Moirang',
       'Mokokchung', 'Montiviot Tea Garden', 'Mulgund', 'Mundargi',
       'Mundwa', 'Musiri', 'Muzhappilangad', 'Nagaur', 'Nagram',
       'Nagri-Parole', 'Naharkatia Kaibatra', 'Nakodar', 'Namchi',
       'Namsai', 'Narainpur Pt V', 'Narasaraopeta', 'Naregal', 'Narkand',
       'Narnaund', 'Narsinghpur', 'Narsingi', 'Nawada', 'Nayana Sagar',
       'Nazira', 'New Delhi', 'Nileshwar', 'No.1 Barjuli T.E.',
       'Nongstoin', 'North Lakhimpur', 'North Vanlaiphai', 'Nurmahal',
       'Ongole', 'Orian', 'Ottapalam', 'Pachmarhi', 'Paithan',
       'Palasbari', 'Paliyad', 'Panchkula', 'Parbatsar', 'Parnera',
       'Partur', 'Parwanoo', 'Patan', 'Patrasayer', 'Pavuluru',
       'Payyanur', 'Pedda Nakkala Palem', 'Peddapalli', 'Pherzawl',
       'Pinjore', 'Pipri', 'Ponda', 'Port Blair', 'Puducherry',
       'Pullambadi', 'Purnaheitupokpi', 'Pushkar', 'Quepem', 'Raipur',
       'Rajagaon', 'Rajgarh', 'Rajgir', 'Rajkot', 'Ramgarh Cantonment',
       'Ramjibanpur', 'Ranchi', 'Rangpo Forest', 'Ranikhet', 'Ranirbazar',
       'Ray', 'Resubelpara', 'Rona', 'Rudraprayag',
       'Sahibzada Ajit Singh Nagar', 'Sairang', 'Saitlaw', 'Salaraitoli',
       'Sandana', 'Sanguem', 'Sankhol', 'Satbhaiya', 'Satrikh',
       'Saualkuchi', 'Sayla', 'Seodhai Kopowhuwa', 'Shahkot', 'Shapar',
       'Sheikhpur Khurd', 'Shillong', 'Shimla', 'Shirahatti', 'Shoranur',
       'Shrirampur', 'Siaha', 'Siddipet', 'Silao', 'Silapathar',
       'Silchar', 'Silvassa', 'Singtam', 'Sircilla', 'Sitapur',
       'Sohagpur', 'Solan', 'Srinagar', 'Subathu', 'Sultanpur',
       'Sultanpur Lodhi', 'Suni', 'Swamikunnu', 'Talaulim', 'Talcher',
       'Talcher Town', 'Talchinan-Sanihati', 'Tengkhal Khunou',
       'Tengnoupal', 'Tezpur', 'Thalassery', 'Thangadh',
       'Thathaiyangarpet', 'Theog', 'Thirukkattupalli', 'Thorapadi',
       'Thoubal', 'Thrissur', 'Thuraiyur', 'Tindivanam', 'Tinsukia',
       'Tiruchirappalli', 'Tirur', 'Torban', 'Tosham', 'Udaipur',
       'Udaipura', 'Uttar Kanchanpur Pt I', 'Valavanur', 'Valsad', 'Vapi',
       'Vatakara', 'Veeraganur', 'Vemulawada', 'Vikravandi', 'Viluppuram',
       'Vinchhiya', 'Vinukonda', 'Vodlemol Cacora', 'Wadgaon Kolhati',
       'Waladgaon', 'Wankaner', 'Warangal', 'Warisaliganj', 'Wokha',
       'Yambem', 'Yelburga', 'Zaidpur']

# Load model
model= load_model()

# Prediction
if st.sidebar.button("Predict AQI Type"):
    input_features = [[statename.index(state),cityname.index(city),PM2_5, PM10, O3, SO2, CO, wind_speed, humidity, temp,aqi]]
    aqi_type = predict_aqi(model, input_features)
    anslist=['Good', 'Moderate', 'Poor', 'Satisfactory']
    st.write(f"### Predicted AQI Type: {anslist[aqi_type]}")


