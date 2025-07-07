
# 🥗 NutriGoal: AI-Powered Personalized Diet Planner

**NutriGoal** is an end-to-end machine learning project that recommends personalized diet plans based on an individual's health metrics, fitness goals, and dietary preferences. It combines data science, web development, and MLOps best practices to deliver goal-driven nutrition recommendations.

---

## 🚀 Features

- 🔍 User Input: Age, height, weight, exercise level, dietary preference
- 🧮 ML-based macro calculator (calories, protein, carbs, fats)
- 🥗 Intelligent meal matcher using real food database
- 🧠 Auto-tagged food goals (`muscle_gain`, `weight_loss`, etc.)
- 📊 Macro-balanced combo generation
- 🌐 Full-stack web app with REST API + React frontend
- ☁️ Deployed on Firebase for public access

---

## 📦 Tech Stack

| Layer         | Tools Used                                 |
|---------------|---------------------------------------------|
| **Frontend**  | React, Tailwind CSS, Blackbox AI Components |
| **Backend**   | Flask (Python), REST API, SQLAlchemy        |
| **Database**  | MySQL Workbench                             |
| **ML/Logic**  | Pandas, Scikit-learn, NumPy, Macro Engine   |
| **Deployment**| Firebase Hosting, GitHub CI/CD (optional)   |

---

## 🔢 How It Works

1. **User Inputs**:
   - Height, weight, age, gender
   - Exercise level (light, 3 days/week, etc.)
   - Goal: muscle gain / weight loss / maintenance
   - Veg / non-veg preference

2. **Macro Engine**:
   - Calculates maintenance calories using Mifflin-St Jeor equation
   - Adjusts macros based on user goal (cut, bulk, maintain)

3. **Meal Recommender**:
   - Selects food combos that best match target macros
   - Filters based on diet preference (veg/non-veg)
   - Returns top combo(s) for the user’s goal

---

## 📊 Sample Input → Output

**Input**:
```json
{
  "height": 178,
  "weight": 75,
  "age": 24,
  "gender": "male",
  "goal": "muscle_gain",
  "exercise_level": "3 days/week weight training",
  "preference": "veg"
}
```

**Output**:
```json
{
  "target_macros": {
    "calories": 2900,
    "protein": 250,
    "carbs": 290,
    "fat": 80
  },
  "recommended_meal": [
    "Tofu Stir Fry",
    "Quinoa Salad",
    "Protein Shake"
  ]
}
```

---

## 📸 Screenshots (Optional)

> _Include screenshots of the app frontend here._

---

## ⚙️ Installation & Run Locally

```bash
# Backend
cd backend
pip install -r requirements.txt
python app.py

# Frontend
cd frontend
npm install
npm start
```

---

## 🌍 Live Demo

> 🔗 https://nutrigoal-diet.web.app

---

## 📚 Learnings & Concepts Demonstrated

- ✅ Feature engineering with real-world nutrition data
- ✅ Rule-based logic + optional ML classification
- ✅ Goal-driven filtering and matching logic
- ✅ RESTful API design with Flask
- ✅ Frontend/backend integration
- ✅ Full pipeline from data to deployment

---

## 📌 Future Improvements

- Add user login (Firebase Auth)
- Track daily intake and meal history
- Include micronutrient recommendations
- Train classification model to auto-tag foods

---

## 👨‍💻 Author

**Paras Vaidya**  
_Data Science | Machine Learning | MLOps_

🔗 [LinkedIn](https://www.linkedin.com/in/parasvaidya/)  
🔗 [GitHub](https://github.com/parasvaidya)

---

## ⭐️ Give it a Star

If you like this project, consider giving it a ⭐ on GitHub!
