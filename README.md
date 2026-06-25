# 🎬 CineMatch – Hybrid Movie Recommendation System

## Overview
CineMatch is a modern web application that combines **collaborative filtering** (matrix factorization) and **content‑based** techniques to deliver personalized movie recommendations. It supports three recommendation modes:
- **Collaborative** – recommendations based on user‑item interaction matrices (SVD).
- **Popular** – top movies by Bayesian‑adjusted popularity.
- **Genre‑Based** – recommendations for a set of favourite genres.
- **Similar** – content‑based similarity to a selected movie.

The UI is built with vanilla HTML, CSS and JavaScript, featuring a dark, glass‑morphic design with vibrant gold accents, smooth micro‑animations and responsive layouts.

## Features
- Fast API powered by **FastAPI** on the backend.
- Interactive tabs for each recommendation strategy.
- Real‑time search suggestions for movie titles.
- Dynamic genre selection with badge styling.
- Visual score arcs and star ratings on movie cards.

## Screenshots
![CineMatch Home Page](file:///C:/Users/HP/.gemini/antigravity-ide/brain/58b5dbcb-1811-4c03-9480-f96ad29ec5a6/home_page_mockup_1782392696026.png)

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the API server: `python app.py`
3. Serve the frontend (e.g., `python -m http.server 5500`).
4. Open the browser at `http://localhost:5500`.

---
*Built with love using FastAPI, vanilla CSS, and modern design principles.*
