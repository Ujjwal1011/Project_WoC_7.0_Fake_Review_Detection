# 🕵️‍♀️ Fake Review Detection Project (DA-IICT Winter of Code)

## 🏗️ Checkpoint 4: Extension Development

### Overview
In this checkpoint, we developed a browser extension to analyze Amazon reviews for authenticity. The extension interacts with a backend server to process and evaluate the reviews.

### Features
- **Real-time Analysis:** The extension allows users to analyze reviews directly on Amazon's website.
- **Language Detection:** Utilizes spaCy to detect if the review is in English.
- **Fake Review Detection:** Employs a pre-trained machine learning model to classify reviews as fake or genuine.

### Files and Directories
- **Extension/**
    - **popup/**
        - `popup.html`: The HTML file for the extension's popup interface.
        - `popup.css`: Styles for the popup interface.
        - `popup.js`: JavaScript for handling user interactions in the popup.
- **server/**
    - `server.py`: Flask server to handle requests from the extension and perform review analysis.
- **background.js**: Handles background tasks for the extension.
- **manifest.json**: Configuration file for the browser extension.
- **content.js**: Injected into Amazon product pages to extract review data and send it to the background script for analysis.


### How to Run
1. **Setup the Server:**
     - Navigate to the `server` directory.
     - Install the required Python packages:
         ```sh
         pip install -r requirements.txt
         ```
     - Run the Flask server:
         ```sh
         python server.py
         ```

2. **Load the Extension:**
     - Open your browser and navigate to the extensions page.
     - Enable "Developer mode".
     - Click "Load unpacked" and select the `Extension` directory.

3. **Use the Extension:**
     - Navigate to an Amazon product page.
     - It will Automaticaly give if the Review are fake of real.



### Contributors
- Ujjwal 

