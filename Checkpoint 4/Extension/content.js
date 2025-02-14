console.log("Content script running on Amazon product page.");

// Global variables
window.fakeReviewCount = 0;
let previousUrl = window.location.href;
let lastCategory = null; // Store the last found category

async function analyzeAndDisplayLabel(reviewData, productCategory) {
  console.log("analyzeAndDisplayLabel called with category:", productCategory); //Check1
  try {
    // Send review to background script for analysis
    const response = await new Promise(resolve => {
      try {
        chrome.runtime.sendMessage(
          { action: "analyzeReview", review: reviewData.text, rating: reviewData.rating, category: productCategory },
          resolve
        );
      } catch (error) {
        console.error("Error sending message:", error);
        resolve({ error: error.message }); // Resolve with an error object
      }
    });

    if (response.error) {
      console.error("Error analyzing review:", response.error);
      return; // Stop processing if there's an error
    }

    let labelText = '';
    let labelColor = '';

    if (response.nonEnglish) {
      labelText = 'Non-English Review';
      labelColor = 'orange';
    } else if (response.isFake) {
      console.log(`Fake review detected for review ID:${reviewData.id}`);
      window.fakeReviewCount++;
      labelText = 'Potential Fake Review';
      labelColor = 'red';
    } else {
      console.log(`Real review detected for review ID: ${reviewData.id}`);
      labelText = 'Real Review';
      labelColor = 'green';
    }

    displayLabel(reviewData.element, labelText, labelColor);

  } catch (error) {
    console.error("Error in analyzeAndDisplayLabel:", error);
  }
}

function displayLabel(reviewElement, text, color) {
  const reviewBody = reviewElement.querySelector('[data-hook="review-body"]');

  if (reviewBody) {
    // Check if a label already exists before adding a new one
    if (reviewElement.querySelector('.review-label')) {
      console.warn(`Review ID: ${reviewElement.id} already has a label. Skipping.`);
      return; // Don't add another label
    }

    const label = document.createElement('span');
    label.innerText = text;

    // Apply CSS styles using classes
    label.classList.add('review-label');
    label.classList.add(`review-label-${color}`); // e.g., review-label-red

    reviewBody.appendChild(label);
  } else {
    console.warn(`Review body not found for review ID: ${reviewElement.id}`);
  }
}

async function processReviews() {
  console.log("content.js: processReviews function called (triggered by pagination event)");
  const reviewElements = document.querySelectorAll('[data-hook="review"]');
  console.log(`content.js: Found ${reviewElements.length} review elements.`);

  if (reviewElements.length === 0) {
    console.log("content.js: No review elements found on this page.");
    return;
  }



  let productCategory = null;
  // Find product category outside of the loop
  const subnav = document.getElementById('nav-subnav');

  if (subnav) {
    const category = subnav.getAttribute('data-category');
    if (category) {
      console.log("Extracted category:", category);
      lastCategory = productCategory; // Update lastCategory
      productCategory = category;
    } else {
      console.warn("data-category attribute not found on #nav-subnav element.");
      productCategory = lastCategory; // Use last known category
    }
  } else {
    console.warn("#nav-subnav element not found.");
    productCategory = lastCategory; // Use last known category
  }

if (productCategory) {
  console.log("Product Category", productCategory);
}


  for (const reviewElement of reviewElements) {
    const reviewId = reviewElement.id;
    // const category = reviewElement.querySelector('[data-hook="review-category"]')?.innerText; //No needed anymore
    const reviewText = reviewElement.querySelector('[data-hook="review-body"] span')?.innerText;
    const ratingElement = reviewElement.querySelector('[data-hook="review-star-rating"]');
    let rating = null;

    if (ratingElement) {
      const ratingMatch = ratingElement.innerText.match(/(\d\.\d)/);
      rating = ratingMatch?.[1] || null;
    } else {
      console.warn(`Rating element not found for review ID: ${reviewId}.`);
    }

    if (!reviewText) {
      console.warn(`Review text not found for review ID: ${reviewId}.`);
      continue;
    }

    const reviewData = {
      id: reviewId,
      text: reviewText,
      rating: rating,
      element: reviewElement,
    };

    console.log(`Processing review ID: ${reviewId} with rating: ${rating} and category: ${productCategory}`);
    await analyzeAndDisplayLabel(reviewData, productCategory);
  }

  console.log("Total fake reviews detected on this page:", window.fakeReviewCount);
}

// Function to check for URL changes
function checkUrlChange() {
  if (window.location.href !== previousUrl) {
    console.log("URL changed from", previousUrl, "to", window.location.href);
    previousUrl = window.location.href;
    processReviews(); // Re-analyze reviews
  }
}

function initializeExtension() {
  processReviews(); // Initial processing of reviews on page load
  setInterval(checkUrlChange, 2000); // Check for URL changes every 2 seconds
}

initializeExtension();

// Inject CSS styles into the page
const style = document.createElement('style');
style.textContent = `
    .review-label {
        font-size: 0.8em; /* Adjust the font size */
        font-weight: bold; /* Make the text bold */
        margin-left: 5px; /* Reduce the margin */
        padding: 2px 5px; /* Add some padding */
        border-radius: 3px; /* Rounded corners */
    }

    .review-label-red {
        color: white; /* White text */
        background-color: red; /* Red background */
    }

    .review-label-green {
        color: white; /* White text */
        background-color: green; /* Green background */
    }

    .review-label-orange {
        color: black;
        background-color: orange; /* Orange background */
    }
`;
document.head.appendChild(style);