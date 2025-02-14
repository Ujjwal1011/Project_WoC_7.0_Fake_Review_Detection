console.log("background.js loaded");

// Listen for tab updates (page loads)
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    try {
        if (changeInfo.status === 'complete' && tab.url.includes('amazon')) {
            console.log("Amazon product page loaded or updated.  Injecting content script.");
            chrome.scripting.executeScript({
                target: { tabId: tabId },
                files: ['content.js']
            });
        }
    } catch (error) {
        console.error("ERROR CAUGHT in chrome.tabs.onUpdated listener!");
    }
});

// Listen for messages from the content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log("Received message from content script:", request); //Add this Line
    if (request.action === "analyzeReview") {
        analyzeReview(request.review, request.rating, request.category) // category
            .then(result => {
                sendResponse(result); // Send back the entire result object
            })
            .catch(error => {
                console.error("Error during analysis:", error);
                sendResponse({ error: error.message });
            });
        return true; // Indicate async response
    }
});

async function analyzeReview(review, rating, category) { // category
    console.log("analyzeReview: Analyzing review:", review, "with category:", category); // category
    const serverUrl = await getServerUrl();

    try {
        const response = await fetch(serverUrl + '/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ review: review, rating: rating, category: category }) // category
        });

        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error in analyzeReview:", error);
        throw error; // Re-throw the error to be handled by the caller
    }
}

function getServerUrl() {
    return new Promise((resolve) => {
        chrome.storage.sync.get({ serverUrl: 'http://localhost:3000' }, (items) => {
            resolve(items.serverUrl);
        });
    });
}