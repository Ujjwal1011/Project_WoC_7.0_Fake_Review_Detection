document.getElementById('saveButton').addEventListener('click', () => {
  const serverUrl = document.getElementById('serverUrl').value;
  chrome.storage.sync.set({ serverUrl: serverUrl }, () => {
    console.log('Server URL saved:', serverUrl);
  });
});

// Load saved server URL when popup opens
chrome.storage.sync.get({ serverUrl: 'http://localhost:3000' }, (items) => {
  document.getElementById('serverUrl').value = items.serverUrl;
});

// Function to request fake review count from content script
function updateFakeReviewCount() {
  chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
    const activeTab = tabs[0];
    if (activeTab) {
      chrome.scripting.executeScript({
        target: {tabId: activeTab.id},
        function: getFakeReviewCountFromContentScript
      }, (injectionResults) => {
        if (injectionResults && injectionResults.length > 0 && injectionResults[0].result !== undefined) {
          const fakeReviewCount = injectionResults[0].result;
          document.getElementById('fakeReviewCount').textContent = fakeReviewCount;
        } else {
          document.getElementById('fakeReviewCount').textContent = "N/A"; // Or some default if count not available
        }
      });
    }
  });
}

// Function that will be executed in content script to get the count
function getFakeReviewCountFromContentScript() {
  return window.fakeReviewCount || 0; // Access the global variable from content.js
}


// Update the fake review count when the popup is opened
document.addEventListener('DOMContentLoaded', updateFakeReviewCount);