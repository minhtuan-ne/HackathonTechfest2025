function fetchText(articleText, sendResponse) {
    fetch('https://hackathontechfest2025.onrender.com/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: articleText
        })
    }).then(response => {
        if (!response.ok) {
            throw new Error("Something wrong with the server");
        }
        return response.json();
    }).then(data => {
        console.log(data);
        sendResponse({
            ok: true,
            label: data.label,
            confidence: data.confidence
        });
    }).catch(err => {
        sendResponse({
            ok: false
        });
    });
}


console.log("Content is injected");
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log("Connected");
    if (request.action === 'automatic-fetch') {
        console.log("work perfectly");
        let articleText = Array.from(document.querySelectorAll('article')).map(article => article.innerText).join(' ');
        if (!articleText.trim()) {
            articleText = document.body.innerText;
        }
        fetchText(articleText, sendResponse);
        return true;
    } else if (request.action === 'manual-fetch') {
        fetchText(request.articleText, sendResponse);
        return true;
    }
});