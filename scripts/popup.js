const automaticCheckButton = document.querySelector('.automatic-check');
automaticCheckButton.onclick = function() {
    const result = document.querySelector('.result');
    const loader = document.querySelector('.loader');
    loader.style.display = 'block';
    result.innerHTML = '';

    chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
        if (tabs.length === 0) {
            console.log("invalid tab");
            return;
        }
        chrome.tabs.sendMessage(tabs[0].id, { action: "automatic-fetch" }, response => {
            loader.style.display = 'none';
            if (response && response.ok) {
                if (response.label === 'FAKE') {
                    if (response.confidence >= 0.5)
                        result.innerHTML = `<p style="margin: auto;">From our calculation, we're confidence that this page is ${Math.round(response.confidence * 100)}% 
                                            <span class="fake">FAKE</span></p>`;
                    else {
                        result.innerHTML = `<p style="margin: auto;">From our calculation, we're confidence that this page is ${Math.round((1-response.confidence) * 100)}% 
                                            <span class="real">REAL</span></p>`;
                    }
                } else {
                    if (response.confidence >= 0.5)
                        result.innerHTML = `<p style="margin: auto;">From our calculation, we're confidence that this page is ${Math.round(response.confidence * 100)}% 
                                            <span class="real">REAL</span></p>`;
                    else {
                        result.innerHTML = `<p style="margin: auto;">From our calculation, we're confidence that this page is ${Math.round((1-response.confidence) * 100)}% 
                                            <span class="fake">FAKE</span></p>`;
                    }
                }
            } else {
                result.innerHTML = `
                    <p class="error">Oops, something wrong with this page</p>
                `
            }
        });
    })
};

const manualCheckButton = document.querySelector('.manual-check');
manualCheckButton.onclick = function() {
    const result = document.querySelector('.result');
    const loader = document.querySelector('.loader');
    const form = document.querySelector('form');
    form.style.display = 'block';
    result.innerHTML = '';
    form.onsubmit = function(e) {
        e.preventDefault();
        loader.style.display = 'block';
        const articleText = document.querySelector('textarea').value;
        chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
            if (tabs.length === 0) {
                console.log("invalid tab");
                return;
            }
            chrome.tabs.sendMessage(tabs[0].id, { action: "manual-fetch", articleText: articleText }, response => {
                loader.style.display = 'none';
                form.style.display = 'none';
                document.querySelector('textarea').value = '';
                if (response && response.ok) {
                    if (response.label === 'FAKE') {
                        if (response.confidence >= 0.5)
                            result.innerHTML = `<p style="margin: auto;">From our calculation, we're confidence that this page is ${Math.round(response.confidence * 100)}% 
                                                <span class="fake">FAKE</span></p>`;
                        else {
                            result.innerHTML = `<p style="margin: auto;">From our calculation, we're confidence that this page is ${Math.round((1-response.confidence) * 100)}% 
                                                <span class="real">REAL</span></p>`;
                        }
                    } else {
                        if (response.confidence >= 0.5)
                            result.innerHTML = `<p style="margin: auto;">From our calculation, we're confidence that this page is ${Math.round(response.confidence * 100)}% 
                                                <span class="real">REAL</span></p>`;
                        else {
                            result.innerHTML = `<p style="margin: auto;">From our calculation, we're confidence that this page is ${Math.round((1-response.confidence) * 100)}% 
                                                <span class="fake">FAKE</span></p>`;
                        }
                    }
                } else {
                    result.innerHTML = `
                        <p class="error">Oops, something wrong with this page</p>
                    `
                }
            });
        })    
    }
}