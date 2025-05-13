document.getElementById('prediction-form').addEventListener('submit', function (event) {
    event.preventDefault();
    
    // Collect form data
    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });
    
    // Send data to Flask backend using AJAX (fetch API)
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.prediction !== undefined) {
            const prediction = result.prediction === 1 ? "Fraudulent" : "Non-Fraudulent";
            document.getElementById('prediction-result').innerText = prediction;
        } else {
            document.getElementById('prediction-result').innerText = "Error: " + result.error;
        }
    })
    .catch(error => {
        document.getElementById('prediction-result').innerText = "Error: " + error.message;
    });
});
