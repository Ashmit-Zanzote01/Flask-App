// quiz.js
document.addEventListener('DOMContentLoaded', () => {
    const totalQuestions = document.querySelectorAll('.question').length; // Get number of questions
    let currentQuestion = 1;

    const nextButtons = document.querySelectorAll('.next-button');  // Select all next buttons
    const submitButton = document.getElementById('quiz-submit');
    const quizForm = document.getElementById('quiz-form');
    const video = document.getElementById('video');

    // Function to display the current question and hide others
    function displayQuestion(questionNumber) {
        const questions = document.querySelectorAll('.question');
        questions.forEach((question, index) => {
            question.style.display = (index + 1 === questionNumber) ? 'block' : 'none';
        });
    }

    // Attach click listeners to NEXT buttons
    nextButtons.forEach(button => {
        button.addEventListener('click', () => {
            moveToNextQuestion();
        });
    });

    // Function to move to the next question
    function moveToNextQuestion() {
        currentQuestion++;
        if (currentQuestion > totalQuestions) {
            currentQuestion = totalQuestions;
        } else {
            displayQuestion(currentQuestion);  // Show the next question
        }

        if (currentQuestion === totalQuestions) {
            const nextButton = document.querySelector(`#question-${currentQuestion - 1} .next-button`);
            nextButton.style.display = 'none';  // Hide the next button
            submitButton.style.display = 'block';  // Show the submit button
        }
    }

    // Initial setup to display the first question
    displayQuestion(currentQuestion);  // Display the first question on page load

    // Set up the video stream (Make sure you have access to the user's camera in the browser)
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    video.play(); // Start playing after metadata is loaded
                    video.style.display = "block";  // Make the video visible only after the stream starts
                };

                // Capture the image inside the submit handler
                submitButton.addEventListener('click', (event) => {
                    event.preventDefault(); // Prevent default form submission

                    // Capture the video feed as an image
                    const canvas = document.createElement('canvas');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;

                    const context = canvas.getContext('2d');
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);

                    const imageData = canvas.toDataURL('image/jpeg');  // Convert canvas to image

                    const formData = new FormData(quizForm);  // Collect form data
                    formData.append('image', imageData);  // Add captured image to form data

                    // Stop the video stream before sending image (crucial!)
                    video.srcObject.getTracks().forEach(track => track.stop());

                    // Submit the form data using fetch
                    fetch('/submit_quiz', {
                        method: 'POST',
                        body: formData
                    })
                        .then(response => response.json())
                        .then(data => {
                            // Update the result text and display the result container
                            const resultText = document.getElementById('result-text');
                            resultText.textContent = data.message;
                            document.getElementById('result-container').style.display = 'block';

                            // Handle chart destruction before creating a new one
                            const ctx = document.getElementById('emotion-chart').getContext('2d');
                            if (ctx) {  // Ensure the canvas context is available
                                let existingChart = Chart.getChart(ctx);
                                if (existingChart) {
                                    existingChart.destroy();  // Destroy previous chart instance
                                }

                                // Create a new chart with the received data
                                new Chart(ctx, {
                                    type: 'bar',
                                    data: {
                                        labels: data.emotionLabels,  // Labels from the server (e.g., Angry, Happy, etc.)
                                        datasets: [{
                                            label: 'Emotion Prediction',
                                            data: data.emotionValues,  // Emotion probability values from the server
                                            backgroundColor: [
                                                'rgba(255, 99, 132, 0.2)',    // Angry
                                                'rgba(255, 159, 64, 0.2)',     // Disgust
                                                'rgba(255, 205, 86, 0.2)',     // Fear
                                                'rgba(75, 192, 192, 0.2)',    // Happy
                                                'rgba(54, 162, 235, 0.2)',    // Sad
                                                'rgba(153, 102, 255, 0.2)',   // Surprise
                                                'rgba(201, 203, 207, 0.2)'    // Neutral
                                            ],
                                            borderColor: [
                                                'rgb(255, 99, 132)',
                                                'rgb(255, 159, 64)',
                                                'rgb(255, 205, 86)',
                                                'rgb(75, 192, 192)',
                                                'rgb(54, 162, 235)',
                                                'rgb(153, 102, 255)',
                                                'rgb(201, 203, 207)'
                                            ],
                                            borderWidth: 1
                                        }]
                                    },
                                    options: {
                                        scales: {
                                            y: {
                                                beginAtZero: true,
                                                max: 1  // Set max to 1 for emotion probabilities
                                            }
                                        }
                                    }
                                });
                            } else {
                                console.error('Canvas context for chart not found.');
                            }

                            // Display the generated graph
                            const graphImg = document.getElementById('emotion-graph');
                            graphImg.src = data.graphPath;
                        })
                        .catch(error => console.error('Error:', error));
                }); // End of submitButton.addEventListener
            }) // End of then(stream)

            .catch(error => {
                console.error('Error accessing camera:', error);
            });

    } else {
        console.error('getUserMedia not supported in this browser.');
    }
});
