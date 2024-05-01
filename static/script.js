document.addEventListener('DOMContentLoaded', function () {
    const video = document.getElementById('video-feed');
    const videoContainer = document.getElementById('video-container');
    const startButton = document.getElementById('start-button');
    const stopButton = document.getElementById('stop-button');
    const loadingContainer = document.querySelector('.loading-container'); // Reference to loading bar element
    const resultContainer = document.getElementById('result-container');
    const infoContainer = document.querySelector('.info-content')
    const resultParagraph = document.getElementById('result');
    const wisdomLookup = {
        'Happy': 'Happiness is not something ready made. It comes from your own actions. - Dalai Lama',
        'Sad': 'The only way out of the labyrinth of suffering is to forgive. - John Green',
        'Angry': 'For every minute you are angry you lose sixty seconds of happiness. - Ralph Waldo Emerson',
        'Disgust': 'The greatest fear in the world is of the opinions of others. - Swami Vivekananda',
        'Surprise': 'The only thing that should surprise us is that there are still some things that can surprise us. - François de La Rochefoucauld',
        'Fear': 'The only thing we have to fear is fear itself. - Franklin D. Roosevelt',
        'Neutral': 'The present moment is filled with joy and happiness. If you are attentive, you will see it. - Thich Nhat Hanh'
    };

    let stream;

    // Initially
    startButton.disabled = false;
    stopButton.style.display = 'none';
    loadingContainer.style.display = 'none';
    video.src = '/static/imgs/indexy.gif';
    video.style.width="";
    videoContainer.style.margin = ''
    resultContainer.style.display = 'none'


    startButton.addEventListener('click', function () {
        if (!stream) {
            startVideoFeed();
        }
    });

    stopButton.addEventListener('click', function () {
        stopVideoFeed();
    });

    function startVideoFeed() {
        loadingContainer.style.display = 'block'; // Show loading spinner
        video.style.width="";
        videoContainer.style.margin = '';  // Center video container in page
        infoContainer.style.display = 'none'
        resultContainer.style.display = 'none'
        startButton.disabled = true;

        fetch('/start_feed')
            .then(response => {
                // Hide loading spinner after 10 seconds
                setTimeout(() => {
                    loadingContainer.style.display = 'none';
                }, 5000); // 10 seconds in milliseconds

                if (response.ok) {
                    console.log('Video feed started successfully');
                    stream = true;
                    video.src = '/video_feed';
                    stopButton.disabled = false;
                    startButton.style.cursor = 'not-allowed'
                    stopButton.style.display = '';
                } else {
                    console.error('Failed to start video feed');
                    loadingContainer.style.display = 'none';
                    infoContainer.style.display = '';
                    alert('Camera is not responding! Please Try Again')
                }
            })
            .catch(error => {
                console.error('Error starting video feed:', error);
                // Hide loading spinner on error
                loadingContainer.style.display = 'none';
                startButton.disabled = false;
                alert("Error occured"  + error);
            });
    }


    function stopVideoFeed() {
        fetch('/stop_feed')
            .then(response => {
                if (response.ok) {
                    stream = false;
                    startButton.disabled = false; // Enable the start button
                    startButton.style.cursor = ''
                    stopButton.disabled = true;
                    stopButton.style.display = 'none';
                    loadingContainer.style.display = 'none';
                    // Fetch the results and display them
                    fetch('/get_results')
                        .then(response => response.json())
                        .then(data => {
                            resultContainer.style.display = 'block'; // Show the result container
                            videoContainer.style.margin = "0 0 0 100px";
                            video.style.width="500px";
                            video.src = 'static/imgs/emotions/' + data.most_confident_emotion +'.gif';
                            let wordsofwisdom = wisdomLookup[data.most_confident_emotion] || 'No wisdom available for this emotion';

                            resultParagraph.innerHTML = `<table style="width: 80%;height:80%; margin:0 auto;">
                            <tr>
                                <th style="padding: 1px;box-shadow: 0 0 10px rgba(0, 0, 0, .3);border-bottom: 2px solid #9ecd7b;">Predicted Emotion</th>
                                <th style="padding: 1px;box-shadow: 0 0 10px rgba(0, 0, 0, .3);border-bottom: 2px solid #9ecd7b;">Model Confidence</th>
                            </tr>
                            <tr>
                                <td style="padding: 2px;box-shadow: 0 0 10px rgba(0, 0, 0, .1);text-align:center;">${data.most_confident_emotion}</td>
                                <td style="padding: 2px;box-shadow: 0 0 10px rgba(0, 0, 0, .1);text-align:center;">${data.accumulated_confidence}</td>
                            </tr>
                            <tr style="height:70%;">
                                <td colspan=2 style="text-align:justify; border-top:2px solid white;border-bottom:2px solid black;font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif;">${wordsofwisdom}</td>
                            </tr>
                            </table>`;
                        })
                        .catch(error => {
                            alert( "Error while getting Results :" + error );

                        });
                } else {
                    console.error('Failed to stop video feed');

                }
            })
            .catch(error => {
                console.error('Error stopping video feed:', error);
                loadingContainer.style.display = 'none';
            });
    }
});
