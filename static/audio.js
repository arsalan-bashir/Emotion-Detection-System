const startButton = document.getElementById('start-button');
const submitButton = document.getElementById('submit-button');
const audioContainer = document.getElementById('audio-container');
const audioPlayer = document.getElementById('audio-player');
const resultContainer = document.getElementById('result-container');
const resultParagraph = document.getElementById('result');
const audioUpload = document.getElementById('audio-upload');
const resetButton = document.getElementById('reset-button');
const mic = document.getElementById('mic');
const infoContainer = document.querySelector('.audio-info-content');

const wisdomLookup = {
    'Happy': 'Happiness is not something ready made. It comes from your own actions. - Dalai Lama',
    'Sad': 'The only way out of the labyrinth of suffering is to forgive. - John Green',
    'Angry': 'For every minute you are angry you lose sixty seconds of happiness. - Ralph Waldo Emerson',
    'Disgust': 'The greatest fear in the world is of the opinions of others. - Swami Vivekananda',
    'Surprise': 'The only thing that should surprise us is that there are still some things that can surprise us. - François de La Rochefoucauld',
    'Fear': 'The only thing we have to fear is fear itself. - Franklin D. Roosevelt',
    'Neutral': 'The present moment is filled with joy and happiness. If you are attentive, you will see it. - Thich Nhat Hanh',
    'Calm': 'Calm mind brings inner strength and self-confidence, so thats very important for good health. - Dalai Lama XIV'
};

let recording = false;
let filename;


startButton.disabled = false;
submitButton.disabled = true;
audioContainer.style.display = 'none';
resultContainer.style.display = 'none';
submitButton.style.cursor = 'not-allowed';

startButton.addEventListener('click', function () {
  if (recording) return;
    audioContainer.style.display = 'none';
    resultContainer.style.display = 'none';
    infoContainer.style.display = 'none';
    mic.src='/static/imgs/mikey.gif';
    mic.style.width = '';

    fetch('/start-recording', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        recording = true;
        filename = data.filename;
        startButton.disabled = true;
        audioContainer.style.display = 'block';
        resultContainer.style.display = 'none';
        submitButton.style.cursor = '';
        submitButton.disabled = true;

        setTimeout(() => {
            submitButton.disabled = false;
            resetButton.style.cursor = '';
            mic.style.width = "50%";
            audioContainer.style.display = 'block';
            audioPlayer.src =  filename;
            mic.src='/static/imgs/mikey.png';
            infoContainer.style.display = 'block';
            startButton.disabled = false;
            mic.style.width = '60%';
        }, 1000);
    })
    .catch(error => {
      alert("Error Occured :: "+ error);
      mic.src='/static/imgs/mikey.png';
    });
});

submitButton.addEventListener('click', function () {
  if (!recording) return;


  fetch('/submit-prediction', { method: 'POST', body: JSON.stringify({ filename }), headers: { 'Content-Type': 'application/json' } })
  .then(response => response.json())
  .then(data => {
      recording = false;
      setTimeout(() => {
            submitButton.disabled = true;
            submitButton.style.cursor = 'not-allowed'
            startButton.style.cursor = '';
            startButton.disabled = false;
            audioContainer.style.display = 'none';
            infoContainer.style.display = 'none';
            resultContainer.style.display = 'block';
            mic.src='/static/imgs/mikey.png';
            let wordsofwisdom = wisdomLookup[data.predicted_emotion] || 'No wisdom available for this emotion';
            resultParagraph.innerHTML = `<table style="width: 80%;height:40%; margin:0 auto;">
                            <tr>
                                <th style="padding: 10px;box-shadow: 0 0 10px rgba(0, 0, 0, .3);border-bottom: 2px solid #9ecd7b;">Predicted Emotion</th>
                                <th style="padding: 10px;box-shadow: 0 0 10px rgba(0, 0, 0, .3);border-bottom: 2px solid #9ecd7b;">${data.predicted_emotion}</th>
                            </tr>
                            <tr style="height:70%;">
                                <td colspan=2 style="text-align:justify; border-bottom:2px solid black;font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif;">${wordsofwisdom}</td>
                            </tr>
                            </table>`;
        }, 500);
    })
  .catch(error => {
      console.error(error);
      startButton.disabled = false;
    });
});


audioUpload.addEventListener('change', () => {
    if (audioUpload.files.length === 0) {
        alert('Please select an audio file.');
        return;
    }

    const formData = new FormData();
    formData.append('audio', audioUpload.files[0]);

    fetch('/upload-recording', {
        method: 'POST',
        body: formData
    }).then(response => response.json())
    .then(data => {
        console.log(data);
        recording = true;
        filename = data.filename;
        startButton.disabled = true;
        startButton.style.cursor = 'not-allowed';
        audioContainer.style.display = 'none';
        resultContainer.style.display = 'none';

        setTimeout(() => {
            submitButton.disabled = false;
            submitButton.style.cursor = '';
            audioContainer.style.display = 'block';
            audioPlayer.src = filename;
        }, 1000);
    });
});

resetButton.addEventListener('click', function () {
    recording = false;
    filename = null;
    audioPlayer.src = '';
    audioContainer.style.display = 'none';
    submitButton.style.cursor = 'not-allowed';
    submitButton.disabled = true;
    location.reload(true); // Reload the page and force cache clearing
});