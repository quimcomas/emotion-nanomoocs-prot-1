//const button = document.getElementById('Categorical');
//const button_1 = document.getElementById('Dimensional');

const MOBILENET_MODEL_PATH = './approach_2/model.json';
//const MOBILENET_MODEL_PATH = './approach_1/model.json';
const DETECTION_MODEL_PATH = './face_detection/';
//const FACE_EXPRESSIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
const FACE_EXPRESSIONS = ["neutral","Happiness","Sadness", "Surprise", "Fear", "Disgust","Anger", "Contempt"]

const IMAGE_SIZE = 224;
//const IMAGE_SIZE = 48;

let mobilenet;
const mobilenetDemo = async () => {
  status('Loading model...');

  mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
  //mobilenet = await tf.loadGraphModel(MOBILENET_MODEL_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  //mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1])).dispose();

  status('');
};

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  status('Detecting emotions...');
  const startTime = performance.now();
  let img = await tf.browser.fromPixels(imgElement,3).toFloat();

  //console.log(img)
  //const img = tf.tensor(imgElement, [IMAGE_SIZE, IMAGE_SIZE, 3])
  const logits = tf.tidy(() => {
    // Reshape to a single-element batch so we can pass it to predict.

    //img = tf.image.resizeBilinear(img, [IMAGE_SIZE, IMAGE_SIZE]);
    //img = img.mean(2);
    //const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    //const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    //const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 1]);

    img = tf.image.resizeBilinear(img, [IMAGE_SIZE, IMAGE_SIZE]);
    const offset = tf.scalar(127.5);
    const normalized = img.sub(offset).div(offset);
    //let im = tf.cast(img,'float32');
    //console.log(im)
    //const batched  = tf.tensor4d(Array.from(im.dataSync()),[1,IMAGE_SIZE,IMAGE_SIZE,3])
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE,3]);

    //console.log(batched)

    // Make a prediction through mobilenet.
    return mobilenet.predict(batched);
  });
  const totalTime = performance.now() - startTime;
  status(`Inference time :  ${Math.floor(totalTime)} ms`);
  return logits
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param values
 */
async function getTopClass(values) {

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });

  return valuesAndIndices[0]
}

const ctx = document.getElementById('chart');
let plot = new Chart(ctx, {
  type: 'line',
  data: {
    datasets: [
      {
        data: 0,
        label: "Neutral",
        borderColor: "#3e95cd",
        backgroundColor: '#3e95cd',
        fill: false
      }, {
        data: 0,
        label: "Happiness",
        borderColor: "#8e5ea2",
        backgroundColor: '#8e5ea2',
        fill: false
      }, {
        data: 0,
        label: "Sadness",
        borderColor: "#3cba9f",
        backgroundColor: '#3cba9f',
        fill: false
      }, {
        data: 0,
        label: "Surprise",
        borderColor: "#e8c3b9",
        backgroundColor: '#e8c3b9',
        fill: false
      }, {
        data: 0,
        label: "Fear",
        borderColor: "#c45850",
        backgroundColor: '#c45850',
        fill: false
      }, {
        data: 0,
        label: "Disgust",
        borderColor: "#a49850",
        backgroundColor: '#a49850',
        fill: false
      }, {
        data: 0,
        label: "Anger",
        borderColor: "#c77950",
        backgroundColor: '#c77950',
        fill: false
      }, {
        data: 0,
        label: "Contempt",
        borderColor: "#B9DEE8",
        backgroundColor: '#B9DEE8',
        fill: false
      }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    title: {
      display: true,
      text: 'Discrete emotion recognition',
      position: 'top'
    },
    legend: {
      display: true,
      position: 'right',
    },
    scales: {
      xAxes: [{
        type: 'time',
        time: {
          unit: 'second'
        }, scaleLabel: {
          display: true,
          labelString: 'Time'
        }
      }],
      yAxes: [{
        ticks: {},
        scaleLabel: {
          display: true,
          labelString: 'Probability'
        }
      },
      ]
    }
  }
});


async function addData(values) {
  plot.data.labels.push(new Date());
  plot.data.datasets[0].data.push(values[0]);
  plot.data.datasets[1].data.push(values[1]);
  plot.data.datasets[2].data.push(values[2]);
  plot.data.datasets[3].data.push(values[3]);
  plot.data.datasets[4].data.push(values[4]);
  plot.data.datasets[5].data.push(values[5]);
  plot.data.datasets[6].data.push(values[6]);
  plot.data.datasets[7].data.push(values[7]);
  plot.update();
}


const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const predictionsElement = document.getElementById('predictions');
let id_name = []

function insert_name(){
  const newArray = document.getElementById("name").value;
  id_name.push(newArray);
  document.getElementById("name").value = "";
}
console.log(id_name)

let array=[];
window.onPlay = async function onPlay() {
  let name_user = id_name[0];
  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');

  const detection = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
  const delay = Date.now();
  array.push(delay);

  //if (detection && id_name.length > 0){
  if (detection && id_name.length > 0 && Date.now()>=array[0]+5000){
    array = []
    const faceCanvases = await faceapi.extractFaces(video, [detection])


    let image64 = faceCanvases[0].toDataURL();
    image64.height = 64
    image64.width = 64
    const prediction = await predict(faceCanvases[0]);

    const values = await prediction.data();

    const topClass = await getTopClass(values)
    //console.log(topClass)


    // TODO(eliot): fix this hack. we should not use private properties

    detection._className = FACE_EXPRESSIONS[topClass.index]
    let emotion_name = detection._className
    detection._classScore = topClass.value
    document.getElementById("emotion").innerHTML = detection._className;
    //console.log(detection)
    drawDetections(video, overlay, detection)
    const current_time= Date.now();
    const visualization = await addData(values)


    const data= {values,emotion_name,name_user,image64};
    const options ={
          method: 'POST',
          headers : {
              'Content-Type': 'application/json',
              'Accept': 'application/json'
          },
          body: JSON.stringify(data)
      };
      //fetch('/api', options).then(res => {console.log(res)});
    const res = await fetch('/api', options);
    const json =  await res.json();
    console.log(json);


  }
  setTimeout(window.onPlay, 300)
  }



function resizeCanvasAndResults(dimensions, canvas, results) {
  const { width, height } = dimensions instanceof HTMLVideoElement
      ? faceapi.getMediaDimensions(dimensions)
      : dimensions
  canvas.width = width
  canvas.height = height

  // resize detections (and landmarks) in case displayed image is smaller than
  // original size
  return faceapi.resizeResults(results, { width, height })
}

function drawDetections(dimensions, canvas, detections) {
  const resizedDetections = resizeCanvasAndResults(dimensions, canvas, detections)
  faceapi.draw.drawDetections(canvas, resizedDetections)
}


async function init() {
  var video = document.getElementById('video');

  await faceapi.loadTinyFaceDetectorModel(DETECTION_MODEL_PATH)
  const stream = await navigator.mediaDevices.getUserMedia({ video: {} })
  video.srcObject = stream

}

window.onload = async function () {
  await mobilenetDemo()
  //console.log(mobilenetDemo())
  init()

}