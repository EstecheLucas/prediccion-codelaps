
const datos = [
  { horsepower: 130, mpg: 18 },
  { horsepower: 165, mpg: 15 },
  { horsepower: 150, mpg: 18 },
  { horsepower: 140, mpg: 16 },
  { horsepower: 198, mpg: 15 },
  { horsepower: 220, mpg: 14 },
  { horsepower: 215, mpg: 14 },
  { horsepower: 225, mpg: 14 },
  { horsepower: 190, mpg: 15 },
  { horsepower: 170, mpg: 15 },
  { horsepower: 160, mpg: 14 },
  { horsepower: 150, mpg: 15 },
  { horsepower: 225, mpg: 14 },
  { horsepower: 95, mpg: 24 },
  { horsepower: 95, mpg: 22 },
  { horsepower: 97, mpg: 18 },
  { horsepower: 85, mpg: 21 },
  { horsepower: 88, mpg: 27 },
  { horsepower: 46, mpg: 26 },
  { horsepower: 87, mpg: 25 }
];


let model;
let hpMean, hpStd;
let mpgMean, mpgStd;


function convertirDatos(datos) {
  const hp = datos.map(d => d.horsepower);
  const mpg = datos.map(d => d.mpg);

 
  const hpTensor = tf.tensor1d(hp);
  const mpgTensor = tf.tensor1d(mpg);


  hpMean = hpTensor.mean();
  hpStd = tf.moments(hpTensor).variance.sqrt();

  mpgMean = mpgTensor.mean();
  mpgStd = tf.moments(mpgTensor).variance.sqrt();


  const hpNorm = hpTensor.sub(hpMean).div(hpStd);
  const mpgNorm = mpgTensor.sub(mpgMean).div(mpgStd);

  return { hpNorm, mpgNorm };
}


function crearModelo() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [1], units: 50, activation: 'sigmoid' }));
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({ optimizer: tf.train.adam(0.1), loss: 'meanSquaredError' });
  return model;
}


async function entrenar() {
  document.getElementById('training-status').innerText = "Entrenando modelo...";
  const { hpNorm, mpgNorm } = convertirDatos(datos);

  model = crearModelo();

  await model.fit(hpNorm.reshape([-1, 1]), mpgNorm.reshape([-1, 1]), {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.getElementById('training-status').innerText = `Epoch ${epoch + 1}: pérdida = ${logs.loss.toFixed(4)}`;
      }
    }
  });

  document.getElementById('training-status').innerText = "Entrenamiento completado.";
}


function predecir() {
  if (!model) {
    alert("Primero debes entrenar el modelo.");
    return;
  }
  const hpInput = parseFloat(document.getElementById('input-hp').value);
  if (isNaN(hpInput)) {
    alert("Por favor, ingresa un número válido para horsepower.");
    return;
  }

 
  const hpNorm = tf.tensor1d([hpInput]).sub(hpMean).div(hpStd);


  const mpgNormPred = model.predict(hpNorm.reshape([-1, 1]));
  const mpgPred = mpgNormPred.mul(mpgStd).add(mpgMean);

  mpgPred.data().then(data => {
    document.getElementById('predicted-result').innerText = `Consumo estimado: ${data[0].toFixed(2)} MPG`;
  });
}


document.getElementById('train').addEventListener('click', entrenar);
document.getElementById('predict').addEventListener('click', predecir);
